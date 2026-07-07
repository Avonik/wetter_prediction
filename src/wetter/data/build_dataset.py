from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from wetter import config
from wetter.data import build_inputs, climatology, features, observations, single_runs

_LAPSE_FREE_AUX = ["rh", "cloud", "wind", "pmsl", "rad"]


def assemble_wide(forecasts: pl.DataFrame) -> pl.DataFrame:
    """Long forecasts -> wide: per-model temperature `t_<model>` + cross-model
    auxiliary means `<var>_mean` (rh/cloud/wind/pmsl/rad)."""
    temp = forecasts.filter(pl.col("variable") == "t")
    wide = temp.pivot(values="value", index=["valid_time", "lead_time_h"], on="model")
    wide = wide.rename({m: f"t_{m}" for m in temp["model"].unique().to_list()})

    aux = forecasts.filter(pl.col("variable") != "t")
    if aux.height > 0:
        aux_agg = aux.group_by(["valid_time", "lead_time_h", "variable"]).agg(
            pl.col("value").mean().alias("v")
        )
        aux_wide = aux_agg.pivot(values="v", index=["valid_time", "lead_time_h"], on="variable")
        aux_wide = aux_wide.rename(
            {c: f"{c}_mean" for c in aux_wide.columns if c not in ("valid_time", "lead_time_h")}
        )
        wide = wide.join(aux_wide, on=["valid_time", "lead_time_h"], how="left")
    return wide


def mean_grid_elevation(forecasts: pl.DataFrame) -> float:
    temp = forecasts.filter(pl.col("variable") == "t")
    per_model = temp.group_by("model").agg(pl.col("grid_elev").first())
    return float(per_model["grid_elev"].mean()) if per_model.height else config.STATION_ELEV_M


def build_canonical(
    obs: pl.DataFrame, forecasts: pl.DataFrame, climatology: pl.DataFrame
) -> pl.DataFrame:
    # the temperature model only needs t_obs; obs also carries p_obs (rain target),
    # which would otherwise collide across the several obs joins below.
    obs = obs.select(["valid_time", "t_obs"])
    wide = assemble_wide(forecasts)
    canon = wide.with_columns(
        (pl.col("valid_time") - pl.duration(hours=pl.col("lead_time_h"))).alias("issue_time")
    )
    # target
    canon = canon.join(obs, on="valid_time", how="left")
    # persistence feature: obs at issue_time
    canon = canon.join(
        obs.rename({"valid_time": "issue_time", "t_obs": "t_obs_at_issue"}),
        on="issue_time",
        how="left",
    )
    # features
    canon = features.add_time_features(canon)
    canon = features.add_multimodel_features(canon)
    clim = climatology.with_columns(
        pl.col("month").cast(pl.Int32), pl.col("hour").cast(pl.Int32)
    )
    canon = canon.join(clim, on=["month", "hour"], how="left")
    canon = features.add_interaction_features(canon)
    for m in config.MODELS:
        if f"t_{m}" in canon.columns:
            canon = features.add_recent_bias(canon, obs, m)
    # real station-vs-grid elevation difference (constant per location, but no longer
    # a placeholder). Near-zero ML value at a single site; see Tier-2 lapse work.
    elev_diff = config.STATION_ELEV_M - mean_grid_elevation(forecasts)
    canon = canon.with_columns(pl.lit(elev_diff).alias("station_vs_grid_elev_diff"))
    return canon.drop_nulls("t_obs").sort(["valid_time", "lead_time_h"])


def build(*, cache_dir: Path | None = None) -> Path:
    obs, forecasts, clim = build_inputs.load_all()
    canon = build_canonical(obs, forecasts, clim)
    out = (cache_dir or config.CURATED_DIR) / "canonical.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    canon.write_parquet(out)
    return out


def build_hourly(
    *,
    cache_dir: Path | None = None,
    start: str | None = None,
    end: str | None = None,
    run_hours: tuple[int, ...] | None = None,
    max_lead_h: int | None = None,
    cached_only: bool = False,
    force: bool = False,
) -> Path:
    """Build the hourly-lead canonical table from Single Runs forecasts. Reuses
    build_canonical: a single-run row carries the same (valid_time, lead_time_h,
    model, variable, value) schema, and issue_time = valid_time - lead = the run time."""
    today = datetime.now(timezone.utc).date().isoformat()
    start = start or config.SINGLE_RUNS_START
    end = end or today
    max_lead_h = max_lead_h or config.HOURLY_MAX_LEAD_H

    obs = observations.fetch_observations(config.FORECAST_START, today)
    runs = single_runs.fetch_runs(
        start, end, run_hours=run_hours, max_lead_h=max_lead_h,
        cached_only=cached_only, force=force,
    )
    clim = climatology.compute_climatology(climatology.fetch_era5(config.OBS_START, today))
    canon = build_canonical(obs, runs.drop("run_time"), clim)

    out = (cache_dir or config.CURATED_DIR) / "canonical_hourly.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    canon.write_parquet(out)
    return out
