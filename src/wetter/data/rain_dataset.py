from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from wetter import config
from wetter.data import features, observations, single_runs


def _precip_agg(wide: pl.DataFrame, pcols: list[str]) -> pl.DataFrame:
    return wide.with_columns(
        pl.mean_horizontal(pcols).alias("precip_mean"),
        pl.max_horizontal(pcols).alias("precip_max"),
        (
            pl.sum_horizontal([(pl.col(c) >= RAIN_THRESH).cast(pl.Float64).fill_null(0.0) for c in pcols])
            / pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Float64) for c in pcols])
        ).alias("precip_prob"),
    )


def _pop_agg(wide: pl.DataFrame, pcols: list[str]) -> pl.DataFrame:
    return wide.with_columns(
        pl.mean_horizontal(pcols).alias("pop_mean"),
        pl.max_horizontal(pcols).alias("pop_max"),
    )

RAIN_THRESH = 0.1  # mm/h counts as "rain"
# full week: skill fades with lead, but a *calibrated* model then honestly reports
# low (near-climatology) probabilities instead of over-confident model agreement.
RAIN_MAX_LEAD = 168


def build_rain_canonical(obs: pl.DataFrame, runs: pl.DataFrame) -> pl.DataFrame:
    """One row per (valid_time, lead): each model's precip forecast + agreement +
    cloud/humidity + time features, target = observed precipitation / rain-yes-no."""
    present = [m for m in config.MODELS if m in runs["model"].unique().to_list()]

    pr = runs.filter(pl.col("variable") == "precip")
    wide = pr.pivot(values="value", index=["valid_time", "lead_time_h"], on="model")
    wide = wide.rename({m: f"precip_{m}" for m in present})
    pcols = [f"precip_{m}" for m in present]
    wide = _precip_agg(wide, pcols)

    aux = runs.filter(pl.col("variable").is_in(["cloud", "rh"]))
    aux_agg = aux.group_by(["valid_time", "lead_time_h", "variable"]).agg(
        pl.col("value").mean().alias("v")
    )
    aux_wide = aux_agg.pivot(values="v", index=["valid_time", "lead_time_h"], on="variable")
    aux_wide = aux_wide.rename(
        {c: f"{c}_mean" for c in aux_wide.columns if c not in ("valid_time", "lead_time_h")}
    )

    canon = wide.join(aux_wide, on=["valid_time", "lead_time_h"], how="left")
    canon = canon.with_columns(
        (pl.col("valid_time") - pl.duration(hours=pl.col("lead_time_h"))).alias("issue_time")
    )
    canon = canon.join(obs.select(["valid_time", "p_obs"]), on="valid_time", how="left")
    # rain persistence: how much did it rain at the moment the run was issued? Strong
    # short-lead signal ("it's raining now -> likely still raining soon") that the NWP
    # forecast alone misses. Left null where the station has no obs at that hour.
    canon = canon.join(
        obs.select(
            pl.col("valid_time").alias("issue_time"),
            pl.col("p_obs").alias("p_obs_at_issue"),
        ),
        on="issue_time",
        how="left",
    )
    canon = features.add_time_features(canon)
    canon = canon.with_columns((pl.col("p_obs") >= RAIN_THRESH).cast(pl.Int8).alias("rain_occ"))
    return (
        canon.drop_nulls("p_obs")
        .filter(pl.col("lead_time_h") <= RAIN_MAX_LEAD)
        .sort(["valid_time", "lead_time_h"])
    )


def build_live_rain_rows(
    current_time: datetime,
    live_fc_long: pl.DataFrame,
    leads: list[int],
    current_precip: float | None = None,
) -> pl.DataFrame:
    """Rain features (per-model precip + agreement + cloud/humidity + time + rain
    persistence) for a forecast issued now, one row per lead — mirrors
    build_rain_canonical. `current_precip` = observed rain now (mm/h), the persistence
    feature `p_obs_at_issue`; None when the station reports nothing."""
    issue = current_time.replace(minute=0, second=0, microsecond=0)
    rows_meta = [(lead, issue + timedelta(hours=lead)) for lead in leads]
    target_times = [vt for _, vt in rows_meta]

    lf = live_fc_long.filter(pl.col("valid_time").is_in(target_times))
    pr = lf.filter(pl.col("variable") == "precip")
    present = [m for m in config.MODELS if m in pr["model"].unique().to_list()]
    wide = pr.pivot(values="value", index="valid_time", on="model")
    wide = wide.rename({m: f"precip_{m}" for m in present})
    wide = _precip_agg(wide, [f"precip_{m}" for m in present])

    pop = lf.filter(pl.col("variable") == "pop")
    if pop.height > 0:
        pop_present = [m for m in config.MODELS if m in pop["model"].unique().to_list()]
        pop_wide = pop.pivot(values="value", index="valid_time", on="model")
        pop_wide = pop_wide.rename({m: f"pop_{m}" for m in pop_present})
        pop_wide = _pop_agg(pop_wide, [f"pop_{m}" for m in pop_present])
        wide = wide.join(pop_wide, on="valid_time", how="left")

    aux = lf.filter(pl.col("variable").is_in(["cloud", "rh"]))
    aux_agg = aux.group_by(["valid_time", "variable"]).agg(pl.col("value").mean().alias("v"))
    aux_wide = aux_agg.pivot(values="v", index="valid_time", on="variable")
    aux_wide = aux_wide.rename({c: f"{c}_mean" for c in aux_wide.columns if c != "valid_time"})

    lead_map = pl.DataFrame(
        {"valid_time": target_times, "lead_time_h": [lead for lead, _ in rows_meta]}
    ).with_columns(
        pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
        pl.col("lead_time_h").cast(pl.Int32),
    )
    df = lead_map.join(wide, on="valid_time", how="left").join(aux_wide, on="valid_time", how="left")
    df = df.with_columns(
        pl.lit(None if current_precip is None else float(current_precip), dtype=pl.Float64).alias(
            "p_obs_at_issue"
        )
    )
    return features.add_time_features(df).sort("lead_time_h")


def build_rain(
    *, cache_dir: Path | None = None, cached_only: bool = False, force_obs: bool = False
) -> Path:
    today = datetime.now(timezone.utc).date().isoformat()
    obs = observations.fetch_observations(config.FORECAST_START, today, force=force_obs)
    runs = single_runs.fetch_runs(config.SINGLE_RUNS_START, today, cached_only=cached_only)
    canon = build_rain_canonical(obs, runs)
    out = (cache_dir or config.CURATED_DIR) / "canonical_rain.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    canon.write_parquet(out)
    return out
