from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from wetter import config
from wetter.data import features
from wetter.models import blend_gbm
from wetter.models.conformal import ConformalizedQuantile

ENGINE_PATH = config.DATA_DIR / "models" / "engine.joblib"
HOURLY_ENGINE_PATH = config.DATA_DIR / "models" / "engine_hourly.joblib"


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _recent_bias_by_lead(canonical: pl.DataFrame, models: list[str]) -> dict[str, dict[int, float]]:
    """Latest historical recent-bias per (model, lead) — used as the live feature value."""
    out: dict[str, dict[int, float]] = {}
    for m in models:
        col = f"recent_bias_{m}"
        if col not in canonical.columns:
            continue
        d = (
            canonical.drop_nulls(col)
            .sort("valid_time")
            .group_by("lead_time_h")
            .agg(pl.col(col).last().alias("v"))
        )
        out[m] = {int(r["lead_time_h"]): float(r["v"]) for r in d.iter_rows(named=True)}
    return out


def train_engine(
    canonical: pl.DataFrame, *, tune_end: str, cal_window_days: int = 120
) -> dict:
    """Tune GBM params on a chronological fold, fit the final point + quantile models,
    calibrate a per-lead CQR adjustment on a recent holdout, and bundle everything
    needed to make live forecasts."""
    feats = blend_gbm.feature_columns(canonical)
    models = [m for m in config.MODELS if f"t_{m}" in canonical.columns]

    tr = canonical.filter(pl.col("valid_time") < _ts(tune_end))
    va = canonical.filter(pl.col("valid_time") >= _ts(tune_end))
    params = blend_gbm.tune_hyperparams(tr, va, feats) if (tr.height and va.height) else dict(
        blend_gbm._PARAMS
    )

    cutoff = canonical["valid_time"].max() - timedelta(days=cal_window_days)
    fit_df = canonical.filter(pl.col("valid_time") < cutoff)
    cal_df = canonical.filter(pl.col("valid_time") >= cutoff)
    if fit_df.height == 0:
        fit_df, cal_df = canonical, canonical

    point = blend_gbm.train_point(fit_df, feats, params)
    qmodels = blend_gbm.train_quantiles(fit_df, feats, params=params)

    cqr_q: dict[int, float] = {}
    for lead in sorted(cal_df["lead_time_h"].unique().to_list()):
        sub = cal_df.filter(pl.col("lead_time_h") == lead)
        if sub.height == 0:
            continue
        qc = blend_gbm.predict_quantiles(qmodels, sub, feats)
        cqr = ConformalizedQuantile().calibrate(sub["t_obs"].to_numpy(), qc[0.1], qc[0.9])
        _, hi = cqr.interval(np.zeros(1), np.zeros(1), alpha=0.2)
        cqr_q[int(lead)] = float(hi[0])

    clim = canonical.select(["month", "hour", "t_clim"]).unique(subset=["month", "hour"])
    elev = (
        float(canonical["station_vs_grid_elev_diff"][0])
        if "station_vs_grid_elev_diff" in canonical.columns
        else 0.0
    )

    return {
        "params": params,
        "features": feats,
        "models": models,
        "leads": sorted(int(x) for x in canonical["lead_time_h"].unique().to_list()),
        "point": point,
        "quantiles": qmodels,
        "cqr_q": cqr_q,
        "climatology": clim,
        "recent_bias": _recent_bias_by_lead(canonical, models),
        "elev_diff": elev,
        "trained_through": canonical["valid_time"].max(),
    }


def save_engine(artifact: dict, path: Path | str = ENGINE_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_engine(path: Path | str = ENGINE_PATH) -> dict:
    return joblib.load(path)


def predict(artifact: dict, live_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (point, lo80, hi80) for rows of `live_df` (must carry the engine's features).

    The point is the median quantile (q0.5), not the separate point model, so it is
    always consistent with the interval; the interval is then clamped to bracket it
    (per-sample quantile crossing can otherwise put the point outside its own band)."""
    feats = artifact["features"]
    q = blend_gbm.predict_quantiles(artifact["quantiles"], live_df, feats)
    leads = live_df["lead_time_h"].to_numpy()
    adj = np.array([artifact["cqr_q"].get(int(lead), 0.0) for lead in leads])
    point = q[0.5]
    lo = np.minimum(q[0.1] - adj, point)
    hi = np.maximum(q[0.9] + adj, point)
    return point, lo, hi


def build_live_rows(
    artifact: dict,
    current_time: datetime,
    current_temp: float,
    live_fc_long: pl.DataFrame,
    leads: list[int] | None = None,
) -> pl.DataFrame:
    """Assemble one feature row per lead for a forecast issued now, mirroring the
    training canonical so the engine can be applied directly. `leads` defaults to
    the leads the engine was trained on (hourly 1..48 or daily 24..168)."""
    leads = leads or artifact.get("leads") or config.LEAD_TIMES_H
    issue = current_time.replace(minute=0, second=0, microsecond=0)
    rows_meta = [(lead, issue + timedelta(hours=lead)) for lead in leads]
    target_times = [vt for _, vt in rows_meta]

    lf = live_fc_long.filter(pl.col("valid_time").is_in(target_times))
    temp = lf.filter(pl.col("variable") == "t")
    wide = temp.pivot(values="value", index="valid_time", on="model")
    wide = wide.rename({m: f"t_{m}" for m in temp["model"].unique().to_list()})

    aux = lf.filter(pl.col("variable") != "t")
    if aux.height > 0:
        aux_agg = aux.group_by(["valid_time", "variable"]).agg(pl.col("value").mean().alias("v"))
        aux_wide = aux_agg.pivot(values="v", index="valid_time", on="variable")
        aux_wide = aux_wide.rename(
            {c: f"{c}_mean" for c in aux_wide.columns if c != "valid_time"}
        )
        wide = wide.join(aux_wide, on="valid_time", how="left")

    lead_map = pl.DataFrame(
        {"valid_time": target_times, "lead_time_h": [lead for lead, _ in rows_meta]}
    ).with_columns(
        pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
        pl.col("lead_time_h").cast(pl.Int32),
    )
    df = lead_map.join(wide, on="valid_time", how="left").with_columns(
        pl.lit(issue).cast(pl.Datetime("us", "UTC")).alias("issue_time"),
        pl.lit(float(current_temp)).alias("t_obs_at_issue"),
        pl.lit(float(artifact["elev_diff"])).alias("station_vs_grid_elev_diff"),
    )
    df = features.add_time_features(df)
    df = features.add_multimodel_features(df)
    clim = artifact["climatology"].with_columns(
        pl.col("month").cast(pl.Int32), pl.col("hour").cast(pl.Int32)
    )
    df = df.join(clim, on=["month", "hour"], how="left")
    df = features.add_interaction_features(df)
    for m, per_lead in artifact["recent_bias"].items():
        df = df.with_columns(
            pl.col("lead_time_h")
            .map_elements(lambda lead: per_lead.get(int(lead)), return_dtype=pl.Float64)
            .alias(f"recent_bias_{m}")
        )
    return df.sort("lead_time_h")


def forecast(
    artifact: dict,
    current_time: datetime,
    current_temp: float,
    live_fc_long: pl.DataFrame,
    leads: list[int] | None = None,
) -> pl.DataFrame:
    """Live forecast table: one row per lead with point + 80% interval."""
    rows = build_live_rows(artifact, current_time, current_temp, live_fc_long, leads=leads)
    point, lo, hi = predict(artifact, rows)
    return rows.select(["lead_time_h", "valid_time"]).with_columns(
        pl.Series("point", point), pl.Series("lo80", lo), pl.Series("hi80", hi)
    )
