from __future__ import annotations
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from wetter import config

RAIN_ENGINE_PATH = config.DATA_DIR / "models" / "rain_engine.joblib"

# exceedance thresholds (mm/h): 0.1 = "rain at all", 1 = notable, 5 = heavy
THRESHOLDS = (0.1, 1.0, 5.0)

# Predictors: physics only (per-model precip + agreement + cloud/humidity) plus rain
# PERSISTENCE (p_obs_at_issue = was it raining when the run was issued?). We deliberately
# EXCLUDE hour/doy/lead_time. Our runs are still ~80% 00-UTC, so lead time and hour-of-day
# stay heavily correlated; letting the model see them makes it memorise date/hour instead of
# reading the precip signal — a backtest with them in scored WORSE than climatology
# (BSS -0.05, 76% of importance on time). Physics + persistence generalises to any issue hour
# and, crucially, lets "it's raining right now" raise the near-term probability.
_FEATURES = [
    "precip_icon_d2", "precip_icon_eu", "precip_icon_global",
    "precip_gfs_seamless", "precip_ecmwf_ifs025",
    "precip_mean", "precip_max", "precip_prob",
    "cloud_mean", "rh_mean",
    "p_obs_at_issue",
]

_PARAMS = dict(
    n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=30,
    subsample=0.8, colsample_bytree=0.8, verbosity=-1,
)


def feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in _FEATURES if c in df.columns]


def _X(df: pl.DataFrame, feats: list[str]):
    # Tolerate a df that is missing a trained feature (e.g. a live row schema slightly
    # behind the engine): add it as null so LightGBM treats it as "missing" instead of
    # crashing the whole page. Keeps predict robust to feature drift.
    missing = [f for f in feats if f not in df.columns]
    if missing:
        df = df.with_columns([pl.lit(None, dtype=pl.Float64).alias(f) for f in missing])
    return df.select(feats).to_pandas()


def train_exceedance(
    train: pl.DataFrame, feats: list[str], thresholds=THRESHOLDS, params: dict | None = None
) -> dict[float, lgb.LGBMClassifier]:
    """One binary classifier per threshold: P(precip >= thr). Skips thresholds with
    too few positive examples (e.g. heavy rain is rare)."""
    X = _X(train, feats)
    models: dict[float, lgb.LGBMClassifier] = {}
    for thr in thresholds:
        y = (train["p_obs"] >= thr).cast(pl.Int8).to_numpy()
        if int(y.sum()) < 20:
            continue
        m = lgb.LGBMClassifier(objective="binary", **(params or _PARAMS))
        m.fit(X, y)
        models[thr] = m
    return models


def predict_exceedance(models, df: pl.DataFrame, feats: list[str]) -> dict[float, np.ndarray]:
    X = _X(df, feats)
    return {thr: m.predict_proba(X)[:, 1] for thr, m in models.items()}


def fit_calibration(models, calib: pl.DataFrame, feats: list[str]) -> dict[float, IsotonicRegression]:
    """Isotonic calibration per threshold on a held-out window (our honesty edge)."""
    cals: dict[float, IsotonicRegression] = {}
    for thr, p in predict_exceedance(models, calib, feats).items():
        y = (calib["p_obs"] >= thr).cast(pl.Int8).to_numpy()
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p, y)
        cals[thr] = iso
    return cals


def apply_calibration(cals, probs: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    return {thr: (cals[thr].predict(p) if thr in cals else p) for thr, p in probs.items()}


def train_rain_engine(canonical: pl.DataFrame) -> dict:
    """Fit the exceedance classifiers on all data. LightGBM's binary probabilities are
    already well-calibrated here (isotonic on a short recent window only capped them),
    so no extra calibration step — `cals` is empty and predictions pass through raw."""
    feats = feature_columns(canonical)
    models = train_exceedance(canonical, feats)
    return {
        "features": feats,
        "models": models,
        "cals": {},
        "thresholds": sorted(models.keys()),
        "base_rate": float(canonical["rain_occ"].mean()),
        "trained_through": canonical["valid_time"].max(),
    }


def save_rain_engine(artifact: dict, path: Path | str = RAIN_ENGINE_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_rain_engine(path: Path | str = RAIN_ENGINE_PATH) -> dict:
    return joblib.load(path)


def predict_rain(artifact: dict, df: pl.DataFrame) -> dict[float, np.ndarray]:
    """Calibrated exceedance probabilities per threshold for `df` (rain features)."""
    raw = predict_exceedance(artifact["models"], df, artifact["features"])
    return apply_calibration(artifact["cals"], raw)

