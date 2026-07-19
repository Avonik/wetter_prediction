from __future__ import annotations
from datetime import timedelta
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
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

_CALIBRATION_EPS = 0.005


class BetaCalibrator:
    """Smooth beta calibration for sparse probability ranges."""

    def __init__(self, eps: float = _CALIBRATION_EPS) -> None:
        self.eps = eps
        self.model = LogisticRegression(C=1e6, max_iter=1000)

    def _features(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), self.eps, 1.0 - self.eps)
        return np.column_stack((np.log(p), np.log1p(-p)))

    def fit(self, p: np.ndarray, y: np.ndarray) -> "BetaCalibrator":
        self.model.fit(self._features(p), np.asarray(y, dtype=int))
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        calibrated = self.model.predict_proba(self._features(p))[:, 1]
        return np.clip(calibrated, self.eps, 1.0 - self.eps)


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


def _enough_calibration_cases(calib: pl.DataFrame, threshold: float) -> bool:
    """Require independent wet/dry hours, not only repeated forecast rows."""
    wet = calib.filter(pl.col("p_obs") >= threshold)
    dry = calib.filter(pl.col("p_obs") < threshold)
    if "valid_time" in calib.columns:
        return wet["valid_time"].n_unique() >= 10 and dry["valid_time"].n_unique() >= 10
    return wet.height >= 20 and dry.height >= 20


def fit_calibration(
    models,
    calib: pl.DataFrame,
    feats: list[str],
    *,
    method: str = "beta",
) -> dict[float, BetaCalibrator | IsotonicRegression]:
    """Fit threshold-wise calibration on a held-out chronological window."""
    if method not in {"beta", "isotonic"}:
        raise ValueError(f"Unknown rain calibration method: {method}")
    cals: dict[float, BetaCalibrator | IsotonicRegression] = {}
    for thr, p in predict_exceedance(models, calib, feats).items():
        if not _enough_calibration_cases(calib, thr):
            continue
        y = (calib["p_obs"] >= thr).cast(pl.Int8).to_numpy()
        calibrator = (
            BetaCalibrator().fit(p, y)
            if method == "beta"
            else IsotonicRegression(out_of_bounds="clip").fit(p, y)
        )
        cals[thr] = calibrator
    return cals


def apply_calibration(cals, probs: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    return {thr: (cals[thr].predict(p) if thr in cals else p) for thr, p in probs.items()}


def train_rain_engine(
    canonical: pl.DataFrame, *, cal_window_days: int = 30, calibration: str = "beta"
) -> dict:
    """Fit classifiers and calibrators with a leak-free chronological holdout."""
    if cal_window_days <= 0:
        raise ValueError("cal_window_days must be positive")
    latest = canonical["valid_time"].max()
    calibration_start = latest - timedelta(days=cal_window_days)
    fit_df = canonical.filter(pl.col("valid_time") < calibration_start)
    calib_df = canonical.filter(pl.col("valid_time") >= calibration_start)
    if fit_df.height == 0 or calib_df.height == 0:
        raise ValueError("Not enough chronological data for rain calibration")

    feats = feature_columns(fit_df)
    models = train_exceedance(fit_df, feats)
    cals = fit_calibration(models, calib_df, feats, method=calibration)
    return {
        "features": feats,
        "models": models,
        "cals": cals,
        "calibration": calibration,
        "calibration_start": calibration_start,
        "fit_through": fit_df["valid_time"].max(),
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
    calibrated = apply_calibration(artifact.get("cals", {}), raw)
    # Exceedance probabilities must be nested: P(>= 5 mm) cannot exceed P(>= 1 mm).
    previous = None
    for threshold in sorted(calibrated):
        p = np.clip(calibrated[threshold], 0.0, 1.0)
        if previous is not None:
            p = np.minimum(p, previous)
        calibrated[threshold] = p
        previous = p
    return calibrated

