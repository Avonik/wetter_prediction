from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score

from wetter.eval import rain_metrics
from wetter.models import rain


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _scores(y: np.ndarray, p: np.ndarray, base_rate: float) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 0.005, 0.995)
    return {
        "brier": rain_metrics.brier(y, p),
        "brier_skill": rain_metrics.brier_skill_score(y, p, base_rate),
        "log_loss": float(log_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "mean_probability": float(np.mean(p)),
        "observed_frequency": float(np.mean(y)),
    }


def evaluate_options(
    canonical: pl.DataFrame,
    *,
    train_end: str,
    cal_end: str,
    threshold: float = 0.1,
) -> dict:
    """Compare raw, isotonic, and beta probabilities on a future holdout.

    Classifiers see only rows before ``train_end``. Calibrators see the following
    window up to ``cal_end``. Every reported metric is from rows after ``cal_end``.
    """
    train_cut = _timestamp(train_end)
    cal_cut = _timestamp(cal_end)
    train = canonical.filter(pl.col("valid_time") < train_cut)
    calib = canonical.filter(
        (pl.col("valid_time") >= train_cut) & (pl.col("valid_time") < cal_cut)
    )
    test = canonical.filter(pl.col("valid_time") >= cal_cut)
    if min(train.height, calib.height, test.height) == 0:
        raise ValueError("Rain backtest requires non-empty train, calibration, and test windows")

    feats = rain.feature_columns(train)
    models = rain.train_exceedance(train, feats, thresholds=(threshold,))
    if threshold not in models:
        raise ValueError(f"Too few events to train threshold {threshold}")
    raw_test = rain.predict_exceedance(models, test, feats)[threshold]
    y = (test["p_obs"] >= threshold).cast(pl.Int8).to_numpy()
    base_rate = float((train["p_obs"] >= threshold).mean())

    options = {"raw": _scores(y, raw_test, base_rate)}
    for method in ("isotonic", "beta"):
        calibrators = rain.fit_calibration(models, calib, feats, method=method)
        calibrated = rain.apply_calibration(calibrators, {threshold: raw_test})[threshold]
        options[method] = _scores(y, calibrated, base_rate)

    return {
        "threshold_mm": threshold,
        "rows": {"train": train.height, "calibration": calib.height, "test": test.height},
        "train_end": train_cut.isoformat(),
        "cal_end": cal_cut.isoformat(),
        "options": options,
    }


def format_options(result: dict) -> str:
    lines = [
        f"Rain threshold: >= {result['threshold_mm']:.1f} mm/h",
        "option       brier     BSS      log-loss  AUC    mean-p  observed",
    ]
    for name, values in result["options"].items():
        lines.append(
            f"{name:<12} {values['brier']:.4f}  {values['brier_skill']:+.3f}  "
            f"{values['log_loss']:.4f}    {values['auc']:.3f}  "
            f"{values['mean_probability']:.3f}   {values['observed_frequency']:.3f}"
        )
    return "\n".join(lines)
