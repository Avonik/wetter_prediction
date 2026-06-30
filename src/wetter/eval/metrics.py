from __future__ import annotations
import numpy as np
from scipy.stats import norm


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - y)))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def bias(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(pred - y))


def skill_score(metric_model: float, metric_ref: float) -> float:
    return 1.0 - metric_model / metric_ref


def crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.clip(sigma, 1e-9, None)
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_from_quantiles(
    y: np.ndarray, quantile_levels: np.ndarray, quantile_preds: np.ndarray
) -> np.ndarray:
    # quantile_preds shape: (n_samples, n_levels); pinball loss averaged over levels
    y = y[:, None]
    diff = y - quantile_preds
    pinball = np.maximum(quantile_levels * diff, (quantile_levels - 1) * diff)
    return 2.0 * pinball.mean(axis=1)
