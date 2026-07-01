from __future__ import annotations
import numpy as np


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Brier score for binary occurrence: mean squared error of the probability.
    Lower is better; 0 = perfect, 0.25 = always-say-50%."""
    return float(np.mean((p - y) ** 2))


def brier_skill_score(y: np.ndarray, p: np.ndarray, base_rate: float | None = None) -> float:
    """1 - Brier/Brier(climatology). >0 = better than always predicting the base rate."""
    base_rate = float(np.mean(y)) if base_rate is None else base_rate
    ref = brier(y, np.full(np.shape(p), base_rate, dtype=float))
    return 1.0 - brier(y, p) / ref if ref > 0 else 0.0


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    """Reliability diagram data: for each probability bin, the mean predicted
    probability vs the observed frequency (calibrated => they match)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    mean_pred = np.array([p[b == k].mean() if np.any(b == k) else np.nan for k in range(n_bins)])
    obs_freq = np.array([y[b == k].mean() if np.any(b == k) else np.nan for k in range(n_bins)])
    counts = np.array([int(np.sum(b == k)) for k in range(n_bins)])
    return mean_pred, obs_freq, counts
