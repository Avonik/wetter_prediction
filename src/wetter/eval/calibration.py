from __future__ import annotations
import numpy as np
from scipy.stats import norm


def pit_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.clip(sigma, 1e-9, None)
    return norm.cdf((y - mu) / sigma)


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def sharpness(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(hi - lo))


def reliability_from_pit(pit: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(pit, bins=n_bins, range=(0.0, 1.0), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts
