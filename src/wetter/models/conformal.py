from __future__ import annotations
import numpy as np


class SplitConformal:
    """Split conformal intervals (symmetric absolute-residual band).

    Assumes exchangeability, which time series violate; mitigate by calibrating
    on a disjoint chronological window. Adaptive conformal (ACI) is a future
    extension.
    """

    def __init__(self) -> None:
        self._residuals: np.ndarray | None = None

    def calibrate(self, y_cal: np.ndarray, point_cal: np.ndarray) -> "SplitConformal":
        self._residuals = np.abs(y_cal - point_cal)
        return self

    def interval(self, point: np.ndarray, alpha: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
        assert self._residuals is not None, "call calibrate first"
        n = self._residuals.size
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        q = float(np.quantile(self._residuals, level, method="higher"))
        return point - q, point + q


class ConformalizedQuantile:
    """Conformalized Quantile Regression (Romano et al., 2019).

    Wraps a quantile model's lower/upper predictions so the interval gains a
    finite-sample coverage guarantee while keeping its input-dependent width.
    Fixes the under-coverage of raw quantile-GBM. Conformity score:
    E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i)); the interval is widened
    (or shrunk) by the (1 - alpha) quantile of E on a calibration window.
    """

    def __init__(self) -> None:
        self._scores: np.ndarray | None = None

    def calibrate(
        self, y_cal: np.ndarray, qlo_cal: np.ndarray, qhi_cal: np.ndarray
    ) -> "ConformalizedQuantile":
        self._scores = np.maximum(qlo_cal - y_cal, y_cal - qhi_cal)
        return self

    def interval(
        self, qlo: np.ndarray, qhi: np.ndarray, alpha: float = 0.2
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self._scores is not None, "call calibrate first"
        n = self._scores.size
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        q = float(np.quantile(self._scores, level, method="higher"))
        return qlo - q, qhi + q
