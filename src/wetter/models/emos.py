from __future__ import annotations
import numpy as np
import polars as pl
from scipy.optimize import minimize

from wetter.eval.metrics import crps_gaussian


class EMOS:
    def __init__(self) -> None:
        self.params = np.array([0.0, 1.0, 1.0, 0.0])  # a, b, c, d

    @staticmethod
    def _mu_sigma(p, m, s):
        a, b, c, d = p
        mu = a + b * m
        sigma = np.sqrt(c * c + d * d * s)
        return mu, sigma

    def fit(self, train: pl.DataFrame) -> "EMOS":
        m = train["t_mean"].to_numpy()
        s = train["t_spread"].fill_null(0.0).to_numpy()
        y = train["t_obs"].to_numpy()

        def loss(p):
            mu, sigma = self._mu_sigma(p, m, s)
            return float(np.mean(crps_gaussian(y, mu, sigma)))

        res = minimize(
            loss, self.params, method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-8},
        )
        self.params = res.x
        return self

    def predict(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        m = df["t_mean"].to_numpy()
        s = df["t_spread"].fill_null(0.0).to_numpy()
        return self._mu_sigma(self.params, m, s)
