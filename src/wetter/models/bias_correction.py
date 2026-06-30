from __future__ import annotations
import numpy as np
import polars as pl

_KEYS = ["lead_time_h", "hour", "month"]


class BiasCorrector:
    def __init__(self) -> None:
        self._table: pl.DataFrame | None = None
        self._global: float = 0.0
        self._model: str = ""

    def fit(self, train: pl.DataFrame, model: str) -> "BiasCorrector":
        self._model = model
        err = train.with_columns((pl.col(f"t_{model}") - pl.col("t_obs")).alias("_e"))
        self._global = float(err["_e"].mean())
        self._table = err.group_by(_KEYS).agg(pl.col("_e").mean().alias("_bias"))
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        assert self._table is not None, "call fit first"
        joined = df.join(self._table, on=_KEYS, how="left").with_columns(
            pl.col("_bias").fill_null(self._global)
        )
        return (joined[f"t_{self._model}"] - joined["_bias"]).to_numpy()
