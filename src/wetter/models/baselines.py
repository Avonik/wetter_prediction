from __future__ import annotations
import numpy as np
import polars as pl


def predict_persistence(df: pl.DataFrame) -> np.ndarray:
    return df["t_obs_at_issue"].to_numpy()


def predict_climatology(df: pl.DataFrame) -> np.ndarray:
    return df["t_clim"].to_numpy()


def predict_raw(df: pl.DataFrame, model: str) -> np.ndarray:
    return df[f"t_{model}"].to_numpy()
