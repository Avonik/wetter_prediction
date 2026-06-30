from __future__ import annotations
from datetime import datetime, timezone

import polars as pl


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def chronological_split(
    df: pl.DataFrame, *, train_end: str, cal_end: str, time_col: str = "valid_time"
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    te1, te2 = _ts(train_end), _ts(cal_end)
    train = df.filter(pl.col(time_col) < te1)
    calib = df.filter((pl.col(time_col) >= te1) & (pl.col(time_col) < te2))
    test = df.filter(pl.col(time_col) >= te2)
    return train, calib, test
