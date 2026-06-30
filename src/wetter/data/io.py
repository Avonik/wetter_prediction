from __future__ import annotations
import calendar
from collections.abc import Callable
from datetime import date
from pathlib import Path

import httpx
import polars as pl

_TIMEOUT = httpx.Timeout(60.0)


def get_json(url: str, params: dict) -> dict:
    resp = httpx.get(url, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def month_chunks(start: str, end: str) -> list[tuple[str, str]]:
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    out: list[tuple[str, str]] = []
    y, m = s.year, s.month
    while (y, m) <= (e.year, e.month):
        first = date(y, m, 1)
        last = date(y, m, calendar.monthrange(y, m)[1])
        lo = max(first, s)
        hi = min(last, e)
        out.append((lo.isoformat(), hi.isoformat()))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def cached_parquet(
    path: Path, builder: Callable[[], pl.DataFrame], *, force: bool = False
) -> pl.DataFrame:
    if path.exists() and not force:
        return pl.read_parquet(path)
    df = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df
