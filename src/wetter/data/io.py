from __future__ import annotations
import calendar
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import httpx
import polars as pl

_TIMEOUT = httpx.Timeout(60.0)
# one shared connection pool (keep-alive) instead of a fresh connection per call
_CLIENT = httpx.Client(
    timeout=_TIMEOUT,
    limits=httpx.Limits(max_connections=16, max_keepalive_connections=16),
)
# concurrency for bulk fetches; 8 in-flight × ~1 s/call ≈ 480/min, under Open-Meteo's 600/min
MAX_CONCURRENCY = 8


def get_json(url: str, params: dict) -> dict:
    """GET JSON via the shared pooled client. Retries transient failures — rate
    limits (429), network hiccups, and empty/invalid JSON bodies (which happen
    under concurrency) — with exponential backoff. Genuine 4xx/5xx (e.g. a 400 for
    a missing model run) raise immediately for the caller to handle."""
    delay = 1.0
    err: Exception = httpx.HTTPError("no attempt made")
    for _ in range(6):
        try:
            resp = _CLIENT.get(url, params=params)
        except httpx.TransportError as e:  # connect/read/protocol error
            err = e
            time.sleep(delay)
            delay = min(delay * 2, 8.0)
            continue
        if resp.status_code == 429:
            err = httpx.HTTPError("429 rate limited")
            time.sleep(delay)
            delay = min(delay * 2, 8.0)
            continue
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError as e:  # empty / invalid JSON body -> transient
            err = e
            time.sleep(delay)
            delay = min(delay * 2, 8.0)
            continue
    raise err


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


def cached_parquet_many(
    items: list[tuple[Path, Callable[[], pl.DataFrame]]],
    *,
    force: bool = False,
    concurrency: int = MAX_CONCURRENCY,
) -> list[pl.DataFrame]:
    """Like cached_parquet but for many (path, builder) jobs — cached ones are read
    instantly; the rest run concurrently (builders do the network I/O). Order preserved."""
    results: dict[int, pl.DataFrame] = {}
    todo: list[tuple[int, Path, Callable[[], pl.DataFrame]]] = []
    for i, (path, builder) in enumerate(items):
        if path.exists() and not force:
            results[i] = pl.read_parquet(path)
        else:
            todo.append((i, path, builder))

    def _run(job):
        i, path, builder = job
        df = builder()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        return i, df

    if todo:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for fut in as_completed([ex.submit(_run, job) for job in todo]):
                i, df = fut.result()
                results[i] = df
    return [results[i] for i in range(len(items))]
