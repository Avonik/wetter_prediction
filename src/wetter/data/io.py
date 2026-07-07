from __future__ import annotations
import calendar
import random
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
# concurrency for bulk fetches. The Single Runs API rejects bursts with
# 429 "Too many concurrent requests", so callers doing many small run queries
# pass a lower value; the archive/forecast APIs tolerate 8.
MAX_CONCURRENCY = 8


class Transient(Exception):
    """A request exhausted its retries for a *transient* reason (rate limit,
    network hiccup, empty/invalid body). The resource may well exist — callers
    must NOT cache this as an empty result; let a later run retry it. Distinct
    from httpx.HTTPStatusError, which is a genuine 4xx/5xx (e.g. a missing run)."""


def get_json(url: str, params: dict) -> dict:
    """GET JSON via the shared pooled client. Retries transient failures — rate
    limits (429), network hiccups, and empty/invalid JSON bodies (which happen
    under concurrency) — with exponential backoff, then raises `Transient`.
    Genuine 4xx/5xx (e.g. a 400 for a missing model run) raise httpx.HTTPStatusError
    immediately so the caller can cache them as legitimately empty."""
    delay = 1.0
    err: Exception = Transient("no attempt made")
    for _ in range(5):
        try:
            resp = _CLIENT.get(url, params=params)
        except httpx.TransportError as e:  # connect/read/protocol error
            err = e
            time.sleep(delay * random.uniform(0.6, 1.4))  # jitter: desync retriers
            delay = min(delay * 2, 8.0)
            continue
        if resp.status_code == 429:
            err = Transient("429 rate limited")
            time.sleep(delay * random.uniform(0.6, 1.4))  # jitter: desync retriers
            delay = min(delay * 2, 8.0)
            continue
        resp.raise_for_status()  # genuine 4xx/5xx -> HTTPStatusError, propagate
        try:
            return resp.json()
        except ValueError as e:  # empty / invalid JSON body -> transient
            err = e
            time.sleep(delay * random.uniform(0.6, 1.4))  # jitter: desync retriers
            delay = min(delay * 2, 8.0)
            continue
    raise Transient(str(err)) from err


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
    instantly; the rest run concurrently (builders do the network I/O). A job that
    fails transiently (e.g. rate-limited) is NOT written and is dropped from the
    result rather than cached empty, so the next run retries it (self-healing).
    Returns the successful frames; order among them is preserved."""
    results: dict[int, pl.DataFrame] = {}
    todo: list[tuple[int, Path, Callable[[], pl.DataFrame]]] = []
    for i, (path, builder) in enumerate(items):
        if path.exists() and not force:
            results[i] = pl.read_parquet(path)
        else:
            todo.append((i, path, builder))

    def _run(job):
        i, path, builder = job
        try:
            df = builder()
        except Exception:  # noqa: BLE001 — transient failure: skip, don't poison the cache
            return i, None
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        return i, df

    if todo:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for fut in as_completed([ex.submit(_run, job) for job in todo]):
                i, df = fut.result()
                if df is not None:
                    results[i] = df
    return [results[i] for i in sorted(results)]
