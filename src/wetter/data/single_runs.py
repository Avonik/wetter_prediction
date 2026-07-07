from __future__ import annotations
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import polars as pl

from wetter import config
from wetter.data import io
from wetter.data.forecasts import VARS

_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
_HOURLY = ",".join(api for _, api in VARS)

_SCHEMA = {
    "run_time": pl.Datetime("us", "UTC"),
    "valid_time": pl.Datetime("us", "UTC"),
    "lead_time_h": pl.Int32,
    "model": pl.Utf8,
    "variable": pl.Utf8,
    "value": pl.Float64,
    "grid_elev": pl.Float64,
}


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=_SCHEMA)


def parse_run(payload: dict, model: str, run_iso: str) -> pl.DataFrame:
    """One issued model run -> long rows (run_time, valid_time, lead_time_h, model,
    variable, value, grid_elev). lead_time_h = valid_time - run_time, in hours."""
    hourly = payload["hourly"]
    run_dt = datetime.fromisoformat(run_iso).replace(tzinfo=timezone.utc)
    base = (
        pl.DataFrame({"time": hourly["time"]})
        .with_columns(pl.col("time").str.to_datetime(time_zone="UTC").alias("valid_time"))
        .select("valid_time")
    )
    elev = float(payload.get("elevation") or config.STATION_ELEV_M)
    frames = []
    for internal, api in VARS:
        if api not in hourly:
            continue
        frames.append(
            base.with_columns(
                pl.lit(run_dt).cast(pl.Datetime("us", "UTC")).alias("run_time"),
                pl.lit(model).alias("model"),
                pl.lit(internal).alias("variable"),
                pl.Series("value", hourly[api], dtype=pl.Float64),
                pl.lit(elev).alias("grid_elev"),
            )
        )
    if not frames:
        return _empty()
    return (
        pl.concat(frames)
        .drop_nulls("value")
        .with_columns(
            (pl.col("valid_time") - pl.col("run_time")).dt.total_hours().cast(pl.Int32).alias(
                "lead_time_h"
            )
        )
        .select(list(_SCHEMA.keys()))
    )


def fetch_runs(
    start: str,
    end: str,
    *,
    models: list[str] = config.MODELS,
    run_hours: tuple[int, ...] | None = None,
    max_lead_h: int | None = None,
    cache_dir: Path | None = None,
    concurrency: int = 1,
    cached_only: bool = False,
    force: bool = False,
) -> pl.DataFrame:
    """Sample issued runs (one per `run_hours` per day) per model, cached per run.
    Missing runs (HTTP 400) are cached as empty so re-runs skip them. With
    `cached_only=True`, only runs already on disk are read — nothing is fetched
    (build from what we have without hitting the rate-limited API)."""
    max_lead_h = max_lead_h or config.HOURLY_MAX_LEAD_H
    if run_hours is None:
        run_hours = config.RUN_HOURS
    root = cache_dir if cache_dir is not None else config.RAW_DIR / "single_runs"
    items: list = []
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    for model in models:
        d = d0
        while d <= d1:
            for hh in run_hours:
                run_iso = f"{d.isoformat()}T{hh:02d}:00"
                path = root / model / f"{d.isoformat()}_{hh:02d}.parquet"

                def builder(run_iso=run_iso, model=model):
                    try:
                        payload = io.get_json(
                            _URL,
                            {
                                "latitude": config.LAT,
                                "longitude": config.LON,
                                "hourly": _HOURLY,
                                "models": model,
                                "run": run_iso,
                                "timezone": "GMT",
                            },
                        )
                    except httpx.HTTPStatusError:
                        # genuine 400 -> this run does not exist; cache as empty.
                        # (Transient rate-limit/network errors propagate and are retried.)
                        return _empty()
                    df = parse_run(payload, model, run_iso)
                    return df.filter(
                        (pl.col("lead_time_h") >= 1) & (pl.col("lead_time_h") <= max_lead_h)
                    )

                items.append((path, builder))
            d = d + timedelta(days=1)
    if cached_only:
        items = [(p, b) for p, b in items if p.exists()]
    frames = io.cached_parquet_many(items, force=force, concurrency=concurrency)
    return pl.concat(frames) if frames else _empty()
