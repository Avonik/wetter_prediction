from __future__ import annotations
from pathlib import Path

import polars as pl

from wetter import config
from wetter.data import io

_URL = "https://api.brightsky.dev/weather"


def parse_weather(payload: dict) -> pl.DataFrame:
    rows = payload.get("weather", [])
    df = pl.DataFrame(
        {
            "valid_time": [r["timestamp"] for r in rows],
            "t_obs": [r.get("temperature") for r in rows],
        }
    )
    return df.with_columns(
        pl.col("valid_time").str.to_datetime(time_zone="UTC"),
        pl.col("t_obs").cast(pl.Float64),
    )


def fetch_observations(
    start: str,
    end: str,
    *,
    station_id: str = config.STATION_ID,
    cache_dir: Path | None = None,
    force: bool = False,
) -> pl.DataFrame:
    base = cache_dir if cache_dir is not None else config.RAW_DIR / "obs"
    frames = []
    for lo, hi in io.month_chunks(start, end):
        path = base / f"brightsky_{station_id}_{lo[:7]}.parquet"

        def builder(lo=lo, hi=hi):
            payload = io.get_json(
                _URL,
                {"dwd_station_id": station_id, "date": lo, "last_date": hi, "tz": "UTC"},
            )
            return parse_weather(payload)

        frames.append(io.cached_parquet(path, builder, force=force))
    return pl.concat(frames).unique(subset="valid_time", keep="last").sort("valid_time")
