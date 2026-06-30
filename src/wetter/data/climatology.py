from __future__ import annotations
from pathlib import Path

import polars as pl

from wetter import config
from wetter.data import io

_URL = "https://archive-api.open-meteo.com/v1/archive"


def parse_era5(payload: dict) -> pl.DataFrame:
    hourly = payload["hourly"]
    return pl.DataFrame(
        {"valid_time": hourly["time"], "t_era5": hourly["temperature_2m"]}
    ).with_columns(
        pl.col("valid_time").str.to_datetime(time_zone="UTC"),
        pl.col("t_era5").cast(pl.Float64),
    )


def fetch_era5(
    start: str, end: str, *, cache_dir: Path | None = None, force: bool = False
) -> pl.DataFrame:
    base = cache_dir if cache_dir is not None else config.RAW_DIR / "era5"
    frames = []
    for lo, hi in io.month_chunks(start, end):
        path = base / f"era5_{lo[:7]}.parquet"

        def builder(lo=lo, hi=hi):
            payload = io.get_json(
                _URL,
                {
                    "latitude": config.LAT,
                    "longitude": config.LON,
                    "hourly": "temperature_2m",
                    "models": "era5",
                    "start_date": lo,
                    "end_date": hi,
                    "timezone": "GMT",
                },
            )
            return parse_era5(payload)

        frames.append(io.cached_parquet(path, builder, force=force))
    return pl.concat(frames).sort("valid_time")


def compute_climatology(era5: pl.DataFrame) -> pl.DataFrame:
    return (
        era5.with_columns(
            pl.col("valid_time").dt.month().cast(pl.Int8).alias("month"),
            pl.col("valid_time").dt.hour().cast(pl.Int8).alias("hour"),
        )
        .group_by("month", "hour")
        .agg(pl.col("t_era5").mean().alias("t_clim"))
        .sort("month", "hour")
    )
