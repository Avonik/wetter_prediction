from __future__ import annotations
from pathlib import Path

import polars as pl

from wetter import config
from wetter.data import io

_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
_LEAD_DAYS = [1, 2, 3, 4, 5, 6, 7]

# (internal short name, Open-Meteo API variable). "t" is the target; the rest are
# predictors (temperature errors correlate with cloud cover, humidity, wind, pressure,
# radiation). All are fetched per lead via the _previous_dayN suffix.
VARS: list[tuple[str, str]] = [
    ("t", "temperature_2m"),
    ("rh", "relative_humidity_2m"),
    ("cloud", "cloud_cover"),
    ("wind", "wind_speed_10m"),
    ("pmsl", "pressure_msl"),
    ("rad", "shortwave_radiation"),
    ("precip", "precipitation"),  # mm/h — for the rain model
]

_HOURLY_VARS = ",".join(
    f"{api}_previous_day{n}" for _, api in VARS for n in _LEAD_DAYS
)


def parse_previous_runs(payload: dict, model: str) -> pl.DataFrame:
    hourly = payload["hourly"]
    base = (
        pl.DataFrame({"time": hourly["time"]})
        .with_columns(pl.col("time").str.to_datetime(time_zone="UTC").alias("valid_time"))
        .select("valid_time")
    )
    elev = float(payload.get("elevation") or config.STATION_ELEV_M)
    frames = []
    for internal, api in VARS:
        for n in _LEAD_DAYS:
            key = f"{api}_previous_day{n}"
            if key not in hourly:
                continue
            frames.append(
                base.with_columns(
                    pl.lit(24 * n, dtype=pl.Int32).alias("lead_time_h"),
                    pl.lit(model).alias("model"),
                    pl.lit(internal).alias("variable"),
                    pl.Series("value", hourly[key], dtype=pl.Float64),
                    pl.lit(elev).alias("grid_elev"),
                )
            )
    return pl.concat(frames).drop_nulls("value")


def fetch_forecasts(
    start: str,
    end: str,
    *,
    models: list[str] = config.MODELS,
    cache_dir: Path | None = None,
    force: bool = False,
) -> pl.DataFrame:
    # "previous_runs2": richer multi-variable schema; kept separate from the
    # earlier temperature-only cache so the new data is fetched cleanly.
    root = (
        cache_dir if cache_dir is not None else config.RAW_DIR / "forecast" / "previous_runs2"
    )
    frames = []
    for model in models:
        for lo, hi in io.month_chunks(start, end):
            path = root / model / f"{lo[:7]}.parquet"

            def builder(lo=lo, hi=hi, model=model):
                payload = io.get_json(
                    _URL,
                    {
                        "latitude": config.LAT,
                        "longitude": config.LON,
                        "hourly": _HOURLY_VARS,
                        "models": model,
                        "start_date": lo,
                        "end_date": hi,
                        "timezone": "GMT",
                    },
                )
                return parse_previous_runs(payload, model)

            frames.append(io.cached_parquet(path, builder, force=force))
    return pl.concat(frames).sort(["model", "variable", "lead_time_h", "valid_time"])
