from __future__ import annotations
from datetime import datetime, timezone

import polars as pl

from wetter import config
from wetter.data import io
from wetter.data.forecasts import VARS

_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_CURRENT_URL = "https://api.brightsky.dev/current_weather"

# Live-only signals. The historical single-runs API does not expose these
# consistently, so keep them out of forecasts.VARS and use them only at serving time.
_LIVE_VARS = VARS + [
    ("pop", "precipitation_probability"),
]


def parse_current_obs(payload: dict) -> tuple[datetime, float]:
    cw = payload["weather"]
    t = datetime.fromisoformat(cw["timestamp"])
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc), float(cw["temperature"])


def fetch_current_obs(*, station_id: str = config.STATION_ID) -> tuple[datetime, float]:
    payload = io.get_json(_CURRENT_URL, {"dwd_station_id": station_id, "tz": "UTC"})
    return parse_current_obs(payload)


def parse_live_forecast(payload: dict, model: str) -> pl.DataFrame:
    hourly = payload["hourly"]
    base = (
        pl.DataFrame({"time": hourly["time"]})
        .with_columns(pl.col("time").str.to_datetime(time_zone="UTC").alias("valid_time"))
        .select("valid_time")
    )
    elev = float(payload.get("elevation") or config.STATION_ELEV_M)
    frames = []
    for internal, api in _LIVE_VARS:
        if api not in hourly:
            continue
        frames.append(
            base.with_columns(
                pl.lit(model).alias("model"),
                pl.lit(internal).alias("variable"),
                pl.Series("value", hourly[api], dtype=pl.Float64),
                pl.lit(elev).alias("grid_elev"),
            )
        )
    return pl.concat(frames).drop_nulls("value")


def fetch_live_forecast(*, models: list[str] = config.MODELS, forecast_days: int = 8) -> pl.DataFrame:
    hourly = ",".join(api for _, api in _LIVE_VARS)
    frames = []
    for model in models:
        payload = io.get_json(
            _FORECAST_URL,
            {
                "latitude": config.LAT,
                "longitude": config.LON,
                "hourly": hourly,
                "models": model,
                "forecast_days": forecast_days,
                "past_days": 1,
                "timezone": "GMT",
            },
        )
        frames.append(parse_live_forecast(payload, model))
    return pl.concat(frames)
