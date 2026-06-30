from __future__ import annotations
from datetime import datetime, timezone

import polars as pl

from wetter import config
from wetter.data import climatology, forecasts, observations


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def load_all() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    obs = observations.fetch_observations(config.FORECAST_START, _today())
    fc = forecasts.fetch_forecasts(config.FORECAST_START, _today())
    era5 = climatology.fetch_era5(config.OBS_START, _today())
    return obs, fc, climatology.compute_climatology(era5)
