from __future__ import annotations
from pathlib import Path

STATION_ID = "06093"
LAT = 53.25
LON = 10.41
STATION_ELEV_M = 62.0

MODELS = ["icon_d2", "icon_eu", "icon_global", "gfs_seamless", "ecmwf_ifs025"]
LEAD_TIMES_H = [24, 48, 72, 96, 120, 144, 168]

OBS_START = "2010-01-01"
FORECAST_START = "2024-01-01"

# Single Runs API (honest hourly-lead forecast history) — ICON-D2 coverage starts ~Sep 2025.
SINGLE_RUNS_START = "2025-09-01"
HOURLY_MAX_LEAD_H = 48

_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CURATED_DIR = DATA_DIR / "curated"
REPORTS_DIR = _ROOT / "reports"
DUCKDB_PATH = DATA_DIR / "wetter.duckdb"


def raw_path(*parts: str) -> Path:
    p = RAW_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
