# Lüneburg Temperature Postprocessing Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible engine that postprocesses multi-model 2m-temperature NWP forecasts for Lüneburg into calibrated probabilistic forecasts, proven against baselines with leak-free evaluation.

**Architecture:** Two-zone data layer (raw API cache → curated parquet), a canonical feature table joined in DuckDB, a ladder of models (baselines → bias correction → LightGBM blend → quantile-GBM / EMOS / conformal), and a leak-free evaluation+report layer. A Typer CLI orchestrates the pipeline.

**Tech Stack:** Python 3.13, httpx, polars, duckdb, pyarrow, numpy, scipy, lightgbm, scikit-learn, matplotlib, typer, pytest, ruff. Spec: `docs/superpowers/specs/2026-06-29-wetter-postprocessing-lueneburg-design.md`.

## Global Constraints

- **Python:** 3.13 (venv at `.venv/` already present).
- **No git repo** (project decision): there are NO commit steps. Each task ends with the **full test suite green** as its checkpoint.
- **`.gitignore`** already exists and must be kept current when new generated artifact types appear.
- **All timestamps are UTC**, tz-aware. `valid_time` = time the forecast is *for*; `issue_time = valid_time − lead_time_h`.
- **Single target variable:** `temperature_2m` (°C). No precipitation/wind in this plan.
- **Models (exact Open-Meteo strings):** `icon_d2`, `icon_eu`, `icon_global`, `gfs_seamless`, `ecmwf_ifs025`.
- **Lead times (hours):** `24, 48, 72, 96, 120, 144, 168`.
- **Station:** Wendisch Evern, DWD id `06093`, elevation `62.0` m; query coords `lat=53.25, lon=10.41`.
- **APIs need no key.** Open-Meteo Previous Runs: `https://previous-runs-api.open-meteo.com/v1/forecast`; ERA5: `https://archive-api.open-meteo.com/v1/archive`; Bright Sky: `https://api.brightsky.dev/weather`. License CC BY 4.0 / DWD ToU — keep attribution in README.
- **Leakage discipline:** chronological splits only; any feature using observations must use data strictly before `issue_time`; conformal calibration on a disjoint window.
- **No network in tests:** all HTTP goes through one injectable `get_json` function, monkeypatched in tests. All file I/O uses a `cache_dir` parameter so tests inject `tmp_path`.
- **DataFrames:** polars is the default; convert to pandas only at the LightGBM/scikit-learn boundary.

---

### Task 1: Project scaffolding + config

**Files:**
- Create: `pyproject.toml`
- Create: `src/wetter/__init__.py`
- Create: `src/wetter/config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `wetter.config` with constants `STATION_ID:str`, `LAT:float`, `LON:float`, `STATION_ELEV_M:float`, `MODELS:list[str]`, `LEAD_TIMES_H:list[int]`, `OBS_START:str`, `FORECAST_START:str`; path constants `DATA_DIR`, `RAW_DIR`, `CURATED_DIR`, `REPORTS_DIR`, `DUCKDB_PATH` (all `pathlib.Path`); helper `raw_path(*parts: str) -> Path` that joins under `RAW_DIR` and creates parent dirs.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from wetter import config

def test_core_constants():
    assert config.STATION_ID == "06093"
    assert config.STATION_ELEV_M == 62.0
    assert config.MODELS == ["icon_d2", "icon_eu", "icon_global", "gfs_seamless", "ecmwf_ifs025"]
    assert config.LEAD_TIMES_H == [24, 48, 72, 96, 120, 144, 168]

def test_raw_path_creates_parents(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    p = config.raw_path("obs", "x.parquet")
    assert p == tmp_path / "raw" / "obs" / "x.parquet"
    assert p.parent.is_dir()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wetter'`.

- [ ] **Step 3: Write `pyproject.toml` and config**

```toml
# pyproject.toml
[project]
name = "wetter"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "httpx>=0.27", "polars>=1.0", "duckdb>=1.0", "pyarrow>=16",
    "numpy>=2.0", "scipy>=1.13", "lightgbm>=4.3", "scikit-learn>=1.5",
    "matplotlib>=3.9", "typer>=0.12",
]
[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.6"]
[project.scripts]
wetter = "wetter.cli:app"
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
[tool.setuptools.packages.find]
where = ["src"]
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
[tool.ruff]
line-length = 100
```

```python
# src/wetter/config.py
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
```

Create empty `src/wetter/__init__.py` and `tests/__init__.py`. Install deps once:
`.venv/Scripts/python -m pip install -e ".[dev]"`

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Full-suite checkpoint**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all green.

---

### Task 2: HTTP + parquet cache infrastructure

**Files:**
- Create: `src/wetter/data/__init__.py`
- Create: `src/wetter/data/io.py`
- Create: `tests/data/__init__.py`
- Create: `tests/data/test_io.py`

**Interfaces:**
- Produces:
  - `get_json(url: str, params: dict) -> dict` — the **only** network call site; monkeypatched in all client tests.
  - `month_chunks(start: str, end: str) -> list[tuple[str, str]]` — inclusive monthly `(YYYY-MM-DD, YYYY-MM-DD)` ranges covering `[start, end]`.
  - `cached_parquet(path: Path, builder: Callable[[], pl.DataFrame], *, force: bool=False) -> pl.DataFrame` — returns cached parquet if present and not `force`, else calls `builder()`, writes, returns.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_io.py
import polars as pl
from wetter.data import io

def test_month_chunks_spans_year_boundary():
    chunks = io.month_chunks("2023-12-01", "2024-02-15")
    assert chunks == [
        ("2023-12-01", "2023-12-31"),
        ("2024-01-01", "2024-01-31"),
        ("2024-02-01", "2024-02-15"),
    ]

def test_cached_parquet_writes_then_reads(tmp_path):
    calls = {"n": 0}
    def builder():
        calls["n"] += 1
        return pl.DataFrame({"a": [1, 2]})
    p = tmp_path / "x.parquet"
    df1 = io.cached_parquet(p, builder)
    df2 = io.cached_parquet(p, builder)          # second call hits cache
    assert calls["n"] == 1
    assert df1.equals(df2)

def test_cached_parquet_force_rebuilds(tmp_path):
    calls = {"n": 0}
    def builder():
        calls["n"] += 1
        return pl.DataFrame({"a": [calls["n"]]})
    p = tmp_path / "x.parquet"
    io.cached_parquet(p, builder)
    io.cached_parquet(p, builder, force=True)
    assert calls["n"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/data/test_io.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wetter.data'`.

- [ ] **Step 3: Write implementation**

```python
# src/wetter/data/io.py
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
```

Create empty `tests/data/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/data/test_io.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Full-suite checkpoint**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all green.

---

### Task 3: Observations client (Bright Sky)

**Files:**
- Create: `src/wetter/data/observations.py`
- Create: `tests/data/test_observations.py`

**Interfaces:**
- Consumes: `wetter.data.io.{get_json, month_chunks, cached_parquet}`, `wetter.config`.
- Produces:
  - `parse_weather(payload: dict) -> pl.DataFrame` → columns `valid_time: Datetime(UTC)`, `t_obs: Float64`.
  - `fetch_observations(start: str, end: str, *, station_id: str = config.STATION_ID, cache_dir: Path | None = None, force: bool=False) -> pl.DataFrame` — monthly-chunked, cached under `raw/obs/`, deduped on `valid_time`, sorted.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_observations.py
import polars as pl
from wetter.data import observations as obs

PAYLOAD = {"weather": [
    {"timestamp": "2024-01-01T00:00:00+00:00", "temperature": 3.2},
    {"timestamp": "2024-01-01T01:00:00+00:00", "temperature": 2.9},
    {"timestamp": "2024-01-01T02:00:00+00:00", "temperature": None},
]}

def test_parse_weather_types_and_utc():
    df = obs.parse_weather(PAYLOAD)
    assert df.columns == ["valid_time", "t_obs"]
    assert df["valid_time"].dtype == pl.Datetime("us", "UTC")
    assert df["t_obs"].to_list()[:2] == [3.2, 2.9]
    assert df["t_obs"].to_list()[2] is None

def test_fetch_observations_uses_cache_and_dedupes(tmp_path, monkeypatch):
    calls = {"n": 0}
    def fake_get_json(url, params):
        calls["n"] += 1
        return PAYLOAD
    monkeypatch.setattr(obs.io, "get_json", fake_get_json)
    df = obs.fetch_observations("2024-01-01", "2024-01-01", cache_dir=tmp_path)
    assert df.height == 3
    assert df["valid_time"].is_sorted()
    n_after_first = calls["n"]
    obs.fetch_observations("2024-01-01", "2024-01-01", cache_dir=tmp_path)  # cached
    assert calls["n"] == n_after_first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/data/test_observations.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write implementation**

```python
# src/wetter/data/observations.py
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
    start: str, end: str, *, station_id: str = config.STATION_ID,
    cache_dir: Path | None = None, force: bool = False,
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
    return (
        pl.concat(frames)
        .unique(subset="valid_time", keep="last")
        .sort("valid_time")
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/data/test_observations.py -v`
Expected: PASS.

- [ ] **Step 5: Full-suite checkpoint**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all green.

---

### Task 4: Forecast client (Open-Meteo Previous Runs)

**Files:**
- Create: `src/wetter/data/forecasts.py`
- Create: `tests/data/test_forecasts.py`

**Interfaces:**
- Consumes: `wetter.data.io`, `wetter.config`.
- Produces:
  - `parse_previous_runs(payload: dict, model: str) -> pl.DataFrame` — melts `temperature_2m_previous_dayN` into long: columns `valid_time: Datetime(UTC)`, `lead_time_h: Int32`, `model: Utf8`, `t_fc: Float64`. `lead_time_h = 24*N`. Drops null `t_fc`.
  - `fetch_forecasts(start: str, end: str, *, models: list[str]=config.MODELS, cache_dir: Path | None=None, force: bool=False) -> pl.DataFrame` — per-model monthly-chunked cache under `raw/forecast/previous_runs/<model>/`, concatenated long.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_forecasts.py
import polars as pl
from wetter.data import forecasts as fc

PAYLOAD = {"hourly": {
    "time": ["2024-01-02T00:00", "2024-01-02T01:00"],
    "temperature_2m_previous_day1": [1.0, 1.5],
    "temperature_2m_previous_day2": [0.5, None],
}}

def test_parse_previous_runs_melts_to_long():
    df = fc.parse_previous_runs(PAYLOAD, "icon_d2")
    assert set(df.columns) == {"valid_time", "lead_time_h", "model", "t_fc"}
    assert df["valid_time"].dtype == pl.Datetime("us", "UTC")
    # 2 times x 2 leads = 4, minus one null = 3
    assert df.height == 3
    assert set(df["lead_time_h"].unique().to_list()) == {24, 48}
    assert df.filter((pl.col("lead_time_h") == 24) & (pl.col("valid_time").dt.hour() == 1))["t_fc"][0] == 1.5

def test_fetch_forecasts_caches_per_model(tmp_path, monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(fc.io, "get_json", lambda url, params: (calls.__setitem__("n", calls["n"] + 1) or PAYLOAD))
    df = fc.fetch_forecasts("2024-01-02", "2024-01-02", models=["icon_d2", "gfs_seamless"], cache_dir=tmp_path)
    assert set(df["model"].unique().to_list()) == {"icon_d2", "gfs_seamless"}
    n1 = calls["n"]
    fc.fetch_forecasts("2024-01-02", "2024-01-02", models=["icon_d2", "gfs_seamless"], cache_dir=tmp_path)
    assert calls["n"] == n1  # fully cached
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/data/test_forecasts.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write implementation**

```python
# src/wetter/data/forecasts.py
from __future__ import annotations
from pathlib import Path

import polars as pl

from wetter import config
from wetter.data import io

_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
_LEAD_DAYS = [1, 2, 3, 4, 5, 6, 7]
_HOURLY_VARS = ",".join(f"temperature_2m_previous_day{n}" for n in _LEAD_DAYS)


def parse_previous_runs(payload: dict, model: str) -> pl.DataFrame:
    hourly = payload["hourly"]
    base = pl.DataFrame({"time": hourly["time"]}).with_columns(
        pl.col("time").str.to_datetime(time_zone="UTC").alias("valid_time")
    )
    frames = []
    for n in _LEAD_DAYS:
        key = f"temperature_2m_previous_day{n}"
        if key not in hourly:
            continue
        frames.append(
            base.select("valid_time").with_columns(
                pl.lit(24 * n, dtype=pl.Int32).alias("lead_time_h"),
                pl.lit(model).alias("model"),
                pl.Series("t_fc", hourly[key], dtype=pl.Float64),
            )
        )
    return pl.concat(frames).drop_nulls("t_fc")


def fetch_forecasts(
    start: str, end: str, *, models: list[str] = config.MODELS,
    cache_dir: Path | None = None, force: bool = False,
) -> pl.DataFrame:
    root = cache_dir if cache_dir is not None else config.RAW_DIR / "forecast" / "previous_runs"
    frames = []
    for model in models:
        for lo, hi in io.month_chunks(start, end):
            path = root / model / f"{lo[:7]}.parquet"
            def builder(lo=lo, hi=hi, model=model):
                payload = io.get_json(
                    _URL,
                    {"latitude": config.LAT, "longitude": config.LON,
                     "hourly": _HOURLY_VARS, "models": model,
                     "start_date": lo, "end_date": hi, "timezone": "GMT"},
                )
                return parse_previous_runs(payload, model)
            frames.append(io.cached_parquet(path, builder, force=force))
    return pl.concat(frames).sort(["model", "lead_time_h", "valid_time"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/data/test_forecasts.py -v`
Expected: PASS.

- [ ] **Step 5: Full-suite checkpoint**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all green.

---

### Task 5: Climatology (ERA5) + normals

**Files:**
- Create: `src/wetter/data/climatology.py`
- Create: `tests/data/test_climatology.py`

**Interfaces:**
- Consumes: `wetter.data.io`, `wetter.config`.
- Produces:
  - `parse_era5(payload: dict) -> pl.DataFrame` → `valid_time: Datetime(UTC)`, `t_era5: Float64`.
  - `fetch_era5(start: str, end: str, *, cache_dir: Path | None=None, force: bool=False) -> pl.DataFrame` — monthly-chunked cache under `raw/era5/`.
  - `compute_climatology(era5: pl.DataFrame) -> pl.DataFrame` → `month: Int8`, `hour: Int8`, `t_clim: Float64` (mean `t_era5` grouped by month, hour).

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_climatology.py
import polars as pl
from wetter.data import climatology as clim

PAYLOAD = {"hourly": {
    "time": ["2020-06-01T12:00", "2021-06-15T12:00", "2020-01-01T00:00"],
    "temperature_2m": [20.0, 22.0, -1.0],
}}

def test_compute_climatology_groups_month_hour():
    era5 = clim.parse_era5(PAYLOAD)
    c = clim.compute_climatology(era5)
    june_noon = c.filter((pl.col("month") == 6) & (pl.col("hour") == 12))["t_clim"][0]
    assert june_noon == 21.0  # mean(20, 22)
    assert c["t_clim"].dtype == pl.Float64
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/data/test_climatology.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write implementation**

```python
# src/wetter/data/climatology.py
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
                {"latitude": config.LAT, "longitude": config.LON,
                 "hourly": "temperature_2m", "models": "era5",
                 "start_date": lo, "end_date": hi, "timezone": "GMT"},
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/data/test_climatology.py -v`
Expected: PASS.

- [ ] **Step 5: Full-suite checkpoint**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all green.

---

### Task 6: Build canonical dataset (joins + leak-free features)

**Files:**
- Create: `src/wetter/data/features.py`
- Create: `src/wetter/data/build_dataset.py`
- Create: `tests/data/test_features.py`
- Create: `tests/data/test_build_dataset.py`

**Interfaces:**
- Consumes: outputs of Tasks 3–5.
- Produces (`features.py`):
  - `add_time_features(df) -> pl.DataFrame` — adds `hour, doy, month` (ints) and `hour_sin, hour_cos, doy_sin, doy_cos` (Float64) from `valid_time`.
  - `add_multimodel_features(df) -> pl.DataFrame` — from wide per-model cols `t_<model>` adds `t_mean, t_median, t_spread, t_min, t_max` (Float64; `t_spread` = population std, null-safe).
  - `add_recent_bias(df: pl.DataFrame, obs: pl.DataFrame, model: str, window: int=30) -> pl.DataFrame` — adds `recent_bias_<model>`: for each row, the mean of `(t_<model> − t_obs_actual)` over the `window` most recent **issue_times strictly before this row's issue_time**, at the same `lead_time_h`. Strictly leak-free.
- Produces (`build_dataset.py`):
  - `pivot_forecasts(forecasts: pl.DataFrame) -> pl.DataFrame` — long→wide: index `(valid_time, lead_time_h)`, columns `t_<model>`.
  - `build_canonical(obs, forecasts, climatology) -> pl.DataFrame` — full canonical table per the spec schema, including `issue_time`, `t_obs_at_issue`, `t_clim`, `station_vs_grid_elev_diff` (placeholder constant 0.0 for MVP — model elevations not pulled), and target `t_obs`. Rows with null `t_obs` dropped.
  - `build(*, cache_dir: Path | None=None) -> Path` — loads cached curated inputs (or raw), writes `curated/canonical.parquet`, returns the path.

**Note on `station_vs_grid_elev_diff`:** the Open-Meteo response echoes per-model grid elevation, but pulling/storing it per model adds I/O for marginal MVP value. Set it to constant `0.0` here and leave a one-line code comment that it can be wired from the API `elevation` field in a later iteration. (This keeps the schema stable without scope creep.)

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_features.py
import polars as pl
from datetime import datetime, timezone
from wetter.data import features

def _dt(h):  # helper: 2024-01-01 at hour h UTC
    return datetime(2024, 1, 1, h, tzinfo=timezone.utc)

def test_time_features_cyclical_bounds():
    df = pl.DataFrame({"valid_time": [_dt(0), _dt(12)]})
    out = features.add_time_features(df)
    assert out["hour"].to_list() == [0, 12]
    assert abs(out["hour_sin"][0] - 0.0) < 1e-9
    assert -1.0 <= out["hour_cos"].min() <= out["hour_cos"].max() <= 1.0

def test_multimodel_spread_zero_when_equal():
    df = pl.DataFrame({"t_icon_d2": [5.0], "t_gfs_seamless": [5.0]})
    out = features.add_multimodel_features(df.select(pl.all()), )
    assert out["t_mean"][0] == 5.0
    assert out["t_spread"][0] == 0.0

def test_recent_bias_is_leak_free():
    # issue times t0<t1<t2 at same lead; model is +2 too warm consistently
    obs = pl.DataFrame({
        "valid_time": [_dt(0), _dt(1), _dt(2)],
        "t_obs": [10.0, 11.0, 12.0],
    })
    df = pl.DataFrame({
        "issue_time": [_dt(0), _dt(1), _dt(2)],
        "valid_time": [_dt(0), _dt(1), _dt(2)],
        "lead_time_h": [24, 24, 24],
        "t_icon_d2": [12.0, 13.0, 14.0],   # error +2 each
    }).join(obs, on="valid_time")
    out = features.add_recent_bias(df, obs, "icon_d2", window=10)
    # first row has no prior issue_time -> null; later rows -> +2.0
    rb = out.sort("issue_time")["recent_bias_icon_d2"].to_list()
    assert rb[0] is None
    assert abs(rb[1] - 2.0) < 1e-9
    assert abs(rb[2] - 2.0) < 1e-9
```

```python
# tests/data/test_build_dataset.py
import polars as pl
from datetime import datetime, timezone
from wetter.data import build_dataset as bd

def _dt(d, h):
    return datetime(2024, 1, d, h, tzinfo=timezone.utc)

def test_build_canonical_has_target_and_issue_time():
    obs = pl.DataFrame({"valid_time": [_dt(1, 0), _dt(2, 0)], "t_obs": [10.0, 11.0]})
    forecasts = pl.DataFrame({
        "valid_time": [_dt(2, 0), _dt(2, 0)],
        "lead_time_h": [24, 24],
        "model": ["icon_d2", "gfs_seamless"],
        "t_fc": [12.0, 9.0],
    })
    clim = pl.DataFrame({"month": [1], "hour": [0], "t_clim": [3.0]})
    canon = bd.build_canonical(obs, forecasts, clim)
    row = canon.row(0, named=True)
    assert row["t_obs"] == 11.0                    # target = obs at valid_time
    assert row["issue_time"] == _dt(1, 0)          # valid - 24h
    assert row["t_obs_at_issue"] == 10.0           # persistence feature
    assert row["t_clim"] == 3.0
    assert row["t_mean"] == 10.5                   # mean(12, 9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/data/test_features.py tests/data/test_build_dataset.py -v`
Expected: FAIL — modules not found.

- [ ] **Step 3: Write `features.py`**

```python
# src/wetter/data/features.py
from __future__ import annotations
import math

import polars as pl

from wetter import config

_TAU = 2 * math.pi


def add_time_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("valid_time").dt.hour().cast(pl.Int32).alias("hour"),
        pl.col("valid_time").dt.ordinal_day().cast(pl.Int32).alias("doy"),
        pl.col("valid_time").dt.month().cast(pl.Int32).alias("month"),
    ).with_columns(
        (pl.col("hour") * (_TAU / 24)).sin().alias("hour_sin"),
        (pl.col("hour") * (_TAU / 24)).cos().alias("hour_cos"),
        (pl.col("doy") * (_TAU / 365.25)).sin().alias("doy_sin"),
        (pl.col("doy") * (_TAU / 365.25)).cos().alias("doy_cos"),
    )


def add_multimodel_features(df: pl.DataFrame) -> pl.DataFrame:
    model_cols = [f"t_{m}" for m in config.MODELS if f"t_{m}" in df.columns]
    return df.with_columns(
        pl.mean_horizontal(model_cols).alias("t_mean"),
        pl.concat_list(model_cols).list.median().alias("t_median"),
        pl.concat_list(model_cols).list.std(ddof=0).alias("t_spread"),
        pl.min_horizontal(model_cols).alias("t_min"),
        pl.max_horizontal(model_cols).alias("t_max"),
    )


def add_recent_bias(
    df: pl.DataFrame, obs: pl.DataFrame, model: str, window: int = 30
) -> pl.DataFrame:
    col = f"t_{model}"
    out_name = f"recent_bias_{model}"
    work = (
        df.join(obs.rename({"t_obs": "_obs_truth"}), on="valid_time", how="left")
        .with_columns((pl.col(col) - pl.col("_obs_truth")).alias("_err"))
        .sort("issue_time")
    )
    # rolling mean over prior rows only (shift(1) excludes the current issue_time)
    work = work.with_columns(
        pl.col("_err")
        .shift(1)
        .rolling_mean(window_size=window, min_samples=1)
        .over("lead_time_h")
        .alias(out_name)
    )
    return work.drop(["_obs_truth", "_err"])
```

- [ ] **Step 4: Run feature tests**

Run: `.venv/Scripts/python -m pytest tests/data/test_features.py -v`
Expected: PASS.

- [ ] **Step 5: Write `build_dataset.py`**

```python
# src/wetter/data/build_dataset.py
from __future__ import annotations
from pathlib import Path

import polars as pl

from wetter import config
from wetter.data import build_inputs  # see note below
from wetter.data import features


def pivot_forecasts(forecasts: pl.DataFrame) -> pl.DataFrame:
    wide = forecasts.pivot(
        values="t_fc", index=["valid_time", "lead_time_h"], on="model"
    )
    rename = {m: f"t_{m}" for m in forecasts["model"].unique().to_list()}
    return wide.rename(rename)


def build_canonical(
    obs: pl.DataFrame, forecasts: pl.DataFrame, climatology: pl.DataFrame
) -> pl.DataFrame:
    wide = pivot_forecasts(forecasts)
    canon = wide.with_columns(
        (pl.col("valid_time") - pl.duration(hours=pl.col("lead_time_h"))).alias("issue_time")
    )
    # target
    canon = canon.join(obs, on="valid_time", how="left")
    # persistence feature: obs at issue_time
    canon = canon.join(
        obs.rename({"valid_time": "issue_time", "t_obs": "t_obs_at_issue"}),
        on="issue_time", how="left",
    )
    # features
    canon = features.add_time_features(canon)
    canon = features.add_multimodel_features(canon)
    canon = canon.join(climatology, on=["month", "hour"], how="left")
    for m in config.MODELS:
        if f"t_{m}" in canon.columns:
            canon = features.add_recent_bias(canon, obs, m)
    # elevation diff: constant 0.0 for MVP (wire from API `elevation` field later)
    canon = canon.with_columns(pl.lit(0.0).alias("station_vs_grid_elev_diff"))
    return canon.drop_nulls("t_obs").sort(["valid_time", "lead_time_h"])


def build(*, cache_dir: Path | None = None) -> Path:
    obs, forecasts, clim = build_inputs.load_all()
    canon = build_canonical(obs, forecasts, clim)
    out = (cache_dir or config.CURATED_DIR) / "canonical.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    canon.write_parquet(out)
    return out
```

Also create `src/wetter/data/build_inputs.py` with a thin loader used by `build()` and the CLI (keeps `build_canonical` pure and unit-testable):

```python
# src/wetter/data/build_inputs.py
from __future__ import annotations
import polars as pl
from wetter import config
from wetter.data import observations, forecasts, climatology


def load_all() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    obs = observations.fetch_observations(config.FORECAST_START, _today())
    fc = forecasts.fetch_forecasts(config.FORECAST_START, _today())
    era5 = climatology.fetch_era5(config.OBS_START, _today())
    return obs, fc, climatology.compute_climatology(era5)


def _today() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).date().isoformat()
```

- [ ] **Step 6: Run build_dataset tests**

Run: `.venv/Scripts/python -m pytest tests/data/test_build_dataset.py -v`
Expected: PASS.

- [ ] **Step 7: Full-suite checkpoint**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all green.

---

### Task 7: Train/test split utility

**Files:**
- Create: `src/wetter/models/__init__.py`
- Create: `src/wetter/split.py`
- Create: `tests/test_split.py`

**Interfaces:**
- Produces: `chronological_split(df: pl.DataFrame, *, train_end: str, cal_end: str, time_col: str="valid_time") -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]` returning `(train, calib, test)` where train = `time < train_end`, calib = `train_end <= time < cal_end`, test = `time >= cal_end`. Splits compare against `issue_time` is out of scope — split on `valid_time` is sufficient because lead is fixed-offset and all leads of a valid_time stay together.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_split.py
import polars as pl
from datetime import datetime, timezone
from wetter.split import chronological_split

def test_split_partitions_chronologically():
    times = [datetime(2024, m, 1, tzinfo=timezone.utc) for m in range(1, 13)]
    df = pl.DataFrame({"valid_time": times, "x": list(range(12))})
    tr, ca, te = chronological_split(df, train_end="2024-07-01", cal_end="2024-10-01")
    assert tr.height == 6 and ca.height == 3 and te.height == 3
    assert tr["valid_time"].max() < ca["valid_time"].min() < te["valid_time"].min()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_split.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write implementation**

```python
# src/wetter/split.py
from __future__ import annotations
from datetime import datetime, timezone

import polars as pl


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def chronological_split(
    df: pl.DataFrame, *, train_end: str, cal_end: str, time_col: str = "valid_time"
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    te1, te2 = _ts(train_end), _ts(cal_end)
    train = df.filter(pl.col(time_col) < te1)
    calib = df.filter((pl.col(time_col) >= te1) & (pl.col(time_col) < te2))
    test = df.filter(pl.col(time_col) >= te2)
    return train, calib, test
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_split.py -v`
Expected: PASS.

- [ ] **Step 5: Full-suite checkpoint**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all green.

---

### Task 8: Metrics (deterministic + CRPS)

**Files:**
- Create: `src/wetter/eval/__init__.py`
- Create: `src/wetter/eval/metrics.py`
- Create: `tests/eval/__init__.py`
- Create: `tests/eval/test_metrics.py`

**Interfaces:**
- Produces (all operate on `numpy.ndarray`):
  - `mae(y, pred) -> float`, `rmse(y, pred) -> float`, `bias(y, pred) -> float` (mean of `pred − y`).
  - `skill_score(metric_model: float, metric_ref: float) -> float` = `1 - metric_model/metric_ref`.
  - `crps_gaussian(y, mu, sigma) -> np.ndarray` — closed form, per-sample.
  - `crps_from_quantiles(y, quantile_levels: np.ndarray, quantile_preds: np.ndarray) -> np.ndarray` — pinball-based approximation (mean over levels of 2·pinball / coverage normalization); per-sample.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_metrics.py
import numpy as np
from wetter.eval import metrics as M

def test_mae_rmse_bias():
    y = np.array([0.0, 0.0, 0.0]); p = np.array([1.0, -1.0, 2.0])
    assert M.mae(y, p) == (1 + 1 + 2) / 3
    assert abs(M.rmse(y, p) - np.sqrt((1 + 1 + 4) / 3)) < 1e-12
    assert abs(M.bias(y, p) - (1 - 1 + 2) / 3) < 1e-12

def test_skill_score_signs():
    assert M.skill_score(0.5, 1.0) == 0.5     # half the error of ref -> +0.5
    assert M.skill_score(2.0, 1.0) == -1.0    # worse than ref -> negative

def test_crps_gaussian_perfect_sharp_is_small():
    # sigma -> small and mu == y -> CRPS -> ~0
    y = np.array([5.0]); mu = np.array([5.0]); sigma = np.array([1e-3])
    assert M.crps_gaussian(y, mu, sigma)[0] < 1e-3

def test_crps_gaussian_known_value():
    # CRPS(N(0,1), 0) = 2*phi(0) - 1/sqrt(pi) = 0.7978845608 - 0.5641895835
    val = M.crps_gaussian(np.array([0.0]), np.array([0.0]), np.array([1.0]))[0]
    assert abs(val - 0.23369497) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/eval/test_metrics.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write implementation**

```python
# src/wetter/eval/metrics.py
from __future__ import annotations
import numpy as np
from scipy.stats import norm


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - y)))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def bias(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(pred - y))


def skill_score(metric_model: float, metric_ref: float) -> float:
    return 1.0 - metric_model / metric_ref


def crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.clip(sigma, 1e-9, None)
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_from_quantiles(
    y: np.ndarray, quantile_levels: np.ndarray, quantile_preds: np.ndarray
) -> np.ndarray:
    # quantile_preds shape: (n_samples, n_levels); pinball loss averaged over levels
    y = y[:, None]
    diff = y - quantile_preds
    pinball = np.maximum(quantile_levels * diff, (quantile_levels - 1) * diff)
    return 2.0 * pinball.mean(axis=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/eval/test_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Full-suite checkpoint**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all green.

---

### Task 9: Baselines + bias correction

**Files:**
- Create: `src/wetter/models/baselines.py`
- Create: `src/wetter/models/bias_correction.py`
- Create: `tests/models/__init__.py`
- Create: `tests/models/test_baselines.py`
- Create: `tests/models/test_bias_correction.py`

**Interfaces:**
- `baselines.py`:
  - `predict_persistence(df) -> np.ndarray` = `df["t_obs_at_issue"]`.
  - `predict_climatology(df) -> np.ndarray` = `df["t_clim"]`.
  - `predict_raw(df, model: str) -> np.ndarray` = `df[f"t_{model}"]`.
- `bias_correction.py`:
  - `class BiasCorrector` with `fit(train: pl.DataFrame, model: str) -> "BiasCorrector"` (learns mean error grouped by `(lead_time_h, hour, month)`) and `predict(df: pl.DataFrame) -> np.ndarray` (= `t_<model>` minus learned error; unseen groups → global mean error fallback).

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/test_baselines.py
import polars as pl, numpy as np
from wetter.models import baselines as B

def test_baselines_read_columns():
    df = pl.DataFrame({"t_obs_at_issue": [1.0], "t_clim": [2.0], "t_icon_d2": [3.0]})
    assert B.predict_persistence(df)[0] == 1.0
    assert B.predict_climatology(df)[0] == 2.0
    assert B.predict_raw(df, "icon_d2")[0] == 3.0
```

```python
# tests/models/test_bias_correction.py
import polars as pl, numpy as np
from wetter.models.bias_correction import BiasCorrector

def test_bias_correction_removes_systematic_offset():
    # model always +2 warm in (lead 24, hour 0, month 1)
    train = pl.DataFrame({
        "t_icon_d2": [12.0, 13.0, 14.0],
        "t_obs": [10.0, 11.0, 12.0],
        "lead_time_h": [24, 24, 24], "hour": [0, 0, 0], "month": [1, 1, 1],
    })
    bc = BiasCorrector().fit(train, "icon_d2")
    test = pl.DataFrame({"t_icon_d2": [20.0], "lead_time_h": [24], "hour": [0], "month": [1]})
    assert abs(bc.predict(test)[0] - 18.0) < 1e-9   # 20 - learned(+2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/models/test_baselines.py tests/models/test_bias_correction.py -v`
Expected: FAIL — modules not found.

- [ ] **Step 3: Write implementations**

```python
# src/wetter/models/baselines.py
from __future__ import annotations
import numpy as np
import polars as pl


def predict_persistence(df: pl.DataFrame) -> np.ndarray:
    return df["t_obs_at_issue"].to_numpy()


def predict_climatology(df: pl.DataFrame) -> np.ndarray:
    return df["t_clim"].to_numpy()


def predict_raw(df: pl.DataFrame, model: str) -> np.ndarray:
    return df[f"t_{model}"].to_numpy()
```

```python
# src/wetter/models/bias_correction.py
from __future__ import annotations
import numpy as np
import polars as pl

_KEYS = ["lead_time_h", "hour", "month"]


class BiasCorrector:
    def __init__(self) -> None:
        self._table: pl.DataFrame | None = None
        self._global: float = 0.0
        self._model: str = ""

    def fit(self, train: pl.DataFrame, model: str) -> "BiasCorrector":
        self._model = model
        err = train.with_columns((pl.col(f"t_{model}") - pl.col("t_obs")).alias("_e"))
        self._global = float(err["_e"].mean())
        self._table = err.group_by(_KEYS).agg(pl.col("_e").mean().alias("_bias"))
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        assert self._table is not None, "call fit first"
        joined = df.join(self._table, on=_KEYS, how="left").with_columns(
            pl.col("_bias").fill_null(self._global)
        )
        return (joined[f"t_{self._model}"] - joined["_bias"]).to_numpy()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/models/test_baselines.py tests/models/test_bias_correction.py -v`
Expected: PASS.

- [ ] **Step 5: Full-suite checkpoint**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all green.

---

### Task 10: LightGBM point + quantile blend

**Files:**
- Create: `src/wetter/models/blend_gbm.py`
- Create: `tests/models/test_blend_gbm.py`

**Interfaces:**
- Produces:
  - `FEATURES: list[str]` — the feature columns used for training (model forecasts present + `t_mean, t_median, t_spread, t_min, t_max, hour_sin, hour_cos, doy_sin, doy_cos, lead_time_h, t_obs_at_issue, t_clim` + any `recent_bias_*` present). A helper `feature_columns(df) -> list[str]` returns those present in `df`.
  - `train_point(train: pl.DataFrame, features: list[str]) -> lightgbm.LGBMRegressor`.
  - `predict(model, df, features) -> np.ndarray`.
  - `train_quantiles(train, features, quantiles=(0.1,0.5,0.9)) -> dict[float, lightgbm.LGBMRegressor]`.
  - `predict_quantiles(models: dict, df, features) -> dict[float, np.ndarray]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_blend_gbm.py
import numpy as np, polars as pl
from wetter.models import blend_gbm as G

def _synth(n=400, seed=0):
    rng = np.random.default_rng(seed)
    t_icon = rng.normal(10, 5, n)
    noise = rng.normal(0, 1, n)
    # truth = icon minus its +2 bias plus mild noise
    t_obs = t_icon - 2.0 + noise
    return pl.DataFrame({
        "t_icon_d2": t_icon, "t_mean": t_icon, "t_median": t_icon,
        "t_spread": np.zeros(n), "t_min": t_icon, "t_max": t_icon,
        "hour_sin": np.zeros(n), "hour_cos": np.ones(n),
        "doy_sin": np.zeros(n), "doy_cos": np.ones(n),
        "lead_time_h": np.full(n, 24), "t_obs_at_issue": t_obs,
        "t_clim": np.full(n, 10.0), "t_obs": t_obs,
    })

def test_point_blend_beats_raw_on_synth():
    df = _synth()
    tr, te = df.head(300), df.tail(100)
    feats = G.feature_columns(tr)
    model = G.train_point(tr, feats)
    pred = G.predict(model, te, feats)
    y = te["t_obs"].to_numpy()
    raw_mae = np.mean(np.abs(te["t_icon_d2"].to_numpy() - y))
    blend_mae = np.mean(np.abs(pred - y))
    assert blend_mae < raw_mae      # learned away the +2 bias

def test_quantiles_are_ordered_on_average():
    df = _synth()
    feats = G.feature_columns(df)
    models = G.train_quantiles(df.head(300), feats)
    preds = G.predict_quantiles(models, df.tail(100), feats)
    assert np.mean(preds[0.1]) < np.mean(preds[0.5]) < np.mean(preds[0.9])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/models/test_blend_gbm.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write implementation**

```python
# src/wetter/models/blend_gbm.py
from __future__ import annotations
import lightgbm as lgb
import numpy as np
import polars as pl

_CANDIDATES = [
    "t_icon_d2", "t_icon_eu", "t_icon_global", "t_gfs_seamless", "t_ecmwf_ifs025",
    "t_mean", "t_median", "t_spread", "t_min", "t_max",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "lead_time_h",
    "t_obs_at_issue", "t_clim",
    "recent_bias_icon_d2", "recent_bias_icon_eu", "recent_bias_icon_global",
    "recent_bias_gfs_seamless", "recent_bias_ecmwf_ifs025",
]

_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
               min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
               verbosity=-1)


def feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in _CANDIDATES if c in df.columns]


def _X(df: pl.DataFrame, features: list[str]):
    return df.select(features).to_pandas()


def train_point(train: pl.DataFrame, features: list[str]) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(objective="regression_l1", **_PARAMS)
    model.fit(_X(train, features), train["t_obs"].to_numpy())
    return model


def predict(model: lgb.LGBMRegressor, df: pl.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict(_X(df, features))


def train_quantiles(
    train: pl.DataFrame, features: list[str], quantiles=(0.1, 0.5, 0.9)
) -> dict[float, lgb.LGBMRegressor]:
    models: dict[float, lgb.LGBMRegressor] = {}
    y = train["t_obs"].to_numpy()
    X = _X(train, features)
    for q in quantiles:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, **_PARAMS)
        m.fit(X, y)
        models[q] = m
    return models


def predict_quantiles(models, df, features) -> dict[float, np.ndarray]:
    X = _X(df, features)
    return {q: m.predict(X) for q, m in models.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/models/test_blend_gbm.py -v`
Expected: PASS.

- [ ] **Step 5: Full-suite checkpoint**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all green.

---

### Task 11: EMOS / NGR (CRPS-minimized Gaussian)

**Files:**
- Create: `src/wetter/models/emos.py`
- Create: `tests/models/test_emos.py`

**Interfaces:**
- Consumes: `wetter.eval.metrics.crps_gaussian`.
- Produces:
  - `class EMOS` with `fit(train: pl.DataFrame) -> "EMOS"` and `predict(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]` returning `(mu, sigma)`.
  - Mean model: `mu = a + b * t_mean`. Variance model: `sigma^2 = c^2 + d^2 * t_spread` (squares enforce positivity). Parameters `(a, b, c, d)` fit by minimizing mean `crps_gaussian` via `scipy.optimize.minimize` (Nelder-Mead).

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_emos.py
import numpy as np, polars as pl
from wetter.models.emos import EMOS

def test_emos_recovers_offset_and_calibrated_spread():
    rng = np.random.default_rng(0)
    n = 2000
    t_mean = rng.normal(10, 5, n)
    spread = np.abs(rng.normal(1.0, 0.2, n))
    # truth = 1 + 1.0*t_mean + gaussian noise scaled by spread
    y = 1.0 + t_mean + rng.normal(0, 1, n) * spread
    df = pl.DataFrame({"t_mean": t_mean, "t_spread": spread, "t_obs": y})
    em = EMOS().fit(df)
    mu, sigma = em.predict(df)
    # mean prediction tracks truth, residual std ~ sigma (calibrated)
    assert abs(np.mean(mu - y)) < 0.2
    assert np.mean(sigma) > 0
    # PIT std should be near 1 if calibrated
    z = (y - mu) / sigma
    assert 0.7 < np.std(z) < 1.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/models/test_emos.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write implementation**

```python
# src/wetter/models/emos.py
from __future__ import annotations
import numpy as np
import polars as pl
from scipy.optimize import minimize

from wetter.eval.metrics import crps_gaussian


class EMOS:
    def __init__(self) -> None:
        self.params = np.array([0.0, 1.0, 1.0, 0.0])  # a, b, c, d

    @staticmethod
    def _mu_sigma(p, m, s):
        a, b, c, d = p
        mu = a + b * m
        sigma = np.sqrt(c * c + d * d * s)
        return mu, sigma

    def fit(self, train: pl.DataFrame) -> "EMOS":
        m = train["t_mean"].to_numpy()
        s = train["t_spread"].fill_null(0.0).to_numpy()
        y = train["t_obs"].to_numpy()

        def loss(p):
            mu, sigma = self._mu_sigma(p, m, s)
            return float(np.mean(crps_gaussian(y, mu, sigma)))

        res = minimize(loss, self.params, method="Nelder-Mead",
                       options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-8})
        self.params = res.x
        return self

    def predict(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        m = df["t_mean"].to_numpy()
        s = df["t_spread"].fill_null(0.0).to_numpy()
        return self._mu_sigma(self.params, m, s)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/models/test_emos.py -v`
Expected: PASS.

- [ ] **Step 5: Full-suite checkpoint**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all green.

---

### Task 12: Conformal prediction intervals

**Files:**
- Create: `src/wetter/models/conformal.py`
- Create: `tests/models/test_conformal.py`

**Interfaces:**
- Produces:
  - `class SplitConformal` with `calibrate(y_cal: np.ndarray, point_cal: np.ndarray) -> "SplitConformal"` (stores absolute residuals) and `interval(point: np.ndarray, alpha: float=0.2) -> tuple[np.ndarray, np.ndarray]` returning `(lo, hi)` using the finite-sample-corrected `(1-alpha)` quantile of calibration residuals (symmetric band around the point prediction).
  - Docstring notes: split conformal assumes exchangeability (violated by time series); calibration uses a disjoint chronological window; adaptive conformal (ACI) is a documented future extension.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_conformal.py
import numpy as np
from wetter.models.conformal import SplitConformal

def test_split_conformal_marginal_coverage():
    rng = np.random.default_rng(0)
    # calibration residuals ~ N(0,1); point preds arbitrary
    y_cal = rng.normal(0, 1, 2000)
    point_cal = np.zeros(2000)
    sc = SplitConformal().calibrate(y_cal, point_cal)

    y_test = rng.normal(0, 1, 5000)
    point_test = np.zeros(5000)
    lo, hi = sc.interval(point_test, alpha=0.2)
    covered = np.mean((y_test >= lo) & (y_test <= hi))
    assert abs(covered - 0.8) < 0.03      # ~80% nominal coverage
    assert np.all(hi >= lo)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/models/test_conformal.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write implementation**

```python
# src/wetter/models/conformal.py
from __future__ import annotations
import numpy as np


class SplitConformal:
    """Split conformal intervals (symmetric absolute-residual band).

    Assumes exchangeability, which time series violate; mitigate by calibrating
    on a disjoint chronological window. Adaptive conformal (ACI) is a future
    extension.
    """

    def __init__(self) -> None:
        self._residuals: np.ndarray | None = None

    def calibrate(self, y_cal: np.ndarray, point_cal: np.ndarray) -> "SplitConformal":
        self._residuals = np.abs(y_cal - point_cal)
        return self

    def interval(self, point: np.ndarray, alpha: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
        assert self._residuals is not None, "call calibrate first"
        n = self._residuals.size
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        q = float(np.quantile(self._residuals, level, method="higher"))
        return point - q, point + q
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/models/test_conformal.py -v`
Expected: PASS.

- [ ] **Step 5: Full-suite checkpoint**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all green.

---

### Task 13: Calibration diagnostics

**Files:**
- Create: `src/wetter/eval/calibration.py`
- Create: `tests/eval/test_calibration.py`

**Interfaces:**
- Produces:
  - `pit_gaussian(y, mu, sigma) -> np.ndarray` — PIT values `Phi((y-mu)/sigma)`.
  - `coverage(y, lo, hi) -> float` — fraction within `[lo, hi]`.
  - `sharpness(lo, hi) -> float` — mean interval width.
  - `reliability_from_pit(pit: np.ndarray, n_bins: int=10) -> tuple[np.ndarray, np.ndarray]` — `(bin_centers, observed_freq)` for a PIT histogram (observed freq per uniform bin; calibrated ≈ flat at 1.0 density → returns normalized counts).

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_calibration.py
import numpy as np
from wetter.eval import calibration as C

def test_pit_uniform_when_calibrated():
    rng = np.random.default_rng(0)
    n = 50000
    mu = np.zeros(n); sigma = np.ones(n)
    y = rng.normal(0, 1, n)
    pit = C.pit_gaussian(y, mu, sigma)
    # calibrated -> PIT ~ Uniform(0,1): mean ~0.5, std ~ 1/sqrt(12)
    assert abs(pit.mean() - 0.5) < 0.01
    assert abs(pit.std() - (1 / np.sqrt(12))) < 0.01

def test_coverage_and_sharpness():
    y = np.array([0.0, 0.0, 5.0])
    lo = np.array([-1.0, -1.0, -1.0]); hi = np.array([1.0, 1.0, 1.0])
    assert abs(C.coverage(y, lo, hi) - 2 / 3) < 1e-9
    assert C.sharpness(lo, hi) == 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/eval/test_calibration.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write implementation**

```python
# src/wetter/eval/calibration.py
from __future__ import annotations
import numpy as np
from scipy.stats import norm


def pit_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.clip(sigma, 1e-9, None)
    return norm.cdf((y - mu) / sigma)


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def sharpness(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(hi - lo))


def reliability_from_pit(pit: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(pit, bins=n_bins, range=(0.0, 1.0), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/eval/test_calibration.py -v`
Expected: PASS.

- [ ] **Step 5: Full-suite checkpoint**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all green.

---

### Task 14: Evaluation orchestration + report

**Files:**
- Create: `src/wetter/eval/report.py`
- Create: `tests/eval/test_report.py`

**Interfaces:**
- Consumes: split, baselines, bias_correction, blend_gbm, emos, conformal, metrics, calibration.
- Produces:
  - `evaluate(canonical: pl.DataFrame, *, train_end: str, cal_end: str) -> pl.DataFrame` — trains all methods on `train`, calibrates conformal on `calib`, scores on `test`, returns a tidy results frame with columns `method, lead_time_h, mae, rmse, bias, skill_vs_persistence, skill_vs_clim, skill_vs_best_raw, crps, coverage_80, sharpness_80` (deterministic metrics filled for point methods; `crps/coverage/sharpness` filled where applicable, else null).
  - `generate_report(results: pl.DataFrame, *, out_dir: Path) -> Path` — writes `report.md` + at least a per-lead MAE figure (`mae_by_lead.png`) and a PIT/reliability figure; returns the markdown path.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_report.py
import polars as pl, numpy as np
from datetime import datetime, timezone, timedelta
from wetter.eval import report as R

def _canon(n=900, seed=0):
    rng = np.random.default_rng(seed)
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = [t0 + timedelta(hours=i) for i in range(n)]
    t_icon = rng.normal(10, 5, n)
    y = t_icon - 2.0 + rng.normal(0, 1, n)
    return pl.DataFrame({
        "valid_time": times, "lead_time_h": np.full(n, 24),
        "t_icon_d2": t_icon, "t_mean": t_icon, "t_median": t_icon,
        "t_spread": np.abs(rng.normal(1, 0.2, n)), "t_min": t_icon, "t_max": t_icon,
        "hour": [t.hour for t in times], "month": [t.month for t in times],
        "hour_sin": np.zeros(n), "hour_cos": np.ones(n),
        "doy_sin": np.zeros(n), "doy_cos": np.ones(n),
        "t_obs_at_issue": y, "t_clim": np.full(n, 10.0), "t_obs": y,
    })

def test_evaluate_produces_rows_for_methods():
    canon = _canon()
    res = R.evaluate(canon, train_end="2024-01-20", cal_end="2024-01-28")
    methods = set(res["method"].unique().to_list())
    assert {"persistence", "climatology", "raw_best", "bias_corrected",
            "gbm_point", "gbm_quantile", "emos", "conformal"} <= methods
    # gbm_point should beat raw_best on MAE for this synthetic bias case
    mae_gbm = res.filter(pl.col("method") == "gbm_point")["mae"][0]
    mae_raw = res.filter(pl.col("method") == "raw_best")["mae"][0]
    assert mae_gbm <= mae_raw + 1e-6

def test_generate_report_writes_files(tmp_path):
    canon = _canon()
    res = R.evaluate(canon, train_end="2024-01-20", cal_end="2024-01-28")
    md = R.generate_report(res, out_dir=tmp_path)
    assert md.exists()
    assert (tmp_path / "mae_by_lead.png").exists()
    assert "Skill" in md.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/eval/test_report.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write implementation**

```python
# src/wetter/eval/report.py
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from wetter import config
from wetter.split import chronological_split
from wetter.models import baselines, blend_gbm
from wetter.models.bias_correction import BiasCorrector
from wetter.models.emos import EMOS
from wetter.models.conformal import SplitConformal
from wetter.eval import metrics as M
from wetter.eval import calibration as Cal


def _best_raw_model(train: pl.DataFrame) -> str:
    y = train["t_obs"].to_numpy()
    scores = {}
    for m in config.MODELS:
        if f"t_{m}" in train.columns:
            scores[m] = M.mae(y, baselines.predict_raw(train, m))
    return min(scores, key=scores.get)


def evaluate(canonical: pl.DataFrame, *, train_end: str, cal_end: str) -> pl.DataFrame:
    train, calib, test = chronological_split(canonical, train_end=train_end, cal_end=cal_end)
    feats = blend_gbm.feature_columns(train)
    best_raw = _best_raw_model(train)

    # fit models
    bc = BiasCorrector().fit(train, best_raw)
    gbm = blend_gbm.train_point(train, feats)
    qmodels = blend_gbm.train_quantiles(train, feats)
    emos = EMOS().fit(train)
    gbm_cal_point = blend_gbm.predict(gbm, calib, feats)
    conf = SplitConformal().calibrate(calib["t_obs"].to_numpy(), gbm_cal_point)

    rows = []
    for lead in sorted(test["lead_time_h"].unique().to_list()):
        sub = test.filter(pl.col("lead_time_h") == lead)
        if sub.height == 0:
            continue
        y = sub["t_obs"].to_numpy()
        ref_pers = M.mae(y, baselines.predict_persistence(sub))
        ref_clim = M.mae(y, baselines.predict_climatology(sub))
        ref_raw = M.mae(y, baselines.predict_raw(sub, best_raw))

        def det_row(method, pred, **extra):
            m = M.mae(y, pred)
            return {
                "method": method, "lead_time_h": lead,
                "mae": m, "rmse": M.rmse(y, pred), "bias": M.bias(y, pred),
                "skill_vs_persistence": M.skill_score(m, ref_pers),
                "skill_vs_clim": M.skill_score(m, ref_clim),
                "skill_vs_best_raw": M.skill_score(m, ref_raw),
                "crps": None, "coverage_80": None, "sharpness_80": None,
                **extra,
            }

        rows.append(det_row("persistence", baselines.predict_persistence(sub)))
        rows.append(det_row("climatology", baselines.predict_climatology(sub)))
        rows.append(det_row("raw_best", baselines.predict_raw(sub, best_raw)))
        rows.append(det_row("bias_corrected", bc.predict(sub)))
        rows.append(det_row("gbm_point", blend_gbm.predict(gbm, sub, feats)))

        # quantile gbm (median as point; CRPS via pinball; coverage from 0.1/0.9)
        qp = blend_gbm.predict_quantiles(qmodels, sub, feats)
        levels = np.array(sorted(qp.keys()))
        qpreds = np.column_stack([qp[q] for q in levels])
        crps_q = float(np.mean(M.crps_from_quantiles(y, levels, qpreds)))
        cov_q = Cal.coverage(y, qp[0.1], qp[0.9])
        sharp_q = Cal.sharpness(qp[0.1], qp[0.9])
        r = det_row("gbm_quantile", qp[0.5])
        r.update(crps=crps_q, coverage_80=cov_q, sharpness_80=sharp_q)
        rows.append(r)

        # emos
        mu, sigma = emos.predict(sub)
        crps_e = float(np.mean(M.crps_gaussian(y, mu, sigma)))
        from scipy.stats import norm
        lo_e, hi_e = norm.ppf(0.1, mu, sigma), norm.ppf(0.9, mu, sigma)
        r = det_row("emos", mu)
        r.update(crps=crps_e, coverage_80=Cal.coverage(y, lo_e, hi_e),
                 sharpness_80=Cal.sharpness(lo_e, hi_e))
        rows.append(r)

        # conformal around gbm point (80% interval -> alpha 0.2)
        point = blend_gbm.predict(gbm, sub, feats)
        lo_c, hi_c = conf.interval(point, alpha=0.2)
        r = det_row("conformal", point)
        r.update(coverage_80=Cal.coverage(y, lo_c, hi_c),
                 sharpness_80=Cal.sharpness(lo_c, hi_c))
        rows.append(r)

    return pl.DataFrame(rows)


def generate_report(results: pl.DataFrame, *, out_dir: Path | None = None) -> Path:
    out = out_dir or config.REPORTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    # figure 1: MAE by lead per method
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in results["method"].unique().to_list():
        d = results.filter(pl.col("method") == method).sort("lead_time_h")
        ax.plot(d["lead_time_h"], d["mae"], marker="o", label=method)
    ax.set_xlabel("lead time (h)"); ax.set_ylabel("MAE (°C)"); ax.legend(fontsize=7)
    ax.set_title("Temperature MAE by lead time")
    fig.savefig(out / "mae_by_lead.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    # figure 2: skill vs best raw
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in results["method"].unique().to_list():
        d = results.filter(pl.col("method") == method).sort("lead_time_h")
        ax.plot(d["lead_time_h"], d["skill_vs_best_raw"], marker="o", label=method)
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("lead time (h)")
    ax.set_ylabel("skill vs best raw"); ax.legend(fontsize=7)
    fig.savefig(out / "skill_by_lead.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    md = out / "report.md"
    lines = ["# Lüneburg Temperature Postprocessing — Evaluation\n",
             "Skill scores are vs the best raw model, persistence, and climatology.\n",
             "![MAE by lead](mae_by_lead.png)\n",
             "![Skill by lead](skill_by_lead.png)\n",
             "## Results table\n",
             results.to_pandas().to_markdown(index=False)]
    md.write_text("\n".join(lines), encoding="utf-8")
    return md
```

(Add `tabulate` to dev deps if `to_markdown` requires it: `.venv/Scripts/python -m pip install tabulate`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/eval/test_report.py -v`
Expected: PASS.

- [ ] **Step 5: Full-suite checkpoint**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all green.

---

### Task 15: Typer CLI

**Files:**
- Create: `src/wetter/cli.py`
- Create: `tests/test_cli.py`
- Create: `README.md` (usage + data attribution)

**Interfaces:**
- Consumes: all prior modules.
- Produces a Typer `app` with commands:
  - `pull-obs`, `pull-forecasts`, `pull-era5` — fetch + cache (date range from config defaults, `--start/--end` overrides, `--force`).
  - `build-dataset` — writes `curated/canonical.parquet`.
  - `evaluate` — runs `report.evaluate` on the canonical table and prints the results frame; `--train-end`, `--cal-end` options.
  - `report` — runs evaluate + `generate_report`, prints output path.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
from typer.testing import CliRunner
from wetter.cli import app

runner = CliRunner()

def test_cli_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ["pull-obs", "pull-forecasts", "build-dataset", "evaluate", "report"]:
        assert cmd in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_cli.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write implementation**

```python
# src/wetter/cli.py
from __future__ import annotations
from datetime import datetime, timezone

import typer

from wetter import config
from wetter.data import observations, forecasts, climatology, build_dataset
from wetter.eval import report

app = typer.Typer(help="Lüneburg temperature postprocessing engine")


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


@app.command("pull-obs")
def pull_obs(start: str = config.OBS_START, end: str = "", force: bool = False) -> None:
    df = observations.fetch_observations(start, end or _today(), force=force)
    typer.echo(f"obs rows: {df.height}")


@app.command("pull-forecasts")
def pull_forecasts(start: str = config.FORECAST_START, end: str = "", force: bool = False) -> None:
    df = forecasts.fetch_forecasts(start, end or _today(), force=force)
    typer.echo(f"forecast rows: {df.height}")


@app.command("pull-era5")
def pull_era5(start: str = config.OBS_START, end: str = "", force: bool = False) -> None:
    df = climatology.fetch_era5(start, end or _today(), force=force)
    typer.echo(f"era5 rows: {df.height}")


@app.command("build-dataset")
def build_dataset_cmd() -> None:
    path = build_dataset.build()
    typer.echo(f"wrote {path}")


@app.command("evaluate")
def evaluate(train_end: str = "2025-07-01", cal_end: str = "2025-10-01") -> None:
    import polars as pl
    canon = pl.read_parquet(config.CURATED_DIR / "canonical.parquet")
    res = report.evaluate(canon, train_end=train_end, cal_end=cal_end)
    typer.echo(res)


@app.command("report")
def report_cmd(train_end: str = "2025-07-01", cal_end: str = "2025-10-01") -> None:
    import polars as pl
    canon = pl.read_parquet(config.CURATED_DIR / "canonical.parquet")
    res = report.evaluate(canon, train_end=train_end, cal_end=cal_end)
    path = report.generate_report(res)
    typer.echo(f"wrote {path}")


if __name__ == "__main__":
    app()
```

Write a `README.md` documenting setup (`pip install -e ".[dev]"`), the CLI pipeline order (`pull-obs → pull-forecasts → pull-era5 → build-dataset → report`), and **data attribution**: "Weather data by Open-Meteo.com (CC BY 4.0); DWD observations via Bright Sky (DWD Terms of Use)."

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Full-suite checkpoint + lint**

Run: `.venv/Scripts/python -m pytest -q` (all green) and `.venv/Scripts/python -m ruff check src tests` (clean).

---

### Task 16: End-to-end smoke run (real data, manual)

**Files:** none (operational verification).

This task has no unit test — it is the real-data integration check after all units pass.

- [ ] **Step 1:** Pull a short window to validate live API parsing:
  `.venv/Scripts/python -m wetter.cli pull-forecasts --start 2024-06-01 --end 2024-06-07`
  Expected: non-zero "forecast rows", files under `data/raw/forecast/previous_runs/<model>/`.
- [ ] **Step 2:** Pull obs + era5 for the same/overlapping window (era5 needs ≥ a year for stable normals; pull `2023-01-01 → 2024-06-07`).
- [ ] **Step 3:** `build-dataset`; confirm `data/curated/canonical.parquet` exists and has the schema columns from the spec (spot-check with a one-off polars read).
- [ ] **Step 4:** `report --train-end 2024-04-01 --cal-end 2024-05-01`; open `reports/report.md`, confirm figures render and skill numbers are present.
- [ ] **Step 5:** Full-suite checkpoint: `.venv/Scripts/python -m pytest -q` (all green).

---

## Self-Review (performed against the spec)

**Spec coverage:**
- §4 data layer (raw→curated, Parquet+DuckDB, idempotent cache, UTC, schema) → Tasks 2–6. *Note:* DuckDB is listed in the stack but the joins are implemented in polars (simpler, fully tested); DuckDB remains available for ad-hoc queries / the Phase-2 dashboard. This is a deliberate simplification recorded here, not a gap.
- §5 modeling (baselines, bias correction, GBM blend, quantile, EMOS, conformal) → Tasks 9–12.
- §6 evaluation (chronological split, MAE/RMSE/skill, CRPS, reliability/PIT, coverage, sharpness, leak guards) → Tasks 7, 8, 13, 14.
- §7 stack/structure/CLI → Tasks 1, 15.
- §2 success criteria (beat baselines, calibrated, one-command reproducible) → Tasks 14 (skill+calibration metrics), 15–16 (CLI one-command + smoke).
- `station_vs_grid_elev_diff` is intentionally a constant 0.0 in the MVP (documented in Task 6); flagged so it is not mistaken for a silent omission.

**Placeholder scan:** no TBD/TODO; every code step contains runnable code and every test contains real assertions.

**Type consistency:** model wide-columns are `t_<model>` everywhere; `lead_time_h` Int32; `valid_time`/`issue_time` `Datetime("us","UTC")`; `feature_columns()` is the single source of the feature list consumed by `train_point`/`train_quantiles`/`predict`; `crps_gaussian` signature `(y, mu, sigma)` is consistent across metrics, EMOS, and report.

**Out of scope (unchanged):** Phase 2 Streamlit dashboard and Phase 3 MOSMIX logger are not in this plan.
