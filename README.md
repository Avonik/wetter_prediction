# Lüneburg Weather Postprocessing Engine

A local statistical-postprocessing project for Lüneburg weather forecasts.

The core selling point is still the same: professional numerical weather models provide
the large-scale physics, and this project learns local station behavior for Lüneburg
(DWD station Wendisch Evern, 06093). In other words, this is Model Output Statistics
(MOS), not a from-scratch weather simulator.

## Current State

The project currently has three user-facing pieces:

- **Hourly temperature forecast**: the main tuned local model used by the website.
  It blends ICON-D2/EU/global, GFS, and ECMWF inputs with local station observations,
  learned bias correction, LightGBM blending, and calibrated uncertainty ranges.
- **Longer evaluation/report pipeline**: the older report flow compares the tuned
  postprocessor against baselines such as persistence, climatology, and the best raw
  model across 24-168 hour lead times.
- **Rain odds**: a separate trained rain model estimates the probability of measurable
  station precipitation and calibrates it on a chronologically held-out recent window.

The displayed percentage is the locally trained and calibrated
`P(precipitation >= 0.1 mm/h)`. Live Open-Meteo precipitation probability is retained in
the API as a diagnostic field, but it does not override the local model. The compact hourly
cards show only `P(precipitation >= 0.1 mm/h)`. Expanding a card reveals the 1 mm and 5 mm
thresholds, the raw amount mean with an explicit caveat, the local temperature interval,
and the individual input-model temperatures.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; use `source .venv/bin/activate` on Unix
pip install -e ".[dev]"
```

With `uv`, the same commands can be run as:

```bash
uv sync --extra dev
uv run wetter --help
```

## Website Pipeline

This is the current app path.

```bash
wetter pull-obs        # DWD station observations via Bright Sky
wetter pull-runs       # Open-Meteo single model runs for hourly lead times
wetter build-hourly    # curated hourly temperature dataset
wetter train-hourly    # tuned hourly temperature engine -> data/models/engine_hourly.joblib
wetter build-rain      # curated rain dataset
wetter train-rain      # trained rain engine -> data/models/rain_engine.joblib
wetter evaluate-rain   # leak-free raw/isotonic/beta probability comparison
wetter serve           # FastAPI website at http://127.0.0.1:8000
```

The website endpoint is `/api/forecast`. A background worker refreshes the forecast every
10 minutes and persists the last successful snapshot under `data/cache/`, so requests keep
returning immediately when an upstream provider is slow or temporarily unavailable. Health
checks are available at `/health/live` and `/health/ready`.

## Report Pipeline

The older report flow is still useful for proving postprocessing skill over longer
lead times.

```bash
wetter pull-obs
wetter pull-forecasts  # Open-Meteo Previous Runs at 24h steps + auxiliary variables
wetter pull-era5       # ERA5 reanalysis for climatology normals
wetter build-dataset   # join + features -> data/curated/canonical.parquet
wetter train           # tuned engine -> data/models/engine.joblib
wetter evaluate        # print metrics table
wetter report          # write timestamped markdown report + figures
```

`wetter report` writes `reports/report_<timestamp>.md` and, when an engine exists,
includes a live forecast section.

## Temperature Model

The temperature engine combines:

- raw model temperatures from ICON-D2, ICON-EU, ICON global, GFS, and ECMWF IFS;
- auxiliary predictors such as humidity, cloud cover, wind, pressure, and radiation;
- local station observations from Wendisch Evern;
- climatology and recent-bias features;
- tuned LightGBM point and quantile models;
- probabilistic postprocessing such as EMOS/NGR, split conformal, and CQR in the
  evaluation/report path.

Metrics include MAE, RMSE, bias, skill scores, CRPS, reliability/PIT, coverage, and
sharpness on chronological train/calibration/test splits.

## Rain Model

The trained rain engine is a set of LightGBM binary exceedance classifiers for hourly
station precipitation thresholds:

- `0.1 mm/h`: measurable rain;
- `1.0 mm/h`: notable rain;
- `5.0 mm/h`: heavy rain.

Its features are intentionally physical rather than calendar-memorized:

- per-model precipitation amounts;
- cross-model precipitation mean/max/agreement;
- mean cloud cover and relative humidity;
- observed precipitation at issue time as a persistence signal.

The classifiers are trained before a recent chronological calibration window. Smooth beta
calibration is the production default; isotonic and raw probabilities remain available in
`wetter evaluate-rain` for reproducible comparison. Thresholds with too few independent
wet hours are deliberately left uncalibrated instead of fitting an unstable correction.

The live website additionally fetches Open-Meteo `precipitation_probability`. Because that
field is not consistently available in the historical single-runs archive, it is exposed
only as `rain_upstream_p` for diagnostics and is not blended into the displayed value.

## Known Limitations

- The available honest single-run rain history currently covers less than a full annual
  cycle, so seasonal calibration should be monitored as more forecasts accumulate.
- Open-Meteo PoP cannot be learned as a blend feature until issue-time values and later
  station outcomes have been collected for a sufficiently long period.
- Bright Sky current-weather precipitation can be missing, so the app separately queries
  recent `/weather` observations and falls back gracefully when the station has no usable
  current precipitation value.

## Data Attribution

- Weather forecast and ERA5 data by [Open-Meteo.com](https://open-meteo.com) - CC BY 4.0.
- DWD station observations via [Bright Sky](https://brightsky.dev) - DWD Terms of Use.
