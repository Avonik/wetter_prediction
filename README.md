# Lüneburg Temperature Postprocessing Engine

A local statistical-postprocessing engine that turns multi-model NWP 2m-temperature
forecasts into **calibrated probabilistic** forecasts for Lüneburg, and proves the added
value honestly against baselines (persistence, climatology, best raw model) with leak-free
evaluation.

This is **not** a weather model. It is Model Output Statistics (MOS) / statistical
postprocessing: professional models supply the large-scale physics; this engine learns the
local bias, the situational model weighting, and the uncertainty calibration.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; use `source .venv/bin/activate` on Unix
pip install -e ".[dev]"
```

## Pipeline (CLI)

Run in order:

```bash
wetter pull-obs            # DWD observations (Wendisch Evern, station 06093) via Bright Sky
wetter pull-forecasts      # Open-Meteo Previous Runs (ICON-D2/EU/global, GFS, ECMWF IFS) + aux vars
wetter pull-era5           # ERA5 reanalysis for climatology normals
wetter build-dataset       # join + features -> data/curated/canonical.parquet
wetter train               # tune GBM, fit & persist the engine -> data/models/engine.joblib
wetter report              # backtest + live forecast -> reports/report_<timestamp>.md + figures
```

`wetter evaluate` prints the metrics table without writing the report. `wetter report`
writes a **timestamped** markdown report (never overwritten) with figures, and — if an
engine has been trained — appends a **live forecast**: the current station temperature plus
the tuned engine's prediction (point + calibrated 80% range) for the next 7 days. All
commands accept `--train-end` / `--cal-end` where relevant.

## Models compared

Baselines (persistence, climatology, raw models) → bias correction → tuned LightGBM blend
(per-model temperatures + auxiliary predictors: humidity, cloud, wind, pressure, radiation,
and interactions) → probabilistic layer: **quantile-GBM**, **EMOS/NGR** (CRPS-minimized
Gaussian), **split conformal**, and **CQR** (conformalized quantile regression). Metrics:
MAE/RMSE/bias + skill scores; CRPS, reliability/PIT,
coverage, sharpness — all per lead time (24–168 h), on a chronological train/calib/test split.

## Data attribution

- Weather forecast & ERA5 data by [Open-Meteo.com](https://open-meteo.com) — CC BY 4.0.
- DWD station observations via [Bright Sky](https://brightsky.dev) — DWD Terms of Use.
