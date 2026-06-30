# Design: Local Temperature Postprocessing Engine for Lüneburg

**Date:** 2026-06-29
**Status:** Approved (pending final spec review)
**Author:** Julian (with Claude)

## 1. Overview

A local statistical-postprocessing engine that takes the 2m-temperature forecasts of
several professional NWP models, corrects their local biases, blends them, and produces
**calibrated probabilistic** forecasts for Lüneburg — and proves the added value honestly
against strong baselines.

This is explicitly **not** a weather model. It is Model Output Statistics (MOS) /
statistical postprocessing: the professional models supply the large-scale physics; this
engine learns the local errors, the situational model weighting, and the uncertainty
calibration. The portfolio claim is methodological rigor (clean pipeline, leak-free
evaluation, calibration, honest skill scores), not "beating ECMWF".

### Decisions locked in during brainstorming
- **Primary goal:** Methodological rigor first, then a thin usable tool on top ("both, sequenced").
- **MVP target variable:** 2m air temperature only. Precipitation/wind are later phases.
- **Forecast-history source:** Open-Meteo **Previous Runs API** as the training/feature core
  (fixed lead times, honest, explicitly built for bias-correction ML).
- **Probabilistic layer:** the **full comparison** — bias correction + quantile-GBM + EMOS/NGR
  + conformal prediction, benchmarked against each other.

## 2. Goals & Success Criteria

- **Location:** Lüneburg (lat 53.25, lon 10.41). Ground truth = DWD station **Wendisch Evern,
  DWD id `06093`** (~5 km SE; the de-facto Lüneburg climate/SYNOP station — there is no
  observation station literally named "Lüneburg").
- **Lead times:** 24, 48, 72, 96, 120, 144, 168 h. (This is the Previous Runs granularity:
  hourly *valid* times, lead time in 24 h steps.)
- **Success means:**
  1. On a **chronological** test set, the engine beats the best raw model **and** persistence
     **and** climatology (positive skill score) — reported **per lead time, individually and
     honestly** (including lead times where it does not win).
  2. The probabilistic forecasts are **calibrated** (reliability/PIT close to diagonal,
     empirical coverage ≈ nominal).
  3. **One command** rebuilds dataset + models + evaluation report reproducibly.

### Honest expectations (stated up front, not a failure mode)
~2.5 years of history (Previous Runs starts Jan 2024) at 24 h lead steps is a moderate
dataset. Gains over raw ICON-D2 for temperature may be small, especially at short lead time.
That is itself a valid, reportable finding. The deliverable's value is the honest,
well-measured comparison — not a dramatic skill jump.

## 3. Scope

**In scope (Phase 1 / MVP):** data pipeline, canonical dataset, baselines, bias correction,
LightGBM blend, all three probabilistic methods (quantile-GBM, EMOS/NGR, conformal),
full leak-free evaluation + report. Temperature only.

**Out of scope (YAGNI):** own NWP/global model, deep learning, precipitation/wind in the MVP,
any deployment/real-time serving in Phase 1, multiple locations.

## 4. Data Layer

### 4.1 Storage philosophy — two zones (raw → curated)

Separate the **raw API cache** from **analysis-ready tables** (bronze/gold pattern).

```
data/                         # fully gitignored; reproducible via CLI
  raw/                        # exactly what the APIs returned — cache & provenance
    obs/        brightsky_06093_2024-01.parquet, ...
    forecast/   previous_runs/icon_d2/2024-01.parquet, .../ecmwf_ifs025/...
    era5/       era5_2024.parquet, ...
    _manifest.json            # coverage, fetch timestamp, model-version note per chunk
  curated/
    obs.parquet               # normalized, long
    forecasts.parquet         # normalized, long: valid_time, lead_time_h, model, t_fc
    climatology.parquet       # ERA5 normals per (month, hour)
    canonical.parquet         # THE feature table models train on
  wetter.duckdb               # DuckDB views/joins over the parquet files
```

### 4.2 Format choice
- **Parquet:** columnar, compressed, typed, language-agnostic, fast scans; works natively with
  polars/pandas/duckdb.
- **DuckDB:** serverless (single file), reads Parquet directly, expresses the trickiest step —
  time-aligned joins of obs ↔ forecasts ↔ climatology — declaratively in SQL, then materializes
  `canonical.parquet`. Chosen for clean, readable join logic and as a queryable artifact, not
  out of necessity (dataset is small).
- **Rejected:** CSV (untyped/slow), SQLite (row-store, weaker for analytical scans), Postgres
  (server overkill).

### 4.3 Sources and their raw shapes
- **Observations (truth):** Bright Sky `/weather` for station `06093` (hourly, from 2010,
  no API key). Pull from 2010 (cheap, one-time). DWD CDC opendata documented as a
  QC-level alternative.
- **Forecasts (features):** Open-Meteo **Previous Runs API**, one request per model. Returns
  hourly valid-time series with columns `temperature_2m_previous_day1..7` → melted to long
  (`lead_time_h ∈ {24..168}`). Models: `icon_d2`, `icon_eu`, `icon_global`, `gfs_seamless`,
  `ecmwf_ifs025`. Coverage Jan 2024 → present.
- **Climatology/context:** ERA5 via archive-api (hourly) for day/hour normals + optional features.

### 4.4 Idempotency & caching
- Pulls run in **monthly chunks**; each chunk is one cache file. Re-runs **skip** existing
  chunks (unless `--force`) → resumable, no duplicate API calls, call-weighting kept low
  (single variable `temperature_2m`).
- `_manifest.json` tracks coverage + fetch time + model-version note (Open-Meteo models upgrade
  over time; the Previous Runs archive itself is fixed history, so re-pull is stable but
  documented).

### 4.5 Time alignment (the critical correctness detail)
- **Everything in UTC.** `valid_time` = the time the forecast is *for*;
  `issue_time = valid_time − lead_time_h`.
- Join: forecast rows **left join** observations on `valid_time` → attaches the target.
  Observation gaps (station downtime) → row without target is dropped from training but flagged.
- Climatology joined on (month, hour).

### 4.6 `canonical.parquet` schema (one row per valid_time × lead_time)

| Group | Columns |
|---|---|
| Keys | `valid_time` (UTC), `lead_time_h`, `issue_time` |
| Per-model forecast | `t_icon_d2`, `t_icon_eu`, `t_icon_global`, `t_gfs`, `t_ecmwf_ifs025` |
| Multi-model features | `t_mean`, `t_median`, `t_spread` (disagreement), `t_min`, `t_max` |
| Time | `hour`, `doy`, `month`, `hour_sin/cos`, `doy_sin/cos` |
| Climatology | `t_clim`, `t_anom_<model>` (model − clim) |
| Persistence / recent error ⚠️ leak-free | `t_obs_at_issue` (obs at `issue_time` — known at forecast time), `recent_bias_<model>` (trailing model error from data **strictly before** `issue_time`) |
| Meta | `station_vs_grid_elev_diff` (station 62 m vs model grid elevations) |
| Target | `t_obs` (observed temp at `valid_time`) |

**Leakage discipline:** `t_obs_at_issue` is legitimate (available at forecast time = the
persistence feature). `recent_bias_<model>` uses a strictly backward-shifted rolling window —
never any information from at or after `issue_time`.

## 5. Modeling

- **Baselines (the honest floor):** persistence (`t_obs_at_issue` carried forward), climatology
  (`t_clim`), and each **raw model**. Every claim is a skill score against these.
- **Bias correction:** per (model, lead_time, hour, month) mean-error subtraction.
- **Point blend:** LightGBM regressor on the canonical table, `lead_time_h` as a feature,
  predicting `t_obs` (the situational "which model to trust when" engine).
- **Probabilistic layer (the comparison):**
  - **Quantile-GBM:** LightGBM quantile objective at 0.1 / 0.5 / 0.9.
  - **EMOS / NGR:** Gaussian predictive distribution, mean = linear combination of model
    forecasts, variance a function of model spread; parameters fit by **CRPS minimization**
    (scipy). The meteorology-classic method.
  - **Conformal prediction:** split/adaptive conformal around the point blend for
    distribution-free coverage. The exchangeability assumption is violated by time series →
    use a time-series-aware variant (blocked / adaptive); calibration on a disjoint window.

Fit strategy: GBM blend uses `lead_time_h` as a feature (single model); EMOS fit per lead time.

## 6. Evaluation & Honesty

- **Split:** chronological — train (2024–2025) / calibration holdout / test (2026).
  Rolling-origin backtest as a stretch goal.
- **Deterministic metrics:** MAE, RMSE, bias + **skill score** vs each baseline, per lead time.
- **Probabilistic metrics:** **CRPS**, **reliability diagram / PIT histogram**, **sharpness**,
  **coverage** of nominal intervals — per method, per lead time.
- **Leakage guards:** chronological split only; conformal calibration on a disjoint window;
  no future-derived features.
- **Output:** a reproducible **markdown report with generated figures** (produced by
  `wetter report`), comparing all methods across lead times; notebooks are for exploration only.

## 7. Tech Stack & Project Structure

Python 3.13 (venv already present). Libraries: `httpx`, `polars` (default dataframe lib;
`pandas` only where a library needs it, e.g. LightGBM/scikit-learn interop), `duckdb`, `pyarrow`,
`numpy`, `scipy`, `lightgbm`, `scikit-learn`, `mapie` (or hand-rolled conformal), `matplotlib`,
`pydantic`, `typer` (CLI), `pytest`, `ruff`.

```
wetter_prediction/
  pyproject.toml
  src/wetter/
    config.py
    data/      observations.py · forecasts.py · climatology.py · build_dataset.py
    models/    baselines.py · bias_correction.py · blend_gbm.py · emos.py · conformal.py
    eval/      metrics.py · calibration.py · report.py
    cli.py
  notebooks/         # exploratory + final report
  tests/
  data/              # parquet cache (gitignored)
  docs/superpowers/specs/
```

CLI: `wetter pull-obs | pull-forecasts | build-dataset | train | evaluate | report`.

## 8. Phase Plan

- **Phase 1 (MVP, rigor):** full pipeline + canonical dataset + baselines + bias correction
  + GBM blend + all three probabilistic methods + full evaluation/report. Temperature only.
- **Phase 2 (tool):** thin **Streamlit dashboard** — live Lüneburg forecast, blended temperature
  + interval, per-model comparison, "which model was historically best", calibration plots.
- **Phase 3 (optional flex):** MOSMIX logger (cron) building a proprietary, growing dataset of
  **real operational** forecasts; later precipitation/wind.

## 9. Risks & Open Questions

- **Modest dataset / small gains:** mitigated by honest per-lead-time reporting (see §2).
- **Precipitation network is separate/denser** than the climate station — irrelevant for the
  temperature MVP, relevant for Phase 3.
- **Open-Meteo model-version inhomogeneity** over the archive window — documented in the manifest;
  acceptable for the MVP scope.
- No blocking open questions.

## Appendix A — Verified data-source reference (as of 2026-06-28/29)

- **Open-Meteo Previous Runs API:** `https://previous-runs-api.open-meteo.com/v1/forecast`.
  Fixed lead-time offsets via `{variable}_previous_day{N}`, N=1..7 (24..168 h before valid time).
  Coverage: most models from Jan 2024 (GFS 2m temp from Mar 2021). Hourly. Free non-commercial,
  no key, CC BY 4.0. Limits ~10k calls/day, call weight scales with vars × date span.
- **Open-Meteo Historical Forecast API:** `https://historical-forecast-api.open-meteo.com/v1/forecast`.
  NOT used as the honest core — it is a stitched near-analysis, not a true multi-day forecast.
- **Open-Meteo Archive (ERA5):** `https://archive-api.open-meteo.com/v1/archive`. `models=era5`
  (1940+, ~25 km) / `era5_land` (1950+, ~11 km). ~5-day latency. Reanalysis = model product.
- **Bright Sky:** `https://api.brightsky.dev/` — `/weather` (obs + MOSMIX merged), `/synop`
  (raw obs). Query by `dwd_station_id=06093` or lat/lon. Historical from 2010. No key; DWD ToU.
- **DWD CDC opendata:** `https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/`
  — historical/recent split with QC levels. Alternative ground-truth source.
- **Grid snapping:** Open-Meteo returns the nearest grid-cell center + elevation per model
  (ICON-D2 ~2 km, ICON-EU ~7 km, ICON-global ~11 km); use returned elevation for height context.
- **No free long-term archive** of issued MOSMIX or ECMWF Open Data forecasts (~2–3 days rolling
  only) → Phase 3 logger must harvest going forward.
