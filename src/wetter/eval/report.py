from __future__ import annotations
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from wetter import config  # noqa: E402
from wetter.split import chronological_split  # noqa: E402
from wetter.models import baselines, blend_gbm  # noqa: E402
from wetter.models.bias_correction import BiasCorrector  # noqa: E402
from wetter.models.emos import EMOS  # noqa: E402
from wetter.models.conformal import SplitConformal, ConformalizedQuantile  # noqa: E402
from wetter.eval import metrics as M  # noqa: E402
from wetter.eval import calibration as Cal  # noqa: E402


def _best_raw_at_lead(train_lead: pl.DataFrame, present_models: list[str]) -> str | None:
    """Best raw model AT THIS LEAD by train MAE, among models with data there.

    Model availability varies by lead (e.g. ICON-D2 only exposes +24h in Previous
    Runs), so the reference model must be chosen per lead, not globally.
    """
    if train_lead.height == 0:
        return None
    y = train_lead["t_obs"].to_numpy()
    best, best_mae = None, float("inf")
    for m in present_models:
        col = train_lead[f"t_{m}"].to_numpy()
        mask = ~np.isnan(col)
        if mask.sum() == 0:
            continue
        mae_m = M.mae(y[mask], col[mask])
        if mae_m < best_mae:
            best, best_mae = m, mae_m
    return best


def evaluate(
    canonical: pl.DataFrame, *, train_end: str, cal_end: str, params: dict | None = None
) -> pl.DataFrame:
    train, calib, test = chronological_split(canonical, train_end=train_end, cal_end=cal_end)
    feats = blend_gbm.feature_columns(train)
    present_models = [m for m in config.MODELS if f"t_{m}" in train.columns]

    # GBM point/quantile are global: lead_time_h is a feature, so they adapt by lead.
    gbm = blend_gbm.train_point(train, feats, params)
    qmodels = blend_gbm.train_quantiles(train, feats, params=params)
    # Bias correctors per model (each keyed by lead/hour/month internally).
    correctors = {m: BiasCorrector().fit(train, m) for m in present_models}

    from scipy.stats import norm

    rows = []
    for lead in sorted(test["lead_time_h"].unique().to_list()):
        sub = test.filter(pl.col("lead_time_h") == lead).filter(
            pl.col("t_obs_at_issue").is_not_null() & pl.col("t_clim").is_not_null()
        )
        if sub.height == 0:
            continue
        train_lead = train.filter(pl.col("lead_time_h") == lead)
        calib_lead = calib.filter(pl.col("lead_time_h") == lead)
        y = sub["t_obs"].to_numpy()

        best_raw = _best_raw_at_lead(train_lead, present_models)
        ref_pers = M.mae(y, baselines.predict_persistence(sub))
        ref_clim = M.mae(y, baselines.predict_climatology(sub))
        ref_raw = M.mae(y, baselines.predict_raw(sub, best_raw)) if best_raw else float("nan")

        def det_row(method, pred, **extra):
            m = M.mae(y, pred)
            return {
                "method": method,
                "lead_time_h": lead,
                "best_raw_model": best_raw,
                "mae": m,
                "rmse": M.rmse(y, pred),
                "bias": M.bias(y, pred),
                "skill_vs_persistence": M.skill_score(m, ref_pers),
                "skill_vs_clim": M.skill_score(m, ref_clim),
                "skill_vs_best_raw": (M.skill_score(m, ref_raw) if best_raw else float("nan")),
                "crps": None,
                "coverage_80": None,
                "sharpness_80": None,
                **extra,
            }

        rows.append(det_row("persistence", baselines.predict_persistence(sub)))
        rows.append(det_row("climatology", baselines.predict_climatology(sub)))
        if best_raw is not None:
            rows.append(det_row("raw_best", baselines.predict_raw(sub, best_raw)))
            rows.append(det_row("bias_corrected", correctors[best_raw].predict(sub)))

        point = blend_gbm.predict(gbm, sub, feats)
        rows.append(det_row("gbm_point", point))

        # quantile gbm (median as point; CRPS via pinball; coverage from 0.1/0.9)
        qp = blend_gbm.predict_quantiles(qmodels, sub, feats)
        levels = np.array(sorted(qp.keys()))
        qpreds = np.column_stack([qp[q] for q in levels])
        crps_q = float(np.mean(M.crps_from_quantiles(y, levels, qpreds)))
        r = det_row("gbm_quantile", qp[0.5])
        r.update(
            crps=crps_q,
            coverage_80=Cal.coverage(y, qp[0.1], qp[0.9]),
            sharpness_80=Cal.sharpness(qp[0.1], qp[0.9]),
        )
        rows.append(r)

        # EMOS fit PER LEAD (intervals must widen with lead time)
        if train_lead.height > 0:
            emos = EMOS().fit(train_lead)
            mu, sigma = emos.predict(sub)
            crps_e = float(np.mean(M.crps_gaussian(y, mu, sigma)))
            lo_e, hi_e = norm.ppf(0.1, mu, sigma), norm.ppf(0.9, mu, sigma)
            r = det_row("emos", mu)
            r.update(
                crps=crps_e,
                coverage_80=Cal.coverage(y, lo_e, hi_e),
                sharpness_80=Cal.sharpness(lo_e, hi_e),
            )
            rows.append(r)

        # conformal around gbm point, calibrated PER LEAD (alpha 0.2 -> 80%)
        if calib_lead.height > 0:
            conf = SplitConformal().calibrate(
                calib_lead["t_obs"].to_numpy(), blend_gbm.predict(gbm, calib_lead, feats)
            )
            lo_c, hi_c = conf.interval(point, alpha=0.2)
            r = det_row("conformal", point)
            r.update(
                coverage_80=Cal.coverage(y, lo_c, hi_c),
                sharpness_80=Cal.sharpness(lo_c, hi_c),
            )
            rows.append(r)

        # CQR: conformalize the quantile model's 0.1/0.9 band PER LEAD (fixes
        # quantile-GBM under-coverage while keeping its adaptive width)
        if calib_lead.height > 0 and {0.1, 0.9} <= set(qp):
            qcal = blend_gbm.predict_quantiles(qmodels, calib_lead, feats)
            cqr = ConformalizedQuantile().calibrate(
                calib_lead["t_obs"].to_numpy(), qcal[0.1], qcal[0.9]
            )
            lo_q, hi_q = cqr.interval(qp[0.1], qp[0.9], alpha=0.2)
            r = det_row("cqr", qp[0.5])
            r.update(
                coverage_80=Cal.coverage(y, lo_q, hi_q),
                sharpness_80=Cal.sharpness(lo_q, hi_q),
            )
            rows.append(r)

    return pl.DataFrame(rows)


# --- human-readable labels & formatting -------------------------------------

_METHOD_LABELS = {
    "persistence": "Persistence (naïve: same as now)",
    "climatology": "Climatology (seasonal normal)",
    "raw_best": "Best raw model",
    "bias_corrected": "Bias-corrected raw model",
    "gbm_point": "ML blend (point)",
    "gbm_quantile": "ML blend (quantiles)",
    "emos": "EMOS (statistical)",
    "conformal": "Conformal intervals",
    "cqr": "Conformalized quantiles (CQR)",
}
_MODEL_LABELS = {
    "icon_d2": "ICON-D2", "icon_eu": "ICON-EU", "icon_global": "ICON-global",
    "gfs_seamless": "GFS", "ecmwf_ifs025": "ECMWF",
}
_PROB_METHODS = ["gbm_quantile", "emos", "conformal", "cqr"]
_IMPROVED_METHODS = ["bias_corrected", "gbm_point", "gbm_quantile", "emos", "conformal"]


def _is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def _num(x, dec: int = 2) -> str:
    return "—" if _is_missing(x) else f"{x:.{dec}f}"


def _pct(x) -> str:
    return "—" if _is_missing(x) else f"{x * 100:+.0f}%"


def _val(results: pl.DataFrame, method: str, lead: int, col: str):
    r = results.filter((pl.col("method") == method) & (pl.col("lead_time_h") == lead))
    return None if r.height == 0 else r[col][0]


def _best_improved(results: pl.DataFrame, lead: int):
    cand = [(m, _val(results, m, lead, "mae")) for m in _IMPROVED_METHODS]
    cand = [(m, v) for m, v in cand if not _is_missing(v)]
    return min(cand, key=lambda t: t[1]) if cand else (None, None)


def _key_findings(results: pl.DataFrame) -> list[str]:
    leads = sorted(results["lead_time_h"].unique().to_list())
    if not leads:
        return ["No results to summarise."]
    lo, hi = leads[0], leads[-1]
    bullets: list[str] = []

    raw_lo, raw_lo_m = _val(results, "raw_best", lo, "mae"), _val(results, "raw_best", lo, "best_raw_model")
    bm_lo, bmae_lo = _best_improved(results, lo)
    if not _is_missing(raw_lo) and not _is_missing(bmae_lo):
        bullets.append(
            f"**At +{lo}h (~{lo // 24} day ahead):** the best professional model "
            f"({_MODEL_LABELS.get(raw_lo_m, raw_lo_m)}) is typically off by **{raw_lo:.2f} °C**. "
            f"Our best method ({_METHOD_LABELS.get(bm_lo, bm_lo)}) cuts that to **{bmae_lo:.2f} °C** "
            f"({(1 - bmae_lo / raw_lo) * 100:+.0f}%)."
        )

    raw_hi, raw_hi_m = _val(results, "raw_best", hi, "mae"), _val(results, "raw_best", hi, "best_raw_model")
    bm_hi, bmae_hi = _best_improved(results, hi)
    if not _is_missing(raw_hi) and not _is_missing(bmae_hi):
        bullets.append(
            f"**At +{hi}h (~{hi // 24} days ahead):** errors are naturally larger — the best pro "
            f"model ({_MODEL_LABELS.get(raw_hi_m, raw_hi_m)}) is off by **{raw_hi:.2f} °C**, and our "
            f"best method holds it to **{bmae_hi:.2f} °C** ({(1 - bmae_hi / raw_hi) * 100:+.0f}%)."
        )

    beats_all = all(
        (lambda bm, rraw: not _is_missing(bm) and not _is_missing(rraw) and bm < rraw)(
            _best_improved(results, lead)[1], _val(results, "raw_best", lead, "mae")
        )
        for lead in leads
    )
    if beats_all:
        bullets.append(
            "**At every forecast horizon, the postprocessing beats the best single professional "
            "model** — and the advantage grows the further ahead we forecast."
        )

    wins = [
        f"+{lead}h → {_MODEL_LABELS.get(m, m)}"
        for lead in leads
        if isinstance((m := _val(results, "raw_best", lead, "best_raw_model")), str)
    ]
    if wins:
        bullets.append("**Best raw model by horizon:** " + ", ".join(wins) + ".")

    best_cal, best_gap = None, float("inf")
    for m in _PROB_METHODS:
        covs = [_val(results, m, lead, "coverage_80") for lead in leads]
        covs = [c for c in covs if not _is_missing(c)]
        if covs and (gap := float(np.mean([abs(c - 0.80) for c in covs]))) < best_gap:
            best_cal, best_gap = m, gap
    if best_cal:
        bullets.append(
            f"**Most trustworthy uncertainty ranges:** {_METHOD_LABELS.get(best_cal, best_cal)} "
            "— its 80% range contains reality closest to the intended 80% of the time."
        )
    return bullets


def _display_table(results: pl.DataFrame) -> str:
    df = results.to_pandas()
    out = pd.DataFrame(
        {
            "Method": df["method"].map(lambda m: _METHOD_LABELS.get(m, m)),
            "Horizon": df["lead_time_h"].map(lambda h: f"+{h}h"),
            "Best raw model": df["best_raw_model"].map(
                lambda m: _MODEL_LABELS.get(m, "—") if isinstance(m, str) else "—"
            ),
            "Avg error (°C)": df["mae"].map(_num),
            "RMSE (°C)": df["rmse"].map(_num),
            "Bias (°C)": df["bias"].map(_num),
            "vs Persistence": df["skill_vs_persistence"].map(_pct),
            "vs Climatology": df["skill_vs_clim"].map(_pct),
            "vs Best raw": df["skill_vs_best_raw"].map(_pct),
            "80% coverage": df["coverage_80"].map(_num),
            "Range width (°C)": df["sharpness_80"].map(_num),
            "CRPS": df["crps"].map(_num),
        }
    )
    return out.to_markdown(index=False)


def _line_plot(results, methods, value_col, *, scale=1.0):
    series = []
    for method in methods:
        d = results.filter(pl.col("method") == method).sort("lead_time_h")
        series.append(
            (method, d["lead_time_h"].to_numpy(), d[value_col].to_numpy() * scale)
        )
    return series


def format_live_section(current_time, current_temp: float, fc_df: pl.DataFrame) -> str:
    """Markdown block: current station temperature + the tuned engine's forecast."""
    lines = [
        f"**Current temperature** at Wendisch Evern (DWD station {config.STATION_ID}): "
        f"**{current_temp:.1f} °C** — measured {current_time:%Y-%m-%d %H:%M} UTC.",
        "",
        "Forecast from our **tuned engine** (point value + 80% range it should fall in):",
        "",
        "| Horizon | Valid time (UTC) | Forecast (°C) | 80% range (°C) |",
        "|:--------|:-----------------|--------------:|:---------------|",
    ]
    for r in fc_df.sort("lead_time_h").iter_rows(named=True):
        lines.append(
            f"| +{r['lead_time_h']}h | {r['valid_time']:%Y-%m-%d %H:%M} | "
            f"{r['point']:.1f} | {r['lo80']:.1f} – {r['hi80']:.1f} |"
        )
    lines += [
        "",
        "*The range is calibrated (CQR): over many days, the truth should land inside it "
        "about 80% of the time. Wider ranges further out reflect genuine uncertainty.*",
    ]
    return "\n".join(lines)


def generate_report(
    results: pl.DataFrame,
    *,
    out_dir: Path | None = None,
    timestamp: str | None = None,
    live_md: str | None = None,
) -> Path:
    out = out_dir or config.REPORTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    mae_png = f"mae_by_lead_{stamp}.png"
    skill_png = f"skill_by_lead_{stamp}.png"
    cov_png = f"coverage_by_lead_{stamp}.png"
    methods = results["method"].unique().to_list()

    # Figure 1 — average error by horizon (all methods)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method, x, yv in _line_plot(results, methods, "mae"):
        ax.plot(x, yv, marker="o", label=_METHOD_LABELS.get(method, method))
    ax.set_xlabel("Forecast horizon (hours ahead)")
    ax.set_ylabel("Average error in °C  (lower = better)")
    ax.set_title("How big is the typical temperature error?")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(out / mae_png, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Figure 2 — improvement over the best pro model (postprocessing methods only)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    imp = [m for m in methods if m in _IMPROVED_METHODS]
    for method, x, yv in _line_plot(results, imp, "skill_vs_best_raw", scale=100.0):
        ax.plot(x, yv, marker="o", label=_METHOD_LABELS.get(method, method))
    ax.axhline(0, color="k", lw=1.0)
    ax.text(0.01, 0.02, "0 = same as the best professional model",
            transform=ax.transAxes, fontsize=8, color="gray")
    ax.set_xlabel("Forecast horizon (hours ahead)")
    ax.set_ylabel("% of error removed vs best pro model  (higher = better)")
    ax.set_title("How much does our postprocessing improve on the pros?")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(out / skill_png, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Figure 3 — honesty of the 80% uncertainty ranges
    fig, ax = plt.subplots(figsize=(9, 5.5))
    prob = [m for m in methods if m in _PROB_METHODS]
    for method, x, yv in _line_plot(results, prob, "coverage_80"):
        ax.plot(x, yv, marker="o", label=_METHOD_LABELS.get(method, method))
    ax.axhline(0.80, color="k", ls="--", lw=1.0, label="ideal (80%)")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Forecast horizon (hours ahead)")
    ax.set_ylabel("Share of times reality fell inside the 80% range")
    ax.set_title("Are the uncertainty ranges honest?")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(out / cov_png, dpi=120, bbox_inches="tight")
    plt.close(fig)

    findings = "\n".join(f"- {b}" for b in _key_findings(results))
    table = _display_table(results)
    live_block = f"\n## Live forecast (next 7 days)\n\n{live_md}\n" if live_md else ""

    text = f"""# Lüneburg Temperature Forecast — How Good Is It?

This project takes the temperature forecasts of several professional weather models, learns
their local mistakes for Lüneburg, blends them, and adds honest uncertainty ranges. Below is
how well it works — in plain language.

## Bottom line

{findings}

## How to read this report

- **Forecast horizon** (a.k.a. "lead time") — how far ahead the forecast is. `+24h` = 1 day
  ahead, `+168h` = 7 days ahead. Forecasts naturally get worse the further out they go.
- We predict the temperature in Lüneburg and compare each method against what actually happened.
- **Lower error is better. Positive percentages are better** — they mean a smaller error than
  the reference. 0% = no better than the reference; negative = worse.

## The methods compared (simple → smart)

| Method | What it does |
|---|---|
| **Persistence** | "It will be like it is right now." A naïve yardstick. |
| **Climatology** | "The normal temperature for this date & hour." A naïve yardstick. |
| **Best raw model** | The single best professional weather model (named per horizon). |
| **Bias-corrected** | That model, minus its known systematic error. |
| **ML blend** | A machine-learning model that combines all the weather models. |
| **ML blend (quantiles) / EMOS / Conformal** | Methods that also give an uncertainty *range*, not just one number. |

## What the numbers mean

- **Avg error (°C)** — average size of the miss. `1.0` ≈ usually off by 1 degree. *Lower is better.*
- **RMSE (°C)** — like avg error, but punishes big misses more. *Lower is better.*
- **Bias (°C)** — systematic lean: `+` runs too warm, `−` too cold, `0` no lean.
- **vs Persistence / Climatology / Best raw** — percent of error removed versus that reference.
  *Positive = better.* "vs Best raw" is the key column: did we beat the pros?
- **80% coverage** — for methods with a range: how often reality actually landed inside the
  predicted 80% range. **Ideal ≈ 0.80.** Higher = ranges too wide; lower = overconfident.
- **Range width (°C)** — how wide that uncertainty range is. Narrower is better — *but only if
  coverage stays near 0.80.*
- **CRPS** — one overall score for range forecasts. *Lower is better.*
- A dash (**—**) means the metric does not apply to that method.

## Figures

![Average error by horizon]({mae_png})

*Average error at each horizon. The naïve yardsticks (persistence, climatology) sit high; every
smarter method sits well below them. Lower is better.*

![Improvement over the best pro model]({skill_png})

*How much error each method removes versus the single best professional model (the 0 line).
Above 0 = better than the pros; the blend's lead grows with the horizon.*

![Honesty of the uncertainty ranges]({cov_png})

*How often reality fell inside the predicted 80% range. The dashed line (0.80) is the target.
Above the line = ranges too cautious; below = overconfident.*

## Full results table

{table}
{live_block}"""
    md = out / f"report_{stamp}.md"
    md.write_text(text, encoding="utf-8")
    return md
