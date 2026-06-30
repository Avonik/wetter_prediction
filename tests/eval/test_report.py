import polars as pl
import numpy as np
from datetime import datetime, timezone, timedelta
from wetter.eval import report as R


def _canon(n=900, seed=0):
    rng = np.random.default_rng(seed)
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = [t0 + timedelta(hours=i) for i in range(n)]
    t_icon = rng.normal(10, 5, n)
    y = t_icon - 2.0 + rng.normal(0, 1, n)
    # persistence proxy: correlated with the target but with realistic error
    # (must NOT equal y, or the persistence baseline MAE is 0 and skill divides by zero)
    persist = y + rng.normal(0, 2.0, n)
    return pl.DataFrame(
        {
            "valid_time": times,
            "lead_time_h": np.full(n, 24),
            "t_icon_d2": t_icon, "t_mean": t_icon, "t_median": t_icon,
            "t_spread": np.abs(rng.normal(1, 0.2, n)), "t_min": t_icon, "t_max": t_icon,
            "hour": [t.hour for t in times], "month": [t.month for t in times],
            "hour_sin": np.zeros(n), "hour_cos": np.ones(n),
            "doy_sin": np.zeros(n), "doy_cos": np.ones(n),
            "t_obs_at_issue": persist, "t_clim": np.full(n, 10.0), "t_obs": y,
        }
    )


def test_evaluate_produces_rows_for_methods():
    canon = _canon()
    res = R.evaluate(canon, train_end="2024-01-20", cal_end="2024-01-28")
    methods = set(res["method"].unique().to_list())
    assert {
        "persistence", "climatology", "raw_best", "bias_corrected",
        "gbm_point", "gbm_quantile", "emos", "conformal",
    } <= methods
    mae_gbm = res.filter(pl.col("method") == "gbm_point")["mae"][0]
    mae_raw = res.filter(pl.col("method") == "raw_best")["mae"][0]
    assert mae_gbm <= mae_raw + 1e-6


def test_evaluate_handles_null_persistence_rows():
    # Observation gaps leave some t_obs_at_issue null in the test window;
    # persistence MAE must stay finite (rows filtered to the evaluable set).
    canon = _canon()
    canon = canon.with_columns(
        pl.when(pl.arange(0, canon.height) % 50 == 0)
        .then(None)
        .otherwise(pl.col("t_obs_at_issue"))
        .alias("t_obs_at_issue")
    )
    res = R.evaluate(canon, train_end="2024-01-20", cal_end="2024-01-28")
    pers_mae = res.filter(pl.col("method") == "persistence")["mae"][0]
    assert pers_mae == pers_mae  # not NaN
    assert pers_mae > 0


def _canon_multilead(n=600, seed=1):
    rng = np.random.default_rng(seed)
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = [t0 + timedelta(hours=i) for i in range(n)]
    frames = []
    for lead in (24, 48):
        gfs = rng.normal(10, 5, n)
        y = gfs - 1.0 + rng.normal(0, 1, n)
        persist = y + rng.normal(0, 2, n)
        icon = gfs + rng.normal(0, 0.5, n) if lead == 24 else np.full(n, np.nan)
        frames.append(
            pl.DataFrame(
                {
                    "valid_time": times,
                    "lead_time_h": np.full(n, lead),
                    "t_icon_d2": icon, "t_gfs_seamless": gfs,
                    "t_mean": gfs, "t_median": gfs, "t_spread": np.zeros(n),
                    "t_min": gfs, "t_max": gfs,
                    "hour": [t.hour for t in times], "month": [t.month for t in times],
                    "hour_sin": np.zeros(n), "hour_cos": np.ones(n),
                    "doy_sin": np.zeros(n), "doy_cos": np.ones(n),
                    "t_obs_at_issue": persist, "t_clim": np.full(n, 10.0), "t_obs": y,
                }
            )
        )
    return pl.concat(frames)


def test_best_raw_is_chosen_per_lead():
    # ICON-D2 exists only at +24h; at +48h the reference must fall back to GFS,
    # and raw_best/bias_corrected must stay finite (Bug 1 regression).
    canon = _canon_multilead()
    res = R.evaluate(canon, train_end="2024-01-15", cal_end="2024-01-20")
    for lead in (24, 48):
        rb = res.filter((pl.col("method") == "raw_best") & (pl.col("lead_time_h") == lead))
        assert rb.height == 1
        assert rb["mae"][0] == rb["mae"][0]  # not NaN
    lead48_model = (
        res.filter((pl.col("method") == "raw_best") & (pl.col("lead_time_h") == 48))
        ["best_raw_model"][0]
    )
    assert lead48_model == "gfs_seamless"


def test_generate_report_timestamped_with_live(tmp_path):
    canon = _canon()
    res = R.evaluate(canon, train_end="2024-01-20", cal_end="2024-01-28")
    md = R.generate_report(
        res, out_dir=tmp_path, timestamp="20240101-000000", live_md="LIVE-SECTION-MARKER"
    )
    # timestamped, not overwritten
    assert md.name == "report_20240101-000000.md"
    for fig in [
        "mae_by_lead_20240101-000000.png",
        "skill_by_lead_20240101-000000.png",
        "coverage_by_lead_20240101-000000.png",
    ]:
        assert (tmp_path / fig).exists()
    text = md.read_text(encoding="utf-8")
    assert "How to read this report" in text
    assert "Bottom line" in text
    assert "Avg error (°C)" in text  # friendly table header
    assert "—" in text  # not-applicable metrics rendered as a dash, not "nan"
    assert "nan" not in text.lower()
    assert "Live forecast" in text and "LIVE-SECTION-MARKER" in text
    assert "mae_by_lead_20240101-000000.png" in text  # image link uses stamped name


def test_format_live_section():
    fc = pl.DataFrame(
        {
            "lead_time_h": [24, 48],
            "valid_time": [
                datetime(2026, 7, 1, 12, tzinfo=timezone.utc),
                datetime(2026, 7, 2, 12, tzinfo=timezone.utc),
            ],
            "point": [20.1, 21.4],
            "lo80": [18.0, 18.5],
            "hi80": [22.5, 24.0],
        }
    )
    md = R.format_live_section(datetime(2026, 6, 30, 11, tzinfo=timezone.utc), 17.3, fc)
    assert "Current temperature" in md
    assert "17.3" in md
    assert "+24h" in md and "+48h" in md
    assert "20.1" in md
