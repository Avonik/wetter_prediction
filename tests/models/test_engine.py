import numpy as np
import polars as pl
from datetime import datetime, timezone, timedelta
from wetter.models import engine


def _canon(n=600, seed=2):
    rng = np.random.default_rng(seed)
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = [t0 + timedelta(hours=i) for i in range(n)]
    frames = []
    for lead in (24, 48):
        base = rng.normal(8, 5, n)
        y = base - 1.0 + rng.normal(0, 1, n)
        frames.append(
            pl.DataFrame(
                {
                    "valid_time": times,
                    "lead_time_h": np.full(n, lead),
                    "t_icon_d2": base, "t_gfs_seamless": base + rng.normal(0, 0.5, n),
                    "t_mean": base, "t_median": base, "t_spread": np.abs(rng.normal(1, 0.2, n)),
                    "t_min": base - 1, "t_max": base + 1,
                    "cloud_mean": rng.uniform(0, 100, n), "wind_mean": rng.uniform(0, 20, n),
                    "rh_mean": rng.uniform(40, 100, n), "pmsl_mean": rng.normal(1013, 8, n),
                    "rad_mean": rng.uniform(0, 600, n),
                    "month": [t.month for t in times], "hour": [t.hour for t in times],
                    "hour_sin": np.zeros(n), "hour_cos": np.ones(n),
                    "doy_sin": np.zeros(n), "doy_cos": np.ones(n),
                    "t_obs_at_issue": y + rng.normal(0, 2, n),
                    "t_clim": np.full(n, 8.0),
                    "station_vs_grid_elev_diff": np.full(n, 22.0),
                    "recent_bias_icon_d2": rng.normal(0, 0.3, n),
                    "t_obs": y,
                }
            )
        )
    return pl.concat(frames)


def test_train_save_load_and_predict(tmp_path):
    canon = _canon()
    art = engine.train_engine(canon, tune_end="2024-01-20", cal_window_days=5)
    assert {"point", "quantiles", "cqr_q", "features", "climatology", "recent_bias"} <= set(art)
    assert set(art["cqr_q"].keys()) == {24, 48}
    assert art["leads"] == [24, 48]

    p = engine.save_engine(art, tmp_path / "engine.joblib")
    art2 = engine.load_engine(p)

    point, lo, hi = engine.predict(art2, canon.head(7))
    assert len(point) == len(lo) == len(hi) == 7
    assert np.all(hi >= lo)  # interval well-formed
    assert np.all((point >= lo) & (point <= hi))  # point always inside its own interval


def test_recent_bias_by_lead_extracts_latest():
    canon = _canon()
    rb = engine._recent_bias_by_lead(canon, ["icon_d2"])
    assert set(rb["icon_d2"].keys()) == {24, 48}


def _live_fc(issue, models=("icon_d2", "gfs_seamless")):
    varmap = {"t": 6.0, "rh": 80.0, "cloud": 50.0, "wind": 10.0, "pmsl": 1010.0, "rad": 100.0}
    rows = []
    for lead in (24, 48, 72, 96, 120, 144, 168):
        vt = issue + timedelta(hours=lead)
        for m in models:
            for v, val in varmap.items():
                rows.append(
                    {"valid_time": vt, "model": m, "variable": v, "value": val, "grid_elev": 30.0}
                )
    return pl.DataFrame(rows).with_columns(pl.col("valid_time").cast(pl.Datetime("us", "UTC")))


def test_build_live_rows_has_all_engine_features_and_predicts():
    canon = _canon()
    art = engine.train_engine(canon, tune_end="2024-01-20", cal_window_days=5)
    issue = datetime(2024, 1, 15, 12, tzinfo=timezone.utc)
    df = engine.build_live_rows(art, issue, 6.0, _live_fc(issue))
    assert df.height == len(art["leads"]) == 2  # defaults to the engine's trained leads
    missing = [c for c in art["features"] if c not in df.columns]
    assert missing == []  # live rows must carry every feature the model expects
    point, lo, hi = engine.predict(art, df)
    assert len(point) == 2
    assert np.all(hi >= lo)


def test_build_live_rows_accepts_custom_leads():
    canon = _canon()
    art = engine.train_engine(canon, tune_end="2024-01-20", cal_window_days=5)
    issue = datetime(2024, 1, 15, 12, tzinfo=timezone.utc)
    df = engine.build_live_rows(art, issue, 6.0, _live_fc(issue), leads=[24])
    assert df.height == 1
    assert df["lead_time_h"].to_list() == [24]
