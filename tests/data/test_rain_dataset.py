from datetime import datetime, timezone

import polars as pl

from wetter.data import rain_dataset as rd


def _dt(d, h):
    return datetime(2025, 9, d, h, tzinfo=timezone.utc)


def _runs():
    # 2 models predict precip at (valid, lead); one model wet, one dry -> prob 0.5
    rows = []
    for m, p in [("icon_d2", 0.5), ("gfs_seamless", 0.0)]:
        rows.append({"valid_time": _dt(2, 0), "lead_time_h": 24, "model": m,
                     "variable": "precip", "value": p, "grid_elev": 22.0})
        rows.append({"valid_time": _dt(2, 0), "lead_time_h": 24, "model": m,
                     "variable": "cloud", "value": 80.0, "grid_elev": 22.0})
    return pl.DataFrame(rows).with_columns(pl.col("valid_time").cast(pl.Datetime("us", "UTC")))


def test_build_rain_canonical_targets_and_agreement():
    obs = pl.DataFrame({"valid_time": [_dt(2, 0)], "p_obs": [0.3]}).with_columns(
        pl.col("valid_time").cast(pl.Datetime("us", "UTC"))
    )
    canon = rd.build_rain_canonical(obs, _runs())
    row = canon.row(0, named=True)
    assert row["p_obs"] == 0.3
    assert row["rain_occ"] == 1  # 0.3 >= 0.1
    assert row["precip_prob"] == 0.5  # 1 of 2 models >= 0.1 mm
    assert row["precip_mean"] == 0.25  # mean(0.5, 0.0)
    assert row["precip_icon_d2"] == 0.5
    assert row["cloud_mean"] == 80.0
    assert row["issue_time"] == _dt(1, 0)


def test_rain_occ_zero_when_dry():
    obs = pl.DataFrame({"valid_time": [_dt(2, 0)], "p_obs": [0.0]}).with_columns(
        pl.col("valid_time").cast(pl.Datetime("us", "UTC"))
    )
    canon = rd.build_rain_canonical(obs, _runs())
    assert canon.row(0, named=True)["rain_occ"] == 0


def test_build_live_rain_rows_keeps_live_pop():
    issue = _dt(1, 0)
    valid = _dt(1, 1)
    rows = []
    for model, pop in [("gfs_seamless", 30.0), ("ecmwf_ifs025", 50.0)]:
        for variable, value in [
            ("precip", 0.0), ("pop", pop), ("cloud", 90.0), ("rh", 80.0),
        ]:
            rows.append(
                {
                    "valid_time": valid, "model": model, "variable": variable,
                    "value": value, "grid_elev": 22.0,
                }
            )
    live = pl.DataFrame(rows).with_columns(pl.col("valid_time").cast(pl.Datetime("us", "UTC")))
    out = rd.build_live_rain_rows(issue, live, [1], current_precip=0.0)
    row = out.row(0, named=True)
    assert row["pop_mean"] == 40.0
    assert row["pop_max"] == 50.0
