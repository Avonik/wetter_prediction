import polars as pl
from datetime import datetime, timezone
from wetter.data import build_dataset as bd


def _dt(d, h):
    return datetime(2024, 1, d, h, tzinfo=timezone.utc)


def _forecasts():
    # long format: temperature for 2 models + cloud for 2 models, with grid elevation
    return pl.DataFrame(
        {
            "valid_time": [_dt(2, 0)] * 4,
            "lead_time_h": [24, 24, 24, 24],
            "model": ["icon_d2", "gfs_seamless", "icon_d2", "gfs_seamless"],
            "variable": ["t", "t", "cloud", "cloud"],
            "value": [12.0, 9.0, 80.0, 60.0],
            "grid_elev": [50.0, 30.0, 50.0, 30.0],
        }
    )


def test_build_canonical_has_target_features_and_elevation():
    obs = pl.DataFrame({"valid_time": [_dt(1, 0), _dt(2, 0)], "t_obs": [10.0, 11.0]})
    clim = pl.DataFrame({"month": [1], "hour": [0], "t_clim": [3.0]})
    canon = bd.build_canonical(obs, _forecasts(), clim)
    row = canon.row(0, named=True)
    assert row["t_obs"] == 11.0  # target = obs at valid_time
    assert row["issue_time"] == _dt(1, 0)  # valid - 24h
    assert row["t_obs_at_issue"] == 10.0  # persistence feature
    assert row["t_clim"] == 3.0
    assert row["t_mean"] == 10.5  # mean(12, 9)
    assert row["cloud_mean"] == 70.0  # mean(80, 60)
    assert row["station_vs_grid_elev_diff"] == 62.0 - 40.0  # station 62 m - mean grid 40 m


def test_build_canonical_reused_for_hourly_leads():
    # single-runs-shaped input (no run_time col, hourly leads 1 & 2) reuses build_canonical
    obs = pl.DataFrame(
        {"valid_time": [_dt(1, 0), _dt(1, 1), _dt(1, 2)], "t_obs": [10.0, 10.5, 11.0]}
    )
    fc = pl.DataFrame(
        {
            "valid_time": [_dt(1, 1), _dt(1, 2), _dt(1, 1), _dt(1, 2)],
            "lead_time_h": [1, 2, 1, 2],
            "model": ["icon_d2", "icon_d2", "gfs_seamless", "gfs_seamless"],
            "variable": ["t", "t", "t", "t"],
            "value": [10.6, 11.2, 10.4, 10.9],
            "grid_elev": [22.0, 22.0, 30.0, 30.0],
        }
    )
    clim = pl.DataFrame({"month": [1, 1, 1], "hour": [0, 1, 2], "t_clim": [3.0, 3.1, 3.2]})
    canon = bd.build_canonical(obs, fc, clim)
    assert sorted(canon["lead_time_h"].unique().to_list()) == [1, 2]
    row = canon.filter(
        (pl.col("lead_time_h") == 1) & (pl.col("valid_time") == _dt(1, 1))
    ).row(0, named=True)
    assert row["issue_time"] == _dt(1, 0)  # = the run time
    assert row["t_obs_at_issue"] == 10.0  # obs at run time (nowcast-fusion signal)
    assert row["t_obs"] == 10.5
