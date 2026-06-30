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
    out = features.add_multimodel_features(df)
    assert out["t_mean"][0] == 5.0
    assert out["t_spread"][0] == 0.0


def test_interaction_features_built_from_aux():
    df = pl.DataFrame(
        {
            "cloud_mean": [20.0, 90.0],
            "rad_mean": [0.0, 600.0],  # night, day
            "wind_mean": [3.0, 3.0],
            "rh_mean": [80.0, 40.0],
            "hour_cos": [1.0, -1.0],
        }
    )
    out = features.add_interaction_features(df)
    assert {"clearnight_cool", "cloud_x_wind", "cloud_x_hourcos", "rh_night"} <= set(out.columns)
    assert out["clearnight_cool"].to_list() == [80.0, 0.0]  # (100-cloud)*night
    assert out["cloud_x_wind"].to_list() == [60.0, 270.0]
    assert out["rh_night"].to_list() == [80.0, 0.0]


def test_interaction_features_noop_without_inputs():
    df = pl.DataFrame({"t_mean": [5.0]})
    out = features.add_interaction_features(df)
    assert out.columns == ["t_mean"]


def test_recent_bias_is_leak_free():
    # issue times t0<t1<t2 at same lead; model is +2 too warm consistently
    obs = pl.DataFrame(
        {"valid_time": [_dt(0), _dt(1), _dt(2)], "t_obs": [10.0, 11.0, 12.0]}
    )
    df = pl.DataFrame(
        {
            "issue_time": [_dt(0), _dt(1), _dt(2)],
            "valid_time": [_dt(0), _dt(1), _dt(2)],
            "lead_time_h": [24, 24, 24],
            "t_icon_d2": [12.0, 13.0, 14.0],  # error +2 each
        }
    ).join(obs, on="valid_time")
    out = features.add_recent_bias(df, obs, "icon_d2", window=10)
    rb = out.sort("issue_time")["recent_bias_icon_d2"].to_list()
    assert rb[0] is None
    assert abs(rb[1] - 2.0) < 1e-9
    assert abs(rb[2] - 2.0) < 1e-9
