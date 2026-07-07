import polars as pl
from wetter.data import live


def test_parse_current_obs():
    payload = {"weather": {"timestamp": "2026-06-29T10:00:00+00:00", "temperature": 17.3}}
    t, temp = live.parse_current_obs(payload)
    assert temp == 17.3
    assert t.tzinfo is not None
    assert t.year == 2026 and t.hour == 10


def test_parse_live_forecast_long_with_elevation():
    payload = {
        "elevation": 25.0,
        "hourly": {
            "time": ["2026-06-30T00:00", "2026-06-30T01:00"],
            "temperature_2m": [15.0, 14.0],
            "cloud_cover": [50.0, 60.0],
        },
    }
    df = live.parse_live_forecast(payload, "icon_d2")
    assert set(df["variable"].unique().to_list()) == {"t", "cloud"}
    assert df["grid_elev"][0] == 25.0
    assert df.filter(pl.col("variable") == "t").height == 2
    assert df["valid_time"].dtype == pl.Datetime("us", "UTC")


def test_parse_live_forecast_includes_live_pop():
    payload = {
        "hourly": {
            "time": ["2026-06-30T00:00"],
            "precipitation_probability": [35],
        },
    }
    df = live.parse_live_forecast(payload, "gfs_seamless")
    row = df.row(0, named=True)
    assert row["variable"] == "pop"
    assert row["value"] == 35.0
