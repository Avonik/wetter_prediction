import polars as pl
from wetter.data import forecasts as fc

PAYLOAD = {
    "elevation": 30.0,
    "hourly": {
        "time": ["2024-01-02T00:00", "2024-01-02T01:00"],
        "temperature_2m_previous_day1": [1.0, 1.5],
        "temperature_2m_previous_day2": [0.5, None],
        "cloud_cover_previous_day1": [80.0, 60.0],
    },
}


def test_parse_previous_runs_melts_variables_and_leads():
    df = fc.parse_previous_runs(PAYLOAD, "icon_d2")
    assert set(df.columns) == {"valid_time", "lead_time_h", "model", "variable", "value", "grid_elev"}
    assert df["valid_time"].dtype == pl.Datetime("us", "UTC")
    # temp: 2 times x 2 leads = 4 minus one null = 3; cloud: 2 -> total 5
    assert df.height == 5
    assert set(df["variable"].unique().to_list()) == {"t", "cloud"}
    assert df["grid_elev"][0] == 30.0
    t24_h1 = df.filter(
        (pl.col("variable") == "t")
        & (pl.col("lead_time_h") == 24)
        & (pl.col("valid_time").dt.hour() == 1)
    )["value"][0]
    assert t24_h1 == 1.5
    cloud = df.filter(pl.col("variable") == "cloud")["value"].to_list()
    assert cloud == [80.0, 60.0]


def test_fetch_forecasts_caches_per_model(tmp_path, monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(
        fc.io, "get_json", lambda url, params: (calls.__setitem__("n", calls["n"] + 1) or PAYLOAD)
    )
    df = fc.fetch_forecasts(
        "2024-01-02", "2024-01-02", models=["icon_d2", "gfs_seamless"], cache_dir=tmp_path
    )
    assert set(df["model"].unique().to_list()) == {"icon_d2", "gfs_seamless"}
    n1 = calls["n"]
    fc.fetch_forecasts(
        "2024-01-02", "2024-01-02", models=["icon_d2", "gfs_seamless"], cache_dir=tmp_path
    )
    assert calls["n"] == n1  # fully cached
