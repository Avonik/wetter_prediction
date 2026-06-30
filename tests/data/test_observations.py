import polars as pl
from wetter.data import observations as obs

PAYLOAD = {
    "weather": [
        {"timestamp": "2024-01-01T00:00:00+00:00", "temperature": 3.2},
        {"timestamp": "2024-01-01T01:00:00+00:00", "temperature": 2.9},
        {"timestamp": "2024-01-01T02:00:00+00:00", "temperature": None},
    ]
}


def test_parse_weather_types_and_utc():
    df = obs.parse_weather(PAYLOAD)
    assert df.columns == ["valid_time", "t_obs"]
    assert df["valid_time"].dtype == pl.Datetime("us", "UTC")
    assert df["t_obs"].to_list()[:2] == [3.2, 2.9]
    assert df["t_obs"].to_list()[2] is None


def test_fetch_observations_uses_cache_and_dedupes(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_get_json(url, params):
        calls["n"] += 1
        return PAYLOAD

    monkeypatch.setattr(obs.io, "get_json", fake_get_json)
    df = obs.fetch_observations("2024-01-01", "2024-01-01", cache_dir=tmp_path)
    assert df.height == 3
    assert df["valid_time"].is_sorted()
    n_after_first = calls["n"]
    obs.fetch_observations("2024-01-01", "2024-01-01", cache_dir=tmp_path)  # cached
    assert calls["n"] == n_after_first
