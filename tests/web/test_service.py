import polars as pl
from datetime import datetime, timedelta, timezone
from wetter.web import service


def test_fetch_current_parses(monkeypatch):
    payload = {
        "weather": {
            "timestamp": "2026-06-30T09:00:00+00:00", "temperature": 18.4,
            "condition": "dry", "icon": "clear-day", "wind_speed": 12.0,
            "relative_humidity": 55, "cloud_cover": 10,
        }
    }
    monkeypatch.setattr(service.io, "get_json", lambda url, params: payload)
    cur = service.fetch_current()
    assert cur["temperature"] == 18.4
    assert cur["icon_raw"] == "clear-day"  # emoji is derived later in assemble()
    assert cur["humidity"] == 55


def test_weather_emoji_specificity():
    assert service._weather_emoji("dry", "clear-day", 5, 0) == "☀️"
    assert service._weather_emoji("dry", "clear-day", 30, 0) == "🌥️"
    assert service._weather_emoji("dry", "partly-cloudy-day", 65, 0) == "⛅"
    assert service._weather_emoji("dry", "cloudy", 95, 0) == "☁️"
    assert service._weather_emoji("rain", "rain", 70, 0.3) == "🌦️"  # showers, some breaks
    assert service._weather_emoji("rain", "rain", 95, 1.0) == "🌧️"  # steady, overcast
    assert service._weather_emoji("dry", "clear-night", 5, 0) == "🌙"
    assert service._weather_emoji("thunderstorm", "thunderstorm", 90, 2) == "⛈️"


def test_current_precip_takes_latest_observed(monkeypatch):
    payload = {"weather": [
        {"timestamp": "2026-07-02T13:00:00+00:00", "precipitation": 0.0},
        {"timestamp": "2026-07-02T14:00:00+00:00", "precipitation": 1.2},  # latest non-null <= issue
        {"timestamp": "2026-07-02T15:00:00+00:00", "precipitation": None},
        {"timestamp": "2026-07-02T16:00:00+00:00", "precipitation": 3.0},  # after issue -> ignored
    ]}
    monkeypatch.setattr(service.io, "get_json", lambda url, params: payload)
    issue = datetime(2026, 7, 2, 15, tzinfo=timezone.utc)
    assert service.current_precip(issue) == 1.2


def test_current_precip_tolerates_failure(monkeypatch):
    def boom(url, params):
        raise RuntimeError("obs down")

    monkeypatch.setattr(service.io, "get_json", boom)
    assert service.current_precip(datetime(2026, 7, 2, 15, tzinfo=timezone.utc)) is None


def test_fetch_alerts_parses(monkeypatch):
    payload = {"alerts": [{
        "headline_de": "Amtliche Warnung vor Windböen", "event_de": "WINDBÖEN",
        "severity": "Moderate", "onset": "2026-07-02T10:00+00:00",
        "expires": "2026-07-02T16:00+00:00", "instruction_de": "Vorsicht",
    }]}
    monkeypatch.setattr(service.io, "get_json", lambda url, params: payload)
    alerts = service.fetch_alerts()
    assert len(alerts) == 1
    assert alerts[0]["headline"] == "Amtliche Warnung vor Windböen"
    assert alerts[0]["severity"] == "moderate"  # lowercased


def test_fetch_alerts_tolerates_failure(monkeypatch):
    def boom(url, params):
        raise RuntimeError("down")

    monkeypatch.setattr(service.io, "get_json", boom)
    assert service.fetch_alerts() == []  # never breaks the page


def test_models_hourly_window_and_labels():
    issue = datetime(2026, 6, 30, 0, tzinfo=timezone.utc)
    rows = []
    for h in [1, 2, 100]:  # 100h is outside the 48h window
        for m, base in [("icon_d2", 10.0), ("ecmwf_ifs025", 11.0)]:
            rows.append(
                {"valid_time": issue + timedelta(hours=h), "model": m,
                 "variable": "t", "value": base + h, "grid_elev": 22.0}
            )
    df = pl.DataFrame(rows).with_columns(pl.col("valid_time").cast(pl.Datetime("us", "UTC")))
    out = service._models_hourly(df, issue, 48)
    assert set(out.keys()) == {"ICON-D2", "ECMWF"}  # pretty labels
    assert len(out["ICON-D2"]) == 2  # h=100 excluded by the window
