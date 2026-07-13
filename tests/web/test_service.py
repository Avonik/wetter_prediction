import polars as pl
from datetime import datetime, timedelta, timezone
import httpx
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


def test_fetch_current_falls_back_to_latest_real_station_observation(monkeypatch):
    forecast_source = 10
    observation_source = 20
    fallback = {
        "sources": [
            {"id": forecast_source, "observation_type": "forecast", "station_name": "WENDISCH EVERN"},
            {"id": observation_source, "observation_type": "current", "station_name": "WENDISCH EVERN"},
        ],
        "weather": [
            {
                "source_id": observation_source,
                "timestamp": "2026-07-13T19:00:00+00:00",
                "temperature": 17.2,
            },
            {
                "source_id": observation_source,
                "timestamp": "2026-07-13T20:00:00+00:00",
                "temperature": 16.8,
                "condition": "dry",
            },
            {
                "source_id": forecast_source,
                "timestamp": "2026-07-13T21:00:00+00:00",
                "temperature": 18.0,
            },
        ],
    }
    calls = []

    def fake_get_json(url, params):
        calls.append((url, params))
        if len(calls) == 1:
            request = httpx.Request("GET", url)
            response = httpx.Response(404, request=request)
            raise httpx.HTTPStatusError("not found", request=request, response=response)
        return fallback

    monkeypatch.setattr(service.io, "get_json", fake_get_json)
    cur = service.fetch_current()

    assert cur["temperature"] == 16.8
    assert cur["time"] == "2026-07-13T20:00:00+00:00"
    assert cur["data_notice"] == {
        "kind": "station_stale",
        "station": "Wendisch Evern",
        "observed_at": "2026-07-13T20:00:00+00:00",
    }
    assert calls[1][1]["dwd_station_id"] == "06093"


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


def test_current_precip_uses_exact_recent_window(monkeypatch):
    seen = {}

    def fake_get_json(url, params):
        seen.update(params)
        return {"weather": []}

    monkeypatch.setattr(service.io, "get_json", fake_get_json)
    issue = datetime(2026, 7, 2, 15, tzinfo=timezone.utc)
    service.current_precip(issue)
    assert seen["date"] == "2026-07-02T11:00:00Z"
    assert seen["last_date"] == "2026-07-02T15:00:00Z"


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


def test_blend_live_pop_uses_probability_floor():
    assert service._blend_live_pop(0.02, 30.0) == 0.3
    assert service._blend_live_pop(0.45, 30.0) == 0.45
    assert service._blend_live_pop(0.2, None) == 0.2


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
