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
    assert cur["icon"] == "☀️"  # clear-day -> sun
    assert cur["humidity"] == 55


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
