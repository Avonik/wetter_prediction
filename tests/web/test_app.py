from fastapi.testclient import TestClient
from wetter.web import app as webapp
from wetter.web import service

client = TestClient(webapp.app)


def test_api_forecast_returns_bundle(monkeypatch):
    monkeypatch.setattr(
        service,
        "bundle",
        lambda force=False: {"place": "Lüneburg", "hourly": [], "daily": []},
    )
    r = client.get("/api/forecast")
    assert r.status_code == 200
    assert r.json()["place"] == "Lüneburg"
    assert r.headers["cache-control"] == "no-cache"


def test_api_forecast_handles_missing_engine(monkeypatch):
    def boom(force=False):
        raise FileNotFoundError("no engine")

    monkeypatch.setattr(service, "bundle", boom)
    r = client.get("/api/forecast")
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "ENGINE_NOT_AVAILABLE"


def test_api_forecast_does_not_expose_internal_error(monkeypatch):
    def boom(force=False):
        raise RuntimeError("secret upstream details")

    monkeypatch.setattr(service, "bundle", boom)
    r = client.get("/api/forecast")
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "FORECAST_NOT_AVAILABLE"
    assert "secret" not in r.text


def test_health_endpoints(monkeypatch, tmp_path):
    monkeypatch.setattr(
        service,
        "cache_status",
        lambda: {
            "has_snapshot": True,
            "stale": False,
            "refreshing": False,
            "last_refresh_failed": False,
        },
    )
    engine_path = tmp_path / "engine.joblib"
    engine_path.touch()
    monkeypatch.setattr(service.engine, "HOURLY_ENGINE_PATH", engine_path)

    assert client.get("/health/live").json() == {"status": "ok"}
    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


def test_index_page_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "Lüneburg" in r.text


def test_api_forecast_passes_force_refresh(monkeypatch):
    seen = []
    monkeypatch.setattr(
        service,
        "bundle",
        lambda force=False: seen.append(force) or {"place": "Lüneburg"},
    )

    assert client.get("/api/forecast?force=true").status_code == 200
    assert seen == [True]
