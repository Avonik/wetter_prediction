from fastapi.testclient import TestClient
from wetter.web import app as webapp
from wetter.web import service

client = TestClient(webapp.app)


def test_api_forecast_returns_bundle(monkeypatch):
    monkeypatch.setattr(
        service, "bundle", lambda force=False: {"place": "Lüneburg", "hourly": [], "daily": []}
    )
    r = client.get("/api/forecast")
    assert r.status_code == 200
    assert r.json()["place"] == "Lüneburg"


def test_api_forecast_handles_missing_engine(monkeypatch):
    def boom(force=False):
        raise FileNotFoundError("no engine")

    monkeypatch.setattr(service, "bundle", boom)
    r = client.get("/api/forecast")
    assert r.status_code == 503
    assert "engine" in r.json()["error"].lower()


def test_index_page_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "Lüneburg" in r.text
