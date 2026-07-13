from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone

import pytest

from wetter.web.cache import ForecastStore


def test_store_persists_and_reloads_snapshot(tmp_path):
    path = tmp_path / "forecast.json"
    store = ForecastStore(lambda: {"value": 42}, path, ttl_s=3600)

    result = store.get()

    assert result["value"] == 42
    assert result["forecast_status"]["stale"] is False
    assert path.exists()

    reloaded = ForecastStore(lambda: pytest.fail("fresh disk snapshot should be used"), path, ttl_s=3600)
    assert reloaded.get()["value"] == 42


def test_stale_snapshot_returns_immediately_while_refresh_runs(tmp_path):
    path = tmp_path / "forecast.json"
    old_time = datetime.now(timezone.utc) - timedelta(hours=1)
    path.write_text(
        json.dumps({"refreshed_at": old_time.isoformat(), "data": {"value": "old"}}),
        encoding="utf-8",
    )
    started = threading.Event()
    release = threading.Event()

    def build():
        started.set()
        assert release.wait(timeout=2)
        return {"value": "new"}

    store = ForecastStore(build, path, ttl_s=60)
    result = store.get()

    assert result["value"] == "old"
    assert result["forecast_status"]["stale"] is True
    assert started.wait(timeout=1)
    release.set()
    assert store.refresh(blocking=True) is True
    assert store.get()["value"] == "new"


def test_failed_background_refresh_keeps_last_known_good_snapshot(tmp_path):
    path = tmp_path / "forecast.json"
    old_time = datetime.now(timezone.utc) - timedelta(hours=1)
    path.write_text(
        json.dumps({"refreshed_at": old_time.isoformat(), "data": {"value": "old"}}),
        encoding="utf-8",
    )
    attempted = threading.Event()

    def build():
        attempted.set()
        raise RuntimeError("upstream down")

    store = ForecastStore(build, path, ttl_s=60)
    result = store.get()

    assert result["value"] == "old"
    assert attempted.wait(timeout=1)
    assert store.status()["has_snapshot"] is True


def test_cold_store_propagates_refresh_failure(tmp_path):
    def build():
        raise RuntimeError("upstream down")

    store = ForecastStore(build, tmp_path / "forecast.json", ttl_s=60)
    with pytest.raises(RuntimeError, match="upstream down"):
        store.get()

