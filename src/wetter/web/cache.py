from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ForecastSnapshot:
    data: dict
    refreshed_at: datetime


class ForecastStore:
    """Thread-safe last-known-good store with disk persistence and background refresh."""

    def __init__(self, builder: Callable[[], dict], path: Path, *, ttl_s: float) -> None:
        self._builder = builder
        self._path = path
        self._ttl_s = ttl_s
        self._snapshot: ForecastSnapshot | None = None
        self._loaded = False
        self._state_lock = threading.RLock()
        self._refresh_lock = threading.Lock()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._last_error: str | None = None

    def _ensure_loaded(self) -> None:
        with self._state_lock:
            if self._loaded:
                return
            self._loaded = True
            if not self._path.exists():
                return
            try:
                payload = json.loads(self._path.read_text(encoding="utf-8"))
                refreshed_at = datetime.fromisoformat(payload["refreshed_at"])
                if refreshed_at.tzinfo is None:
                    refreshed_at = refreshed_at.replace(tzinfo=timezone.utc)
                self._snapshot = ForecastSnapshot(payload["data"], refreshed_at)
                logger.info("Loaded persisted forecast snapshot from %s", self._path)
            except (OSError, ValueError, KeyError, TypeError):
                logger.exception("Could not load persisted forecast snapshot")

    def _is_stale(self, snapshot: ForecastSnapshot) -> bool:
        return (_utc_now() - snapshot.refreshed_at).total_seconds() >= self._ttl_s

    def _persist(self, snapshot: ForecastSnapshot) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp = self._path.with_name(f"{self._path.name}.{os.getpid()}.tmp")
        payload = {
            "refreshed_at": snapshot.refreshed_at.isoformat(),
            "data": snapshot.data,
        }
        temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temp.replace(self._path)

    def refresh(self, *, blocking: bool = True) -> bool:
        """Build and publish a new snapshot. Return False if another refresh owns the lock."""
        if not self._refresh_lock.acquire(blocking=blocking):
            return False
        started = _utc_now()
        try:
            data = self._builder()
            snapshot = ForecastSnapshot(data=data, refreshed_at=_utc_now())
            try:
                self._persist(snapshot)
            except (OSError, TypeError, ValueError):
                logger.exception("Could not persist forecast snapshot")
            with self._state_lock:
                self._snapshot = snapshot
                self._loaded = True
                self._last_error = None
            elapsed = (_utc_now() - started).total_seconds()
            logger.info("Forecast snapshot refreshed in %.2fs", elapsed)
            return True
        except Exception as exc:
            with self._state_lock:
                self._last_error = type(exc).__name__
            logger.exception("Forecast snapshot refresh failed")
            raise
        finally:
            self._refresh_lock.release()

    def refresh_async(self) -> None:
        if self._refresh_lock.locked():
            return

        def run() -> None:
            try:
                self.refresh(blocking=False)
            except Exception:  # noqa: BLE001 - the last-known-good snapshot remains available
                pass

        threading.Thread(target=run, name="forecast-refresh", daemon=True).start()

    def get(self, *, force: bool = False) -> dict:
        self._ensure_loaded()
        with self._state_lock:
            snapshot = self._snapshot

        if force:
            self.refresh(blocking=True)
            with self._state_lock:
                snapshot = self._snapshot
        elif snapshot is None:
            # Only a truly cold installation waits. Once a snapshot exists, requests never
            # block on upstream weather services again.
            self.refresh(blocking=True)
            with self._state_lock:
                snapshot = self._snapshot
        elif self._is_stale(snapshot):
            self.refresh_async()

        if snapshot is None:  # defensive: refresh() either publishes or raises
            raise RuntimeError("No forecast snapshot available")

        stale = self._is_stale(snapshot)
        result = dict(snapshot.data)
        result["forecast_status"] = {
            "stale": stale,
            "refreshing": self._refresh_lock.locked(),
            "refreshed_at": snapshot.refreshed_at.isoformat(),
        }
        return result

    def status(self) -> dict:
        self._ensure_loaded()
        with self._state_lock:
            snapshot = self._snapshot
            last_error = self._last_error
        return {
            "has_snapshot": snapshot is not None,
            "stale": self._is_stale(snapshot) if snapshot else None,
            "refreshing": self._refresh_lock.locked(),
            "last_refresh_failed": last_error is not None,
        }

    def start(self) -> None:
        with self._state_lock:
            if self._worker and self._worker.is_alive():
                return
            self._stop.clear()
            self._worker = threading.Thread(
                target=self._worker_loop, name="forecast-refresh-worker", daemon=True
            )
            self._worker.start()

    def _worker_loop(self) -> None:
        poll_s = min(max(self._ttl_s / 2, 5.0), 60.0)
        while not self._stop.is_set():
            try:
                self.get()
            except Exception:  # noqa: BLE001 - retry later; health endpoint exposes readiness
                pass
            self._stop.wait(poll_s)

    def stop(self) -> None:
        self._stop.set()
        worker = self._worker
        if worker and worker.is_alive():
            worker.join(timeout=2.0)
