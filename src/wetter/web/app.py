from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from wetter.web import service

_STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    service.start_background_refresh()
    try:
        yield
    finally:
        service.stop_background_refresh()


app = FastAPI(title="Lüneburg Weather Engine", lifespan=lifespan)


@app.get("/api/forecast")
def api_forecast(force: bool = False):
    try:
        return JSONResponse(service.bundle(force=force), headers={"Cache-Control": "no-cache"})
    except FileNotFoundError:
        return JSONResponse(
            {
                "error": {
                    "code": "ENGINE_NOT_AVAILABLE",
                    "message": "The forecast model is not available.",
                }
            },
            status_code=503,
        )
    except Exception:  # noqa: BLE001 - details are logged by the snapshot store
        return JSONResponse(
            {
                "error": {
                    "code": "FORECAST_NOT_AVAILABLE",
                    "message": "No forecast is currently available. Please try again shortly.",
                }
            },
            status_code=503,
        )


@app.get("/health/live")
def health_live():
    return {"status": "ok"}


@app.get("/health/ready")
def health_ready():
    status = service.cache_status()
    ready = service.engine.HOURLY_ENGINE_PATH.exists() and status["has_snapshot"]
    return JSONResponse(
        {"status": "ready" if ready else "not_ready", "forecast_cache": status},
        status_code=200 if ready else 503,
    )


@app.get("/")
def index():
    return FileResponse(_STATIC / "index.html")


app.mount("/static", StaticFiles(directory=_STATIC), name="static")
