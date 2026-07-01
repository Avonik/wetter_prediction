from __future__ import annotations
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from wetter.web import service

_STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Lüneburg Weather Engine")


@app.get("/api/forecast")
def api_forecast(force: bool = False):
    try:
        return JSONResponse(service.bundle(force=force))
    except FileNotFoundError:
        return JSONResponse(
            {"error": "No trained engine. Run `wetter build-hourly`, `wetter train-hourly`, "
                      "`wetter train` first."},
            status_code=503,
        )
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=502)


@app.get("/")
def index():
    return FileResponse(_STATIC / "index.html")


app.mount("/static", StaticFiles(directory=_STATIC), name="static")
