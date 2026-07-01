from __future__ import annotations
import sys
from datetime import datetime, timezone

import typer

# polars renders tables with Unicode box-drawing chars; force UTF-8 stdout so
# `wetter evaluate` does not crash on a Windows cp1252 console.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

from wetter import config
from wetter.data import (
    build_dataset, climatology, forecasts, live, observations, rain_dataset, single_runs,
)
from wetter.eval import report
from wetter.models import engine, rain

app = typer.Typer(help="Lüneburg temperature postprocessing engine")


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


@app.command("pull-obs")
def pull_obs(start: str = config.OBS_START, end: str = "", force: bool = False) -> None:
    df = observations.fetch_observations(start, end or _today(), force=force)
    typer.echo(f"obs rows: {df.height}")


@app.command("pull-forecasts")
def pull_forecasts(start: str = config.FORECAST_START, end: str = "", force: bool = False) -> None:
    df = forecasts.fetch_forecasts(start, end or _today(), force=force)
    typer.echo(f"forecast rows: {df.height}")


@app.command("pull-era5")
def pull_era5(start: str = config.OBS_START, end: str = "", force: bool = False) -> None:
    df = climatology.fetch_era5(start, end or _today(), force=force)
    typer.echo(f"era5 rows: {df.height}")


@app.command("build-dataset")
def build_dataset_cmd() -> None:
    path = build_dataset.build()
    typer.echo(f"wrote {path}")


@app.command("pull-runs")
def pull_runs(start: str = config.SINGLE_RUNS_START, end: str = "", force: bool = False) -> None:
    df = single_runs.fetch_runs(start, end or _today(), force=force)
    typer.echo(f"single-run rows: {df.height}")


@app.command("build-hourly")
def build_hourly_cmd(force: bool = False) -> None:
    path = build_dataset.build_hourly(force=force)
    typer.echo(f"wrote {path}")


@app.command("build-rain")
def build_rain_cmd(force_obs: bool = False) -> None:
    path = rain_dataset.build_rain(force_obs=force_obs)
    typer.echo(f"wrote {path}")


@app.command("train-rain")
def train_rain_cmd() -> None:
    import polars as pl

    canon = pl.read_parquet(config.CURATED_DIR / "canonical_rain.parquet")
    art = rain.train_rain_engine(canon)
    path = rain.save_rain_engine(art)
    typer.echo(f"trained rain engine -> {path}")
    typer.echo(f"thresholds {art['thresholds']} | base rate {art['base_rate']:.3f}")


@app.command("train-hourly")
def train_hourly(tune_end: str = "2026-04-01", cal_window_days: int = 30) -> None:
    import polars as pl

    canon = pl.read_parquet(config.CURATED_DIR / "canonical_hourly.parquet")
    art = engine.train_engine(canon, tune_end=tune_end, cal_window_days=cal_window_days)
    path = engine.save_engine(art, engine.HOURLY_ENGINE_PATH)
    typer.echo(f"trained hourly engine -> {path}")
    typer.echo(f"leads {art['leads'][0]}..{art['leads'][-1]}h  params: {art['params']}")


@app.command("serve")
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the Lüneburg weather website."""
    import uvicorn

    typer.echo(f"Lüneburg weather → http://{host}:{port}")
    uvicorn.run("wetter.web.app:app", host=host, port=port, log_level="warning")


@app.command("forecast")
def forecast_cmd(hours: int = 24) -> None:
    """Hourly Lüneburg temperature forecast for the next N hours (the 'app' view)."""
    if not engine.HOURLY_ENGINE_PATH.exists():
        typer.echo("No hourly engine yet — run `wetter build-hourly` then `wetter train-hourly`.")
        raise typer.Exit(1)
    art = engine.load_engine(engine.HOURLY_ENGINE_PATH)
    current_time, current_temp = live.fetch_current_obs()
    leads = [h for h in art["leads"] if 1 <= h <= hours]
    fc = engine.forecast(art, current_time, current_temp, live.fetch_live_forecast(), leads=leads)
    md = report.format_live_section(current_time, current_temp, fc)
    typer.echo(md)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = config.REPORTS_DIR / f"forecast_{stamp}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# Lüneburg — hourly temperature forecast (next {hours} h)\n\n{md}\n", encoding="utf-8")
    typer.echo(f"\nwrote {out}")


@app.command("train")
def train(tune_end: str = "2025-10-01", cal_window_days: int = 120) -> None:
    import polars as pl

    canon = pl.read_parquet(config.CURATED_DIR / "canonical.parquet")
    art = engine.train_engine(canon, tune_end=tune_end, cal_window_days=cal_window_days)
    path = engine.save_engine(art)
    typer.echo(f"trained tuned engine -> {path}")
    typer.echo(f"tuned params: {art['params']}")


@app.command("evaluate")
def evaluate(train_end: str = "2025-07-01", cal_end: str = "2025-10-01") -> None:
    import polars as pl

    canon = pl.read_parquet(config.CURATED_DIR / "canonical.parquet")
    params = engine.load_engine()["params"] if engine.ENGINE_PATH.exists() else None
    res = report.evaluate(canon, train_end=train_end, cal_end=cal_end, params=params)
    typer.echo(res)


@app.command("report")
def report_cmd(train_end: str = "2025-07-01", cal_end: str = "2025-10-01") -> None:
    import polars as pl

    canon = pl.read_parquet(config.CURATED_DIR / "canonical.parquet")
    params, live_md = None, "_Run `wetter train` first to enable the live forecast._"
    if engine.ENGINE_PATH.exists():
        art = engine.load_engine()
        params = art.get("params")
        try:
            current_time, current_temp = live.fetch_current_obs()
            fc = engine.forecast(art, current_time, current_temp, live.fetch_live_forecast())
            live_md = report.format_live_section(current_time, current_temp, fc)
        except Exception as exc:  # noqa: BLE001
            live_md = f"_Live forecast unavailable: {type(exc).__name__}: {exc}_"
    res = report.evaluate(canon, train_end=train_end, cal_end=cal_end, params=params)
    path = report.generate_report(res, live_md=live_md)
    typer.echo(f"wrote {path}")


if __name__ == "__main__":
    app()
