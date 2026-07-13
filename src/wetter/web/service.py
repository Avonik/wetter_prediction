from __future__ import annotations
import logging
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

import httpx
import polars as pl

from wetter import config
from wetter.data import io, live, observations, rain_dataset
from wetter.models import engine
from wetter.models import rain as rain_model
from wetter.web.cache import ForecastStore

_TTL_S = 600  # serve a cached bundle for 10 min to avoid hammering the upstream APIs
_SNAPSHOT_PATH = config.DATA_DIR / "cache" / "forecast_snapshot.json"
logger = logging.getLogger(__name__)

_ALERTS_URL = "https://api.brightsky.dev/alerts"
_RECENT_WEATHER_URL = "https://api.brightsky.dev/weather"
_CURRENT_FALLBACK_DAYS = 7


def _weather_emoji(condition, icon_raw, cloud_cover, precip) -> str:
    """Granular sky symbol from condition + cloud cover + precipitation."""
    is_night = "night" in (icon_raw or "")
    cond = (condition or "").lower()
    if cond == "thunderstorm":
        return "⛈️"
    if cond == "snow":
        return "❄️"
    if cond in ("sleet", "hail"):
        return "🌨️"
    if cond == "fog":
        return "🌫️"
    if cond == "rain" or (precip or 0) > 0.05:
        # showers (some breaks) vs steady rain (fully overcast)
        return "🌧️" if (cloud_cover if cloud_cover is not None else 100) >= 85 else "🌦️"
    cc = cloud_cover if cloud_cover is not None else 0
    if is_night:
        return "🌙" if cc < 40 else "☁️"
    if cc < 15:
        return "☀️"      # klar
    if cc < 50:
        return "🌥️"      # sonnig, wenig Wolken
    if cc < 85:
        return "⛅"       # heiter bis wolkig
    return "☁️"          # bedeckt

# pretty labels for the raw model lines
MODEL_LABELS = {
    "icon_d2": "ICON-D2", "icon_eu": "ICON-EU", "icon_global": "ICON",
    "gfs_seamless": "GFS", "ecmwf_ifs025": "ECMWF",
}

_TZ = ZoneInfo("Europe/Berlin")
_DOW = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]


def _current_fields(w: dict) -> dict:
    t = datetime.fromisoformat(w["timestamp"]).astimezone(timezone.utc)
    return {
        "time": t.isoformat(),
        "temperature": w.get("temperature"),
        "condition": w.get("condition"),
        "icon_raw": w.get("icon"),  # Bright Sky icon string (day/night hint); emoji set in assemble
        "wind_speed": w.get("wind_speed", w.get("wind_speed_10")),
        "humidity": w.get("relative_humidity"),
        "cloud_cover": w.get("cloud_cover"),
        "precip": w.get("precipitation", w.get("precipitation_60")),
    }


def _latest_station_observation() -> tuple[dict, str]:
    """Return the newest real observation for our station, excluding forecast rows."""
    now = datetime.now(timezone.utc)
    lo = now - timedelta(days=_CURRENT_FALLBACK_DAYS)
    payload = io.get_json(
        _RECENT_WEATHER_URL,
        {
            "dwd_station_id": config.STATION_ID,
            "date": lo.isoformat().replace("+00:00", "Z"),
            "last_date": now.isoformat().replace("+00:00", "Z"),
            "tz": "UTC",
        },
    )
    sources = {s["id"]: s for s in payload.get("sources", [])}
    observed = [
        row
        for row in payload.get("weather", [])
        if sources.get(row.get("source_id"), {}).get("observation_type")
        in {"current", "historical", "synop"}
        and row.get("temperature") is not None
    ]
    if not observed:
        raise LookupError("No recent station observation available")
    latest = max(observed, key=lambda row: datetime.fromisoformat(row["timestamp"]))
    source = sources.get(latest.get("source_id"), {})
    station_name = (source.get("station_name") or "Wendisch Evern").title()
    return latest, station_name


def fetch_current() -> dict:
    try:
        payload = io.get_json(
            live._CURRENT_URL, {"dwd_station_id": config.STATION_ID, "tz": "UTC"}
        )
        return _current_fields(payload["weather"])
    except (httpx.HTTPStatusError, io.Transient, KeyError, TypeError, ValueError):
        w, station_name = _latest_station_observation()
        current = _current_fields(w)
        current["data_notice"] = {
            "kind": "station_stale",
            "station": station_name,
            "observed_at": current["time"],
        }
        return current


def fetch_alerts() -> list[dict]:
    """Active DWD weather warnings for Lüneburg (via Bright Sky /alerts)."""
    try:
        payload = io.get_json(_ALERTS_URL, {"lat": config.LAT, "lon": config.LON, "tz": "UTC"})
    except Exception:  # noqa: BLE001  (warnings are optional; never break the page)
        logger.warning("Weather alerts are currently unavailable", exc_info=True)
        return []
    out = []
    for a in payload.get("alerts", []) or []:
        out.append(
            {
                "headline": a.get("headline_de") or a.get("event_de") or "Wetterwarnung",
                "event": a.get("event_de"),
                "severity": (a.get("severity") or "").lower(),
                "onset": a.get("onset"),
                "expires": a.get("expires"),
                "instruction": a.get("instruction_de"),
            }
        )
    return out


def current_precip(issue: datetime) -> float | None:
    """Most recent OBSERVED precipitation (mm/h) at the station — the rain-persistence
    signal 'is it raining right now'. Read from the same DWD /weather obs the model was
    trained on (p_obs); Bright Sky's /current_weather frequently omits precipitation."""
    try:
        lo = (issue - timedelta(hours=4)).isoformat().replace("+00:00", "Z")
        hi = issue.isoformat().replace("+00:00", "Z")
        payload = io.get_json(
            observations._URL,
            {"dwd_station_id": config.STATION_ID, "date": lo, "last_date": hi, "tz": "UTC"},
        )
    except Exception:  # noqa: BLE001 — persistence is optional; never break the page
        logger.warning("Current precipitation observation is unavailable", exc_info=True)
        return None
    best = None
    for r in payload.get("weather", []) or []:
        p = r.get("precipitation")
        if p is None:
            continue
        t = datetime.fromisoformat(r["timestamp"]).astimezone(timezone.utc)
        if t <= issue and (best is None or t >= best[0]):
            best = (t, float(p))
    return best[1] if best else None


def _rain_category(p: float | None, p1: float | None) -> str:
    if p is None or p < 0.2:
        return "trocken"
    if p1 is not None and p1 >= 0.4:
        return "Regen"
    if p >= 0.5:
        return "leichter Regen"
    return "evtl. Schauer"


def _blend_live_pop(model_p: float, pop_percent: float | None) -> float:
    """Use upstream live PoP as a floor for the displayed chance of any rain.

    The trained rain model predicts observed hourly precipitation >= 0.1 mm. That is
    good for measurable rain, but it under-reads trace drizzle in the first hours when
    NWP amount fields are 0.0 while probability-of-precipitation is still elevated.
    """
    if pop_percent is None:
        return model_p
    return max(model_p, min(max(float(pop_percent) / 100.0, 0.0), 1.0))


def _rain_bundle(
    issue: datetime, live_fc_long: pl.DataFrame, current_precip: float | None = None
) -> tuple[dict, dict]:
    """Our calibrated P(rain) over the whole week: hourly detail (0-48 h shown) plus a
    daily rain chance (max hourly probability per German day) — all from our model, so
    long-range days honestly regress toward climatology instead of over-confident 100%."""
    rain_hourly: dict[str, dict] = {}
    daily_chance: dict[str, float] = {}
    try:
        art = rain_model.load_rain_engine()
    except (FileNotFoundError, OSError):
        return rain_hourly, daily_chance

    rows = rain_dataset.build_live_rain_rows(
        issue, live_fc_long, list(range(1, 169)), current_precip
    )
    probs = rain_model.predict_rain(art, rows)
    p01, p1 = probs.get(0.1), probs.get(1.0)
    if p01 is None:
        return rain_hourly, daily_chance
    for i, row in enumerate(rows.iter_rows(named=True)):
        vt = row["valid_time"]
        p = _blend_live_pop(float(p01[i]), row.get("pop_mean"))
        pm = float(p1[i]) if p1 is not None else None
        rain_hourly[vt.isoformat()] = {"p": round(p, 2), "cat": _rain_category(p, pm)}
        d = vt.astimezone(_TZ).strftime("%d.%m.")
        daily_chance[d] = max(daily_chance.get(d, 0.0), p)
    return rain_hourly, daily_chance


def _models_hourly(live_fc_long: pl.DataFrame, issue: datetime, hours: int) -> dict:
    hi = issue + timedelta(hours=hours)
    temp = live_fc_long.filter(
        (pl.col("variable") == "t")
        & (pl.col("valid_time") >= issue)
        & (pl.col("valid_time") <= hi)
    )
    out = {}
    for m in [m for m in config.MODELS if m in temp["model"].unique().to_list()]:
        d = temp.filter(pl.col("model") == m).sort("valid_time")
        out[MODEL_LABELS.get(m, m)] = [
            {"t": r["valid_time"].isoformat(), "v": r["value"]}
            for r in d.iter_rows(named=True)
        ]
    return out


def _our_daily_highlow(our_hourly: pl.DataFrame, days: int = 7) -> list[dict]:
    """Daily high/low per calendar day in German time, from OUR model's hourly
    forecast across the full week (the engine now spans 1..168 h)."""
    per_day = (
        our_hourly.with_columns(
            pl.col("valid_time").dt.convert_time_zone("Europe/Berlin").dt.date().alias("d")
        )
        .group_by("d")
        .agg(pl.col("point").min().alias("tmin"), pl.col("point").max().alias("tmax"))
        .sort("d")
    )
    # start at tomorrow: "today" would only cover the remaining hours from now, which is
    # misleading late in the day (past rain / the day's real high already happened).
    today = datetime.now(_TZ).date()
    per_day = per_day.filter(pl.col("d") > today).head(days)
    return [
        {
            "dow": _DOW[r["d"].weekday()],
            "date": r["d"].strftime("%d.%m."),
            "date_iso": r["d"].isoformat(),
            "tmin": round(float(r["tmin"])),
            "tmax": round(float(r["tmax"])),
        }
        for r in per_day.iter_rows(named=True)
    ]


def _fc_rows(fc: pl.DataFrame) -> list[dict]:
    return [
        {
            "t": r["valid_time"].isoformat(),
            "lead": int(r["lead_time_h"]),
            "point": round(float(r["point"]), 1),
            "lo": round(float(r["lo80"]), 1),
            "hi": round(float(r["hi80"]), 1),
        }
        for r in fc.sort("lead_time_h").iter_rows(named=True)
    ]


def assemble(current: dict, live_fc_long: pl.DataFrame, hourly_art: dict) -> dict:
    issue = datetime.fromisoformat(current["time"]).replace(minute=0, second=0, microsecond=0)
    ctemp = float(current["temperature"])

    # Bright Sky's current obs sometimes omits wind/cloud/humidity — fill from the
    # live model forecast at the current hour so the hero card is complete.
    current = dict(current)
    data_notice = current.pop("data_notice", None)
    at_issue = live_fc_long.filter(pl.col("valid_time") == issue)

    def _now_var(var: str):
        s = at_issue.filter(pl.col("variable") == var)["value"]
        return round(float(s.mean()), 0) if s.len() else None

    if current.get("wind_speed") is None:
        current["wind_speed"] = _now_var("wind")
    if current.get("cloud_cover") is None:
        current["cloud_cover"] = _now_var("cloud")
    if current.get("humidity") is None:
        current["humidity"] = _now_var("rh")
    current["icon"] = _weather_emoji(
        current.get("condition"), current.get("icon_raw"),
        current.get("cloud_cover"), current.get("precip"),
    )

    # tolerate engines trained before the "leads" field existed
    all_leads = [lead for lead in (hourly_art.get("leads") or range(1, 169)) if 1 <= lead <= 168]
    fc_all = engine.forecast(hourly_art, issue, ctemp, live_fc_long, leads=all_leads)
    hourly = _fc_rows(fc_all.filter(pl.col("lead_time_h") <= 48))  # strip + chart (next 48 h)
    daily = _our_daily_highlow(fc_all, 7)  # our model's daily high/low over the week

    # rain: calibrated hourly P(rain) + daily rain chance. Feed the current observed rain
    # (persistence) from the DWD obs, falling back to Bright Sky's current_weather field.
    cprecip = current_precip(issue)
    if cprecip is None:
        cprecip = current.get("precip")
    rain_hourly, daily_chance = _rain_bundle(issue, live_fc_long, cprecip)
    for h in hourly:
        r = rain_hourly.get(h["t"])
        h["rain_p"] = r["p"] if r else None
        h["rain_cat"] = r["cat"] if r else None
    for d in daily:
        d["rain_p"] = daily_chance.get(d["date"])

    return {
        "place": "Lüneburg",
        "station": "Wendisch Evern · DWD 06093",
        "current": current,
        "hourly": hourly,
        "daily": daily,
        "models_hourly": _models_hourly(live_fc_long, issue, 48),
        "data_notice": data_notice,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@lru_cache(maxsize=1)
def _load_hourly_engine() -> dict:
    return engine.load_engine(engine.HOURLY_ENGINE_PATH)


def _build_bundle() -> dict:
    hourly_art = _load_hourly_engine()
    data = assemble(fetch_current(), live.fetch_live_forecast(), hourly_art)
    data["alerts"] = fetch_alerts()
    return data


_FORECAST_STORE = ForecastStore(_build_bundle, _SNAPSHOT_PATH, ttl_s=_TTL_S)


def bundle(*, force: bool = False) -> dict:
    return _FORECAST_STORE.get(force=force)


def start_background_refresh() -> None:
    _FORECAST_STORE.start()


def stop_background_refresh() -> None:
    _FORECAST_STORE.stop()


def cache_status() -> dict:
    return _FORECAST_STORE.status()
