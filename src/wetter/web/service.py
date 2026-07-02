from __future__ import annotations
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import polars as pl

from wetter import config
from wetter.data import io, live, rain_dataset
from wetter.models import engine
from wetter.models import rain as rain_model

_TTL_S = 600  # serve a cached bundle for 10 min to avoid hammering the upstream APIs
_CACHE: dict = {"ts": -1e18, "data": None}

_ALERTS_URL = "https://api.brightsky.dev/alerts"


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


def fetch_current() -> dict:
    payload = io.get_json(live._CURRENT_URL, {"dwd_station_id": config.STATION_ID, "tz": "UTC"})
    w = payload["weather"]
    t = datetime.fromisoformat(w["timestamp"]).astimezone(timezone.utc)
    return {
        "time": t.isoformat(),
        "temperature": w.get("temperature"),
        "condition": w.get("condition"),
        "icon_raw": w.get("icon"),  # Bright Sky icon string (day/night hint); emoji set in assemble
        "wind_speed": w.get("wind_speed"),
        "humidity": w.get("relative_humidity"),
        "cloud_cover": w.get("cloud_cover"),
        "precip": w.get("precipitation"),  # mm/h now (is it raining?)
    }


def fetch_alerts() -> list[dict]:
    """Active DWD weather warnings for Lüneburg (via Bright Sky /alerts)."""
    try:
        payload = io.get_json(_ALERTS_URL, {"lat": config.LAT, "lon": config.LON, "tz": "UTC"})
    except Exception:  # noqa: BLE001  (warnings are optional; never break the page)
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


def _rain_category(p: float | None, p1: float | None) -> str:
    if p is None or p < 0.2:
        return "trocken"
    if p1 is not None and p1 >= 0.4:
        return "Regen"
    if p >= 0.5:
        return "leichter Regen"
    return "evtl. Schauer"


def _rain_bundle(issue: datetime, live_fc_long: pl.DataFrame) -> tuple[dict, dict]:
    """Our calibrated P(rain) over the whole week: hourly detail (0-48 h shown) plus a
    daily rain chance (max hourly probability per German day) — all from our model, so
    long-range days honestly regress toward climatology instead of over-confident 100%."""
    rain_hourly: dict[str, dict] = {}
    daily_chance: dict[str, float] = {}
    try:
        art = rain_model.load_rain_engine()
    except (FileNotFoundError, OSError):
        return rain_hourly, daily_chance

    rows = rain_dataset.build_live_rain_rows(issue, live_fc_long, list(range(1, 169)))
    probs = rain_model.predict_rain(art, rows)
    p01, p1 = probs.get(0.1), probs.get(1.0)
    if p01 is None:
        return rain_hourly, daily_chance
    for i, vt in enumerate(rows["valid_time"].to_list()):
        p = float(p01[i])
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

    # rain: calibrated hourly P(rain) + daily rain chance
    rain_hourly, daily_chance = _rain_bundle(issue, live_fc_long)
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
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def bundle(*, force: bool = False) -> dict:
    if not force and _CACHE["data"] is not None and (time.monotonic() - _CACHE["ts"] < _TTL_S):
        return _CACHE["data"]
    hourly_art = engine.load_engine(engine.HOURLY_ENGINE_PATH)
    data = assemble(fetch_current(), live.fetch_live_forecast(), hourly_art)
    data["alerts"] = fetch_alerts()
    _CACHE.update(ts=time.monotonic(), data=data)
    return data
