const LANGUAGE_KEY = "wetter.language";

const I18N = {
  de: {
    htmlLang: "de",
    locale: "de-DE",
    title: "Lüneburg Wetter · lokal getunt",
    languageLabel: "Sprache",
    switchAria: "Auf englisches Layout umschalten",
    humidity: "Feuchte",
    cloudCover: "Bewölkung",
    hoursTitle: "Nächste Stunden",
    hoursSubtitle: "Temperatur + kalibrierte Regenwahrscheinlichkeit, Stunde für Stunde.",
    chartTitle: "48-Stunden-Verlauf & Modellvergleich",
    chartSubtitle:
      '<b style="color:#ffd166">Unser Modell</b> (dicke Linie + 80 %-Band) gegen die großen Wettermodelle (gestrichelt).',
    dailyTitle: "7-Tage-Vorhersage",
    dailySubtitle: "Tageshöchst- und Tiefstwerte · unser feingetuntes Modell (deutsche Zeit).",
    note:
      "<strong>Wie diese Vorhersage entsteht.</strong> Sie kombiniert mehrere Profi-Wettermodelle " +
      "(ICON-D2, ICON-EU, ECMWF, GFS), korrigiert deren lokale Fehler für die Station Wendisch Evern " +
      "und bezieht die aktuelle Messung mit ein. Ergebnis: Temperatur mit Unsicherheitsband und " +
      "eine <em>kalibrierte</em> Regenwahrscheinlichkeit — sagt sie 30 %, regnet es über viele Tage " +
      "auch etwa in 30 % der Fälle. Die Genauigkeit ist gegen die tatsächlich gemessenen Werte " +
      "geprüft (Backtest). Kein offizielles Wetterprodukt.",
    loading: "Lade aktuelle Vorhersage…",
    footerData:
      'Wetterdaten: <a href="https://open-meteo.com">Open-Meteo</a> (CC BY 4.0) · ' +
      'Stationsdaten: <a href="https://brightsky.dev">Bright Sky</a> / DWD · ' +
      "kein offizielles Wetterprodukt.",
    supportProject: "Projekt unterstützen",
    updatedPrefix: "Stand:",
    currentFallback: "aktuell",
    until: "bis",
    errorTitle: "⚠️ Vorhersage konnte nicht geladen werden.",
    bandLabel: "80 %-Band",
    ourModelLabel: "Unser Modell",
    conditions: {
      clear: "klar",
      cloudy: "bewölkt",
      dry: "trocken",
      fog: "Nebel",
      hail: "Hagel",
      rain: "Regen",
      sleet: "Schneeregen",
      snow: "Schnee",
      thunderstorm: "Gewitter",
      trocken: "trocken",
    },
  },
  en: {
    htmlLang: "en",
    locale: "en-GB",
    title: "Lüneburg Weather · locally tuned",
    languageLabel: "Language",
    switchAria: "Switch to German layout",
    humidity: "Humidity",
    cloudCover: "Clouds",
    hoursTitle: "Next Hours",
    hoursSubtitle: "Temperature + calibrated rain probability, hour by hour.",
    chartTitle: "48-Hour Trend & Model Comparison",
    chartSubtitle:
      '<b style="color:#ffd166">Our model</b> (thick line + 80% band) against the major weather models (dashed).',
    dailyTitle: "7-Day Forecast",
    dailySubtitle: "Daily highs and lows · our locally tuned model (German time).",
    note:
      "<strong>How this forecast is made.</strong> It combines several professional weather models " +
      "(ICON-D2, ICON-EU, ECMWF, GFS), corrects their local bias for the Wendisch Evern station, " +
      "and includes the latest observation. The result: temperature with an uncertainty band and " +
      "a <em>calibrated</em> rain probability — when it says 30%, rain occurs on roughly 30% of similar days. " +
      "Accuracy is checked against the values actually measured in a backtest. Not an official weather product.",
    loading: "Loading current forecast…",
    footerData:
      'Weather data: <a href="https://open-meteo.com">Open-Meteo</a> (CC BY 4.0) · ' +
      'Station data: <a href="https://brightsky.dev">Bright Sky</a> / DWD · ' +
      "not an official weather product.",
    supportProject: "Support the project",
    updatedPrefix: "Updated:",
    currentFallback: "current",
    until: "until",
    errorTitle: "⚠️ Forecast could not be loaded.",
    bandLabel: "80% band",
    ourModelLabel: "Our model",
    conditions: {
      clear: "clear",
      cloudy: "cloudy",
      dry: "dry",
      fog: "fog",
      hail: "hail",
      rain: "rain",
      sleet: "sleet",
      snow: "snow",
      thunderstorm: "thunderstorm",
      trocken: "dry",
      "leichter regen": "light rain",
      "evtl. schauer": "possible showers",
    },
  },
};

const DOW_EN = { Mo: "Mon", Di: "Tue", Mi: "Wed", Do: "Thu", Fr: "Fri", Sa: "Sat", So: "Sun" };

const state = {
  lang: readLanguage(),
  forecast: null,
  error: "",
};

const fmtTemp = (v) => (v == null ? "–" : Math.round(v) + "°");
const $ = (id) => document.getElementById(id);
const tr = () => I18N[state.lang];
const locale = () => tr().locale;

function readLanguage() {
  try {
    return localStorage.getItem(LANGUAGE_KEY) === "en" ? "en" : "de";
  } catch {
    return "de";
  }
}

function saveLanguage() {
  try {
    localStorage.setItem(LANGUAGE_KEY, state.lang);
  } catch {
    // A blocked localStorage should not block the layout switch.
  }
}

function applyTranslations() {
  document.documentElement.lang = tr().htmlLang;
  document.title = tr().title;

  document.querySelectorAll("[data-i18n]").forEach((el) => {
    el.textContent = tr()[el.dataset.i18n];
  });
  document.querySelectorAll("[data-i18n-html]").forEach((el) => {
    el.innerHTML = tr()[el.dataset.i18nHtml];
  });

  const toggle = $("langToggle");
  if (toggle) {
    toggle.checked = state.lang === "en";
    toggle.setAttribute("aria-label", tr().switchAria);
  }
}

function setLanguage(lang) {
  if (!I18N[lang] || lang === state.lang) return;
  state.lang = lang;
  saveLanguage();
  applyTranslations();

  if (state.forecast) render(state.forecast);
  if (state.error) renderError();
}

function initLanguageToggle() {
  const toggle = $("langToggle");
  if (!toggle) return;
  applyTranslations();
  toggle.addEventListener("change", () => setLanguage(toggle.checked ? "en" : "de"));
}

async function load() {
  try {
    const res = await fetch("/api/forecast");
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || res.statusText);
    render(data);
  } catch (e) {
    showError(e.message);
  }
}

function showError(msg) {
  $("loading").hidden = true;
  state.error = msg;
  renderError();
}

function renderError() {
  const el = $("error");
  el.hidden = false;
  el.innerHTML = `<p>${tr().errorTitle}</p><p class="small">${state.error}</p>`;
}

function render(d) {
  state.forecast = d;
  state.error = "";
  $("loading").hidden = true;
  $("error").hidden = true;
  $("content").hidden = false;
  $("station").textContent = d.station;

  const gen = new Date(d.generated_at);
  $("updated").textContent =
    `${tr().updatedPrefix} ` +
    gen.toLocaleString(locale(), {
      hour: "2-digit",
      minute: "2-digit",
      day: "2-digit",
      month: "2-digit",
      timeZone: "Europe/Berlin",
    });

  const c = d.current;
  $("hero-icon").textContent = c.icon || "🌡️";
  $("hero-temp").textContent = fmtTemp(c.temperature);
  const cond = formatCondition(c.condition);
  const raining = c.precip != null && c.precip > 0.05;
  $("hero-cond").textContent = raining ? `${cond} · 🌧️ ${c.precip} mm/h` : cond;
  $("c-wind").textContent = c.wind_speed != null ? Math.round(c.wind_speed) + " km/h" : "–";
  $("c-hum").textContent = c.humidity != null ? Math.round(c.humidity) + " %" : "–";
  $("c-cloud").textContent = c.cloud_cover != null ? Math.round(c.cloud_cover) + " %" : "–";
  tintHero(c.temperature);

  renderWarnings(d.alerts);
  renderHourStrip(d.hourly);
  renderChart(d);
  renderDaily(d.daily);
}

function formatCondition(condition) {
  if (!condition) return tr().currentFallback;
  const key = String(condition).trim().toLowerCase();
  return tr().conditions[key] || condition;
}

function renderWarnings(alerts) {
  const el = $("warnings");
  if (!alerts || !alerts.length) {
    el.hidden = true;
    el.innerHTML = "";
    return;
  }
  el.hidden = false;
  el.innerHTML = alerts
    .map((a) => {
      const sev = (a.severity || "").toLowerCase();
      const until = a.expires
        ? " · " +
          tr().until +
          " " +
          new Date(a.expires).toLocaleString(locale(), {
            weekday: "short",
            hour: "2-digit",
            minute: "2-digit",
            timeZone: "Europe/Berlin",
          })
        : "";
      return (
        `<div class="warn warn-${sev}"><span class="warn-ic">⚠️</span>` +
        `<div><div class="warn-h">${a.headline}</div>` +
        `<div class="warn-m">${a.event || ""}${until}</div></div></div>`
      );
    })
    .join("");
}

function renderHourStrip(hourly) {
  const el = $("hourstrip");
  el.innerHTML = "";
  for (const h of hourly.slice(0, 12)) {
    const hr = new Date(h.t).toLocaleTimeString(locale(), {
      hour: "2-digit",
      minute: "2-digit",
      timeZone: "Europe/Berlin",
    });
    const card = document.createElement("div");
    card.className = "hourcard";
    const rain = h.rain_p != null ? "💧 " + Math.round(h.rain_p * 100) + "%" : "";
    card.innerHTML =
      `<div class="hr">${hr}</div>` +
      `<div class="ht">${Math.round(h.point)}°</div>` +
      `<div class="hrange">${Math.round(h.lo)}–${Math.round(h.hi)}°</div>` +
      `<div class="hrain">${rain}</div>`;
    el.appendChild(card);
  }
}

function tintHero(t) {
  if (t == null) return;
  const hue = Math.max(0, Math.min(210, 210 - (t + 5) * 7)); // cold→blue, warm→orange
  $("hero").style.background =
    `linear-gradient(135deg, hsla(${hue},70%,55%,0.32), rgba(255,255,255,0.06))`;
}

const COLORS = ["#6db3ff", "#9b8cff", "#5fd0c0", "#ff9f7a", "#d98cff"];

function renderChart(d) {
  const labels = d.hourly.map((h) =>
    new Date(h.t).toLocaleTimeString(locale(), { hour: "2-digit", timeZone: "Europe/Berlin" })
  );
  const lo = d.hourly.map((h) => h.lo);
  const hi = d.hourly.map((h) => h.hi);
  const pt = d.hourly.map((h) => h.point);

  const ds = [
    { label: "_lo", data: lo, borderColor: "transparent", pointRadius: 0, fill: false },
    {
      label: tr().bandLabel, data: hi, borderColor: "transparent",
      backgroundColor: "rgba(255,209,102,0.18)", pointRadius: 0, fill: "-1",
    },
    {
      label: tr().ourModelLabel, data: pt, borderColor: "#ffd166", borderWidth: 3,
      pointRadius: 0, tension: 0.35, fill: false,
    },
  ];

  let ci = 0;
  for (const [name, series] of Object.entries(d.models_hourly)) {
    const map = Object.fromEntries(series.map((s) => [s.t, s.v]));
    const aligned = d.hourly.map((h) => (h.t in map ? map[h.t] : null));
    ds.push({
      label: name, data: aligned, borderColor: COLORS[ci % COLORS.length],
      borderWidth: 1.4, pointRadius: 0, tension: 0.35, fill: false, borderDash: [4, 3],
    });
    ci++;
  }

  if (window._chart) window._chart.destroy();
  window._chart = new Chart($("hourlyChart"), {
    type: "line",
    data: { labels, datasets: ds },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          labels: {
            color: "#dce6fb", usePointStyle: true, boxWidth: 8,
            filter: (i) => !i.text.startsWith("_"),
          },
        },
        tooltip: { callbacks: { label: (x) => `${x.dataset.label}: ${x.parsed.y?.toFixed(1)} °C` } },
      },
      scales: {
        x: { ticks: { color: "#aebbd4", maxTicksLimit: 12 }, grid: { color: "rgba(255,255,255,0.06)" } },
        y: { ticks: { color: "#aebbd4", callback: (v) => v + "°" }, grid: { color: "rgba(255,255,255,0.08)" } },
      },
    },
  });
}

function renderDaily(daily) {
  const el = $("daily");
  el.innerHTML = "";
  for (const day of daily) {
    const card = document.createElement("div");
    card.className = "day";
    const rain = day.rain_p != null ? "💧 " + Math.round(day.rain_p * 100) + "%" : "";
    card.innerHTML =
      `<div class="dow">${formatDayName(day)}</div><div class="date">${formatDayDate(day)}</div>` +
      `<div class="hi">${day.tmax}°</div>` +
      `<div class="lo">↓ ${day.tmin}°</div>` +
      `<div class="drain">${rain}</div>` +
      `<div class="spark"></div>`;
    el.appendChild(card);
  }
}

function formatDayName(day) {
  if (day.date_iso) {
    return dateFromIsoDay(day.date_iso).toLocaleDateString(locale(), {
      weekday: "short",
      timeZone: "Europe/Berlin",
    });
  }
  return state.lang === "en" ? DOW_EN[day.dow] || day.dow : day.dow;
}

function formatDayDate(day) {
  if (!day.date_iso) return day.date;
  const options =
    state.lang === "en"
      ? { day: "2-digit", month: "short", timeZone: "Europe/Berlin" }
      : { day: "2-digit", month: "2-digit", timeZone: "Europe/Berlin" };
  return dateFromIsoDay(day.date_iso).toLocaleDateString(locale(), options);
}

function dateFromIsoDay(value) {
  return new Date(`${value}T12:00:00Z`);
}

initLanguageToggle();
load();
