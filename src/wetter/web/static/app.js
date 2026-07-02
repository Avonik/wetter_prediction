const fmtTemp = (v) => (v == null ? "–" : Math.round(v) + "°");

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
  document.getElementById("loading").hidden = true;
  const el = document.getElementById("error");
  el.hidden = false;
  el.innerHTML = `<p>⚠️ Vorhersage konnte nicht geladen werden.</p><p class="small">${msg}</p>`;
}

function render(d) {
  document.getElementById("loading").hidden = true;
  document.getElementById("content").hidden = false;
  document.getElementById("station").textContent = d.station;
  const gen = new Date(d.generated_at);
  document.getElementById("updated").textContent =
    "Stand: " + gen.toLocaleString("de-DE", { hour: "2-digit", minute: "2-digit", day: "2-digit", month: "2-digit" });

  const c = d.current;
  document.getElementById("hero-icon").textContent = c.icon || "🌡️";
  document.getElementById("hero-temp").textContent = fmtTemp(c.temperature);
  const cond = c.condition || "aktuell";
  const raining = c.precip != null && c.precip > 0.05;
  document.getElementById("hero-cond").textContent = raining
    ? `${cond} · 🌧️ ${c.precip} mm/h` : cond;
  document.getElementById("c-wind").textContent = c.wind_speed != null ? Math.round(c.wind_speed) + " km/h" : "–";
  document.getElementById("c-hum").textContent = c.humidity != null ? Math.round(c.humidity) + " %" : "–";
  document.getElementById("c-cloud").textContent = c.cloud_cover != null ? Math.round(c.cloud_cover) + " %" : "–";
  tintHero(c.temperature);

  renderWarnings(d.alerts);
  renderHourStrip(d.hourly);
  renderChart(d);
  renderDaily(d.daily);
}

function renderWarnings(alerts) {
  const el = document.getElementById("warnings");
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
        ? " · bis " +
          new Date(a.expires).toLocaleString("de-DE", {
            weekday: "short", hour: "2-digit", minute: "2-digit", timeZone: "Europe/Berlin",
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
  const el = document.getElementById("hourstrip");
  el.innerHTML = "";
  for (const h of hourly.slice(0, 12)) {
    const hr = new Date(h.t).toLocaleTimeString("de-DE", {
      hour: "2-digit", minute: "2-digit", timeZone: "Europe/Berlin",
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
  document.getElementById("hero").style.background =
    `linear-gradient(135deg, hsla(${hue},70%,55%,0.32), rgba(255,255,255,0.06))`;
}

const COLORS = ["#6db3ff", "#9b8cff", "#5fd0c0", "#ff9f7a", "#d98cff"];

function renderChart(d) {
  const labels = d.hourly.map((h) =>
    new Date(h.t).toLocaleTimeString("de-DE", { hour: "2-digit" })
  );
  const lo = d.hourly.map((h) => h.lo);
  const hi = d.hourly.map((h) => h.hi);
  const pt = d.hourly.map((h) => h.point);

  const ds = [
    { label: "_lo", data: lo, borderColor: "transparent", pointRadius: 0, fill: false },
    {
      label: "80 %-Band", data: hi, borderColor: "transparent",
      backgroundColor: "rgba(255,209,102,0.18)", pointRadius: 0, fill: "-1",
    },
    {
      label: "Unser Modell", data: pt, borderColor: "#ffd166", borderWidth: 3,
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
  window._chart = new Chart(document.getElementById("hourlyChart"), {
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
  const el = document.getElementById("daily");
  el.innerHTML = "";
  for (const day of daily) {
    const card = document.createElement("div");
    card.className = "day";
    const rain = day.rain_p != null ? "💧 " + Math.round(day.rain_p * 100) + "%" : "";
    card.innerHTML =
      `<div class="dow">${day.dow}</div><div class="date">${day.date}</div>` +
      `<div class="hi">${day.tmax}°</div>` +
      `<div class="lo">↓ ${day.tmin}°</div>` +
      `<div class="drain">${rain}</div>` +
      `<div class="spark"></div>`;
    el.appendChild(card);
  }
}

load();
