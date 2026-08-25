/* ---- config ------------------------------------------------------------ */
const METRICS = {
  new:  { label: "New cutting", unit: "ha",    prop: (m) => `newly_${m}_ha`,
          stops: [0.01, 2, 5] },
  rate: { label: "Cutting rate", unit: "ha/yr", prop: (m) => `rate_${m}_ha_yr`,
          stops: [0.05, 0.5, 1.5] },
  pct:  { label: "% bare now",   unit: "%",     prop: (m) => `pct_${m}`,
          stops: [0.5, 2, 5] },
};
const COLORS = ["#357a5f", "#f2d16b", "#ef8a3c", "#e04a34"];

const state = { method: "gng", metric: "new", desig: "", county: "" };
let FEATURES = [];
let selectedName = null;

/* ---- map ---------------------------------------------------------------- */
const map = new maplibregl.Map({
  container: "map",
  style: {
    version: 8,
    sources: {
      carto: {
        type: "raster",
        tiles: [
          "https://a.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}.png",
          "https://b.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}.png",
          "https://c.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}.png",
        ],
        tileSize: 256,
        attribution: "© OpenStreetMap contributors © CARTO",
      },
      cartoLabels: {
        type: "raster",
        tiles: [
          "https://a.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}.png",
          "https://b.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}.png",
        ],
        tileSize: 256, attribution: "© CARTO",
      },
    },
    layers: [
      { id: "bg", type: "background", paint: { "background-color": "#eef2ee" } },
      { id: "carto", type: "raster", source: "carto",
        paint: { "raster-opacity": 0.55, "raster-saturation": -0.55 } },
    ],
  },
  center: [-8.75, 53.55], zoom: 7.6,
});
map.addControl(new maplibregl.NavigationControl(), "top-right");

/* ---- helpers ------------------------------------------------------------ */
const M = () => METRICS[state.metric];
const curProp = () => M().prop(state.method);
const val = (p) => p[curProp()] ?? 0;

function colorFor(v) {
  const s = M().stops;
  return v >= s[2] ? COLORS[3] : v >= s[1] ? COLORS[2]
       : v >= s[0] ? COLORS[1] : COLORS[0];
}

function fillExpr() {
  const s = M().stops, p = curProp();
  return ["step", ["coalesce", ["get", p], 0],
    COLORS[0], s[0], COLORS[1], s[1], COLORS[2], s[2], COLORS[3]];
}

// proportional-symbol radius: area-perception (sqrt) scaled per metric so a
// worst-in-class site reads big and a quiet one reads small, on any zoom.
const RADIUS_REF = { new: 60, rate: 2, pct: 15 };
function radiusExpr(extra = 0) {
  const p = curProp(), ref = RADIUS_REF[state.metric];
  return ["+", extra, ["interpolate", ["linear"],
    ["sqrt", ["min", 1, ["/", ["max", 0, ["coalesce", ["get", p], 0]], ref]]],
    0, 5, 1, 30]];
}

function centroid(geom) {
  try {
    let ring;
    if (geom.type === "MultiPolygon") {
      ring = geom.coordinates.map((poly) => poly[0])
        .sort((a, b) => b.length - a.length)[0];
    } else if (geom.type === "Polygon") {
      ring = geom.coordinates[0];
    } else { return null; }
    if (!ring || !ring.length) return null;
    let x = 0, y = 0;
    ring.forEach((c) => { x += c[0]; y += c[1]; });
    return [x / ring.length, y / ring.length];
  } catch (e) { return null; }
}

function pointsFC() {
  return {
    type: "FeatureCollection",
    features: FEATURES.map((f) => ({
      c: centroid(f.geometry), p: f.properties,
    })).filter((o) => o.c).map((o) => ({
      type: "Feature",
      geometry: { type: "Point", coordinates: o.c },
      properties: o.p,
    })),
  };
}

function filterExpr() {
  const f = ["all"];
  if (state.county) f.push(["==", ["get", "county"], state.county]);
  if (state.desig === "SAC") f.push(["in", "SAC", ["get", "designation"]]);
  else if (state.desig === "SPA") f.push(["in", "SPA", ["get", "designation"]]);
  else if (state.desig === "NHA-only")
    f.push(["==", ["get", "designation"], "NHA"]);
  return f.length > 1 ? f : null;
}

function passesFilter(p) {
  if (state.county && p.county !== state.county) return false;
  if (state.desig === "SAC" && !p.designation.includes("SAC")) return false;
  if (state.desig === "SPA" && !p.designation.includes("SPA")) return false;
  if (state.desig === "NHA-only" && p.designation !== "NHA") return false;
  return true;
}

const fmt = (n, d = 1) => (n === undefined || n === null) ? "–" : Number(n).toFixed(d);

/* ---- sparkline (inline SVG) -------------------------------------------- */
function sparkline(years, series, color) {
  const pts = years.map((y, i) => [y, series[i]]).filter((p) => p[1] != null);
  if (pts.length < 2) return `<div class="empty">Not enough clear years</div>`;
  const w = 320, h = 56, pad = 8;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y1 = Math.max(...ys, 0.001);
  const X = (x) => pad + (w - 2 * pad) * (x - x0) / (x1 - x0 || 1);
  const Y = (y) => (h - pad) - (h - 2 * pad) * (y / y1);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
  const area = `${line}L${X(x1).toFixed(1)},${h - pad}L${X(x0).toFixed(1)},${h - pad}Z`;
  const dots = pts.map((p) =>
    `<circle cx="${X(p[0]).toFixed(1)}" cy="${Y(p[1]).toFixed(1)}" r="2.6" fill="${color}"/>`).join("");
  return `<div class="spark">
    <svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
      <path d="${area}" fill="${color}" opacity="0.16"/>
      <path d="${line}" fill="none" stroke="${color}" stroke-width="2"/>
      ${dots}
    </svg>
    <div class="cap"><span>${x0}: ${fmt(ys[0])} ha</span>
      <span>${x1}: ${fmt(ys[ys.length - 1])} ha</span></div>
  </div>`;
}

/* ---- detail card -------------------------------------------------------- */
function badges(desig) {
  return desig.split(" + ").map((d) => {
    const cls = d === "SAC" ? "sac" : d === "SPA" ? "spa" : "";
    return `<span class="badge ${cls}">${d}</span>`;
  }).join(" ");
}

/* ---- expanding bog card ------------------------------------------------- */
const GNG_C = "#2880ff", NDVI_C = "#ff4d4d";
let cardP = null, yearIdx = 0, frontId = "card-img-a", aspectSet = false;
let playTimer = null;

const $ = (id) => document.getElementById(id);

function twoLineSpark(years, gng, ndvi, idx) {
  const w = 640, h = 96, pad = 12;
  const ymax = Math.max(...[...gng, ...ndvi].filter((v) => v != null), 0.001);
  const x0 = years[0], x1 = years[years.length - 1];
  const X = (i) => pad + (w - 2 * pad) * (years[i] - x0) / ((x1 - x0) || 1);
  const Y = (v) => (h - pad) - (h - 2 * pad) * (v / ymax);
  const path = (s) => s.map((v, i) => v == null ? "" :
    `${i && s[i - 1] != null ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join("");
  const dots = (s, c) => s.map((v, i) => v == null ? "" :
    `<circle cx="${X(i).toFixed(1)}" cy="${Y(v).toFixed(1)}" r="${i === idx ? 4 : 2.6}"
       fill="${c}"/>`).join("");
  const grid = years.map((y, i) =>
    `<line x1="${X(i).toFixed(1)}" y1="${pad - 4}" x2="${X(i).toFixed(1)}" y2="${h - pad + 4}"
       stroke="#ffffff" stroke-opacity="0.05"/>`).join("");
  const mx = X(idx).toFixed(1);
  return `<div><span class="lg"><i style="background:${GNG_C}"></i>GNG</span>
      <span class="lg"><i style="background:${NDVI_C}"></i>NDVI</span></div>
    <svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
      ${grid}
      <line x1="${mx}" y1="2" x2="${mx}" y2="${h - 2}" stroke="#4bd8a0"
        stroke-width="1.5" stroke-dasharray="3 3"/>
      <path d="${path(ndvi)}" fill="none" stroke="${NDVI_C}" stroke-width="2.2"/>
      <path d="${path(gng)}" fill="none" stroke="${GNG_C}" stroke-width="2.2"/>
      ${dots(ndvi, NDVI_C)}${dots(gng, GNG_C)}
    </svg>`;
}

function updateYearUI() {
  const p = cardP, y = p.years[yearIdx];
  const g = p.gng_series[yearIdx], n = p.ndvi_series[yearIdx];
  $("year-pill").textContent = y;
  $("card-year-read").innerHTML =
    `Bare peat in ${y}: <b style="color:${GNG_C}">GNG ${fmt(g)}</b> ·
     <b style="color:${NDVI_C}">NDVI ${fmt(n)}</b> ha`;
  $("card-spark").innerHTML =
    twoLineSpark(p.years, p.gng_series, p.ndvi_series, yearIdx);
}

function showYear(idx) {
  yearIdx = idx;
  $("year-range").value = String(idx);
  const p = cardP, y = p.years[idx];
  const back = $(frontId === "card-img-a" ? "card-img-b" : "card-img-a");
  const front = $(frontId);
  const noimg = $("card-noimg");
  back.onload = () => {
    if (!aspectSet) {
      document.querySelector(".img-stack").style.paddingBottom =
        (100 * back.naturalHeight / back.naturalWidth).toFixed(2) + "%";
      aspectSet = true;
    }
    back.classList.add("show"); front.classList.remove("show");
    frontId = back.id; noimg.classList.add("hidden");
  };
  back.onerror = () => { noimg.classList.remove("hidden"); };
  noimg.classList.add("hidden");
  back.src = `data/tiles/${p.code}_${y}.jpg`;
  updateYearUI();
}

function stopPlay() {
  if (playTimer) { clearInterval(playTimer); playTimer = null; }
  $("play-btn").classList.remove("playing");
  $("play-btn").textContent = "▶";
}
function togglePlay() {
  if (playTimer) { stopPlay(); return; }
  $("play-btn").classList.add("playing");
  $("play-btn").textContent = "⏸";
  let i = yearIdx >= cardP.years.length - 1 ? 0 : yearIdx;
  showYear(i);
  playTimer = setInterval(() => {
    i = (i + 1) % cardP.years.length;
    showYear(i);
  }, 1000);
}

function statTiles(p) {
  const pair = (k, g, n) =>
    `<div class="tile-stat"><div class="k">${k}</div>
      <div class="vs"><span class="vg" style="color:${GNG_C}">${g}</span>
      <span class="vn" style="color:${NDVI_C}">${n}</span>
      <span class="u">GNG · NDVI</span></div></div>`;
  const one = (k, v, u) =>
    `<div class="tile-stat"><div class="k">${k}</div>
      <div class="vs"><span class="vg">${v}</span><span class="u">${u}</span></div></div>`;
  return pair(`New cutting ${p.span}`, fmt(p.newly_gng_ha), fmt(p.newly_ndvi_ha)) +
    pair("Cutting rate ha/yr", fmt(p.rate_gng_ha_yr, 2), fmt(p.rate_ndvi_ha_yr, 2)) +
    pair("Re-vegetated ha", fmt(p.reveg_gng_ha), fmt(p.reveg_ndvi_ha)) +
    one("Cloud-clear both ends", fmt(p.both_cov_pct, 0) + "%", "");
}

function openCard(p) {
  stopPlay();
  cardP = p; aspectSet = false; frontId = "card-img-a";
  $("card-img-a").classList.remove("show");
  $("card-img-b").classList.remove("show");
  $("card-img-a").removeAttribute("src"); $("card-img-b").removeAttribute("src");
  $("card-name").textContent = p.name;
  $("card-meta").textContent = `${p.county} · ${fmt(p.site_ha, 0)} ha protected`;
  $("card-badges").innerHTML = badges(p.designation);

  const range = $("year-range");
  range.max = String(p.years.length - 1);
  $("year-ticks").innerHTML = p.years.map((y) => `<span>${y}</span>`).join("");

  const vd = $("card-valid");
  if (p.plots_2022 != null) {
    vd.classList.remove("hidden");
    vd.innerHTML = `<span class="vd-icon">✓</span>
      <span><b>NPWS-documented hotspot.</b> ${p.plots_2022} turf plots were
      officially recorded as cut here in 2022 — our detector independently
      flags ${fmt(p.newly_gng_ha)} ha of new bare peat, without any labels.</span>`;
  } else {
    vd.classList.add("hidden"); vd.innerHTML = "";
  }
  $("card-stats").innerHTML = statTiles(p);

  showYear(p.years.length - 1);
  $("card-backdrop").classList.remove("hidden");
  requestAnimationFrame(() => $("card-backdrop").classList.add("open"));
}

function closeCard() {
  stopPlay();
  const bd = $("card-backdrop");
  bd.classList.remove("open");
  setTimeout(() => bd.classList.add("hidden"), 320);
}

$("year-range").addEventListener("input", (e) => { stopPlay(); showYear(+e.target.value); });
$("play-btn").addEventListener("click", togglePlay);
$("card-close").addEventListener("click", closeCard);
$("card-backdrop").addEventListener("click", (e) => {
  if (e.target.id === "card-backdrop") closeCard();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeCard();
  else if (e.key === " " && !$("card-backdrop").classList.contains("hidden")) {
    e.preventDefault(); togglePlay();
  }
});

/* ---- headline insight --------------------------------------------------- */
function renderInsight() {
  const withCut = FEATURES.filter((f) => f.properties.newly_gng_ha > 0.5);
  const totalNew = FEATURES.reduce((s, f) => s + (f.properties.newly_gng_ha || 0), 0);
  const top = [...FEATURES].sort((a, b) =>
    b.properties.newly_gng_ha - a.properties.newly_gng_ha)[0];
  const hs = FEATURES.filter((f) => f.properties.plots_2022 != null
    && f.properties.newly_gng_ha > 0.5);

  const valid = hs.length
    ? `<div class="ins-valid">
         <div class="ins-valid-hd">✓ Cross-checked against NPWS records</div>
         <div class="ins-valid-body">The detector independently flags
           ${hs.length} of the government-documented turf-cutting hotspots —
           ${hs.slice(0, 3).map((f) => f.properties.name.replace(/ (Bog )?SAC.*/, ""))
             .join(", ")} — where cutting was officially recorded.</div>
       </div>`
    : "";

  document.getElementById("insight").innerHTML = `
    <div class="ins-head">
      <span class="ins-big">${fmt(totalNew, 0)}<span class="ins-unit">ha</span></span>
      <span class="ins-cap">of new bare peat detected across
        ${withCut.length} protected bogs, 2018 → 2026</span>
    </div>
    <div class="ins-row">
      <div class="ins-cell"><b>${FEATURES.length}</b><span>bogs analysed</span></div>
      <div class="ins-cell"><b>${withCut.length}</b><span>show new cutting</span></div>
      <div class="ins-cell"><b>${top ? fmt(top.properties.newly_gng_ha, 0) : "–"}</b>
        <span>ha worst site</span></div>
    </div>
    ${valid}
    <div class="ins-ctx">Up to 90% of Irish peatlands are degraded; they hold
      two-thirds of the nation's soil carbon. Ireland was referred to the EU
      Court of Justice (2024) over failure to stop bog destruction, yet only
      ~18 of 57 raised-bog SACs are monitored on the ground.</div>`;
}

/* ---- ranking ------------------------------------------------------------ */
function renderRanking() {
  const rows = FEATURES.filter((f) => passesFilter(f.properties))
    .map((f) => ({ p: f.properties, v: val(f.properties) }))
    .sort((a, b) => b.v - a.v).slice(0, 14);
  const max = Math.max(...rows.map((r) => r.v), M().stops[2]);
  document.getElementById("rank-label").textContent =
    `by ${M().label.toLowerCase()} (${M().unit}) · ${state.method.toUpperCase()}`;
  const ol = document.getElementById("rank-list");
  if (!rows.length) { ol.innerHTML = `<div class="empty">No sites match.</div>`; return; }
  ol.innerHTML = rows.map((r, i) => `
    <li class="rank-item ${r.p.name === selectedName ? "sel" : ""}"
        data-name="${r.p.name.replace(/"/g, "&quot;")}">
      <span class="rank-idx">${i + 1}</span>
      <div class="rank-body">
        <div class="rank-name">${r.p.name.replace(" NHA", "")}</div>
        <div class="rank-bar"><i style="width:${Math.max(3, 100 * r.v / max)}%;
          background:${colorFor(r.v)}"></i></div>
      </div>
      <span class="rank-val">${fmt(r.v, state.metric === "rate" ? 2 : 1)}</span>
    </li>`).join("");
  ol.querySelectorAll(".rank-item").forEach((li) =>
    li.addEventListener("click", () => selectByName(li.dataset.name, true)));
}

/* ---- legend ------------------------------------------------------------- */
function renderLegend() {
  const s = M().stops, u = M().unit;
  const labels = [`< ${s[0]}`, `${s[0]}–${s[1]}`, `${s[1]}–${s[2]}`, `> ${s[2]}`];
  document.getElementById("legend").innerHTML =
    `<div class="lg-note">Circle colour &amp; size — ${M().label.toLowerCase()} (${u})
       per bog · ◍ ring = NPWS-documented site</div>` +
    labels.map((t, i) =>
      `<span class="lg"><span class="dot" style="background:${COLORS[i]}"></span>${t} ${u}</span>`
    ).join("");
}

/* ---- selection ---------------------------------------------------------- */
function selectByName(name, fly) {
  selectedName = name;
  const f = FEATURES.find((x) => x.properties.name === name);
  if (!f) return;
  if (history.replaceState)
    history.replaceState(null, "", "#" + encodeURIComponent(name));
  openCard(f.properties);
  if (map.getLayer("bog-sel"))
    map.setFilter("bog-sel", ["==", ["get", "name"], name]);
  renderRanking();
  if (fly && map.getLayer("bog-circles")) {
    const b = new maplibregl.LngLatBounds();
    eachCoord(f.geometry, (c) => b.extend(c));
    if (!b.isEmpty()) map.fitBounds(b, { padding: 90, maxZoom: 12.5, duration: 700 });
  }
}

function eachCoord(geom, cb) {
  const rings = geom.type === "Polygon" ? geom.coordinates
    : geom.coordinates.flat();
  rings.forEach((r) => r.forEach(cb));
}

/* ---- refresh all view state -------------------------------------------- */
function refresh() {
  if (map.getLayer("bog-circles")) {
    const flt = filterExpr();
    map.setPaintProperty("sites-fill", "fill-color", fillExpr());
    map.setFilter("sites-fill", flt);
    map.setFilter("sites-line", flt);
    map.setPaintProperty("bog-circles", "circle-color", fillExpr());
    map.setPaintProperty("bog-circles", "circle-radius", radiusExpr());
    map.setFilter("bog-circles", flt);
    map.setPaintProperty("bog-sel", "circle-radius", radiusExpr(5));
    map.setPaintProperty("bog-hotspot", "circle-radius", radiusExpr(4.5));
    // keep hotspot rings within the active filter too
    const hs = [">", ["coalesce", ["get", "plots_2022"], -1], -1];
    map.setFilter("bog-hotspot", flt ? ["all", flt, hs] : hs);
  }
  renderRanking();
  renderLegend();
}

/* ---- controls ----------------------------------------------------------- */
function wireSeg(id, key) {
  document.querySelectorAll(`#${id} button`).forEach((b) =>
    b.addEventListener("click", () => {
      document.querySelectorAll(`#${id} button`).forEach((x) => x.classList.remove("on"));
      b.classList.add("on");
      state[key] = b.dataset.v;
      refresh();
    }));
}
wireSeg("method", "method");
wireSeg("metric", "metric");
document.getElementById("filter-desig").addEventListener("change", (e) => {
  state.desig = e.target.value; refresh();
});
document.getElementById("filter-county").addEventListener("change", (e) => {
  state.county = e.target.value; refresh();
});

/* ---- load --------------------------------------------------------------- */
// Fetch data once; render the sidebar as soon as it arrives, independent of
// the map's WebGL init, so ranking/detail work even if the map is slow.
const dataPromise = fetch("data/sites.geojson").then((r) => r.json());

dataPromise.then((fc) => {
  FEATURES = fc.features;
  FEATURES.forEach((f) => {
    const p = f.properties, s = p.site_ha || 1;
    p.pct_ndvi = Math.round(1000 * (p.bare_now_ndvi_ha || 0) / s) / 10;
    p.pct_gng = Math.round(1000 * (p.bare_now_gng_ha || 0) / s) / 10;
  });
  document.getElementById("coverage").textContent = `· ${FEATURES.length} bogs`;
  renderInsight();
  renderRanking();
  renderLegend();
  // deep-link: #Site%20Name selects a bog on load (shareable)
  const hash = decodeURIComponent(location.hash.replace(/^#/, ""));
  if (hash && FEATURES.some((f) => f.properties.name === hash))
    selectByName(hash, false);
});

map.on("load", async () => {
  const fc = await dataPromise;

  // actual bog shapes — subtle fill, shown mostly on zoom-in for context
  map.addSource("sites", { type: "geojson", data: fc });
  map.addLayer({ id: "sites-fill", type: "fill", source: "sites",
    paint: { "fill-color": fillExpr(),
      "fill-opacity": ["interpolate", ["linear"], ["zoom"], 8, 0.12, 12, 0.5] } });
  map.addLayer({ id: "sites-line", type: "line", source: "sites",
    paint: { "line-color": "#2b5a49", "line-width": 0.7, "line-opacity": 0.55 } });

  // place labels on top of the muted basemap, kept light
  map.addLayer({ id: "labels", type: "raster", source: "cartoLabels",
    paint: { "raster-opacity": 0.85 } });

  // proportional symbols — the primary, always-legible layer
  map.addSource("points", { type: "geojson", data: pointsFC() });
  map.addLayer({ id: "bog-circles", type: "circle", source: "points",
    paint: {
      "circle-radius": radiusExpr(),
      "circle-color": fillExpr(),
      "circle-opacity": 0.9,
      "circle-stroke-color": "#ffffff",
      "circle-stroke-width": 1.4,
    } });
  map.addLayer({ id: "bog-hotspot", type: "circle", source: "points",
    filter: [">", ["coalesce", ["get", "plots_2022"], -1], -1],
    paint: { "circle-radius": radiusExpr(4.5), "circle-color": "rgba(0,0,0,0)",
      "circle-stroke-color": "#0b7d63", "circle-stroke-width": 2.4,
      "circle-stroke-opacity": 0.95 } });
  map.addLayer({ id: "bog-sel", type: "circle", source: "points",
    filter: ["==", ["get", "name"], "___none___"],
    paint: { "circle-radius": radiusExpr(6), "circle-color": "rgba(0,0,0,0)",
      "circle-stroke-color": "#0f766e", "circle-stroke-width": 3 } });

  const b = new maplibregl.LngLatBounds();
  FEATURES.forEach((f) => b.extend(centroid(f.geometry)));
  if (!b.isEmpty()) map.fitBounds(b, { padding: 60, maxZoom: 9, duration: 0 });

  const pick = (e) => selectByName(e.features[0].properties.name, false);
  map.on("click", "bog-circles", pick);
  map.on("mouseenter", "bog-circles", () => map.getCanvas().style.cursor = "pointer");
  map.on("mouseleave", "bog-circles", () => map.getCanvas().style.cursor = "");

  refresh();
});
