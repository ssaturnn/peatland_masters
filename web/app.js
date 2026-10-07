/* ---- config ------------------------------------------------------------ */
const METRICS = {
  new:  { label: "Newly bare", unit: "ha",    prop: (m) => `newly_${m}_ha`,
          stops: [0.01, 2, 5] },
  rate: { label: "Bare-area trend", unit: "ha/yr", prop: (m) => `rate_${m}_ha_yr`,
          stops: [0.05, 0.5, 1.5] },
  pct:  { label: "% bare now",   unit: "%",     prop: (m) => `pct_${m}`,
          stops: [0.5, 2, 5] },
  peak: { label: "Peak bare, any year", unit: "ha", prop: (m) => `peak_${m}_ha`,
          stops: [0.01, 2, 5] },
};
const COLORS = ["#357a5f", "#f2d16b", "#ef8a3c", "#e04a34"];

const state = { method: "gng", metric: "new", desig: "", county: "" };
let FEATURES = [];
let selectedName = null;

// /private/ serves this same page behind a password and adds same-day
// PlanetScope 3 m imagery (research licence: never on the public site).
const PRIVATE = location.pathname.startsWith("/private");
let PLANET = {}, PLANET_CITE = "";
const psKey = (p, y) => `${p.code}_${y}`;

/* ---- map ---------------------------------------------------------------- */
const map = new maplibregl.Map({
  container: "map",
  style: {
    version: 8,
    sources: {
      esri: {
        type: "raster",
        tiles: ["https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"],
        tileSize: 256,
        attribution: "Esri, HERE, Garmin © OpenStreetMap contributors",
      },
      esriRef: {
        type: "raster",
        tiles: ["https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}"],
        tileSize: 256, attribution: "Esri",
      },
    },
    layers: [
      { id: "bg", type: "background", paint: { "background-color": "#eef2ee" } },
      { id: "esri", type: "raster", source: "esri",
        paint: { "raster-opacity": 0.9 } },
      { id: "esri-ref", type: "raster", source: "esriRef",
        paint: { "raster-opacity": 0.75 } },
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
const RADIUS_REF = { new: 60, rate: 2, pct: 15, peak: 60 };
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

// NPWS plot records exist for 2021 and 2022: a documented site is judged in
// those years, not on the first-vs-last change, which a later quiet year hides
const DOC_YEARS = [2021, 2022];
const docYears = (p) => DOC_YEARS.filter((y) => p[`plots_${y}`] != null);
const gngIn = (p, y) => { const i = p.years.indexOf(y); return i < 0 ? null : p.gng_series[i]; };
const HOTSPOT_FILTER = ["any", ...DOC_YEARS.map((y) =>
  [">", ["coalesce", ["get", `plots_${y}`], -1], -1])];
// coastal sites: the upper intertidal fringe is never under water on a clear
// scene, so it survives the water exclusion and inflates the figures
const isUpperBound = (p) => (p.water_excl_pct || 0) > 25;
const shortName = (p) => p.name.replace(/ Bog (SAC|NHA)$/, "").replace(/ (SAC|NHA)$/, "");

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
let playTimer = null, showOverlay = true, showPS = true, swipePct = 50;

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
    `Candidate bare peat in ${y}: <b style="color:${GNG_C}">GNG ${fmt(g)}</b> ·
     <b style="color:${NDVI_C}">NDVI ${fmt(n)}</b> ha`;
  $("card-spark").innerHTML =
    twoLineSpark(p.years, p.gng_series, p.ndvi_series, yearIdx);
  if (p.dates && p.dates[yearIdx]) {
    const date = document.createElement("div");
    date.textContent = `Satellite acquisition: ${p.dates[yearIdx]}`;
    $("card-year-read").appendChild(date);
  }
  const ps = PLANET[psKey(p, y)];
  if (ps) {
    const line = document.createElement("div");
    line.className = "ps-read";
    // gap to the Sentinel-2 scene shown now, which can differ from the one the
    // PlanetScope tile was ordered for when a site's season window changed
    const s2 = p.dates && p.dates[yearIdx];
    const gap = s2 ? Math.round((Date.parse(ps.date) - Date.parse(s2)) / 864e5) : ps.delta_days;
    line.textContent = `PlanetScope 3 m: ${ps.date}` +
      (gap ? ` (${gap > 0 ? "+" : ""}${gap} d from Sentinel-2)` : " (same day)") +
      (ps.clear_pct != null ? ` · ${Math.round(ps.clear_pct)}% of the bog clear` : "") +
      ` · ${ps.instruments.join(", ")}`;
    line.classList.toggle("ps-warn", ps.clear_pct != null && ps.clear_pct < 90);
    $("card-year-read").appendChild(line);
    if (showPS && ps.candidates) {
      const note = document.createElement("div");
      note.className = "ps-cite";
      note.textContent = "Yellow: GNG on the 3 m image. PlanetScope has no SWIR band, so " +
        "open water and burn scars can appear there; the 10 m detector rejects them.";
      $("card-year-read").appendChild(note);
    }
    if (showPS) {
      const cite = document.createElement("div");
      cite.className = "ps-cite";
      cite.textContent = `Imagery: ${PLANET_CITE}`;
      $("card-year-read").appendChild(cite);
    }
  }
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
  back.src = `data/tiles/${p.code}_${y}${showOverlay ? "" : "c"}.jpg`;
  updatePS();
  updateYearUI();
}

/* ---- PlanetScope swipe (private view only) ------------------------------ */
function updatePS() {
  const key = psKey(cardP, cardP.years[yearIdx]), ps = PLANET[key];
  const on = !!ps && showPS;
  const btn = $("ps-btn"), img = $("card-img-ps");
  btn.classList.toggle("hidden", !ps);
  btn.classList.toggle("ovl-on", on);
  btn.textContent = on ? "PlanetScope 3 m: on" : "PlanetScope 3 m: off";
  if (on) {
    img.src = `/private/planet/${key}_ps${showOverlay ? "" : "c"}.jpg`;
    setSwipe(swipePct);
  }
  img.classList.toggle("show", on);
  $("swipe").classList.toggle("hidden", !on);
  stackEl.classList.toggle("comparing", on);
  $("card-legend").innerHTML = `<span><i class="sw sw-gng"></i>GNG${on ? " 10 m" : ""}</span>
    <span><i class="sw sw-ndvi"></i>NDVI</span>` +
    (on && ps.candidates ? `<span><i class="sw sw-ps"></i>GNG 3 m</span>` : "");
}

function setSwipe(pct) {
  swipePct = Math.max(0, Math.min(100, pct));
  $("card-img-ps").style.clipPath = `inset(0 0 0 ${swipePct}%)`;
  $("swipe").style.left = `${swipePct}%`;
  $("swipe").setAttribute("aria-valuenow", String(Math.round(swipePct)));
}

// drag anywhere on the image: Sentinel-2 left of the line, PlanetScope right
const stackEl = document.querySelector(".img-stack");
let swiping = false;
const swipeTo = (e) => {
  const r = stackEl.getBoundingClientRect();
  setSwipe(100 * (e.clientX - r.left) / r.width);
};
stackEl.addEventListener("pointerdown", (e) => {
  if (!stackEl.classList.contains("comparing")) return;
  swiping = true; stackEl.setPointerCapture(e.pointerId); swipeTo(e);
});
stackEl.addEventListener("pointermove", (e) => { if (swiping) swipeTo(e); });
["pointerup", "pointercancel"].forEach((t) =>
  stackEl.addEventListener(t, () => { swiping = false; }));
$("swipe").addEventListener("keydown", (e) => {
  if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
  e.preventDefault();
  setSwipe(swipePct + (e.key === "ArrowLeft" ? -5 : 5));
});
$("ps-btn").addEventListener("click", () => {
  showPS = !showPS; updatePS(); updateYearUI();
});

function toggleOverlay() {
  showOverlay = !showOverlay;
  const b = $("ovl-btn");
  b.textContent = `Overlays: ${showOverlay ? "on" : "off"}`;
  b.classList.toggle("ovl-on", showOverlay);
  $("card-legend").style.visibility = showOverlay ? "visible" : "hidden";
  showYear(yearIdx);
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
  const trend = !p.mk_trend ? "Not assessed" : p.mk_trend === "increasing"
    ? `<span style="color:${NDVI_C}">▲ rising</span>`
    : p.mk_trend === "decreasing"
      ? `<span style="color:${GNG_C}">▼ falling</span>` : "— none";
  const trendU = !p.mk_trend ? "No test result in this dataset" : p.mk_trend !== "none"
    ? `Mann–Kendall z=${fmt(p.mk_z, 1)}, p=${fmt(p.mk_p, 3)}` : "not significant";
  return pair(`Newly bare ${p.span}`, fmt(p.newly_gng_ha), fmt(p.newly_ndvi_ha)) +
    one("Statistical trend", trend, trendU) +
    pair("Bare-area trend ha/yr", fmt(p.rate_gng_ha_yr, 2), fmt(p.rate_ndvi_ha_yr, 2)) +
    one("Cloud-clear both ends", fmt(p.both_cov_pct, 0) + "%", "") +
    (p.water_excl_pct > 5
      ? one("Water excluded", fmt(p.water_excl_pct, 0) + "%", "open or tidal water on a clear scene")
      : "");
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
  const dy = docYears(p);
  if (dy.length) {
    const found = dy.some((y) => (gngIn(p, y) ?? 0) > 0.5);
    const perYear = dy.map((y) => {
      const a = gngIn(p, y);
      return `${p[`plots_${y}`]} plots in ${y}, ${a == null
        ? "no clear scene that year" : `${fmt(a)} ha detected`}`;
    }).join("; ");
    vd.classList.remove("hidden");
    vd.classList.toggle("miss", !found);
    vd.innerHTML = `<span class="vd-icon">${found ? "✓" : "!"}</span>
      <span><b>NPWS-documented cutting.</b> ${perYear}.
      ${found ? "" : "The detector finds no candidate bare peat in the documented years. "}
      Compared in the years the plots were recorded; plot counts and hectares are
      different measures.</span>`;
  } else {
    vd.classList.add("hidden"); vd.innerHTML = "";
  }

  const notes = [];
  if (isUpperBound(p)) notes.push(`<b>Coastal site: read these figures as an upper
    bound.</b> Water seen on any clear scene (${fmt(p.water_excl_pct, 0)}% of the site) is
    excluded, but the upper intertidal fringe is never under water on a clear
    acquisition and can still pass as bare surface.`);
  if (p.mk_trend === "decreasing") notes.push(`<b>The falling trend is partly scene
    timing.</b> Up to 2022 most clear scenes were in April, afterwards mostly in May and
    June, when the same surfaces read greener.`);
  $("card-warn").innerHTML = notes.map((n) => `<p>${n}</p>`).join("");
  $("card-warn").classList.toggle("hidden", !notes.length);
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
$("ovl-btn").addEventListener("click", toggleOverlay);
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
  const nUp = FEATURES.filter((f) => f.properties.mk_trend === "increasing").length;
  const nDown = FEATURES.filter((f) => f.properties.mk_trend === "decreasing").length;
  const documented = FEATURES.map((f) => f.properties).filter((p) => docYears(p).length);
  const found = documented.filter((p) => docYears(p).some((y) => (gngIn(p, y) ?? 0) > 0.5));
  const missed = documented.filter((p) => !found.includes(p));
  const ub = top && isUpperBound(top.properties);

  const valid = documented.length
    ? `<div class="ins-valid">
         <div class="ins-valid-hd">Sites with documented cutting</div>
         <div class="ins-valid-body">NPWS recorded turf plots at ${documented.length} of these
           bogs in 2021–2022. In those years the detector finds candidate bare peat at
           ${found.length}: ${found.map(shortName).join(", ")}.${missed.length
             ? ` It finds none at ${missed.map(shortName).join(", ")}.` : ""}</div>
       </div>`
    : "";
  const link = (href, text) => `<a href="${href}" target="_blank" rel="noopener">${text}</a>`;

  document.getElementById("insight").innerHTML = `
    <div class="ins-head">
      <span class="ins-big">${fmt(totalNew, 0)}<span class="ins-unit">ha</span></span>
      <span class="ins-cap">of newly bare candidate surface across
        ${FEATURES.length} protected bogs, first vs last clear survey year${ub
          ? `. ${fmt(top.properties.newly_gng_ha, 0)} ha of it is ${top.properties.name.replace(/ NHA$/, "")},
             a coastal site where the figure is an upper bound` : ""}</span>
    </div>
    <div class="ins-row">
      <div class="ins-cell"><b>${FEATURES.length}</b><span>bogs analysed</span></div>
      <div class="ins-cell"><b>${nDown}<i class="tr-down">▼</i>
        · ${nUp}<i class="tr-up">▲</i></b>
        <span>significant falling · rising trend (p&lt;0.05)</span></div>
      <div class="ins-cell"><b>${withCut.length}</b><span>have >0.5 ha newly bare</span></div>
      <div class="ins-cell"><b>${top ? fmt(top.properties.newly_gng_ha, 0) : "–"}</b>
        <span>${ub ? "ha largest change (coastal, upper bound)" : "ha largest candidate change"}</span></div>
    </div>
    ${nDown ? `<div class="ins-note">Falling trends are partly scene timing: up to 2022 most
      clear scenes were in April, afterwards mostly in May and June.</div>` : ""}
    ${valid}
    <div class="ins-ctx">82% of Irish peatlands are damaged to some extent
      (${link("https://www.epa.ie/publications/research/reports/research-401-peatland-properties-influencing-greenhouse-gas-emissions-and-removal.php", "EPA, 2022")}).
      In March 2024 the European Commission referred Ireland to the EU Court of Justice
      for failing to protect raised- and blanket-bog SACs from turf cutting
      (${link("https://ec.europa.eu/commission/presscorner/detail/en/ip_24_1232", "European Commission")}),
      and NPWS monitored cutting on the ground at 28 of the 57 SACs concerned from 2016
      (${link("https://iwt.ie/press-release-unlawful-turf-cutting/", "Irish Wildlife Trust, 2020")}).</div>`;
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
        <div class="rank-name">${r.p.mk_trend === "increasing"
          ? '<span class="tr-up" title="statistically significant rising trend">▲</span> '
          : r.p.mk_trend === "decreasing"
            ? '<span class="tr-down" title="statistically significant falling trend">▼</span> '
            : ""}${r.p.name.replace(" NHA", "")}${isUpperBound(r.p)
          ? ' <span class="ub" title="coastal site: the intertidal fringe can inflate this figure">upper bound</span>'
          : ""}</div>
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
       per bog · ◍ ring = NPWS-documented site${state.metric === "peak"
         ? " · the peak is often an April scene of 2018–2022, when dry vegetation can read as bare"
         : ""}</div>` +
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
    history.replaceState(null, "", location.pathname + "#" + encodeURIComponent(name));
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
    const hs = HOTSPOT_FILTER;
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
function derive(p) {
  const s = p.site_ha || 1;
  p.pct_ndvi = Math.round(1000 * (p.bare_now_ndvi_ha || 0) / s) / 10;
  p.pct_gng = Math.round(1000 * (p.bare_now_gng_ha || 0) / s) / 10;
  // the largest area in any survey year: a site cut in 2021 and quiet since
  // shows here, where the first-vs-last change hides it
  const peak = (series) => Math.max(0, ...(series || []).filter((v) => v != null));
  p.peak_gng_ha = peak(p.gng_series);
  p.peak_ndvi_ha = peak(p.ndvi_series);
}

// Fetch data once; render the sidebar as soon as it arrives, independent of
// the map's WebGL init, so ranking/detail work even if the map is slow.
const dataPromise = fetch("data/sites.geojson")
  .then((r) => {
    if (!r.ok) throw new Error(`Dataset HTTP ${r.status}`);
    return r.json();
  })
  .then((fc) => {
    if (fc.type !== "FeatureCollection" || !Array.isArray(fc.features))
      throw new Error("Invalid map dataset");
    // derived metrics live on the features, so the map layers and the sidebar
    // read the same values whichever of them initialises first
    fc.features.forEach((f) => derive(f.properties));
    return fc;
  })
  .catch(() => {
    $("coverage").textContent = "· data unavailable";
    $("insight").textContent = "The results could not be loaded. Please reload the page.";
    return null;
  });

/* ---- PlanetScope list (private view) ------------------------------------ */
function renderPlanetList() {
  const rows = Object.entries(PLANET).map(([key, ps]) => {
    const [code, year] = key.split("_");
    const f = FEATURES.find((x) => x.properties.code === code);
    return f && { name: f.properties.name, year: Number(year), ps };
  }).filter(Boolean).sort((a, b) => a.name.localeCompare(b.name));
  if (!rows.length) return;
  const evalRows = rows.filter((r) => r.ps.source !== "release");
  const bogs = new Set(rows.map((r) => r.name)).size;
  const years = rows.map((r) => r.year);
  $("planet").innerHTML = `<h2>PlanetScope 3 m <span>· within 3 days of Sentinel-2</span></h2>
    <p class="ps-hint">In the bog card for ${rows.length} bog-years on ${bogs} bogs,
      ${Math.min(...years)}–${Math.max(...years)}, most of them 2024–2026. Open any bog and
      drag across the image to compare 10 m with 3 m.</p>
    <div class="ps-sub">Evaluation site-years</div>
    <ul>${evalRows.map((r) => `<li><button class="ps-item"
        data-name="${r.name.replace(/"/g, "&quot;")}" data-year="${r.year}">
        <span>${r.name.replace(/ (SAC|NHA)$/, "")}</span>
        <span class="ps-date">${r.ps.date}</span></button></li>`).join("")}</ul>`;
  $("planet").classList.remove("hidden");
  $("planet").querySelectorAll(".ps-item").forEach((b) => b.addEventListener("click", () => {
    showPS = true;
    selectByName(b.dataset.name, true);
    const i = cardP.years.indexOf(Number(b.dataset.year));
    if (i >= 0 && i !== yearIdx) showYear(i);
  }));
}

if (PRIVATE) {
  $("private-note").classList.remove("hidden");
  $("private-link").innerHTML =
    `<a href="/private/findings/">Findings (RU/EN)</a> · <a href="/private/briefing/">Research briefing</a> · <a href="/">Public map</a>`;
}
const planetPromise = PRIVATE
  ? fetch("/private/planet/manifest.json")
      .then((r) => (r.ok ? r.json() : null)).catch(() => null)
  : Promise.resolve(null);

Promise.all([dataPromise, planetPromise]).then(([fc, planet]) => {
  if (!fc) return;
  FEATURES = fc.features;
  document.getElementById("coverage").textContent = `· ${FEATURES.length} bogs`;
  renderInsight();
  renderRanking();
  renderLegend();
  if (planet && planet.tiles) {
    PLANET = planet.tiles; PLANET_CITE = planet.citation || "";
    renderPlanetList();
  }
  // deep-link: #Site%20Name selects a bog on load (shareable);
  // #Site%20Name|2022 also opens that year
  const [hash, year] = decodeURIComponent(location.hash.replace(/^#/, "")).split("|");
  if (hash && FEATURES.some((f) => f.properties.name === hash)) {
    selectByName(hash, false);
    const i = cardP.years.indexOf(Number(year));
    if (i >= 0 && i !== yearIdx) showYear(i);
  }
});

function whenStyleReady(fn) {
  if (map.isStyleLoaded()) { fn(); return; }
  // styledata may fire before raster sources finish, with no later styledata
  // event. The load event reliably releases the overlay initialization.
  map.once("load", fn);
}
whenStyleReady(async () => {
  const fc = await dataPromise;
  if (!fc) return;

  // actual bog shapes — subtle fill, shown mostly on zoom-in for context
  map.addSource("sites", { type: "geojson", data: fc });
  map.addLayer({ id: "sites-fill", type: "fill", source: "sites",
    paint: { "fill-color": fillExpr(),
      "fill-opacity": ["interpolate", ["linear"], ["zoom"], 8, 0.12, 12, 0.5] } });
  map.addLayer({ id: "sites-line", type: "line", source: "sites",
    paint: { "line-color": "#2b5a49", "line-width": 0.7, "line-opacity": 0.55 } });

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
    filter: HOTSPOT_FILTER,
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
