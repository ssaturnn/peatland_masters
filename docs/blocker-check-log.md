# Blocker-check log — initial spike

First pass at the code, done specifically to find blockers before
committing to the full build. Every check below was run for real on the
development machine; the numbers are actual program output.

**Verdict: no blockers. The whole pipeline runs end to end.**

## What was checked

| # | Check | Result | Notes |
|---|---|---|---|
| 1 | Geospatial stack installs (rasterio, geopandas, shapely, pyproj, scikit-learn) | **Pass** | Clean wheel install on macOS; no system GDAL needed |
| 2 | Load NPWS boundaries, reproject to ITM | **Pass** | 171 polygons, 57 West-of-Ireland bog sites, 23,656 ha |
| 3 | Sentinel-2 access, programmatic | **Pass** | Planetary Computer STAC; scene over target at 0.27% cloud; windowed COG read, no full-tile download, no auth friction |
| 4 | NDVI computation | **Pass** | Range 0.02–0.75 over a bog site |
| 5 | Clip detection to polygon + area in hectares | **Pass** | Site area from imagery = 221.1 ha vs NPWS record 221 ha — a clean cross-check that CRS/reproject/area are correct |
| 6 | GNG on multi-spectral pixels | **Pass** | 6-band stack; trains in 0.3 s; after long-edge pruning yields 3 spectral clusters; isolates a low-NDVI (0.21) bare-peat cluster; 96.5% pixel agreement with the NDVI baseline inside the site |

## The pipeline that now runs

```
NPWS shapefile ─▶ pick bog site ─▶ WGS84 bbox
                                     │
Planetary Computer STAC ◀────────────┘
   │  (search low-cloud Sentinel-2 L2A scene)
   ▼
windowed read: red + NIR (COG over HTTP)
   │
   ▼
NDVI ─▶ low-NDVI mask (bare / cut peat)
   │
   ▼
reproject site polygon into image CRS ─▶ clip mask to polygon
   │
   ▼
area in hectares  +  % of site exposed
```

## Scripts

| Script | Does |
|---|---|
| `scripts/01_check_env.py` | Import + version check for the whole stack |
| `scripts/02_load_boundaries.py` | Load NPWS boundaries, summarise, confirm CRS/bbox |
| `scripts/03_fetch_imagery.py` | Search + windowed read of a Sentinel-2 scene |
| `scripts/04_ndvi_baseline.py` | Full chain: imagery → NDVI → mask → clip → area |
| `scripts/05_gng_cluster.py` | GNG on a 6-band stack; compares to the NDVI baseline |

Modules live in `src/peatland/` (`config`, `boundaries`, `imagery`,
`detect`, `geo`, `gng`).

## Note on GNG segmentation

Growing Neural Gas builds a single connected graph, so its raw connected
components give one cluster. The standard fix is applied: cut edges
longer than *mean + factor·std* of edge length. A long edge bridges two
spectrally distant prototypes — a boundary between land-cover manifolds —
so removing them makes the components correspond to real clusters while
the fine topology inside each cluster is kept. This is the
`GrowingNeuralGas.prune_long_edges()` step. It is a documented technique,
not a workaround, and the pruning factor becomes one of the
hyper-parameters for the sensitivity analysis.

## What this de-risks

- **Data access is free and works headless** — no scraping, no paid
  imagery, no gated portal, no manual coordinate entry. The single
  biggest worry (getting satellite data programmatically) is gone.
- **Georeferencing and area are solved** — the site-area cross-check
  (221.1 vs 221 ha) proves the coordinate handling is right.
- **The environment is reproducible** — `requirements.txt` installs
  cleanly from scratch.

## What is deliberately still a stub (next, not blockers)

- K-means and GNG detectors (only the NDVI baseline is wired up so far).
- Cloud masking via the SCL band (current test scenes were near-clear).
- Change detection between two epochs.
- Legal-vs-decommissioned separation.
- Ground-truth set for precision/recall.
