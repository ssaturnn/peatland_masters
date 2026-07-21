# peatland_masters

**Working title.** Satellite-based detection of industrial turf-cutting in
legally-protected Irish peatlands.

| | |
|---|---|
| Programme          | MSc Computing — Research Methods for Computer Scientists |
| Institution        | Atlantic Technological University, Galway |
| Project window     | January 2026 — February 2027 |
| Author             | Aleksandr Turchaninov |
| Supervisor         | Dr. Brian McGinley |
| Study region       | West of Ireland — Galway, Mayo, Roscommon |

---

## 1. Motivation

Ireland's protected peatlands — Special Areas of Conservation (SAC),
Special Protection Areas (SPA), Natural Heritage Areas (NHA), and proposed
NHAs (pNHA) — are designated under the EU Habitats Directive, the Birds
Directive, and the Irish Wildlife Amendment Act. Industrial and domestic
turf-cutting inside these designations has been prohibited for over a
decade.

Enforcement is a persistent problem. The European Commission has issued
multiple infringement proceedings against Ireland for failing to protect
Annex I bog habitats, and the National Parks and Wildlife Service (NPWS)
enforcement capacity is chronically small relative to the geographic
extent of the designated sites. The EU Nature Restoration Regulation
(August 2024, in force from 2024) obliges member states to halt further
degradation and to report progress — sharpening the need for a scalable
monitoring capability that does not depend on manual site visits.

## 2. Research question

> Can freely-available multi-spectral satellite imagery, combined with
> unsupervised clustering methods, reliably detect and quantify
> industrial-scale turf-cutting activity inside legally-protected Irish
> peatlands, at a resolution useful for enforcement and restoration
> reporting?

### Sub-questions

1. Which spectral signatures reliably discriminate active turf-cutting
   surfaces (exposed peat, milled peat stockpiles, drain networks) from
   intact and semi-intact bog surface, across seasons and cloud
   conditions?
2. Does a Growing-Neural-Gas-based unsupervised clustering of
   multi-spectral pixel vectors match or improve upon a K-means baseline
   and a simple NDVI-threshold baseline on a hand-labelled ground-truth
   set?
3. Can change-detection between two Sentinel-2 epochs (candidate: 2020
   vs 2025) identify newly-cut area within protected sites and support
   a per-site trend indicator?

## 3. Data

### 3.1 Satellite imagery (input)

| Source | Resolution | Licence | Access |
|---|---|---|---|
| Sentinel-2 (ESA Copernicus) | 10 m VIS+NIR, 20 m SWIR | Copernicus open | Copernicus Data Space Ecosystem, Sentinel Hub, Google Earth Engine |
| Landsat 8/9 (USGS)          | 30 m                    | Public domain    | USGS EarthExplorer, Google Earth Engine |

Sentinel-2 is the primary source. Landsat is a fallback for long-term
archive comparisons.

### 3.2 Protected-area boundaries (mask)

- **NPWS SAC / SPA / NHA / pNHA** shapefiles, all in Irish Transverse
  Mercator (EPSG:2157).
- **CORINE Land Cover 2018** peat classes (EPA), as a cross-check.

### 3.3 Ground truth

- NPWS enforcement records — obtainable via Access to Information on the
  Environment (AIE) request.
- EU Commission infringement case documentation for Ireland — public.
- Investigative reporting (Irish Times, thejournal.ie) as reference for
  known cutting sites.
- Manual annotation of 20–30 reference sites against high-resolution
  imagery, for use as evaluation ground truth only.

## 4. Method

### 4.1 Preprocessing

- Sentinel-2 tiles clipped to Galway / Mayo / Roscommon extent.
- Reprojection from UTM to ITM (EPSG:2157) for alignment with NPWS
  polygons.
- Cloud masking via the Sentinel-2 SCL band.
- Compose per-pixel spectral vectors across selected bands.

### 4.2 Classification — three approaches, compared

| Approach | Role | Notes |
|---|---|---|
| **Baseline A · NDVI threshold** | First-pass detection | Exposed peat has NDVI < ~0.2; simple thresholding |
| **Baseline B · K-means clustering** | Established multi-spectral baseline | Fixed *k*, per-tile |
| **Proposed · Growing Neural Gas** | Main contribution | Adaptive node count, topology-preserving graph over pixel spectral vectors, Delaunay-like adjacency between land-cover clusters |

GNG has a substantial remote-sensing lineage via Kohonen SOMs and their
descendants; applying it here to the specific problem of illegal
turf-cutting detection inside protected sites is the applied contribution.
The output graph — nodes as spectral prototypes, edges as topological
adjacency — provides both a classification and an interpretable structure
that can be inspected by a domain expert.

### 4.3 Masking and area estimation

- Intersect the classified "exposed peat" raster with each NPWS polygon
  using `shapely.ops.unary_union` / `intersection`.
- Compute area per site by summing pixel counts × per-pixel area
  (Sentinel-2 10 m band → 100 m² per pixel → 100 pixels per hectare).

### 4.4 Change detection

- Repeat classification for two epochs (e.g. 2020 and 2025).
- Compute per-cell change in classified state.
- Report newly-exposed area per site (candidate signal for new
  cutting).

## 5. Evaluation

- **Precision / recall** against the hand-labelled ground-truth patches.
- **Confusion matrix** across land-cover classes.
- **Quantitative comparison** vs. NDVI-threshold and K-means baselines —
  GNG must show a measurable improvement (or a documented trade-off) to
  justify its use.
- **Site-level rankings** — sanity-check top-N flagged sites against
  public enforcement records and press reports.
- **Sensitivity analysis** on key hyper-parameters (GNG node budget,
  edge age, SCL cloud tolerance, spectral band selection).

## 6. Deliverables

- Reproducible Python pipeline, open-source (MIT or BSD).
- GeoJSON / shapefile export of detected polygons per protected site.
- Interactive web map (MapLibre + `maplibre-gl-terradraw`) — clickable
  protected sites showing detected activity, area, and change over time.
- MSc thesis document.
- Optional stretch: one conference-style write-up.

## 7. Rough timeline (13 months)

| Months | Milestone |
|---|---|
| 1–2   | Environment; Sentinel-2 for Galway/Mayo/Roscommon; NPWS overlay working |
| 3–4   | NDVI + K-means baselines; ground-truth collection |
| 5–7   | GNG multi-spectral clustering; comparative evaluation |
| 8–9   | Change detection between two epochs; per-site area estimation |
| 9–10  | Web-map front end; GeoJSON export |
| 10–11 | Final validation; sensitivity analysis |
| 11–13 | Thesis write-up and viva preparation |

## 8. Locked decisions

- Region: Galway / Mayo / Roscommon.
- Method core: multi-spectral clustering, GNG as primary technique, NDVI
  and K-means as baselines.
- **Peat depth is not an input** — per supervisor guidance. Removes
  reliance on unavailable national peat-depth rasters and simplifies the
  data pipeline significantly.
- No proprietary imagery scraping (Google Maps and equivalents). Free,
  openly-licensed sources only.

## 9. Open decisions — to lock down early

1. **Temporal scope.** Single 2025 snapshot vs. change detection between
   2020 and 2025. Change detection is roughly twice the effort but
   materially strengthens the thesis narrative.
2. **Coverage.** SAC only (EU legal weight), NHA only (national), or
   both. Recommendation: both, with SAC prioritised in reporting.
3. **Ground-truth acquisition path.** AIE request to NPWS, EU
   infringement documentation, and 20–30 manually annotated sites.
4. **Hybrid method inclusion.** Whether a CNN-based semantic segmentation
   (e.g. U-Net) is added as an additional baseline or as a preprocessing
   stage upstream of the GNG.
5. **Legal / illegal separation.** How to distinguish Bord na Móna
   decommissioned works and Coillte forestry drainage from illegal
   cutting — likely a rule based on site ownership and designation status.

## 10. References — starting bibliography

- Fritzke, B. (1994). *A Growing Neural Gas Network Learns Topologies*,
  NIPS 7.
- Hagenauer, J. & Helbich, M. (2013). *Contextual Neural Gas for spatial
  clustering and analysis*. International Journal of Geographical
  Information Science, 27(2), 251–266.
- Farrell, C. et al. (2024). Peatland restoration study. *Restoration
  Ecology*.
- EU Nature Restoration Regulation (Regulation (EU) 2024/1991).
- NPWS. Designated site boundary data
  (`npws.ie/maps-and-data/designated-site-data/download-boundary-data`).
- Copernicus Data Space Ecosystem — Sentinel-2 access
  (`dataspace.copernicus.eu`).

*Full literature review is a Phase-1 output.*

## 11. Repository conventions

- English throughout: code, comments, docs, commit messages.
- One idea per commit; small, reviewable diffs.
- See `CLAUDE.md` for attribution / authorship directives that apply to
  any AI-assisted work in this repository.

## Licence

To be finalised before the first code commit (MIT or BSD 3-clause).
