# Concept & Build Plan

This document captures the original idea for the project and turns it into
a concrete list of what the program actually has to do. The formal
research plan lives in the top-level `README.md`; this is the practical,
"how does the idea become code" companion to it.

---

## 1. Where the idea started

Ireland's protected peatlands — Special Areas of Conservation (SAC),
Natural Heritage Areas (NHA), and the rest of the designated network —
are legally protected, yet turf is still being cut inside some of them on
an industrial scale. Some sites are genuinely left alone and recovering;
others are still being stripped. Nobody has a cheap, repeatable way to
see which is which from above.

That gap is the whole project: **look at the protected peatlands from
satellite, find the ones where industrial turf-cutting is still
happening, and measure how much.**

## 2. The initial notes, cleaned up

The raw brainstorm, organised into what each line actually asks for:

- **Find where the law is working.** Which protected peatlands show no
  fresh cutting — i.e. the designation is doing its job.
- **Find where it is not working.** Which protected peatlands are still
  being cut or drained, with nothing being done to restore them.
- **Use the view from satellite.** Detect industrial turf-cutting from
  satellite imagery instead of site visits.
- **Get the coordinates.** Pull the GPS location of each detected cutting
  area straight off the imagery.
- **Measure the area.** Work out how many hectares are affected inside
  each protected site.
- **Use the legal boundaries.** Overlay the official protected-site
  outlines so detections can be attributed to a specific SAC / NHA.

## 3. Turning the rough notes into something buildable

A few of the original ideas need to be pinned down or corrected before
they become code:

| Original thought | What we actually do | Why |
|---|---|---|
| "Maybe scrape from Google Maps" | Use **Sentinel-2** (ESA Copernicus) instead | Google Maps imagery is proprietary and its terms forbid automated access. Sentinel-2 is free, openly licensed, 10 m resolution, and revisits every ~5 days. |
| "Can you extract GPS coordinates from the map?" | Yes — Sentinel-2 tiles are already georeferenced. Every pixel maps to a coordinate through the raster's affine transform. | No manual coordinate work; `rasterio` gives pixel → metres, then reproject to lat/lon. |
| "Get GPS traces / land boundaries" | Use the **NPWS designated-site shapefiles** for the legal boundaries | These are the authoritative outlines of every SAC / NHA in Irish Transverse Mercator (EPSG:2157). We already have the NHA layer downloaded and loading. |
| "Area calculation is tricky, need to ask AI" | Not tricky once the raster is georeferenced: **area = pixel count × per-pixel area** | A Sentinel-2 10 m pixel is 100 m² = 0.01 ha, so 100 pixels = 1 ha. Clip the detected pixels to the protected polygon and sum. |
| "Measure maybe with AI" | Classify each pixel with unsupervised clustering (Growing Neural Gas), with NDVI-threshold and K-means as baselines | Exposed/cut peat has a distinct spectral signature; clustering separates it from intact bog without needing labelled training data. |

## 4. What the program will do — the pipeline

End to end, the tool takes a region and two dates and produces a ranked,
mapped list of protected peatlands with detected cutting activity.

### Stage 1 — Data ingestion
- Pull Sentinel-2 tiles covering Galway / Mayo / Roscommon for two epochs
  (candidate: 2020 and 2025).
- Load NPWS SAC / NHA / SPA / pNHA boundary shapefiles.
- *(The NHA loader already works — see `notes/poc-early/` for the pyshp
  reader that filters bog sites by county.)*

### Stage 2 — Preprocessing
- Clip imagery to the region of interest.
- Drop clouds using the Sentinel-2 scene-classification (SCL) band.
- Reproject imagery to EPSG:2157 so it aligns with the NPWS polygons.
- Stack the chosen bands into a per-pixel spectral vector.

### Stage 3 — Detection (three methods, compared)
| Method | Role |
|---|---|
| NDVI threshold | Simple first-pass baseline — exposed peat has very low NDVI |
| K-means clustering | Established multi-spectral baseline |
| **Growing Neural Gas** | Main method — adaptive, topology-preserving clustering of the spectral vectors; identifies the "exposed / cut peat" cluster |

### Stage 4 — Geolocation & area
- Convert detected pixels to coordinates via the raster transform.
- Intersect the detected mask with each protected-site polygon.
- Report affected area per site (pixel count × 0.01 ha).

### Stage 5 — Change detection
- Run the classification for both epochs.
- Diff the results to find newly-exposed peat inside each site — the
  signal that cutting is ongoing rather than historical.

### Stage 6 — Output
- Per-site table: designation, detected cut area, change since the
  earlier epoch, and a simple flag (clean / active / worsening).
- GeoJSON / shapefile export of the detected polygons.
- Interactive web map (MapLibre + `maplibre-gl-terradraw`): click a
  protected site to see its status and history.

## 5. Build order

Rough sequence for the implementation — each step produces something that
runs before the next one starts.

1. **Ingest + overlay** — one Sentinel-2 tile clipped to one county,
   NPWS boundaries drawn on top. Proves the georeferencing and the CRS
   alignment.
2. **NDVI baseline** — threshold exposed peat, clip to protected
   polygons, print area per site. First end-to-end result.
3. **K-means baseline** — multi-spectral clustering, same output shape.
4. **GNG detector** — the real method; compare against the two
   baselines on a hand-labelled ground-truth set.
5. **Change detection** — add the second epoch and the diff.
6. **Web map + export** — make the results explorable and shareable.
7. **Evaluation + sensitivity** — precision/recall, hyper-parameter
   sweeps, final numbers for the thesis.

## 6. What we already know is fine

Confirmed before starting, so no nasty surprises mid-build:

- **Data is free and open.** Sentinel-2 (Copernicus) and the NPWS
  boundary shapefiles are both openly licensed. No scraping, no paid
  imagery, no gated portals.
- **Peat depth is not needed.** The task is surface detection of cutting
  activity, not sub-surface volume — so the absence of a national
  peat-depth raster does not block anything.
- **Georeferencing and area are solved problems.** Standard `rasterio` /
  `shapely` operations, not research questions.
- **Growing Neural Gas fits the problem.** Self-organising networks have
  a long track record in remote-sensing land-cover classification, so
  using GNG here rests on established ground while the application to
  protected-peatland cutting detection is the new part.

## 7. Still to lock down

- **Temporal scope** — single 2025 snapshot, or 2020 → 2025 change
  detection. Change detection is more work but a much stronger story.
- **Coverage** — SAC only, NHA only, or both. Leaning toward both, with
  SAC prioritised because of its EU legal weight.
- **Ground truth** — how we assemble the labelled reference set (NPWS
  enforcement records via AIE request, EU infringement documentation,
  and a set of manually annotated sites).
- **Legal vs illegal** — how to separate decommissioned Bord na Móna
  works and Coillte forestry drainage from actual illegal cutting,
  probably by site ownership and designation status.
