# Testing and validation

**9 September 2026:** earlier numerical examples below describe historical,
uncorrected detector versions and must not be reused as current accuracy results.
The Sentinel-2 baseline offset was missing from the reader. See
[the final-month plan and controlled radiometry check](completion-plan.md).
The current tests cover harmonization, valid-AOI clustering, shared baselines,
bounded GNG prediction and reference-label scoring. Release validation can require
matching image provenance and the current detector version. Independent accuracy
labels are still required.

How correctness is checked at each level — from pure functions up to the
scientific claim that a detected patch is really cut peat.

## 1. Unit tests (deterministic, no network)

`tests/` — run with `python3 -m pytest tests/ -q`. These pin the maths
and logic that everything else depends on:

| File | Checks |
|---|---|
| `test_geo.py` | pixel area (10 m → 0.01 ha), mask-area summation, empty mask |
| `test_detect.py` | NDVI formula + divide-by-zero guard, range in [-1,1], threshold mask |
| `test_change_preprocess.py` | change classes (newly-bare / re-vegetated / stable), both-valid gating, SCL cloud mask, clear-fraction, designation labels, cutting-rate slope |
| `test_gng.py` | GNG grows > 2 nodes, is deterministic for a fixed seed, edge-pruning splits two separated blobs, `predict` labels every row |

They use tiny synthetic arrays, so they are fast and independent of
imagery or the network.

## 2. Data-integrity validation (generated dataset)

`scripts/validate_dataset.py` reads the produced `web/data/sites.geojson`
and asserts invariants that must hold for the map to be trustworthy:

- every feature carries the required properties;
- `ndvi_series` / `gng_series` lengths match `years`;
- bare area never exceeds the protected site area;
- coverage is a valid percentage; change areas are non-negative;
- **area cross-check:** the imagery-derived site area matches the NPWS
  register within 8% (a georeferencing / reprojection sanity check).

It exits non-zero on any failure, so it can gate a deploy.

## 3. Method validation (the scientific claim)

Unit tests prove the code is correct; they do **not** prove a detected
patch is genuinely cut peat. That is an accuracy question, addressed by:

- **Area cross-check** (already automated): imagery site area vs the NPWS
  register — confirms the geometry/area chain is right.
- **Two-detector agreement:** GNG vs the NDVI baseline are compared per
  site; large disagreement flags a site to inspect.
- **Cloud-aware change:** only pixels clear in both epochs are compared,
  so cloud cannot masquerade as change.
- **Point-based accuracy assessment (planned):** sample a few hundred
  random points across sites, label each by eye against high-resolution
  imagery, and report precision / recall / a confusion matrix. The method
  is unsupervised, so no training labels are needed — labelling is for
  evaluation only.

## 3a. External validation against documented cutting (result)

The detector was cross-checked against the only public "ground truth" that
exists: the Department of Housing / NPWS records of turf plots cut without
consent in raised-bog SACs, reported by *thejournal.ie* and the *Irish
Times*. The three SACs with published 2022 plot counts were run through the
unsupervised pipeline **with no labels**:

| SAC (2022 record)      | Documented plots | Detected new bare peat 2018→2024 |
|------------------------|------------------|----------------------------------|
| Monivea (most-cut SAC) | 49               | 19.2 ha                          |
| Barroughter            | 42               | 2.9 ha                           |
| Callow                 | 31               | 8.6 ha                           |

All three documented hotspots are flagged as actively cutting, and
**Monivea — the most-cut SAC on record — is by a wide margin the largest
detection**, so the method clearly tracks real activity without any manual
annotation. The exact ordering of the two smaller sites does not match the
plot counts (Barroughter has more *plots* but less detected *area* than
Callow): plot count and hectares are different measures, the records are for
2022 while the detection is a 2018→2024 change, and n = 3 is small. So this
is corroboration of the signal, not a precision figure; a proper accuracy
assessment (§3) still needs plot-level or point-level reference data.

**Where GNG beats the NDVI baseline — a worked example.** At Rosroe Bog SAC
(2018) the NDVI baseline flags 1.2 ha of "bare peat" that is in fact the
site's open lough (SWIR ≈ 0.02, NIR ≈ 0.00 — water, not peat). GNG rejects
it via the SWIR signature and reports 0 ha, while still matching NDVI on the
real bare peat in later years. This is the detector's designed contribution:
NDVI-level sensitivity, with the full spectrum removing the water and shadow
a vegetation index cannot tell apart from bare peat.

## 3b. Seasonal confound and the season-max design (visual QA finding)

Zoom-level QA of per-year tiles exposed a methodological trap: with one
scene per year chosen only by cloud-freeness, the acquisition date drifted
between May and September across years — and the visible bare area of the
same bog can differ several-fold between a mid-May scene (freshly cut
banks, turf spread to dry) and a late-August one (banks darkened, turf
gathered, vegetation flushed). The interannual "trend" was partly a
phenology lottery.

Fix, now standard in the pipeline: every year is sampled in a **fixed
seasonal window (1 May – 15 July, the cutting season)**, and the year's
figure is the **maximum bare area over all cloud-clear scenes** in that
window — the peak visible extraction state, robust to single-date timing.
Residual caveat: a year whose only clear scenes fall late in the window
(e.g. Monivea 2023, June-only) can still under-report.

**Scene timing across years (v3 release check, September 2026).** On the
raised bogs, the large early values of the series come mostly from April and
early-May scenes; Sonnagh, for example, reads 75 ha on 21 April 2018 and 53 ha
on 25 April 2021. From 2023, most clear scenes fall between mid-May and June,
when the same bogs read near zero. The ten significant falling trends are
therefore partly confounded with acquisition timing.

Some of the decline appears real. On the 24 April 2024 scene, Callow and
Cloonchambers read 0.7 and 1.2 ha, against 2.6–8.4 ha in April 2020–2022. The
two-date check in `docs/multitemporal-gng.md` targets this ambiguity directly.

## 3c. Longitudinal validation on the most-cut bog on record

Monivea Bog SAC is the State's most-documented cutting site (70 banks cut
in 2013; 51 plots in 2021 and 49 in 2022 — the highest of any SAC, per
NPWS records released under FOI). Our season-max series, built with no
knowledge of those records:

| Year | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|------|------|------|------|------|------|------|------|------|------|
| Detected ha | 1.9 | 0.0 | 2.7 | 2.3 | **40.9** | 2.2* | 19.2 | 15.6 | 21.7 |

Quiet baseline through 2018–2021, an explosion to 40.9 ha in May 2022 —
immediately following the documented 2021/22 cutting maximum — and
sustained 15–22 ha activity through 2026. (*2023 had no clear May scene;
June-only acquisition under-reports.) The detector independently
reproduces the documented activity profile of the worst-offending bog.

## 3d. Burn discrimination (QA finding → detector improvement)

Series QA flagged Moorfield Bog NHA with a 62 ha spike in May 2025 that
vanished the following year — implausible as cutting. Inspection showed a
**spring bog fire**: the low-NDVI scar mimics bare peat to a vegetation
index, but its NBR = (NIR−SWIR2)/(NIR+SWIR2) is strongly negative
(−0.13), while genuine cut peat measured on Monivea sits at +0.09
(p10 +0.02). A per-pixel NBR > 0 guard now separates the two:

| Moorfield, May 2025 | NDVI baseline | GNG |
|---------------------|---------------|-----|
| Detected "bare peat" | 61.9 ha (the burn) | 3.0 ha (real cut banks only) |

With this the GNG detector rejects **three** classes of false positive
that defeat a pure NDVI threshold — open water (Rosroe), deep shadow, and
burn scars (Moorfield) — while keeping the baseline's sensitivity on real
cut surfaces (Monivea: 37.3 of 40.4 ha retained). Figures:
`outputs/rosroe_demo.png`, `outputs/moorfield_burn_demo.png`.

## 3e. Blanket-bog phenology (QA finding → adaptive contrast threshold)

Visual QA of the Atlantic blanket bogs (Achill, Erris, Connemara) exposed
the last big false-positive class: their vegetation — Molinia and Calluna —
stays winter-brown well into May, so on an early-season scene most of the
bog sits just under the fixed NDVI threshold with **no peat exposed at
all**. Doogort East Bog 2023 was the worst case: 223 ha "detected" on a
31 May scene, visually just a uniformly brown bog.

The fix follows from what fresh cutting actually is: an **anomaly against
the bog's own vegetated matrix**, not an absolute reflectance class. The
detector now caps its threshold at (p75 NDVI over the bog − 0.18) per
scene. On a green scene (matrix ≈ 0.6) the cap is inactive and the fixed
0.25 applies; on a senescent scene (matrix ≈ 0.35) it tightens to ≈ 0.17,
rejecting the in-distribution brown while keeping genuinely dark fresh
peat (NDVI ≈ 0.10). Control results (`scripts/17_control_cases.py`):

| Control case | Before | After | Truth |
|---|---|---|---|
| Doogort East 2023 (brown blanket bog) | 223.4 ha | 19.0 ha | no visible cutting |
| Moorfield 2025 (burn scar)            | 4.0 ha   | 3.0 ha  | burn, not cutting |
| Monivea 2021 (51 documented plots)    | 2.2 ha   | 3.3 ha  | documented cutting |

The NDVI baseline on the same Doogort scene still reports 224 ha — the
clearest demonstration yet of why the thesis method is needed. Two
consequences are accepted and recorded rather than hidden: (i) a year
whose only clear scenes are senescent can under-report (the per-year
`matrix_series` value marks such low-contrast years); (ii) documented
raised-bog hotspots measured only on brown May scenes (Monivea 2022)
lose most of their previously reported area — the earlier figures were
themselves partly phenology inflation, and the register plot counts
(tens of small cut banks) are more consistent with the new, smaller
areas than with 40 ha.

Supporting changes in the same pass: geometric **cloud-shadow
projection** (cloud pixels projected along the anti-solar azimuth for
400–2000 m cloud heights; dark-NIR pixels under the projection are
masked — SCL's own shadow class badly under-detects), a **month-spread
scene pool** (the clearest scene of each month enters the season-max
pool first, so a wide window cannot fill the pool with same-month
duplicates), per-day scene de-duplication, and an earlier **15 April
season start for raised bogs** (their cutting season opens in April;
blanket bogs keep the 1 May start because of the phenology above).

## 3f. Intertidal flats (QA finding → ever-water exclusion)

After the radiometry correction, visual QA of the largest remaining
detection — Tullaghan Bay And Bog NHA, a coastal estuary complex —
showed the mask sitting on the bay's **tidal mudflats**, not on peat.
At low tide an exposed flat is dark wet sediment: low NDVI, low
brightness, moisture-bearing — spectrally indistinguishable from fresh
bare peat in a single scene. The year series (390 → 99 → 486 ha)
tracked the tide state at acquisition, not extraction.

The multi-temporal record disambiguates what a single scene cannot: a
tidal pixel is open water (SCL class 6) on *some* clear scene. The
pipeline now unions SCL water across every clear scene of a site and
excludes those pixels from detection in all years (`ever_water_mask`,
covered by unit tests; the exclusion is stored with the site record so
rendered images use exactly the masks the numbers used). Tullaghan Bay
drops from 490 ha to 23–44 ha across the series, and the year-to-year
swing no longer tracks the tide. Visual QA of the corrected tiles shows
the residual is mixed: some patches on the bog margins, but also a thin
fringe along the shoreline — upper-intertidal pixels that were never
under water on any clear acquisition, so the union cannot catch them.
A 1–2 pixel buffer around the ever-water mask is the next refinement;
until then Tullaghan Bay's figure is an upper bound, not a cutting
estimate.

## 3g. Broken cumulus and haze fields (QA finding → v3 cloud screen)

Building the blind reference sample exposed the last large error source. A
contact sheet of the eight frozen evaluation scenes showed cloud directly on
sampled points in four of them — popcorn cumulus over Ederglen 2018,
isolated cumulus over Doogort East 2023 and Cloonchambers 2022, and a haze
field with parallax colour fringes over Moorfield 2025 — although every one
had passed the quality gates (SCL clear fraction 0.87–0.99, bright-haze
fraction ≤ 0.04). There are two causes. The SCL misses small cumulus, and
the brightness haze gate had been calibrated on offset-inflated reflectance,
so after the harmonisation it lets cloud fields through. On top of that,
the season-max rule preferentially *selects* such scenes, because cloud
edges add low-NDVI pixels.

Detector v3 adds two screens. They were calibrated on the cloud-affected
calibration scenes only (Doogort East 2023, Moorfield 2025) and then
checked on the rest:

- **per pixel:** blue reflectance > 0.16 is cloud (bog surfaces sit at
  0.03–0.10). The mask is dilated by 3 pixels (30 m) to remove the darker,
  parallax-shifted cloud edges, and it also seeds the shadow projection;
- **per scene:** a HOT-style haze index (blue − 0.5·red, after Zhang et
  al., 2002). A scene where more than 15 % of the bog exceeds 0.02 is
  dropped: 0.57 on the hazy Moorfield scene vs ≤ 0.063 on usable scenes.

Share of GNG-detected pixels removed by the pixel screen on the frozen
scenes:

| Scene | Role | Sky | Effect |
|---|---|---|---|
| Doogort East 2023 | calibration | cumulus | 99 % removed |
| Moorfield 2025 | calibration | haze field | scene rejected (HOT 0.57) |
| Monivea 2022 | calibration | clear | 0 % removed |
| Tullaghan Bay 2026 | calibration | clear | 20 % removed |
| Ederglen 2018 | held-out | popcorn cumulus | scene rejected (HOT 0.60) |
| Cloonchambers 2022 | held-out | one cumulus | 16 % removed |
| Corliskea 2021 | held-out | clear | 12 % removed |
| Moycullen 2018 | held-out | clear | 13 % removed |

So the Doogort East 2023 detection that survived the phenology fix
(10.7 ha) was almost entirely cloud. The control run
(`scripts/17_control_cases.py`) confirms it end to end. Doogort East
2023's cumulus scene is now gated and its season maximum falls to
0.5 ha. The documented hotspots hold: Monivea 3.3 ha (2021) and 4.3 ha
(2022), Callow 2.6 ha (2022). The Moorfield 2025 burn stays at 0 ha for
GNG, while the NDVI baseline still reports 2.0 ha of burn. Hazy scenes
that the HOT gate now drops include some June acquisitions, so a few
summer-only years may lose their observation; that is conservative
(no false detections), but it reduces coverage. On clear scenes the screen removes
0–20 % of detections, all among the brightest detected surfaces. Whether
those are sand, tracks or genuinely pale, dry peat is exactly what the
labelled sample will show; losing some very dry peat is a recorded
limitation.

## 3h. Reference sample and labelling protocol

Accuracy is assessed on a stratified random sample of pixels from frozen
scenes. `scripts/12_method_comparison.py` freezes the exact scene the
published release used for each site-year, on the release's grid and with
its tidal exclusion. `scripts/13_sample_points.py --strata agreement` then
draws three strata per site-year:

- GNG-detected;
- **disagreement** — GNG negative but NDVI, rules or K-means positive,
  sampled directly because that is where the methods differ;
- all-negative.

Stratum populations and sample sizes are recorded, so population-weighted
estimates are available (Olofsson et al., 2014). Site-years are split into
calibration sites (used while tuning thresholds) and held-out sites (never
inspected during tuning). Held-out accuracy is the headline figure.

Labels are assigned blind with `scripts/18_label_tool.py`. For each point
the annotator sees the Sentinel-2 acquisition in true colour and NIR false
colour (330 m close-up and 1.2 km context), plus very-high-resolution
imagery — never any detector output. Each 10 m pixel is labelled bare_peat,
vegetated, water, burn, other or unsure, by its dominant cover on the
acquisition date. False colour is the main test for brown areas:
senescent vegetation usually stays red, bare peat does not. The exception
is a drought year (2018): fully dead grass can turn beige in false colour
too, so there the extent, texture and absence of cut-bank geometry decide.
This exception was added to the guide after the first five labels. The
evidence used and
free-text notes are stored with every label, and unsure points are left out
of the metrics.

Date-matched reference: for six of the eight site-years, PlanetScope 3 m
surface-reflectance frames from the same day as the Sentinel-2 acquisition
are shown next to the Sentinel-2 chips. They were ordered through Planet's
Education and Research programme with `scripts/20_planet_reference.py`
(Planet Team, 2026). Unlike the basemap imagery, they show the state on the
date, at three times the resolution, which settles most bare-peat versus
dry-grass cases. The two 2018 site-years (Moycullen, Ederglen) have no
usable same-week PlanetScope frame. There the reference rests on the
Sentinel-2 acquisition alone, supported by basemap context.

Limitation: where PlanetScope is missing, the reference is visual
interpretation of the same acquisition the detectors saw. It is
independent of them in method, but not in data source.

Planet Team (2026). Planet Application Program Interface: In Space for Life
on Earth. San Francisco, CA. https://api.planet.com

## 4. Web app checks

- `node --check web/app.js` — JavaScript syntax gate.
- Headless-Chrome screenshot of the running site — confirms the sidebar,
  ranking, sparkline and map render (the sidebar renders independently of
  the map's WebGL init, so ranking/detail work even if the map is slow).

## Quick command

```bash
python3 -m pytest tests/ -q            # unit tests
python3 scripts/validate_dataset.py    # dataset invariants + area cross-check
node --check web/app.js                # web syntax
```
