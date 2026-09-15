# PlanetScope case study: Monivea Bog SAC, 23 April 2022

## Finding

After cross-sensor NDVI calibration, PlanetScope detects **5.3685 ha** of
candidate bare surface at 3 m, compared with **4.2800 ha** in the frozen
Sentinel-2 v3 GNG mask: a difference of **+1.0885 ha**. The calibrated
PlanetScope result contains **5,965 pixels in 96 patches**. Its GNG variant
produces the same mask.

Aggregating the calibrated 3 m mask to 10 m with a **bare fraction >= 0.5**
produces **515 candidate cells (5.15 ha)**. On common comparison support,
**360 cells** overlap the frozen Sentinel-2 candidates, giving **IoU =
0.617496**. There are **155 cells found only by aggregated PlanetScope** and
**68 found only by Sentinel-2**. A nominal 10 m morphological opening removes
**0.8748 ha (16.2951%)** of the native PlanetScope candidates, a proxy for
small objects, narrow features and protrusions.

The calibration reproduces the paired-sensor relationship:
**NDVI_S2 = 1.175830 × NDVI_PS10 − 0.123077**, with **r = 0.971473**,
**RMSE = 0.023633 NDVI units**, and **n = 28,682**. All finite nonwater paired
site cells enter the fit, independently of both detectors' candidate masks.
These results quantify finer candidate structure and disagreement after
harmonisation. They do not establish additional confirmed turf cutting or
attribute every difference to spatial resolution.

All reported numbers come from
[summary.json](../outputs/planet_case_study/summary.json).

## Inputs and reproducibility

Run [scripts/21_planet_case_study.py](../scripts/21_planet_case_study.py) from
the repository root:

```sh
python3 scripts/21_planet_case_study.py
python3 -m pytest tests/ -q
```

The defaults are `--run outputs/evaluation/2026-09-13-v3-sample` and
`--out outputs/planet_case_study`. Both options accept repository-relative
paths. The output directory must be inside the repository and separate from
the read-only input run. The script reads the frozen prediction directly;
it does not rerun the Sentinel-2 detector or download imagery.

The input run contains `002352_2022.npz`, `002352_2022.json`,
`planet/002352_2022/ps_mosaic.tif`, and `planet/manifest.json`. Their file
names relative to the run and SHA-256 hashes are recorded in the summary.
The implementation uses NumPy, Rasterio, SciPy, Matplotlib and the existing
`src/peatland/gng.py`. Unit tests use synthetic arrays and pytest.

The saved Sentinel-2 scene is
`S2B_MSIL2A_20220423T115359_R023_T29UNV_20240605T041341`, detector version
`2026-09-13-cloud-screen-v3`. Its metadata identifies Monivea as a
calibration site. The Planet manifest records two PSB.SD items:

| Item | Acquisition, UTC | Manifest clear percentage |
| --- | --- | ---: |
| `20220423_113512_00_2407` | 2022-04-23 11:35:12.001344 | 96% |
| `20220423_113514_28_2407` | 2022-04-23 11:35:14.289864 | 74% |

The manifest reports same-day acquisition, a successful order and full
geometric coverage. Its clear percentages describe source items; they do
not establish that every assessed pixel is cloud-free.

## Assessed area and grids

Reproject the saved Sentinel-2 `inside` mask onto the native PlanetScope
grid using nearest-neighbour resampling, then intersect it with valid
PlanetScope data. All four selected bands must have valid raster masks,
finite values and nonzero values. Read blue/green/red/NIR as bands 1/2/3/4
for a four-band mosaic or 2/4/6/8 for an eight-band mosaic. Divide stored
values by 10,000 to obtain reflectance.

Both grids use EPSG:32629 in metres, with north-up pixels. PlanetScope
pixels occupy 9 m² and Sentinel-2 cells 100 m². Their origins differ and
10/3 is not an integer, so aggregation uses exact rectangle intersections
in map coordinates. The script records both transforms and rejects grids
without a shared projected metre CRS or with rotation.

| Area/support | Result |
| --- | ---: |
| Sentinel-2 `inside` area | 286.8300 ha |
| Sentinel-2 valid `inside` area | 286.8200 ha |
| PlanetScope valid assessed area | 286.7958 ha; 318,662 pixels |
| Invalid PlanetScope pixels inside the reprojected site | 0 |
| All paired 10 m site cells used for calibration | 28,682 |
| Comparison cells with >= 95% assessed PlanetScope coverage | 28,334; 283.3400 ha |
| Valid Sentinel-2 site cells excluded from comparison by coverage | 348 |

Native assessed areas differ slightly because the boundary is rasterized
on different grids. Calibration includes every valid paired site cell with
positive PlanetScope assessed support. The detector comparison retains the
95% coverage cutoff to avoid majority labels based on small observed
fractions of boundary cells. These domains depend on coverage and validity,
not candidate labels. All 428 frozen Sentinel-2 candidates remain in the
comparison domain.

## Cross-sensor calibration

For each Sentinel-2 10 m cell, area-average each PlanetScope reflectance
band over its valid assessed overlap. Compute NDVI and NDWI from those mean
bands, rather than averaging native index values:

```text
NDVI_PS10 = (mean_NIR - mean_red) / (mean_NIR + mean_red)
NDWI_PS10 = (mean_green - mean_NIR) / (mean_green + mean_NIR)
NDVI_S2  = (S2_NIR - S2_red) / (S2_NIR + S2_red)
NDWI_S2  = (S2_green - S2_NIR) / (S2_green + S2_NIR)
```

Ratios use a 1e-9 denominator epsilon. Select all `valid & inside`
Sentinel-2 cells with positive assessed PlanetScope coverage and finite
paired indices. Exclude a cell if **NDWI > 0 on either sensor**. Here, zero
paired cells are excluded for water or nonfinite indices, leaving **28,682**.
No Sentinel-2 prediction, PlanetScope candidate mask, low-NDVI subset or
training subsample enters calibration.

The pure function `fit_ndvi_calibration` fits ordinary least squares with
an intercept and equal weight for every selected cell:

```text
a = sum((PS10 - mean_PS10) * (S2 - mean_S2)) / sum((PS10 - mean_PS10)^2)
b = mean_S2 - a * mean_PS10
RMSE = sqrt(mean((S2 - (a * PS10 + b))^2))
```

| Fit statistic | Value |
| --- | ---: |
| Slope, a | 1.175829788 |
| Intercept, b | -0.123077350 |
| Pearson correlation, r | 0.971472779 |
| In-sample RMSE, NDVI units | 0.023632507 |
| Paired cells, n | 28,682 |

As a boundary-support sensitivity check, restricting the fit to the
28,334 comparison cells gives **a = 1.172997232**, **b = −0.121790591**,
**r = 0.970681781**, and **RMSE = 0.023438490**. The main result uses the
fit from all 28,682 paired cells.

![Paired 10 m NDVI density and fitted line](../outputs/planet_case_study/ndvi_calibration.png)

Apply the main fit to every native PlanetScope NDVI value without clipping:

```text
NDVI_cal = a * NDVI_PS + b
t = max(0.10, min(0.25, p75(assessed NDVI_cal) - 0.18))
candidate = assessed AND NDVI_cal < t AND native_NDWI <= 0
```

The site p75 changes from **0.534987** to **0.505976**. Recomputing the
adaptive threshold from this calibrated percentile gives **0.25**, because
the cap still applies. This corresponds to a native PlanetScope NDVI cutoff
of approximately **0.317289**. Water exclusion still uses native green/NIR
NDWI; no reflectance-band or NDWI calibration is applied. There are zero
positive-NDWI native pixels in the assessed mosaic. No area sieve, extra
brightness threshold or morphological cleanup changes the candidate mask.

As a diagnostic after fitting, the median NDVI on the 428 common frozen
candidate cells is **0.234592** for Sentinel-2, **0.297760** for PlanetScope
at 10 m before calibration and **0.227038** after calibration. The median
paired PlanetScope-minus-Sentinel-2 difference changes from **+0.064831** to
**−0.006009** there. These candidate-restricted diagnostics are not fit inputs.

### Calibrated GNG variant

Reuse `src/peatland/gng.py` on six features: blue, green, red, NIR,
**calibrated NDVI**, and native NDWI. Z-score each feature over the assessed
PlanetScope pixels using its mean and standard deviation plus 1e-9. Train
on 8,000 pixels sampled without replacement, with seed 42, 15,000 steps
and at most 80 nodes; retain the existing implementation's other defaults.
Assign every assessed pixel to its nearest learned prototype.

Apply the calibrated NDVI gate to the pixel. Reject water where the pixel
NDWI > 0 or the de-standardized prototype NDWI > 0. This follows the frozen
pipeline's physical water-labelling idea and can only remove candidates
from the direct rule. There is no additional centroid-NDVI gate.

The graph has **80 nodes**, **zero water-labelled nodes**, and changes
**zero pixels / 0 ha** relative to the calibrated rule; their native-grid
IoU is **1.0**. With no positive-NDWI site pixels, this water decision adds
no rejection here. A positive affine NDVI transformation also leaves its
z-scored feature coordinate effectively unchanged. The changed physical
pixel gate accounts for the increased area relative to naive transfer.

## Calibrated comparison metrics

### Native areas and patch sizes

Count patches with **8-connectivity**, allowing diagonal connections, and
no minimum mapping unit. Native statistics use each sensor's assessed
area. The summary records every patch size, quantiles and histograms for
both PlanetScope variants, the frozen mask and their aggregated masks.

| Metric | Calibrated PS rules/GNG, 3 m | Frozen S2 GNG, 10 m |
| --- | ---: | ---: |
| Candidate area | 5.3685 ha | 4.2800 ha |
| Candidate pixels | 5,965 | 428 |
| Connected patches | 96 | 13 |
| Minimum patch | 0.0009 ha | 0.0100 ha |
| Median patch | 0.0018 ha | 0.1100 ha |
| p90 patch | 0.02385 ha | 0.6820 ha |
| Largest patch | 2.4021 ha | 2.0100 ha |

| Patch area interval, ha | Calibrated PS rules/GNG count | S2 GNG count |
| --- | ---: | ---: |
| [0, 0.01) | 79 | 0 |
| [0.01, 0.05) | 9 | 5 |
| [0.05, 0.1) | 1 | 1 |
| [0.1, 0.5) | 4 | 5 |
| [0.5, 1) | 2 | 1 |
| >= 1 | 1 | 1 |

### Aggregation and overlap

Sum exact areas of intersection between PlanetScope pixels and target
Sentinel-2 cells. Bare fraction is candidate area divided by observed
assessed area in the cell; coverage is that observed area divided by the
full 100 m². A cell is bare at **fraction >= 0.5**. Compare labels only on
Sentinel-2 `valid & inside` cells with **coverage >= 0.95**. Fractions with
zero support are NaN. Low-coverage cells retain their observed fractions
in the arrays but are excluded from comparison, rather than treated as
negative observations. IoU is intersection divided by union.

| Common 10 m comparison | Calibrated PS rules and GNG |
| --- | ---: |
| PlanetScope bare cells / majority-labelled area | 515 / 5.15 ha |
| Frozen S2 candidate cells / area | 428 / 4.28 ha |
| Intersection / union | 360 / 583 cells |
| IoU | 0.617496 |
| Cells found only at 3 m, after aggregation | 155 |
| Cells found only at 10 m | 68 |
| Largest PlanetScope bare fraction | 1.0 |
| Actual PlanetScope candidate area within comparison cells | 5.364491 ha |
| PlanetScope patches after aggregation | 16 |
| Aggregated patch minimum / median / p90 / maximum | 0.01 / 0.035 / 0.875 / 2.41 ha |

The actual area inside comparison cells sums fractional overlaps; the
5.15 ha majority-labelled area counts whole cells. These are different
quantities. The small difference between 5.364491 ha and native 5.3685 ha
comes from support boundaries.

At native sampling, **3,879 PlanetScope candidate pixels (3.4911 ha)** lie
inside the nearest-reprojected frozen Sentinel-2 footprint and **2,086
pixels (1.8774 ha)** lie outside. Outside-footprint area includes edge
and alignment differences; it is not an independently verified gain.

### Narrow-feature share

Binary opening with a nominal **10 m diameter disk**, sampled by native
pixel-centre distances, removes **972 pixels / 0.8748 ha**, or **16.2951%**
of calibrated candidate area. The result is identical for the GNG variant.
At 3 m sampling the active footprint is a 3 × 3 block. This is a coarse
proxy for small objects, narrow features and boundary protrusions: some
9 m wide shapes can survive, while small objects can disappear regardless
of elongation. Opening is used only for this statistic, not to clean the
reported candidate mask.

## Naive transfer: the short lesson

Applying the unchanged formula directly to uncalibrated PlanetScope NDVI
gives a threshold of **0.25** and just **0.0225 ha**, comprising **25 pixels
in 11 patches**. Its GNG variant is identical. No 10 m cell reaches 50%
bare fraction: **IoU = 0**, **0 PlanetScope-only cells**, and **428
Sentinel-2-only cells**. The nominal 10 m opening removes all 25 pixels.

The raw PlanetScope NDVI offset excludes most of the surface corresponding
to Sentinel-2 candidates. This naive transfer therefore cannot measure a
resolution benefit. It is retained under `naive_transfer` in the summary,
with the same complete metrics and separate saved masks, as a baseline for
the calibrated comparison.

## Figures and saved arrays

Both map figures show the **calibrated 3 m mask** as the main PlanetScope
result. They use matching geographic extents, native grids, nearest-neighbour
display, common RGB scaling (reflectance × 3.2, gamma 1.4), candidate outlines,
north arrows, scale bars and the Planet citation. RGB values are not calibrated.

The zoom is the full 400 × 400 m window with the largest frozen Sentinel-2
candidate count, with ties resolved by first row then column. It contains
**264 S2 candidate cells / 2.64 ha**. Its EPSG:32629 bounds (west, south,
east, north) are **(520925.091757, 5912424.060384, 521325.091757,
5912824.060384)**. Candidate density supplies a reproducible cutting-margin
proxy; recent activity has not been independently verified.

![Whole assessed site](../outputs/planet_case_study/whole_site.png)

![Cutting-margin zoom](../outputs/planet_case_study/cutting_margin_zoom.png)

All three PNGs are saved at **150 dpi**. The maps are 2040 × 1245 pixels;
the calibration density plot is 990 × 840 pixels.

- [summary.json](../outputs/planet_case_study/summary.json): input file names
  and hashes, provenance, calibration coefficients and selection, thresholds,
  all calibrated metrics, the naive baseline, grids and limitations.
- [masks.npz](../outputs/planet_case_study/masks.npz): main calibrated masks
  under `planet_rules` and `planet_gng`, their fractions, aggregated labels
  and opening-loss masks; equivalent baseline arrays prefixed `naive_`;
  native and calibrated NDVI; paired 10 m NDVI/NDWI; `calibration_support`
  and `comparison_support`; assessed masks, coverage, transforms and CRS.
  Fractions and coverage are in [0, 1], not percentages.
- [whole_site.png](../outputs/planet_case_study/whole_site.png),
  [cutting_margin_zoom.png](../outputs/planet_case_study/cutting_margin_zoom.png),
  and [ndvi_calibration.png](../outputs/planet_case_study/ndvi_calibration.png).

## Limitations

- **Single date and calibration site:** the fit and comparison describe
  Monivea on this date. RMSE is an in-sample spectral residual, not an
  independent validation error or a measure of classification accuracy.
- **No SWIR or quality-mask guards:** PlanetScope lacks the Sentinel-2 NBR
  burn and SWIR water/deep-shadow guards. Neither SCL nor UDM2 cloud/shadow
  screening is applied here. Burn, shadow and residual cloud may affect
  candidates despite the NDWI water exclusion.
- **Co-registration:** a possible 1–2 pixel displacement is not estimated
  or corrected. This is 3–6 m in PlanetScope pixels or 10–20 m in S2 cells,
  enough to change edge agreement and narrow-feature counts.
- **Remaining cross-sensor differences:** linear NDVI calibration reduces
  the systematic offset but does not equalize band responses, atmospheric
  correction, spatial response or pixel mixing. Both sensors have error;
  OLS treats PlanetScope NDVI as the predictor and fits S2 as the response.
  Applying a relationship fitted at 10 m to native 3 m NDVI assumes that it
  transfers across scale. A 3 m output grid alone does not establish 3 m
  effective optical resolving power.
- **Morphology and boundaries:** patch counts and opening loss depend on
  connectivity, discretization and assessed-area boundaries. The narrow
  share is not a direct measurement of cutting-strip width.
- **No ground truth:** area differences and IoU measure candidate-output
  agreement. They do not establish precision, recall or additional confirmed
  turf-cutting area. Detector guards and remaining spectral differences also
  prevent treating the calibrated comparison as a pure resolution experiment.

## PlanetScope across the release (private map)

Beyond this case study, PlanetScope was ordered for the password-protected
view of the web map on 15 September 2026 (`scripts/23_planet_release.py`).

**Targets.** Each bog's latest survey year, which is the year its card opens
on, plus the NPWS-documented site-years for 2021 and 2022.

**Selection rules.** The search covered ±3 days around the exact release
Sentinel-2 date. Frames had to be at least 50% clear and had to cover at
least 90% of the clip.

**Clipping.** Clips follow the site boundary buffered by 250 m, not the
bounding box, to save quota.

| Site-years | Count |
|---|---:|
| Ordered (78 latest year, 5 documented) | 83 |
| Same day as Sentinel-2 | 50 |
| 1 day apart | 17 |
| 2–3 days apart | 16 |
| Withheld: under 50% of the bog clear (UDM2) | 11 |
| Shown in the bog card | 72 |
| No qualifying frame | 4 |

The four without a frame are Barroughter 2022, Raford River 2026,
Cloonshanville 2026 and Tullaghan Bog (Roscommon) 2026. Clipped AOIs total
574 km². The conservative quota ledger counts the full AOI once per ordered
frame and records 719.5 km². Like the evaluation mosaics, these orders are
not harmonised to Sentinel-2.

**Cloud screening.** Frames were selected on scene-level clear percentages,
which do not guarantee that the bog itself is clear. Each tile is therefore
checked against Planet's own UDM2 mask. For every pixel, the clear flag comes
from the frame that supplies that pixel in the mosaic. The tile then reports
the clear share of the bog.

- Eleven site-years fall below 50% clear and are withheld from the map. They
  include Monivea 2026 (24% clear) and Doogort East 2026 (33%).
- The 78 tiles shown (6 evaluation, 72 release) have a median of 100% clear.
- Eleven of the tiles shown are between 50% and 90% clear, and the card
  flags them.

UDM2 counts light haze as not clear, so this screen is conservative.

These mosaics are for visual comparison in the bog card. A swipe switches
between the published Sentinel-2 tile and PlanetScope on the same extent.
The frozen v3 GNG outline is recomputed on the exact release scene and
checked against the published area. No PlanetScope detection is run outside
Monivea. Where the frames are 1–3 days apart, the surface can differ, for
example if turf was turned or collected in between. The swipe is therefore
context, not a same-instant reference.

## Citation

Planet Team (2026). Planet Application Program Interface: In Space for Life on Earth.
San Francisco, CA. https://api.planet.com
