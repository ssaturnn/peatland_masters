# What 10 m misses: PlanetScope 3 m against Sentinel-2 across 218 bog site-years

The Monivea case study ([planet-case-study.md](planet-case-study.md)) asks what the
10 m detector misses on one site-year. This repeats the identical calculation on
every bog-year that has a cloud-screened PlanetScope mosaic, so the answer is a
distribution rather than a single number.

Study version `2026-09-16-planet-resolution-v1`. The released detector v3
(`2026-09-13-cloud-screen-v3`) is unchanged and is the reference throughout. **No
reference label is read here.** Every figure below is candidate bare surface and
agreement between two candidate masks, never precision, recall or confirmed turf
cutting.

## Method

For each site-year with a PlanetScope tile on the private map:

1. **Assessed area.** The Sentinel-2 `inside` mask is reprojected onto the Planet
   grid and intersected with valid Planet data. Cloudy Planet pixels are removed
   with Planet's own UDM2 clear flag, taken frame by frame in the order the mosaic
   was merged. Tiles below 50% clear were already withheld upstream.
2. **Calibration.** Planet bands are area-averaged to the 10 m grid, NDVI and NDWI
   are computed from those mean bands, and Sentinel-2 NDVI is regressed on Planet
   NDVI by ordinary least squares over every finite, non-water paired cell. No
   candidate mask enters the fit.
3. **Detection at 3 m.** The calibrated NDVI feeds the same rule as the release
   detector: `NDVI < max(0.10, min(0.25, site p75 − 0.18))`, with `NDWI > 0` water
   removed.
4. **Comparison.** The 3 m mask is aggregated back to the 10 m grid by exact area
   weighting; a cell counts as bare at a bare fraction of 0.5 or more where Planet
   covers at least 95% of it. That label is compared with the frozen v3 GNG mask on
   the cells where both sensors are assessed.

Code: `scripts/25_planet_resolution.py` and `src/peatland/cross_sensor.py`, the
module the case study now shares. Outputs: `outputs/planet_resolution/`
(`summary.json`, per-site masks, figures).

## What entered the study

| Site-years | Count |
|---|---:|
| Compared, no failures | 218 |
| Sentinel-2 has candidates | 77 |
| PlanetScope finds candidates where Sentinel-2 finds none | 31 |
| Both sensors empty | 110 |
| Sentinel-2 candidates of at least 0.5 ha | 24 |

By survey year: 64 site-years from 2024, 77 from 2025 and 68 from 2026, plus the
nine earlier years that carry documented cutting (2021–2023). Most are the quiet
years of quiet bogs: in 110 of 218 neither sensor reports anything, which is
agreement of a sort but says nothing about resolution. The numbers below therefore
separate the site-years where the 10 m detector actually fires.

## Cross-sensor calibration

Across all 218 scenes the relationship is consistent with Monivea's:

| Fit | Median | p25 – p75 | Range |
|---|---:|---|---|
| Slope *a* | 1.178 | 1.126 – 1.253 | 0.145 – 1.527 |
| Intercept *b* | −0.145 | −0.211 – −0.092 | −0.440 – 0.597 |
| Pearson *r* | 0.973 | 0.944 – 0.984 | 0.116 – 0.996 |
| RMSE (NDVI) | 0.024 | 0.021 – 0.034 | 0.011 – 0.113 |

PlanetScope reads systematically greener than Sentinel-2 at every level, exactly as
at Monivea, which is why an uncalibrated threshold transfer is meaningless.

Thirteen of the 218 scenes correlate too weakly for the transfer to be trusted
(*r* < 0.8) and are listed in `summary.json` under
`totals.unreliable_calibrations`. They are mostly blanket bogs, or bogs whose
assessed surface is nearly uniform, so there is little NDVI range for the fit to
work with. Their 3 m areas should not be read as resolution effects.

## Agreement and area

| Measure | Value |
|---|---|
| Pooled IoU over all compared cells | **0.30** (6,561 shared, 8,518 only at 3 m, 6,829 only at 10 m) |
| IoU per site-year where Sentinel-2 fires | median 0.19 (n = 77), p25 0.01, p75 0.38, max 0.92 |
| IoU on site-years with at least 0.5 ha | median 0.31 (n = 24), p25 0.19, p75 0.49 |
| Pooled candidate area, 3 m ÷ 10 m | **1.13** (150.8 ha against 133.9 ha on common support) |
| Area ratio per site-year | median 1.29 (n = 77); on the larger sites 1.38 |
| Narrow-feature share at 3 m | median 0.34 where Sentinel-2 fires (n = 67); 0.26 on the larger sites |

On the seven site-years that carry documented NPWS cutting (Monivea 2021 and 2022,
Callow 2021 and 2022, Cloonchambers 2021 and 2022, Corliskea 2021) the picture is
tighter and consistent: **median IoU 0.52, median area ratio 1.20, median narrow
share 0.23**. Where the detector fires on real cutting, 3 m sees roughly a fifth
more candidate area, in the same places, and about a quarter of that area sits in
features narrower than 10 m.

### Why the pooled figures fell when the study grew

An earlier version of this study covered 78 site-years and reported a pooled IoU of
0.44 and a pooled area ratio of 1.33. Adding the 2024 and 2025 years of every bog
took the study to 218 site-years and moved those to 0.30 and 1.13. The drop is a
property of what was added, not a change in what the sensors see:

- The added years are mostly **quiet bogs with tiny candidate areas**. Where a bog
  has three or four candidate cells, one cell of displacement — from
  co-registration, from the 3 m/10 m boundary, or from the three-day gap between
  acquisitions — is enough to drive the IoU to zero. Half of the site-years where
  Sentinel-2 fires now sit below 0.19, and the p25 is 0.01.
- They include **more blanket bog**, where the comparison is weakest (below).
- The larger-area subset moved far less (IoU 0.45 to 0.31 on n = 14 → 24), and the
  **documented-cutting subset did not move at all**: IoU 0.52, ratio 1.20, narrow
  share 0.23, exactly as before.

The last point is what matters for the thesis. Adding 140 mostly empty site-years
dilutes every pooled statistic; it does not change what the two sensors show where
turf cutting is actually documented.

### Raised bogs and blanket bogs behave differently

| Bog type | Site-years | With Sentinel-2 candidates | Median IoU | Median area ratio | Median *r* |
|---|---:|---:|---:|---:|---:|
| Raised | 174 | 58 | 0.19 | 1.35 | 0.977 |
| Blanket | 44 | 19 | 0.11 | 0.85 | 0.943 |

The comparison works better on raised bogs, which is where the documented cutting
and the thesis's hotspots are. On blanket bogs the two sensors barely agree on which
cells are bare: their surfaces are more uniform, the calibration is weaker, the
candidate areas are small enough that a one-cell shift destroys the overlap, and at
3 m the ratio is below one, so PlanetScope finds *less* than Sentinel-2 there. That
is a limitation of the cross-sensor comparison on blanket bog, not evidence about
either detector's accuracy.

## Where PlanetScope finds what Sentinel-2 rejects

Thirty-one site-years have 3 m candidates and no 10 m candidate at all. Two
mechanisms explain the largest of them, and both are the missing SWIR bands:

| Site-year | 3 m candidates | Why Sentinel-2 reports nothing |
|---|---:|---|
| Moorfield 2025 | 4.10 ha | The 2025 burn scar. Sentinel-2 rejects it with the NBR burn guard; PlanetScope has no SWIR and cannot. |
| Rosroe 2026 | 1.25 ha | The site's open lough and its margins. Sentinel-2 rejects dark water on SWIR; PlanetScope has only NDWI. |

`example_disagreement.png` is Moorfield 2025 for exactly this reason.

**The largest single source of 3 m-only cells is open water.** Moycullen Bogs NHA,
a lake-studded blanket-bog complex, appears twice at the top of the list: 2024
contributes 1,940 cells that are bare at 3 m and not at 10 m, and 2026 another
1,656. Together that is **42% of every 3 m-only cell in the study**, from two
site-years out of 218. On the common support the site reports 19.9 ha at 3 m against
0.6 ha at 10 m in 2024, and 17.1 ha against 0.9 ha in 2026.

Measured on the Sentinel-2 scene, those cells are water, not peat:

| Moycullen | 3 m-only cells | Mean NDVI | Mean SWIR1 | Mean visible | Below the v3 SWIR floor (0.07) |
|---|---:|---:|---:|---:|---:|
| 2024 | 1,940 | −0.48 | 0.021 | 0.004 | 96% |
| 2026 | 1,656 | −0.73 | 0.027 | 0.009 | 97% |

The v3 detector removes them through its SWIR water floor. PlanetScope has only the
green–NIR water index, and peat-stained water is dark in both of those bands, so the
index does not fire and the calibrated NDVI rule sees a dark, non-vegetated surface.
The extra 3 m area is therefore not a straightforward gain: a large part of it is
surface the 10 m detector deliberately excludes, and without SWIR the 3 m mask
cannot make that distinction.

## Does clustering add anything at 3 m?

The four-band GNG variant was fitted on all 218 scenes alongside the rules. It
changed the mask on **two** site-years — Lough Gall 2025 (332 pixels, 0.30 ha) and
Tullaghan Bay 2026 (106 pixels, 0.10 ha) — and was pixel-identical on the other
**216 of 218**. This extends the single-date finding from the eight frozen
Sentinel-2 scenes to 218 PlanetScope scenes: with the physical guards in place,
clustering on a single date changes essentially nothing.

## Figures

- `distributions.png` — IoU, area ratio and narrow share across the site-years.
- `example_agreement.png` — Derrinlough 2024, the best agreement in the study
  (IoU 0.92).
- `example_disagreement.png` — Moorfield 2025, the burn scar only PlanetScope flags.
- `example_largest.png` — Tullaghan Bay 2026, the largest 3 m candidate area
  (38.6 ha).

## Limitations

- No reference label is used, so nothing here is accuracy. Both masks are candidate
  bare surface, and neither has been confirmed as turf cutting.
- One date per site-year. Any difference mixes resolution with whatever changed
  between the two acquisitions; the frames are within three days of Sentinel-2, not
  simultaneous.
- PlanetScope has no SWIR, so the burn and the water/deep-shadow guards are
  unavailable. The 3 m mask is the less protected of the two, and at Moycullen that
  alone accounts for 42% of the study's 3 m-only cells.
- Co-registration is neither estimated nor corrected. One to two pixels is 3–6 m and
  changes edge agreement and the narrow-feature count. On the many site-years with
  only a few candidate cells, that is the difference between an IoU of 0.5 and 0.
- The calibration is fitted per scene at 10 m and applied to native 3 m pixels,
  which assumes the relationship transfers across scale; its residuals are
  in-sample. Thirteen scenes fail the *r* ≥ 0.8 check outright.
- The narrow share is a morphological opening proxy: it includes small objects and
  boundary protrusions and is not a measured cutting-strip width. Over all 133
  site-years with 3 m candidates its median is 0.62, because a bog whose entire 3 m
  mask is a few isolated specks has every one of them removed by the opening; the
  0.34 and 0.26 figures above are the meaningful ones.
- Per-site ratios on tiny areas are unstable — the maximum is 33 on less than a
  hectare — which is why the pooled and the at-least-0.5 ha figures are quoted
  alongside the medians.
- UDM2 counts light haze as not clear, so the cloud screen is conservative.

## Reproduce

```sh
python3 scripts/25_planet_resolution.py --gng
python3 scripts/25_planet_resolution.py --key 002352_2022    # one site-year
```

## Citation

Planet Team (2026). Planet Application Program Interface: In Space for Life on Earth.
San Francisco, CA. https://api.planet.com
