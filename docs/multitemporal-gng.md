# Two-date seasonal detector: calibration, frozen application and tidal buffer

This is an experimental method: detector `2026-09-15-multitemporal-gng-v2`, scene
rules `2026-09-15-season-pairs-v5`. It is **not** part of the released detector v3
(`2026-09-13-cloud-screen-v3`), which stays frozen. No reference labels were
opened or used. Everything below reports candidate areas and agreement with the
frozen v3 masks, not accuracy. Scoring waits for the author's reviewed labels
(see the last section).

## Why a second date

On seven of the eight frozen evaluation scenes, GNG, K-means and the spectral
rules give identical single-date masks. The physical guards do the work. The
hardest remaining ambiguity is a pale April strip, which can be freshly cut peat
or last year's dead grass. The two diverge over the season: cut peat stays bare,
while dead grass greens up by June. The falling trends on raised bogs are also
confounded with scene timing (April scenes up to 2022, mostly mid-May to June
after that; `docs/testing-and-validation.md`).

The two-date detector keeps a pixel only if it is bare at an early date **and**
still bare at a late date.

## Method

**Scene pairs.** Each frozen site-year gets two scenes, both reprojected onto the
frozen evaluation grid:

- **Early:** 1 April to 15 May, target 25 April. The frozen v3 scene itself is used
  when it falls in this window.
- **Late:** 1 to 30 June, target 15 June, with a declared fallback of 1 to 15 July.

Candidate scenes are tried in a fixed order: June before the July fallback, then
nearest to the target date, then scene cloud cover. A scene is accepted when it
passes the v3 scene gates on the whole bog: clear fraction at least 0.85, and
bright-haze and HOT haze fractions at most 0.15. Each scene carries the v3
per-pixel validity mask (SCL, blue-band cloud screen with a 30 m buffer, and
projected cloud shadow). A quick SCL screen (clear fraction at least 0.85 on the
bog) skips obviously cloudy scenes before a full read.

**Features.** For each date: NDVI, NBR (NIR/SWIR2), SWIR1 and MNDWI. The detector
uses both dates plus the late-minus-early changes, 12 features in all.

**Physical gate.** A pixel passes when it meets all of these conditions:

- at both dates, NDVI is below that date's adaptive threshold,
  min(0.25, max(p75 NDVI − 0.18, 0.10)), computed on that date's own clear bog
  pixels;
- NBR > 0 at both dates;
- SWIR1 ≥ 0.07 at both dates;
- MNDWI ≤ 0 at both dates;
- mean visible reflectance < 0.30 at both dates;
- NDVI green-up (late minus early) ≤ 0.10.

These are the v3 thresholds applied to two dates. Each date's threshold is
computed separately, so clouds on the late scene cannot move the early
threshold.

**Rules and GNG.**

- *Rules:* a pixel is a candidate if it passes the gate.
- *Trajectory GNG:* a GNG with 80 nodes is trained on 8,000 standardised 12-feature
  pixels drawn from the common clear support (15,000 steps, seed 42). A candidate
  must pass the gate itself and must also belong to a prototype that passes the
  gate.

**Support.** Results exist only where both dates are clear, inside the site and
outside the ever-water mask. Elsewhere the answer is unknown and is never counted
as zero. Persistent candidates are always a subset of the early-date
candidates, so a site with no early candidates has an empty result without
needing a late scene.

Code: `src/peatland/multitemporal.py`, `src/peatland/temporal_experiment.py`,
`scripts/24_multitemporal_gng.py`. Tests: `tests/test_multitemporal.py`,
`tests/test_temporal_experiment.py`.

## Scene rules decided on calibration evidence

The first rule set required a clear late scene over the whole bog. Monivea 2022
then had no late scene, so a candidate-pixel variant was tried on the
calibration site-years only. It accepted a late scene when:

- at least 85% of the early candidates were valid on the late scene, and
- the HOT and bright-haze fractions measured on those candidates were at most
  0.15.

The attempt records are kept in `outputs/multitemporal_gng/rejected_candidate_rule/`.
The variant was **rejected** before any held-out imagery was fetched:

| Late scene | Whole-bog clear | Whole-bog HOT | HOT on early candidates | Variant |
|---|---:|---:|---:|---|
| Doogort East, 22 Jun 2023 | 0.941 | 0.052 | 0.205 | rejected a clear scene |
| Tullaghan Bay, 14 Jun 2026, tile 29UMV | 0.950 | 0.023 | 0.182 | rejected |
| Tullaghan Bay, 14 Jun 2026, tile 29UMA | 0.986 | 0.011 | 0.000 | accepted |
| Monivea, 2 Jul 2022 | 0.953 | 0.335 | 0.850 | rejected (hazy either way) |

The HOT index (blue − 0.5·red) measures haze on vegetated land. On bright, dry,
bare peat it exceeds its 0.02 threshold on clear scenes. It even differs between
two tiles of the same acquisition day. Measured on candidate pixels, it therefore
rejects clear scenes.

The variant also rescued nothing. Monivea's own candidates were clouded in every
June scene (best SCL clear fraction on the candidates 0.45, 4 June) and on 7 and
9 July (0.78, 0.41). The late acceptance therefore reverted to the v3 whole-bog
gates. A share of the early candidates seen on the late scene is still recorded,
as a diagnostic only.

## Calibration site-years

Areas are in hectares. The v3 column is the frozen single-date GNG on the v3
scene date, which differs from the pair dates for Doogort East and Tullaghan Bay.

| Site-year | Early | Late | Status | Early candidates | Persistent rules | Persistent GNG | Frozen v3 GNG |
|---|---|---|---|---:|---:|---:|---:|
| Monivea 2022 | 23 Apr (frozen) | none | unavailable | 4.28 | – | – | 4.28 (23 Apr) |
| Moorfield 2025 | 14 May (frozen) | none | no early candidates | 0.00 | 0.00 | 0.00 | 0.00 (14 May) |
| Doogort East 2023 | 23 Apr | 22 Jun | available | 0.44 | 0.05 | 0.00 | 0.21 (31 May) |
| Tullaghan Bay 2026 | 29 Apr | 14 Jun | available | 23.61 | 15.76 | 10.00 | 31.95 (11 Jul) |

- **Monivea 2022** is the documented-cutting calibration site, and it has no
  usable late scene.
  - 18 scenes were checked (12 in June, 6 in the July fallback).
  - The best whole-bog SCL clear fractions were 0.68 (4 June), 0.46 (9 July) and
    0.37 (7 July).
  - 2 July passed the SCL screen but failed the haze gate (clear 0.953,
    HOT 0.335).
  - On a single date, its early candidates (4.28 ha) equal the frozen v3 mask.
- **Moorfield 2025** is the burn control. It has no early candidates, as v3 has
  none, so the result is empty by construction. No late scene passed either:
  - 22 scenes were checked;
  - the best was 23 June at SCL clear 0.70;
  - 20 June was 0.72 clear but hazy (HOT 0.59).
- **Doogort East 2023:**
  - 0.44 ha of early candidates;
  - 0.39 ha of them are dropped by the late date;
  - the trajectory GNG keeps none;
  - v3 reports 0.21 ha on 31 May, all outside the trajectory result.
- **Tullaghan Bay 2026:**
  - 96.5% of the early candidates are seen on the late scene, and 7.03 ha of them
    are dropped by the late date.
  - On the common support, v3 covers 31.25 ha and the trajectory GNG 10.00 ha, with
    7.91 ha shared (IoU 0.24).
  - 23.34 ha of v3 are not persistent; 2.09 ha are persistent but not in v3 on its
    date.
  - Most v3 and trajectory candidates sit on the estuary shore (figure below).

On two dates, clustering is no longer neutral. At Tullaghan Bay the trajectory
GNG keeps 10.00 ha of the 15.76 ha of rule candidates: only 5 of its 80
prototypes pass the gate. On single dates the GNG and rule masks were identical
on seven of eight scenes. Whether the extra removals are right can only be
judged with labels.

**Green-up sensitivity** (maximum NDVI increase allowed; 0.10 was frozen):

| Site-year | 0.05 | 0.10 | 0.15 |
|---|---|---|---|
| Tullaghan Bay 2026, rules / GNG | 14.13 / 9.29 | 15.76 / 10.00 | 16.02 / 10.04 |
| Doogort East 2023, rules / GNG | 0.05 / 0.00 | 0.05 / 0.00 | 0.05 / 0.00 |

The result is not sensitive to this threshold, so the defaults were frozen
unchanged. They are the v3 values plus a green-up limit of 0.10. Only two
calibration site-years had pairs, which is too few for any tuning.

Figures: `outputs/multitemporal_gng/calibration/figures/`. Summary:
`outputs/multitemporal_gng/calibration/summary.json`.

## Frozen application to all eight site-years

`frozen_params.json` records the version, parameters, scene rules and hashes. It
was written at 2026-09-15T18:19:47Z, one second before the first held-out pair
folder was created. `apply` refuses to run if the detector code, the parameters
or the scene-selection code differ from that record.

| Site-year | Role | Early | Late | Status | Early candidates | Persistent rules | Persistent GNG | Frozen v3 GNG |
|---|---|---|---|---|---:|---:|---:|---:|
| Monivea 2022 | calibration | 23 Apr (frozen) | none | unavailable | 4.28 | – | – | 4.28 (23 Apr) |
| Moorfield 2025 | calibration | 14 May (frozen) | none | no early candidates | 0.00 | 0.00 | 0.00 | 0.00 (14 May) |
| Doogort East 2023 | calibration | 23 Apr | 22 Jun | available | 0.44 | 0.05 | 0.00 | 0.21 (31 May) |
| Tullaghan Bay 2026 | calibration | 29 Apr | 14 Jun | available | 23.61 | 15.76 | 10.00 | 31.95 (11 Jul) |
| Cloonchambers 2022 | held-out | 23 Apr (frozen) | none | unavailable | 7.60 | – | – | 7.60 (23 Apr) |
| Corliskea 2021 | held-out | 15 Apr (frozen) | 27 Jun | available | 4.01 | 0.00 | 0.00 | 4.01 (15 Apr) |
| Moycullen 2018 | held-out | none | not searched | unavailable | – | – | – | 52.57 (16 May) |
| Ederglen 2018 | held-out | 2 May | 26 Jun | available | 0.00 | 0.00 | 0.00 | 0.00 (22 May) |

Notes on the held-out site-years:

- **Cloonchambers 2022** has the same weather as Monivea (same tile, same year).
  18 late scenes were checked (12 in June, 6 in July). The best whole-bog SCL
  clear fraction was 0.29 (4 June). Its 7.60 ha of early candidates cannot be
  checked.
- **Corliskea 2021:**
  - The late scene (27 June, clear fraction 1.00) sees all 4.01 ha of early
    candidates, and none of them is still bare. Every pixel of the frozen v3
    detection of 15 April is dropped by the late date (see the figure).
  - The pale April strips green up by late June.
  - The two-date test cannot say whether they were dead grass, or cut or
    disturbed surfaces that re-vegetated or were covered within ten weeks. The
    reviewed labels on this held-out site-year decide.
- **Moycullen 2018** has no early scene, so its late window was never
  searched.
  - Its frozen v3 scene (16 May) falls one day after the early window.
  - None of the 18 April and May scenes passed the v3 gates. The best SCL clear
    fraction was 0.72 (21 April); 6 May was 0.82 clear with HOT 0.21.
  - The 52.57 ha v3 detection, the largest in the sample, cannot be checked.
- **Ederglen 2018** has no early candidates on 2 May, and v3 has none on
  22 May. Both methods give zero.

Overall, the pair exists for four of the eight site-years. It is unnecessary for
one (Moorfield) and unavailable for three. Of the five site-years where v3
detects 4 ha or more, the pair exists for two (Tullaghan Bay, Corliskea).

Where the pair exists, the persistence test removes most of the v3 detection:

- **Tullaghan Bay:** 7.91 of the 31.25 ha that v3 detects on the common support
  (25%) are persistent; another 2.09 ha are persistent but outside v3.
- **Corliskea:** none of 4.01 ha.
- **Doogort East:** none of 0.21 ha.

Outputs:
- masks: `outputs/multitemporal_gng/masks/`;
- figures: `outputs/multitemporal_gng/figures/`;
- summary: `outputs/multitemporal_gng/apply_summary.json`.

## Tidal fringe buffer (Tullaghan Bay 2026, calibration only)

The ever-water mask (a union of SCL water over all clear scenes) misses the upper
intertidal fringe (`docs/testing-and-validation.md` §3f). Here the mask is grown
by 0, 1 or 2 pixels (8-neighbour) before detection, on the frozen 11 July 2026
scene. The 0-pixel run reproduces the frozen v3 GNG mask pixel for pixel.

| Buffer | Extra area excluded | GNG | Rules | GNG change | v3 GNG inside the new ring | GNG changed outside the ring |
|---|---:|---:|---:|---:|---:|---:|
| 0 px | 0.00 | 31.95 | 29.64 | 0.00 | 0.00 | 0.00 |
| 1 px (10 m) | 48.56 | 27.60 | 24.28 | −4.35 | 6.13 | 1.92 |
| 2 px (20 m) | 91.30 | 22.25 | 20.17 | −9.70 | 10.78 | 1.08 |

19% of the frozen v3 detection (6.13 ha) lies within 10 m of an ever-water
pixel, and 34% (10.78 ha) within 20 m.

The GNG also changes outside the ring, by 1–2 ha, because the network is
retrained on a different pixel set. The rules have no training step, but their
adaptive threshold is recomputed on the smaller area.

Without labels, there is no way to tell whether the ring is upper-intertidal mud
or genuine margin peat. The buffer should be adopted only if the reviewed
Tullaghan Bay calibration points support it. It changes the release, so it
would need a new detector version.

Output: `outputs/multitemporal_gng/tidal_buffer.json`.

## Feasibility

June cloud cover in the West of Ireland is the binding constraint. With the v3
scene gates:

- the pair exists for four of the eight site-years;
- among the five site-years with a v3 detection of 4 ha or more, it exists for
  only two;
- neither raised-bog site-year of 2022 (Monivea, Cloonchambers) has a usable June
  or early-July scene.

A method that needs a clear June scene will be unavailable for a large share of
site-years. Where a pair exists, it can serve as a complement that flags which
early candidates stay bare. It cannot replace the single-date season maximum.

## Limitations

- No reference labels were used. Areas and agreement with v3 describe candidate
  outputs, not accuracy.
- For three of the four site-years with a pair, the pair dates differ from the
  v3 date, so disagreement mixes method and date effects.
- Window edges matter. Moycullen 2018's frozen scene falls one day outside the
  early window, and the window then had no usable scene.
- The held-out result that everything is dropped rests on one site-year
  (Corliskea 2021). It is not evidence of a general rate.
- The late window can miss cutting that starts after the early date. It also
  cannot separate re-vegetating cut peat from grass.
- Two usable calibration pairs are too few to tune anything. The parameters are
  the v3 ones plus a declared green-up limit.
- The July fallback extends the late window up to the end of the raised-bog season
  (15 July). It was declared before calibration results were seen, but it was
  added while the first pairs were being fetched.

## Reproduce

```sh
python3 scripts/24_multitemporal_gng.py fetch-calibration
python3 scripts/24_multitemporal_gng.py calibrate
python3 scripts/24_multitemporal_gng.py freeze    # before any held-out imagery
python3 scripts/24_multitemporal_gng.py apply
python3 scripts/24_multitemporal_gng.py tidal
```

## Scoring once the labels are reviewed

`apply` saves `prediction_trajectory_gng` and `prediction_trajectory_rules` masks
per site-year (`outputs/multitemporal_gng/masks/`). To add their values to the
sampled points and score all methods on the same points:

```sh
python3 scripts/24_multitemporal_gng.py predictions \
    --points outputs/evaluation/2026-09-13-v3-sample/annotation/accuracy_points.csv \
    --output outputs/multitemporal_gng/accuracy_points_trajectory.csv
python3 scripts/14_score_points.py outputs/multitemporal_gng/accuracy_points_trajectory.csv \
    --labels outputs/evaluation/2026-09-13-v3-sample/annotation/labels.csv \
    --methods ndvi rules kmeans gng trajectory_gng trajectory_rules --common-support
```

A point outside a site-year's two-date support gets an empty value, never 0.
`--common-support` scores every method on the points where all of them have a
value. The unweighted sample metrics then describe that subset only, and the
population weights no longer apply. Report the held-out and calibration site-years
separately.
