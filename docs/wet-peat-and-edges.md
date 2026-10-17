# Wet peat and edge pixels: two follow-up checks on v5

Status: 17 October 2026. Both checks were run after v5 was frozen and tested
(`docs/v5-swir-peat.md`). They do not change the frozen detector or its test
figures; they explain its two weak points and test one extension.

## 1. Why the checks

The area-weighted v5 test left two questions:

- Industrial recall was 53%. Nearly all of the weighted omission came from five
  missed peat points, four of them on the south Mountdillon area in spring 2024,
  after harvesting had stopped and the fields were being rewetted. The missed
  points are dark in every band: SWIR1 0.15–0.25 (v5 needs 0.22), NBR above zero
  (v5 needs below zero), NIR 0.06–0.23, against 0.36 for vegetation.
- Protected precision was 54% per point (77% by area). The 37 false positives
  are spectrally the same as the true detections (median SWIR1 0.25 for both,
  NBR −0.04 against −0.10). The annotators' notes keep placing them "at the edge
  of a cut face" or "between turf strips".

## 2. Wet peat (`scripts/49_wet_peat.py`)

`v5.wet_peat_mask` adds pixels that v5 rejects but that are dark and not
green: NIR < 0.20, NBR < 0.30, NDVI < 0.60, MNDWI ≤ −0.30, SWIR1 ≥ 0.14. The
thresholds were read off the five missed industrial points, so the original
sample cannot test them. A new sample was drawn instead: 164 points on the
same 14 frozen scenes (test and external), stratified by v5 detection, the
wet-peat signature and the rest, none of them in the original sample. Points
carry opaque ids and no site name, so annotators saw neither the site, the
season nor the stratum. Five blind passes as for the main test (Fleiss' κ 0.79;
0.80 for bare peat against the rest; 127 of 164 unanimous).

Five-pass majority labels of the new points:

| | Wet-peat stratum | Rest | v5 |
|---|---:|---:|---:|
| Industrial (6 scenes) | 40 bare / 60 | 1 / 24 | 22 / 24 |
| Protected (8 site-years) | 5 bare / 40 | 0 / 16 | – |

Pooled with the original sample and post-stratified by the finer map
(area-weighted, stratified bootstrap 95% intervals):

| Scope | Map | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Industrial test | v5 | 90% (82–97) | 62% (50–74) | 73% (64–82) |
| | v5 + wet peat | 84% (78–90) | 82% (67–100) | 83% (74–92) |
| Industrial external | v5 | 93% (84–100) | 68% (63–72) | 78% (73–83) |
| | v5 + wet peat | 88% (80–95) | 100% | 94% (89–97) |
| Protected | v5 | 77% (67–86) | 42% (33–59) | 55% (45–68) |
| | v5 + wet peat | 14% (9–17) | 100% | 24% (17–29) |
| | v5 + wet peat within 30 m of v5 | 61% (41–78) | 98% (94–100) | 75% (58–87) |

On production fields the wet-peat signature is mostly peat (40 of 60) and
adding it raises area-weighted F1 by 10–16 points. On intact bog the same
signature is mostly wet heath and hollows (35 of 40), so adding it everywhere
ruins precision. Four of the five bare protected wet-peat points lie within
22 m of a v5 detection (the fifth 206 m away); the vegetated ones lie a median
of 316 m away. They are freshly cut faces beside the drying turf that v5 does
see.

So v5 finds the dry part of active turf cutting and misses much of its wet
part: the protected recall, not estimable from the original sample, is about
42% (the estimate moves between about 25% and 45% with how the wet stratum is
split, because it rests on five bare points). The 30 m rule was chosen after
seeing these labels and rests on those five points; it is an exploratory
result, not a validated detector.

## 3. Edge pixels (`scripts/50_block_check.py`)

v5's SWIR bands are sampled at 20 m, so a vegetated 10 m pixel beside a cut
face can carry part of its signal. All 176 protected test points were
relabelled blind with a 30 m question: is exposed peat covering about a tenth
or more of the 3 × 3-pixel block around the point? Chips show the block on
PlanetScope 3 m and Sentinel-2; ids are opaque. Five passes (pairwise κ
0.67–0.91, mean 0.79; 160 of 176 points with at least four matching votes).

| | Per point | By area |
|---|---:|---:|
| Precision, peat in the pixel | 54% (43 / 80) | 77% (67–86) |
| Precision, peat within the 30 m block | 79% (63 / 80) | 87% (78–93) |

23 of the 37 pixel-level false positives have exposed peat in their block. By
site, detected points with peat in the block: Callow 11/12, Corbo 12/12,
Barroughter 9/12, Corliskea 11/12, Cloonchambers 8/8, Bellanagare 6/12,
Monivea 6/12. Of the 96 non-detected points, 2 have peat in the block.

The remaining 14 false positives are concentrated at Bellanagare (5, the
control bog with no documented cutting) and Monivea (5): dry high-bog surface
with the peat signature and no cut geometry.

## 4. Persistence through the summer (`scripts/51_summer_persistence.py`)

v5 was run on every clear scene from 15 May to 15 September of each protected
test site-year (6–11 scenes, frozen test scene reproduced exactly). True
detections appear on a median of 4 other scenes, false positives on 3.
Requiring a detection on at least one other scene removes 8 false positives
and 2 true ones (per-point precision 54% → 59%, area-weighted F1 87% → 88%);
stricter rules lose more true detections than false ones. The false positives
are as persistent as cut faces, which fits the edge explanation above, and
dry-surface false positives at Bellanagare are the short-lived ones.

## 5. What this means

- On production and cutaway bog, v5 plus the wet-peat signature is the better
  map (F1 83–94% by area against 73–78%).
- On protected bogs, most v5 false positives sit at real cutting: as a
  locator of turf cutting, v5 is right within 30 m for 87% of the detected
  area. Its real weakness there is omission of wet, freshly cut peat.
- Growing v5 into adjacent wet-peat pixels is the obvious fix and works on
  this sample, but needs an independent sample before it is used.

All labels come from model-assisted blind passes, as in the main test.
