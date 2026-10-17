# Exposed peat has a short-wave infrared signature (detector v5)

Status: 17 October 2026. Reference labels are the majority of five blind
model-assisted passes (section 4, `scripts/48_multipass_consensus.py`); they
have not been reviewed point by point by the author.

## 1. Why the v1–v4 detectors failed

Two independent checks showed that the spring candidates of v1–v4 were not
exposed peat:

- **Summer check** (`scripts/31_summer_check.py`). On all 8 spring site-years of
  the two evaluation samples, 98–100% of the GNG candidate pixels have a
  same-summer Sentinel-2 median NDVI of 0.50 or more. Exposed peat does not grow
  a canopy within a season; winter-brown Molinia does. PlanetScope summer frames
  of Callow, Cloonchambers, Slieve and Loughatorick South confirm it.
- **Positive control** (`scripts/32_positive_control.py`). On the Bord na Móna
  Mountdillon production bogs (about 2,000 ha of milled peat in summer 2018) the
  v3 GNG detector finds 3 ha. Dry milled peat has a median NDVI of 0.42 (peat is
  dark in red as well as near infrared) and a median NBR of −0.41 (SWIR2 above
  NIR). The v1–v4 rule required NDVI below about 0.25 and NBR above 0: the burn
  guard removed every exposed-peat pixel and the NDVI threshold kept pale dead
  grass.

`scripts/33_detection_limit.py` makes this exact: a strip of milled-peat
spectrum implanted into six protected-bog scenes is never flagged by the v1–v4
rule, at any width up to a full 20 m pixel.

## 2. The v5 rule

`src/peatland/v5.py` tests the SWIR signature directly:

| Condition | Physical meaning |
|---|---|
| NBR < 0 | SWIR2 above NIR: exposed peat or a burn scar |
| SWIR1 ≥ 0.22 | dry peat is bright in SWIR1; char, shadow and wet ground are dark |
| SWIR1 / SWIR2 ≥ 1.25 | fresh burns have a flatter SWIR (median 1.14 against 1.35) |
| MNDWI ≤ −0.30 | not water or saturated sediment |

Thresholds were set on two calibration scenes only: south Mountdillon in 2018
(spring and summer) and the Moorfield 2025 spring burn. They keep 82–85% of the
peat pixels and 4% of the burn pixels. All results below are on other scenes.

## 3. Detection limit

Linear mixing of the milled-peat spectrum into 6 protected-bog scenes; a strip
of width w covers w/10 of a 10 m band and w/20 of a 20 m SWIR band.

| Strip width | v5 flags | v1–v4 rule flags |
|---:|---:|---:|
| 3 m | 7% | 0% |
| 4 m | 24% | 0% |
| 5 m | 58% | 0% |
| 6 m | 81% | 0% |
| 8 m | 97% | 0% |
| ≥ 10 m | ≥ 99.7% | 0% |

The NDVI of the mixed pixel stays at 0.46 at every width: in the visible and
near infrared, exposed peat is indistinguishable from a spring bog. This is
also why PlanetScope (no SWIR band) cannot carry the detection; it serves as
the 3 m reference.

## 4. Independent test (`outputs/evaluation/2026-10-08-v5-test`)

Selection (`scripts/34_v5_test_run.py`), fixed before any prediction was seen:
four industrial site-years not used for calibration (west Mountdillon summer
2018 and spring 2020; south Mountdillon summer 2022 and spring 2024) and eight
protected raised-bog SACs in summer (six with NPWS-documented cutting in that
year, two controls). 272 points stratified by v5, PlanetScope within 4 days of
every Sentinel-2 scene.

Reference labels: five blind passes by three different models (passes A–C one
model, D and E two others), each labelling every point from the same chips
without seeing the other passes or any prediction. The label is the majority;
an unsure vote abstains. Fleiss' κ 0.76 over the five passes (0.77 for bare
peat against the rest), pairwise Cohen's κ 0.67–0.88; 210 of 272 points
unanimous, 247 with at least four of five votes. Against the earlier two-pass
proposal, 8 points changed bare-peat status (7 of them earlier unsure).

| Scope | Method | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Industrial (95 points) | v5 | 88% | 89% | 88% |
| Industrial | NDVI, rules, K-means, GNG | – | 0% | 0% |
| Protected, summer (176 points) | v5 | 54% | 100% | 70% |
| Protected, summer | NDVI, rules, K-means, GNG | – | 0% | 0% |

On the points with at least four of five votes: industrial F1 92% (87 points),
protected 70% (160 points).

These are per-point figures. The sample takes 12 points from each stratum
(v5-detected and not), so detections are over-represented and the per-point
recall is optimistic. Scaled to the pixel population of each site-year
(Olofsson et al., 2014; stratified bootstrap 95% intervals):

| Scope | Precision | Recall | F1 |
|---|---:|---:|---:|
| Industrial | 88% (77–96) | 53% (39–82) | 66% (53–85) |
| Protected, summer | 77% (65–86) | not estimable | – |
| External | 93% (83–100) | 54% (35–100) | 69% (50–97) |

On industrial fields v5 is precise but misses about half of the exposed peat
by area, mostly wet peat below the SWIR1 threshold. On protected bogs none of
the 96 sampled non-detected points was bare peat, so recall cannot be
estimated from this sample: those points stand for about 403,000 pixels, and
the 95% upper bound on missed peat (3.8%) would allow far more missed peat
than detected. Weighted (Olofsson) estimates are in
`annotation/accuracy.provisional_v5_5pass.md`; the two-pass figures (industrial
F1 89%) are in `accuracy.provisional_v5.md`. The protected recall of 100% means no
undetected sample point was labelled bare peat; with 12 undetected points per
site it is not a precise estimate.

A supplementary blind sample drawn after this test (`docs/wet-peat-and-edges.md`)
targeted dark, wet pixels that v5 leaves out. On protected bogs 5 of 40 such
points were bare peat, freshly cut faces beside v5 detections. With that
sample the protected recall becomes estimable: about 42% (33–59%), so v5
finds the dry part of active cutting and misses much of its wet part. The
same check raises industrial recall estimates slightly (62% test, 68%
external) because it adds sample points in the undetected stratum.

By site (v5 detected points labelled bare peat): Callow 10/12, Corbo 11/12,
Barroughter 10/12, Bellanagare 4/12, Monivea 3/12, Corliskea 3/12,
Cloonchambers 2/8. False positives are mixed pixels on cut faces and
cut-over strips; PlanetScope was hazy at Corliskea and Cloonchambers. A blind
relabel with a 30 m question found exposed peat within 30 m of 23 of the 37
false positives; with that tolerance the protected precision is 79% per point
and 87% (78–93%) by area (`docs/wet-peat-and-edges.md`).

### External test (`outputs/evaluation/2026-10-08-v5-external`)

A second group of production bogs south of Athlone (area BNMX), added after v5
was frozen, about 30 km from the calibration area: summer 2018 and spring
2023, 60 points stratified by v5, PlanetScope within 4 days, five blind passes
as above (Fleiss' κ 0.89; 0.93 for bare peat against the rest; 54 of 60
unanimous). One point changed status against the two-pass proposal.

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| v5 | 93% | 90% | 92% |
| NDVI, rules, K-means, GNG | – | 0% | 0% |

## 5. Survey of the raised-bog SACs (`scripts/35_v5_survey.py`)

Clearest summer scene per SAC in 2021 and 2022 (46 site-years):

| Group | Site-years | Median exposed-peat share |
|---|---:|---:|
| Cutting documented by NPWS that year | 9 | 0.18% |
| No record | 37 | 0.00% |

One-sided Mann–Whitney p = 0.0008 (0.0008 after removing site-years within
13 months of a FIRMS fire). The v3 GNG finds nothing in summer on either group.
The largest undocumented value, Cloonmoylan 2021 (39 ha), follows a fire on 31 May 2020, but the dense time series (`docs/timing-and-fire.md`) shows the 2021 signal equally on burnt and unburnt bog: a dry spell, not the fire. The v5 area does not scale with the plot count
(Spearman ρ −0.47, n 7, p 0.29): a plot is a few metres of face, and the area
seen depends on how the turf is spread.

## 6. Industrial time series

v5 area on the Mountdillon areas (`outputs/positive-control/v5_series.json`)
falls from about 1,600–1,700 ha (south, 2018 and spring 2020) to 550–1,250 ha in 2021–2025 after
industrial harvesting stopped at the end of 2020, as fields are rewetted and
colonised. The series is noisy: wet peat after rain has a lower SWIR1 and drops
below the threshold.

## 7. Limitations

- Reference labels come from five blind model-assisted passes, not from field
  survey or point-by-point expert review; agreement between passes is reported
  above.
- v5 also flags burnt peat; FIRMS separates large fires only.
- Wet peat can fall below the SWIR1 threshold; a wet-peat extension is tested
  in `docs/wet-peat-and-edges.md`.
- The detection limit assumes linear mixing and a strip centred on the pixel.

## 8. Web map release (`scripts/36_v5_release.py`)

For every published bog and year 2018–2026 the map uses the clearest summer
scene (15 June – 31 August) passing the shared gates on the release grid, maps
v5 exposed peat and renders the card tiles (`web/data/tiles-v5/`). A summer is
shown but left out of the bog's figures when it follows a NASA FIRMS fire
within 500 m by up to 13 months, or when one contiguous patch covers 10 ha or
more: turf plots are far smaller, so such a patch most likely marks a fire that
FIRMS did not record or has not yet published (the keyless archive ends in
2024). Per bog the map reports the peak and mean summer area, the number of
summers with at least 0.1 ha, the change between the first and last clear
summer and a least-squares trend.
