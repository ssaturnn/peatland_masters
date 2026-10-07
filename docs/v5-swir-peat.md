# Exposed peat has a short-wave infrared signature (detector v5)

Status: 8 October 2026. Reference labels are model pre-labels (two blind passes)
pending author review; figures are provisional.

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
every Sentinel-2 scene, two blind labelling passes (agreement 94.5%,
κ 0.88).

| Scope | Method | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Industrial (92 points) | v5 | 87% | 91% | 89% |
| Industrial | NDVI, rules, K-means, GNG | – | 0% | 0% |
| Protected, summer (172 points) | v5 | 54% | 100% | 70% |
| Protected, summer | NDVI, rules, K-means, GNG | – | 0% | 0% |

Weighted (Olofsson) estimates are in
`annotation/accuracy.provisional_v5.md`. The protected recall of 100% means no
undetected sample point was labelled bare peat; with 12 undetected points per
site it is not a precise estimate.

By site (v5 detected points labelled bare peat): Callow 9/11, Corbo 10/11,
Barroughter 9/11, Bellanagare 4/11, Monivea 3/12, Corliskea 4/12,
Cloonchambers 2/8. False positives are mixed pixels on cut faces and
cut-over strips; PlanetScope was hazy at Corliskea and Cloonchambers.

### External test (`outputs/evaluation/2026-10-08-v5-external`)

A second group of production bogs south of Athlone (area BNMX), added after v5
was frozen, about 30 km from the calibration area: summer 2018 and spring
2023, 60 points stratified by v5, PlanetScope within 4 days, two blind passes
(agreement 96.7%, κ 0.94).

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| v5 | 93% | 93% | 93% |
| NDVI, rules, K-means, GNG | – | 0% | 0% |

## 5. Survey of the raised-bog SACs (`scripts/35_v5_survey.py`)

Clearest summer scene per SAC in 2021 and 2022 (46 site-years):

| Group | Site-years | Median exposed-peat share |
|---|---:|---:|
| Cutting documented by NPWS that year | 9 | 0.18% |
| No record | 37 | 0.00% |

One-sided Mann–Whitney p = 0.0008 (0.0008 after removing site-years within
13 months of a FIRMS fire). The v3 GNG finds nothing in summer on either group.
The largest undocumented value, Cloonmoylan 2021 (39 ha), follows a fire on
31 May 2020 recorded by FIRMS: burnt peat is exposed peat too, so v5 maps it;
the FIRMS record separates it. The v5 area does not scale with the plot count
(Spearman ρ −0.47, n 7, p 0.29): a plot is a few metres of face, and the area
seen depends on how the turf is spread.

## 6. Industrial time series

v5 area on the Mountdillon areas (`outputs/positive-control/v5_series.json`)
falls from about 1,600–1,700 ha (south, 2018–2020) to 550–1,050 ha after
industrial harvesting stopped at the end of 2020, as fields are rewetted and
colonised. The series is noisy: wet peat after rain has a lower SWIR1 and drops
below the threshold.

## 7. Limitations

- Reference labels are model pre-labels; the author should confirm at least
  the bare-peat points.
- v5 also flags burnt peat; FIRMS separates large fires only.
- Wet peat can fall below the SWIR1 threshold.
- The detection limit assumes linear mixing and a strip centred on the pixel.
