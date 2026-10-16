# Is it a hidden peat fire, and is it cut peat or dry grass? (October 2026)

Supervisor's questions: could the exposed peat that v5 finds be an
underground (smouldering) peat fire that NASA FIRMS does not record, and can we
tell cut peat from dried grass? Four checks, all on data already used in the
project plus Landsat surface temperature.

Scripts: `scripts/44_thermal_check.py`, `scripts/45_dry_grass_check.py`,
`scripts/47_fire_behaviour_firms.py`. Outputs (gitignored):
`outputs/evaluation/thermal-check.json` and `.png`,
`outputs/evaluation/dry-grass-check.json`,
`outputs/evaluation/fire-behaviour-firms.json`.

## 1. Surface temperature (Landsat 8/9 thermal band)

Landsat Collection 2 Level-2 surface temperature (ST_B10), 15 clear summer
overpasses 2017–2025 on which both Castle Ffrench East and a Bord na Móna
Mountdillon area are cloud-free (at least 80% clear). Classes: Castle Ffrench
East worked fields (pixels flagged v5 on at least 40% of the clear
April–September scenes of 2017–2023, 46.5 ha) against the intact bog of the
same site (never flagged); Mountdillon exposed peat (v5 on the 2018 summer
scene) against vegetation (NDVI ≥ 0.6) in the same rectangle. Mountdillon is
milled or abandoned production bog, not burning.

| Same overpass, median of 15 dates | Value | Range |
|---|---:|---:|
| Castle fields minus intact bog | +4.5 K | +0.9 to +8.9 |
| Mountdillon exposed peat minus vegetation | +6.8 K | +3.1 to +9.2 |
| Castle fields minus Mountdillon exposed peat | +0.3 K | −2.7 to +6.4 |
| Hottest 1% minus median, Castle fields | 1.8 K | 1.1 to 3.9 |
| Hottest 1% minus median, Mountdillon peat | 5.8 K | 3.1 to 8.3 |

The fields are warmer than the intact bog by the same amount that known,
non-burning bare peat is warmer than vegetation: a dark, dry, non-transpiring
surface heats in the sun. They are not warmer than Mountdillon peat, and they
have no hot outliers (their hottest 1% sits less than 4 K above their median,
less than in Mountdillon). From 2024 the field-minus-bog difference drops to
2–4.5 K, consistent with less exposed peat after working declined.

Limits. Landsat is a daytime (about 11:30) sensor, so solar heating masks a
weak subsurface source; a night-time scene would be a sharper test. The
thermal band is 100 m, resampled to 30 m: a smouldering pocket of a few
square metres is diluted about a hundredfold in one pixel and cannot be ruled
out by this check. A smouldering area of tens of metres would show as a hot
outlier of several kelvin, and none is seen.

## 2. Behaviour through time (Castle Ffrench East)

From the field-exposure series (276 clear Sentinel-2 scenes) and ERA5 rain at
each scene (`scripts/42_rain_check.py`):

- 18 April–September scenes where the fields drop from exposed (≥ 60%) to
  dark (< 20%) earlier in the same season. Median rain in the 3 days before:
  16 mm, against 0.8 mm on exposed scenes; 15 of the 18 follow at least 5 mm.
- 7 of the 18 come back to ≥ 60% later in the same season, median 10 days
  later (5 days at the quickest). The others fall in late August–September
  with no later clear summer scene, and the fields are exposed again the next
  spring.
- SWIR1/SWIR2 of the field pixels on dry scenes (≥ 50% exposed): median 1.46,
  5th percentile 1.37, none below 1.2 (248,194 pixel observations). The fresh
  Cloonmoylan burn two days after its fire had 1.11.

A burn scar does not switch off with rain and back on within days, and it has
a flat SWIR. The fields behave like dry peat that wets and dries.

## 3. FIRMS near every v5 detection

FIRMS on disk: the keyless VIIRS (375 m) and MODIS (1 km) country archives for
2018–2024 (`docs/firms-burns.md`); 2025–2026 need a MAP_KEY and are not
checked. For each bog-year of the v5 release with at least 0.1 ha of exposed
peat (233 bog-years, 60 bogs), any vegetation-fire detection of any confidence
within 1 km of the boundary, two years either side:

- 185 of 233 bog-years have no FIRMS detection at all (770 of 942 ha); 48 have
  one (172 ha). Of the 167 bog-years up to 2024, 122 are silent.
- Only 20 bog-years (111 ha) have a detection in the same or the previous
  year. Nine are already flagged after-fire in the release and left out of
  its figures (Doogort 2019, Cloonmoylan 2020–2021, Tullaghan (Roscommon)
  2018, Callow 2018–2019, Flughany 2018, Moycullen 2023, Monivea 2022). Two
  scenes predate the fire (Moorfield and Moycullen, 28 June 2018, fires on 1
  and 10 July). The other nine are small areas (0.15–5.5 ha) with a fire
  500 m–1 km away or more than 400 days earlier, outside the release's flag.
- Castle Ffrench East and Killaclogher, the two large workings, have no FIRMS
  detection within 1 km in 2018–2024.

FIRMS misses small and smouldering fires (it detects flaming fronts with a
375 m to 1 km pixel), so a silent FIRMS does not prove there was no fire. That
is why checks 1 and 2 matter: they test the exposed surface itself.

## 4. Cut peat or dry grass

The old spring NDVI detector's candidates that are green by summer (median
summer NDVI ≥ 0.5) are winter-dead Molinia and sedge: 17,568 pixels on 8
April–May site-years. On the same spring scenes v5 flags **7 of them
(0.04%)**. Almost all fail NBR < 0 (17,541 pixels); the SWIR1 and ratio tests
fail only 171 and 12.

| Median | Dry grass (spring) | Industrial exposed peat (Mountdillon 2018) |
|---|---:|---:|
| NDVI | 0.24 | 0.42 |
| NBR | +0.11 | −0.36 |
| SWIR1 | 0.43 | 0.33 |
| SWIR1/SWIR2 | 1.51 | 1.36 |

Dry grass is brighter than peat in near infrared and keeps NBR positive; peat
has SWIR2 above NIR. NBR is what separates them, not NDVI.

In the labelled v5 test (272 points; reference labels are model pre-labels
from two blind passes, pending review), v5 flags 128 points: 80 bare peat,
40 vegetated, 7 unsure, 1 other. By the annotator notes the 40 vegetated
ones are 29 mixed pixels at a cut face or strip edge, 5 revegetating cutaway,
4 tracks, 2 other vegetation; **none mentions dry or dead grass**. (The raw
point precision here, 67%, is unweighted over both scopes; the stratified
figures in `docs/v5-swir-peat.md` stand.)

## Conclusion

- **Hidden fire.** No sign of one at Castle Ffrench East: the fields are as
  warm as non-burning bare peat and have no hot spots; the exposure switches
  off with rain and back on within days; their SWIR ratio is far from a burn.
  FIRMS is silent at both large workings. Small smouldering pockets below the
  100 m thermal pixel cannot be excluded by satellite data alone.
- **Cut peat or dry grass.** v5 flags 0.04% of known dry-grass pixels; its
  errors are mixed pixels at cut edges, not dry grass. Drought is the
  remaining confusion: in a dry spell, degraded bog surfaces can pass the SWIR
  test (Cloonmoylan 2021, `docs/timing-and-fire.md`).
