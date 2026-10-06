# Fires on the monitored bogs: NASA FIRMS, 2018–2024

This checks when the 82 published bogs burned, using NASA FIRMS active-fire
detections, and places each fire among the survey years the release publishes.
It adds context to each fire from the weather before it and the legal calendar.
The aim is to find burn scars that could pass as candidate bare peat, and to
give the detector's spectral burn guard an external reference.

Study version `2026-09-28-firms-burns-v1`. No reference label is read. Detector
v3 is unchanged.

## Data and coverage

| Source | Pixel | Years used | Access |
|---|---|---|---|
| VIIRS S-NPP (`viirs-snpp`) | 375 m | 2018–2024 | keyless FIRMS country archive |
| VIIRS NOAA-20 (`viirs-jpss1`) | 375 m | 2018–2024 | keyless FIRMS country archive |
| MODIS Terra/Aqua (`modis`) | 1 km | 2018–2024 | keyless FIRMS country archive |

The yearly country archives stop at 2024: a year is published there some
months after it ends. **2025 and 2026 need the FIRMS area API with a free
MAP_KEY**, which the user registers with an email address at
<https://firms.modaps.eosdis.nasa.gov/api/map_key/>. The script is ready for it
(`--api-start`), but that path has not been run. As a result, the Moorfield Bog/Farm
Cottage burn of May 2025 cannot be checked yet. FIRMS has no detection at that
site in 2018–2024.

Weather comes from the Open-Meteo historical archive (ERA5-based reanalysis,
keyless) at each bog's centroid.

## Method

- **Matching.** A detection belongs to a bog when it lies within one pixel
  footprint of the boundary: 375 m for VIIRS, 1 km for MODIS. The distance to
  the boundary is kept, and 0 m means inside the bog. FIRMS non-vegetation
  types (volcano, static land source, offshore) are dropped.
- **Events.** Detections on the same bog less than three days apart form one
  event.
- **Weather.** For the 30 days before each event: total rain, days since the
  last day with at least 1 mm, and mean temperature. The rain is compared with
  the median of the same calendar window in 2016–2024. An event is flagged
  *dry* after a spell of 10 or more dry days, or with under half the usual rain.
  The flag is a descriptive heuristic, not a drought index.
- **Calendar.** Section 40 of the Wildlife Act prohibits burning vegetation on
  uncultivated land from 1 March to 31 August. Events in that window are flagged.
- **Release context.** Each event is placed against the published scene of the
  same year (was the fire before or after it?) and the GNG area of that year and
  the next.

Code: `scripts/26_firms_burns.py`, tests `tests/test_firms_burns.py`. Outputs
(gitignored): `outputs/firms/detections.csv`, `outputs/firms/events.json`.

## Results

FIRMS has **117 detections within a footprint of 17 of the 82 bogs**, grouped into
**22 fire events**. Thirteen events have at least one detection inside the
boundary. Fifteen rest on a single sensor and eight on a single detection.

| Year | Events | Bogs | With a detection inside the bog |
|---|---:|---:|---:|
| 2018 | 5 | 5 | 4 |
| 2019 | 2 | 2 | 2 |
| 2020 | 9 | 8 | 5 |
| 2021 | 0 | 0 | 0 |
| 2022 | 1 | 1 | 1 |
| 2023 | 3 | 3 | 1 |
| 2024 | 2 | 2 | 0 |

Twenty-one of the 22 events fall in the closed season (1 March to 31 August).
The exception is one distant detection in December 2023.

| Fire | Bog | Detections | Nearest | Rain 30 d / usual | Dry days before | Published GNG, that year → next |
|---|---|---:|---:|---|---:|---|
| 2018-05-06 | Tullaghan Bog (Roscommon) | 10 | inside | 102 / 85 mm | 2 | 0.51 (21 Apr) → 0.00 |
| 2018-06-28 | Callow | 3 | inside | 56 / 103 mm | 7 | 0.00 (scene the same day) → 0.33 |
| 2018-06-28 | Flughany | 4 | inside | 61 / 110 mm | 7 | 0.02 (3 Jul, after the fire) |
| 2018-07-01 | Moorfield Bog NHA | 14 | inside | 39 / 102 mm | 10 | 0.00 (29 May) → not published |
| 2018-07-10 | Moycullen | 1 | 903 m | 56 / 94 mm | 4 | 52.6 (16 May) → 0.42 |
| 2019-04-21 | Barroughter | 1 | inside | 60 / 78 mm | 5 | 0.02 (11 May, after the fire) |
| 2019-04-22 | Doogort East | 17 | inside | 66 / 112 mm | 0 | 0.07 (21 Jun, after the fire) |
| 2020-04-20 | Callow | 11 | inside | 26 / 98 mm | 1 | 3.08 (15 Apr) → 4.31 |
| 2020-04-20 | Tullaghanrock | 7 | inside | 26 / 98 mm | 1 | 0.02 (5 May, after the fire) |
| 2020-04-25 | Carrowbehy/Caher | 1 | 242 m | 18 / 96 mm | 12 | 4.08 (15 Apr) → 3.79 |
| 2020-05-05 | Callow | 1 | 202 m | 41 / 91 mm | 3 | 3.08 (15 Apr) → 4.31 |
| 2020-05-24 | Barroughter | 1 | inside | 63 / 71 mm | 0 | 0.00 (15 Apr) → 0.01 |
| 2020-05-31 | **Cloonmoylan** | 24 | inside | 30 / 69 mm | 7 | **0.31 (15 Apr) → 5.72** |
| 2020-06-03 | **Funshin** | 5 | inside | 35 / 69 mm | 10 | **0.25 (15 Apr) → 1.51** |
| 2020-06-03 | Clooncullaun | 2 | 559 m | 35 / 69 mm | 10 | 0.44 (15 Apr) → 0.42 |
| 2020-06-03 | Leaha | 2 | 421 m | 35 / 69 mm | 10 | 0.00 (15 Apr) → 0.00 |
| 2022-04-03 | Monivea | 1 | inside | 56 / 119 mm | 0 | 4.28 (23 Apr, after the fire) |
| 2023-04-07 | Doogort East | 2 | 262 m | 243 / 129 mm | 0 | 0.21 (31 May, after the fire) |
| 2023-06-08 | Moycullen | 6 | inside | 31 / 103 mm | 17 | 0.70 (22 Jun, after the fire) |
| 2023-12-01 | Derrynagran | 1 | 368 m | 91 / 114 mm | 4 | 0.00 (30 May) → 0.00 |
| 2024-04-03 | Tullaghan Bay | 2 | 162 m | 137 / 141 mm | 0 | year not published |
| 2024-06-23 | Derrycanan | 1 | 368 m | 72 / 85 mm | 1 | 0.02 (24 Apr) → 0.00 |

### What the fires show about the detector

**The burn guard holds on fresh scars that the release scene saw.**

- Doogort East burned in April 2019 (17 detections inside the bog). The
  scene two months later reports 0.07 ha.
- Tullaghanrock burned two weeks before its May 2020 scene and reports
  0.02 ha.

In both cases the NBR guard kept the scar out of the candidates.

**Recovering scars are the leak, as expected.** The year after a fire can
look like bare peat: the scar has lost its negative NBR but not yet regained
its vegetation. Two bogs burned in late May and early June 2020, after that
year's April scene, and their published GNG area jumps in the next year:

- Cloonmoylan: 0.31 → 5.72 ha.
- Funshin: 0.25 → 1.51 ha.

Cloonmoylan's 5.72 ha in 2021 is the peak of one of the ten significant falling
trends. Part of that trend may therefore come from a burn, not from cutting.

Callow burned on 20 April 2020, days after its 15 April scene, and then reads
4.31 ha in 2021. Callow is also an NPWS-documented cutting site in 2021–2022, so
there the two causes cannot be separated from these data.

**Worth a pixel-level check.**

- Monivea has one low-power detection inside the bog on 3 April 2022, 20 days
  before the scene that reports 4.28 ha.
- Moycullen has six detections two weeks before its June 2023 scene
  (0.70 ha).

Whether any of those candidate pixels are burn is a question for the reference
labels and for the burn layer proposed below.

### Weather and calendar

- The spring 2020 fires followed a markedly dry month: 18–41 mm of rain
  in the 30 days before, against 69–98 mm usually, with 7–12 dry days on
  several bogs.
- The June–July 2018 fires fell in the 2018 drought, with about half the
  usual rain.
- Several fires came with no drought at all: Doogort East on 7 April 2023
  after 243 mm, and Tullaghan Bay on 3 April 2024 after the usual rain, both
  with rain on the day before.
- 11 of the 22 events meet the dry heuristic.
- All but one event fell in the period when burning vegetation is prohibited.

**The cause of a fire cannot be inferred from satellites.** Deliberate
burning, accidental ignition and a spreading fire look the same to FIRMS. The
weather says whether conditions favoured a fire. The calendar says whether
burning was lawful on uncultivated land. Neither says who or what started it.

## Limitations

- **Footprints are large.** A VIIRS detection stands for a pixel of about
  375 m and a MODIS one for about 1 km. Both are far coarser than a cutting
  strip, and a detection within the buffer may be a fire just outside the
  bog. Seven events have no detection inside the boundary.
- **Omission is common.** Small or smouldering fires, fires under cloud and
  fires between satellite overpasses are often missed. **The absence of a
  detection does not mean that a bog did not burn.** For example, 2021 has no
  event at all.
- **Commission errors are possible.** Eight events rest on a single detection.
  FIRMS confidence is recorded with every detection in `detections.csv`.
- **Coverage stops at 2024** until a MAP_KEY is used, so the 2025 and 2026
  release years have no fire information.
- **The weather is modelled** (reanalysis at the bog centroid), not
  measured on the bog, and the dry flag is a heuristic.

## Proposal for detector v4 (not implemented)

1. **Burn layer from events.** Rasterise each event's detections (a VIIRS or
   MODIS footprint around each point, clipped to the bog) and mark:
   - those pixels as *burnt* from the event date to the end of that season;
   - the same pixels as *recovering* in the following season.

   Pixels in either state are not candidate bare peat. This targets the
   Cloonmoylan and Funshin pattern above, which the NBR guard cannot catch.
2. **Burn prototypes in GNG.** Train the GNG as now, then label as *burn* every
   prototype whose pixels overlap the burnt layer much more than chance. Pixels
   that map to those prototypes elsewhere are then excluded too. This is where
   clustering could add something the single-date rules do not: it spreads a
   sparse, coarse external label to spectrally similar pixels.
3. **Evaluate before adoption.** Judge both changes on the frozen reference
   sample, held-out sites first, like every other v4 candidate. The 14 points
   labelled *burn* are the natural check.

## Reproduce

```sh
python3 scripts/26_firms_burns.py                  # 2018-2024, keyless
FIRMS_MAP_KEY=... python3 scripts/26_firms_burns.py --api-start 2025-01-01
```

Sources: NASA FIRMS (<https://firms.modaps.eosdis.nasa.gov>), VIIRS 375 m
active fire product (S-NPP, NOAA-20) and MODIS Collection 6.1 active fires;
Open-Meteo historical weather API (<https://open-meteo.com>), ERA5-based.
