# Timing of working and the fire signal (supervisor's questions, October 2026)

Scripts: `scripts/38_time_series.py` (every clear Sentinel-2 scene of one bog),
`scripts/39_cutting_timing.py` (vegetation-to-peat transitions),
`scripts/40_fire_signal.py` (burn mapped by dNBR, signal through time),
`scripts/41_landsat_history.py` (Landsat 1984–2025, same SWIR rule).

## Castle Ffrench East Bog NHA (largest recurring working in the release)

- 276 clear Sentinel-2 scenes, 2017–2026. About 47 ha of fields are exposed
  every spring and summer and drop out of v5 every autumn and winter, when
  the peat is wet (NDVI on the fields stays at the bare-peat level, 0.4–0.55).
- Share of April–September scenes with at least half of the fields exposed:
  2017 50%, 2018 75%, 2019 67%, 2020 83%, 2021 64%, 2022 78%, 2023 67%,
  2024 33%, 2025 8%, 2026 0%. Working continued every season up to 2023,
  declined in 2024 and stopped by 2025.
- PlanetScope 3 m: even milling strips on 10 August 2022; ponded water and a
  patchy surface on 19 June 2025.
- Landsat 1987–2025: the fields are already present on the earliest clear
  scene (1988). v5 does not flag them in 1987–1989 because the peat was wet on
  those dates, so one scene a year shows dry exposed peat, not when a field
  was opened. The start of working predates the SWIR archive.
- New vegetation-to-peat transitions after 2017 cover 7 ha and are dated to a
  median bracket of 15 days between clear scenes (3 days at best).

## Fires (FIRMS dates; burn mapped by dNBR ≥ 0.15, independent of v5)

- Doogort East Bog NHA, 22 April 2019, 66 ha: 12–15 days later NBR −0.16 to
  −0.07, SWIR1/SWIR2 1.29–1.33, and v5 flags 39–52% of the burn as exposed
  peat; 42–60 days later 2–16%; from 112 days the burn is indistinguishable
  from unburnt bog.
- Cloonmoylan Bog SAC, 31 May 2020, 279 ha: two days later NBR −0.24 and
  SWIR1/SWIR2 1.11, so v5 rejects the fresh burn; regrowth within 2–4 months.
  In April–July 2021 v5 flags 7–19% of the bog equally in burnt and unburnt
  parts: a dry spell, not the fire. The 39 ha of Cloonmoylan 2021 in the
  summer survey are therefore drought, not post-fire exposure.

## What this means

Fire and working differ in duration: peat bared by fire stays exposed for one
to two months, worked fields every season for years. A dense time series with
the FIRMS date separates them. v5 is sensitive to surface moisture in both
directions: wet peat drops out, and in a drought the drying surface of a
degraded bog can be flagged.
