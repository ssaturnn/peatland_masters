# Sentinel-1 radar check (supervisor's question, October 2026)

Script: `scripts/43_sentinel1_check.py`. Outputs in `outputs/time-series/`:
`001244_s1_fields.json/.png`, `002381_s1_fire.json/.png`,
`000248_s1_fire.json/.png` (backscatter stacks cached as `*_s1_*.npz`).

## Method

- Sentinel-1 C-band SAR, Planetary Computer collection `sentinel-1-rtc`
  (radiometrically terrain-corrected gamma0, VV and VH, 10 m, UTM 29N),
  read onto each bog's 10 m release grid and converted to dB.
- One relative orbit per bog so the incidence angle is fixed: orbit 96
  (descending) for Castle Ffrench East and Doogort East, orbit 23 (descending)
  for Cloonmoylan, because the edge of the orbit-96 swath crosses Cloonmoylan
  (70–77% coverage). Every slice of a date is mosaicked; a date is used only
  if it covers at least 80% of the bog.
- Thinning: at most one date per 6 days (fires) or 12 days (Castle Ffrench
  East, 2017–2026). 283 dates at Castle Ffrench East, 119 at Doogort, 109 at
  Cloonmoylan.
- Worked fields: the same mask as `001244_fields.json`, pixels flagged v5
  exposed peat on at least 30% of clear May–August Sentinel-2 scenes (47.3 ha).
  Intact bog: inside the boundary, flagged on under 2% of those scenes and at
  least 30 m from the fields (136.9 ha).
- Rain: ERA5 daily precipitation at the bog (Open-Meteo archive), sum of the
  3 days before the acquisition, as in `scripts/42_rain_check.py`.
- Burn: the dNBR mask of `scripts/40_fire_signal.py` (pre-fire median of 270
  days minus post-fire median of 45 days ≥ 0.15), unburnt control with
  dNBR < 0.05.

## Castle Ffrench East: worked fields

- **Radar on the fields is driven by moisture, in the opposite direction from
  SWIR.** April–September, field VV against rain in the 3 days before: Spearman
  ρ +0.53 (p 6e-12, n 149); intact bog ρ +0.30. Wet bare peat backscatters
  strongly and dry milled peat weakly. Against the v5 field share on
  Sentinel-2 scenes within 2 days: field VV ρ −0.59, fields − intact VV
  ρ −0.59, VH ρ −0.57 (n 74, p ≤ 1.1e-7). Radar and SWIR record the same
  wetting and drying of the fields.
- **Separating fields from intact bog: VH, not VV.** Per-pixel
  April–September medians, AUC of fields against intact bog (0.5 = no
  separation): VV 0.28–0.63, so no stable separation; VH 0.12–0.37, so
  fields have lower VH (less volume scattering from vegetation), with the
  strongest separation in 2019–2022 (AUC 0.12–0.15).
- **The radar shows the 2024–2026 decline independently of optical imagery
  and cloud.** Fields − intact, April–September medians:
  - VH: −1.88 dB in 2017–2023 (−2.11 dB in 2019–2022) against −0.84 dB in
    2024–2026, Mann-Whitney p 1.3e-11.
  - VV: −0.35 dB against +0.67 dB, p 2e-5.
  - VV on dry days (< 2 mm in 3 days): −1.2 to −2.5 dB every year 2018–2023,
    −0.05 to +0.09 dB in 2024–2026.
  - VH AUC rises from 0.12–0.15 (2019–2022) to 0.32–0.37 (2024–2026).

  The fields stop behaving like bare milled peat: the dry-day contrast
  disappears and VH (vegetation volume) approaches that of the intact bog.
- **Cloud gaps.** April–September dates with Sentinel-1 cover over any orbit:
  108–120 a season with two satellites (2017–2021, 2025–2026), 52–59 with
  one (2022–2024). Clear Sentinel-2 scenes on the fields: 6–14 a season. The
  radar gives about ten times more dates, but each one measures roughness and
  moisture, not exposed peat itself.

## Fires

| | Doogort East, 22 Apr 2019 (66 ha) | Cloonmoylan, 31 May 2020 (279 ha) |
|---|---|---|
| Orbit | 96 desc | 23 desc |
| VV burnt − unburnt before the fire | −0.48 ± 0.20 dB | −0.79 ± 0.31 dB |
| 0–30 days after | +1.17 | +0.16 |
| 1–4 months | +1.40 to +1.59 | +0.44 to +0.67 |
| 4–12 months | +2.09 to +2.22 | +0.78 to +1.12 |
| 12–15 months | +0.99 | +0.49 |
| VH burnt − unburnt, before → 4–12 months | +0.06 → +1.5 to +1.7 | −1.06 → −0.6 to −0.9 |

- **VV over the burn rises 0.8–2.0 dB above its pre-fire contrast within days, and up to 2.7 dB over the following year**
  (Doogort: +2.0 dB 3 days after; Cloonmoylan: +0.8 dB the day after) and
  **stays raised for more than a year**. At both sites the difference is still
  more than 2 standard deviations from the pre-fire level 15 months later.
  Burning removes the vegetation canopy and litter, so the surface (rough, wet
  peat) scatters more directly.
- For comparison, the optical signal lasts much less: v5 exposed peat 1–2
  months, NBR recovery 3–4 months (`docs/timing-and-fire.md`).
- VH responds less consistently (strongly at Doogort, weakly at
  Cloonmoylan).

## Conclusion

1. **Detecting working:** Sentinel-1 does not map exposed peat by itself (VV
   gives no stable separation; VH separates fields from intact bog at
   AUC 0.12–0.37, i.e. 0.63–0.88 the other way round). It does confirm the
   decline of working at Castle Ffrench East after 2023 independently of
   optical data and weather: both the VH and the dry-day VV contrast
   collapse in 2024–2026.
2. **Moisture:** radar does not remove the moisture problem. It is itself
   dominated by moisture (field VV against rain ρ +0.53), with the sign
   opposite to SWIR. Conditioning on dry days still shows the decline.
3. **Fire:** clear and long-lived. A burn raises VV by 0.8–2.7 dB for about a
   year, far longer than the optical signal. This makes radar useful to flag
   past fires that the optical record or FIRMS missed. It does not show
   whether a fire smouldered below the surface.
4. **Cloud gaps:** about ten times more dates than clear Sentinel-2 scenes,
   useful for timing changes; it measures surface state, not peat.

Limits: one bog for working and two fires; one orbit each; the field mask
comes from Sentinel-2 v5, so the radar test is of agreement and separability,
not an independent map; ERA5 rain is a 0.25° reanalysis.
