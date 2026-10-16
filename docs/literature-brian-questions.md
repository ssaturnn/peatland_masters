# Prior work for the supervisor's questions (October 2026)

Every reference below was checked against Crossref, OpenAlex, Semantic
Scholar or the publisher's page. **[unverified]** marks a claim taken from
secondary text. **(inference)** marks our reasoning about the v5 rule, not a
published finding.

## 1. Can MODIS/VIIRS (FIRMS) detect smouldering or underground peat fires?

Only partly. They detect surface smouldering when it is large and hot enough,
but not truly subsurface burning, and they miss small fires, fires under cloud
and fires not burning at overpass time. A FIRMS-flagged year is reliable; an
unflagged year is not evidence of no fire.

- NASA FIRMS FAQ: MODIS "routinely detects both flaming and smoldering fires
  1000 m² in size". Cloud, heavy smoke or canopy can obscure a fire. Overpasses
  about 10:30/22:30 (Terra) and 13:30/01:30 (Aqua, S-NPP VIIRS).
  https://www.earthdata.nasa.gov/data/tools/firms/faq
- Giglio, Schroeder & Justice (2016). The Collection 6 MODIS active fire
  detection algorithm and fire products. *Remote Sensing of Environment* 178.
  doi:10.1016/j.rse.2016.02.054. Detection probability is driven by fire size.
- Schroeder et al. (2014). The New VIIRS 375 m active fire detection data
  product. *RSE* 143:85–96. doi:10.1016/j.rse.2013.12.008. Smaller fires of the
  same temperature are detected; cloud-flagged pixels are skipped.
- Burke et al. (2019). *Drones* 3:23. doi:10.3390/drones3010023. Smouldering
  500–700 °C against 1500–1800 °C for flaming; peat-fire surface temperatures
  50–400 °C; smouldering can lie over 1 m below unburnt material.
- Sofan, Bruce, Jones & Marsden (2019). *Remote Sensing* 11:465.
  doi:10.3390/rs11040465. VIIRS 375 m detected 71% of ground-truthed peat fires
  in Central Kalimantan; night-time VIIRS agreed better with Landsat-8.
- Elvidge et al. (2015). *Environmental Research Letters* 10:065002.
  doi:10.1088/1748-9326/10/6/065002. Smouldering needs night-time long-wave IR.
- Scholten et al. (2021). Overwintering fires in boreal forests. *Nature*
  593:399–404. doi:10.1038/s41586-021-03437-y. Invisible while smouldering;
  found only when they re-emerged in spring.
- Cawkwell et al., EGU21-8083 (abstract): FIRMS misses small, short-lived fires
  under Irish cloud. https://meetingorganizer.copernicus.org/EGU21/EGU21-8083.html

## 2. Can remote sensing show an underground fire is present or absent?

No single signal proves it. Evidence combines high-resolution thermal or SWIR
hot spots (ideally at night), persistent or re-emerging hot spots, post-fire
subsidence or coherence change in InSAR, and diffuse vegetation die-off.
Ground or drone thermal survey is the reliable confirmation. No Irish or
British satellite study of subsurface smouldering was found.

- Hu, Ban & Nascetti (2021). *Int. J. Appl. Earth Obs. Geoinf.* 101:102347.
  doi:10.1016/j.jag.2021.102347. Sentinel-2 B11/B12 active fire; cool, small
  fires are the main omissions. (inference) A hot pixel raises B12 more than
  B11, lowering SWIR1/SWIR2, so active burning fails the v5 ratio test.
- Khakim et al. (2020). *JAG* 84:101953. doi:10.1016/j.jag.2019.101953.
  Sentinel-1 SBAS: post-fire peat subsidence 6.4× the pre-fire rate.
- Hrysiewicz, Holohan, Donohue & Cushnan (2023). *RSE* 291:113516.
  doi:10.1016/j.rse.2023.113516. Ballynafagh bog (Kildare) 2019 fire: VV
  backscatter rose; coherence, displacement and soil moisture were not
  noticeably affected.
- FLARES project (UCC/EPA): Landsat-8/Sentinel-2 burned area for Ireland
  2015–2021 with a peatland case study.
  https://epa.ie/our-services/research/epa-funded-research/epa-funded-projects/research-data-table-dev/fire-land-and-atmospheric-remote-sensing-of-emissions-projections-policy-and-land-use-and-cover-synthesis.php

## 3. Has anyone done similar work?

Yes for industrial extraction, land-use classes and bare-peat cover. A
plot-scale detector of domestic turf cutting is largely missing.

- Connolly (2019). *Irish Geography* 51(2). doi:10.55650/igj.2018.1371.
  Irish peatland land use 2005–06 (77%); western seaboard excluded for cloud.
- Habib, Ingle, Saunders & Connolly (2024). *Scientific Reports* 14:1171.
  doi:10.1038/s41598-024-51660-0. Sentinel-2 random forest, 7 classes incl.
  cutover and cutaway, 89% overall accuracy.
- Connolly & Holden (2011). *Irish Geography*. doi:10.1080/00750778.2011.615558.
- O'Connell, Connolly & Holden (2011). *Proc. SPIE* 8174.
  doi:10.1117/12.898562. Raised-bog change attributed mainly to turf cutting
  at bog margins, plus a 36 ha burn.
- O'Connell et al. (2014). *JAG*. doi:10.1016/j.jag.2014.03.006.
- Bhatnagar et al. (2018). *PIAHS* 380:9. doi:10.5194/piahs-380-9-2018.
- Tampuu et al. (2021). *JAG* 98:102309. doi:10.1016/j.jag.2021.102309.
- Torabi Haghighi et al. (2018). *Land Degradation & Development*.
  doi:10.1002/ldr.3122.
- Trippier et al. (2020, 2021). JNCC Reports 667 and 674 (Sentinel-2 % bare
  peat; overprediction outside summer).
  https://data.jncc.gov.uk/data/958df51f-2e7c-4d2b-92f0-eac84c2a86af/JNCC-Report-667-FINAL-WEB.pdf
  https://data.jncc.gov.uk/data/66e8cb57-0113-4eea-82d0-1c26b20513eb/JNCC-Report-674-FINAL-WEB.pdf

## 4. What does Sentinel-1 give for peatlands?

- Tampuu et al. (2021): active milled fields had 6-day VV coherence below 0.2
  against about 0.6 for abandoned fields and open bog; heavy rain on one date
  of a pair can mimic surface works.
- Tampuu et al. (2020). *Remote Sensing* 12:1566. doi:10.3390/rs12101566.
- Alshammari et al. (2018). *Remote Sensing* 10:1103. doi:10.3390/rs10071103.
- Alshammari et al. (2020). *JGR Biogeosciences*. doi:10.1029/2018JG004953.
- Bradley et al. (2022). *Earth Surface Dynamics* 10:261.
  doi:10.5194/esurf-10-261-2022.
- Hrysiewicz et al. (2024). *RSE* 311:114232. doi:10.1016/j.rse.2024.114232.
- Asmuß, Bechtold & Tiemeyer (2019). *Remote Sensing* 11:1659.
  doi:10.3390/rs11141659. Backscatter vs water table, mean Spearman 0.45.
- Bechtold et al. (2018). *Remote Sensing* 10:536. doi:10.3390/rs10040536.

## 5. Bare peat against dry or senescent grass

- Nagler, Daughtry & Goward (2000). *RSE* 71:207–215.
  doi:10.1016/S0034-4257(99)00082-6. Dry litter absorbs near 2.1 µm, soils do
  not (Cellulose Absorption Index).
- Nagler et al. (2003). *RSE*. doi:10.1016/j.rse.2003.06.001.
- Quemada & Daughtry (2016). *Remote Sensing* 8:660. doi:10.3390/rs8080660.
  NDTI = (SWIR1 − SWIR2)/(SWIR1 + SWIR2) works only when dry. (inference)
  SWIR1/SWIR2 ≥ 1.25 equals NDTI ≥ 0.11, which dry litter also passes; NBR < 0
  is what excludes it (confirmed in `docs/fire-and-grass-checks.md`: 0.04%).
- Heremans, Wepener & Somers (2024). IGARSS.
  doi:10.1109/IGARSS53475.2024.10642634. Molinia cover from monthly
  Sentinel-2, poor transfer between years.
- Burdun et al. (2020). *Remote Sensing* 12:2936. doi:10.3390/rs12182936.
  SWIR tracks peatland water table.

## Gaps to state in the thesis

- No published satellite validation of underground smouldering detection in
  Ireland or Britain.
- No published plot-scale detector of domestic turf cutting.
- No peat-specific ECOSTRESS or Sentinel-3 SLSTR validation found.
