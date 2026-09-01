# NPWS monitoring method & data-access routes

Research notes answering the supervisor's questions (Sept 2026).

## 1. How NPWS actually detects turf-cutting

Assembled from primary sources (Irish Wildlife Manuals 81 & 62, the 2014
NHA Review, and departmental statements reported by Noteworthy):

1. **Ortho-imagery interpretation.** NPWS holds a bespoke imagery series
   ("NPWS Designated Raised Bog Orthophotos 2010") and compares it with
   the OSi/Tailte national series (1995 / 2000 / 2005 / later 25 cm
   epochs). Analysts identify individual turf plots (cut banks) along
   the high-bog margin and flag those with fresh bare faces or spread
   peat as "active" — the source of the per-plot counts (e.g. Monivea
   51 plots in 2021). The 2014 NHA Review's Table 2 lists per-site
   "active turf-plots in past 7 years" produced exactly this way.
2. **Aerial + satellite monitoring.** The Department states it monitors
   "through aerial monitoring and use of satellite imagery"; the
   European Commission ran its own aerial survey (2012).
3. **Ranger patrols and site visits** verify and enforce.
4. **Periodic condition surveys** (IWM 81 methods, pp. 5–8): pre-survey
   photo-interpretation, then field ecotope mapping with GPS, producing
   Annex I habitat polygons; cutting impact is quantified by comparing
   successive surveys (~45 ha of Degraded Raised Bog lost 2004/05–2011/13).

**Key precedent:** Habib et al. 2024 (Remote Sens. Ecol. Conserv.)
applied a U-Net to the same 25 cm national imagery to map drains on all
Irish raised bogs (~80% accuracy) — evidence that the national imagery
supports automated analysis.

### Implementing an NPWS-comparable method (plan)

Their unit of account is the **plot**, ours is the **hectare**. To
compare like-for-like on 2–3 documented hotspots (Monivea, Callow,
Cloonchambers):

1. digitise turf-plot boundaries along the bog margin once (visible as
   striped banks on any high-res epoch);
2. for each year, mark a plot "active" if our detected bare-peat mask
   intersects it (fresh cutting face);
3. compare our active-plot counts per year against the published NPWS
   counts — a direct, method-level validation, far stronger than area
   correlation.

This is a realistic thesis work-package once reference imagery access
is confirmed.

## 2. Tailte Éireann access — the route exists and ATU is in it

- Ireland's **National Mapping Agreement (NMA)** gives educational
  institutions non-commercial access to most Tailte national mapping.
  **ATU participates**: see the ATU Library maps guide
  (atlantictu.libguides.com/maps/requests) with a Map Data Request Form.
- **Step 1:** email **Library@atu.ie** citing the NMA; request 25 cm
  digital orthophotography tiles (specify epochs + ITM extents over the
  study bogs). The guide lists "Aerial" products; whether ortho
  GeoTIFFs are in the standard bundle needs confirming.
- **Step 2 (if needed):** escalate to **customer.services@tailte.ie**
  with the academic justification (draft:
  `docs/tailte-eireann-email-draft.md`).
- Note: 1995–2018 ortho epochs are freely *viewable* in GeoHive
  (geohive.ie) — usable for point labelling by eye even before any
  licence lands.
- Licence note: outputs must acknowledge Tailte Éireann copyright.

## 3. Coillte & EPA viewers — context layers, not imagery

- **EPA (gis.epa.ie):** OGC WMS/WFS with CORINE land cover (peat class),
  Teagasc/EPA peat-soils maps, Copernicus HRL. Free download policy.
  Useful as context/masks; hosts no aerial imagery.
- **Coillte:** estate boundaries on data.gov.ie (CC-BY, shapefile +
  ArcGIS FeatureServer, ITM). Useful to mask afforested bog margins and
  identify Coillte-owned decommissioned works (our legal/illegal
  separation task). No imagery.

## 4. OpenAerialMap — checked, not useful

API queried over Galway/Mayo/Roscommon: **zero records**; whole island:
18 one-off drone captures, none near a bog. Only relevance: a free CC-BY
publication platform if we ever fly our own drone survey.

## Consolidated contacts

| Purpose | Contact |
|---|---|
| Tailte via ATU (NMA) | Library@atu.ie + ATU Map Data Request Form |
| Tailte direct | customer.services@tailte.ie |
| Free ortho viewing | geohive.ie (1995–2018 epochs) |
| EPA layers | gis.epa.ie/geoserver/wms (GetCapabilities) |
| Coillte boundaries | data.gov.ie/dataset/coilltepublicboundaries |
