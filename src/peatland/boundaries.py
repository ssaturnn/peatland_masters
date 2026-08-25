"""Load and filter NPWS protected-site boundaries."""

import geopandas as gpd

from . import config


def load_nha(path=None):
    """Load the NPWS NHA boundary shapefile as a GeoDataFrame in ITM."""
    path = path or config.NPWS_NHA_SHP
    gdf = gpd.read_file(path)
    # Force everything to Irish Transverse Mercator for consistent overlay.
    if gdf.crs is None or gdf.crs.to_epsg() != 2157:
        gdf = gdf.to_crs(config.ITM)
    return gdf


def west_bog_sites(gdf):
    """Filter to bog-named sites in Galway / Mayo / Roscommon."""
    counties = set(config.TARGET_COUNTIES.keys())
    mask = gdf["COUNTY"].isin(counties) & gdf["SITE_NAME"].str.contains(
        "bog", case=False, na=False
    )
    return gdf.loc[mask].copy()


def _load_layer(path):
    gdf = gpd.read_file(path)
    if gdf.crs is None or gdf.crs.to_epsg() != 2157:
        gdf = gdf.to_crs(config.ITM)
    return gdf


def load_sac(path=None):
    """Load the NPWS SAC (Special Area of Conservation) boundaries in ITM."""
    return _load_layer(path or config.NPWS_SAC_SHP)


def load_spa(path=None):
    """Load the NPWS SPA (Special Protection Area) boundaries in ITM."""
    return _load_layer(path or config.NPWS_SPA_SHP)


def _overlap_fraction(geom, other_union):
    try:
        return geom.intersection(other_union).area / geom.area if geom.area else 0.0
    except Exception:
        return 0.0


def _union(gdf):
    return gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") \
        else gdf.geometry.unary_union


def designation_label(in_sac, in_spa):
    parts = ["NHA"]
    if in_sac:
        parts.append("SAC")
    if in_spa:
        parts.append("SPA")
    return " + ".join(parts)


# Turf plots cut in 2022 without State consent, from the Dept. of Housing /
# NPWS records reported by thejournal.ie / Irish Times — used for external
# cross-checking of the detector (keyed by SAC site code).
DOCUMENTED_PLOTS_2022 = {
    "002352": 49,  # Monivea Bog SAC (Galway) — most-cut SAC in 2022
    "000231": 42,  # Barroughter Bog SAC (Galway)
    "000595": 31,  # Callow Bog SAC (Roscommon)
    # Corliskea/Trien/Cloonfelliv (002110) is a reported hotspot too, but the
    # public source gives no exact 2022 plot count, so it is left unlabelled
    # rather than assumed zero.
}

# 2021 plot counts (NPWS records via FOI, reported by Noteworthy/thejournal.ie
# Jan 2022; 282 plots across SACs in 2021, ~320/yr average since 2012).
DOCUMENTED_PLOTS_2021 = {
    "002352": 51,  # Monivea Bog SAC — the most-cut SAC (70 banks back in 2013)
    "000595": 22,  # Callow Bog SAC
    "000600": 22,  # Cloonchambers Bog SAC (Roscommon)
    "002349": 11,  # Corbo Bog SAC (Roscommon; also 11 plots in 2012)
}


def _primary_region_county(county_str):
    for c in str(county_str).split(","):
        c = c.strip()
        if c in config.TARGET_COUNTIES:
            return c
    return None


def _raised_bog_sacs(max_ha=1500):
    """Region raised-bog SACs: SAC sites named '...bog' in Galway/Mayo/
    Roscommon, small enough to be raised bogs (excludes the huge blanket-bog
    complexes). Dissolved to one feature per site, county normalised.
    """
    sac = load_sac()
    sac = sac[sac["SITE_NAME"].str.contains("bog", case=False, na=False)].copy()
    sac["cty"] = sac["COUNTY"].apply(_primary_region_county)
    sac = sac[sac["cty"].notna() & (sac["HA"] < max_ha)]
    sac = sac.dissolve(
        by="SITECODE",
        aggfunc={"SITE_NAME": "first", "cty": "first", "HA": "sum",
                 "URL": "first"},
        as_index=False,
    ).reset_index(drop=True)
    sac = sac.rename(columns={"cty": "COUNTY"})
    return sac


def build_sites(min_overlap=0.05, include_sac=True, sac_max_ha=1500):
    """Canonical site set for the analysis.

    * NHA bogs in Galway/Mayo/Roscommon (verified peat), each tagged with
      whether it also lies within an SAC/SPA (stronger legal status).
    * optionally, the region's raised-bog SACs — these carry the EU's
      strongest protection and include the sites where illegal cutting is
      actually documented, so the detector can be cross-checked against the
      public plot-count records (`plots_2022`).

    Returns a GeoDataFrame (ITM) with in_sac/in_spa, a 'designation' string,
    'source' (NHA/SAC) and 'plots_2022'.
    """
    west = west_bog_sites(load_nha()).dissolve(
        by="SITECODE",
        aggfunc={"SITE_NAME": "first", "COUNTY": "first", "HA": "sum",
                 "URL": "first"},
        as_index=False,
    ).reset_index(drop=True)
    sac_u = _union(load_sac())
    spa_u = _union(load_spa())

    west["sac_frac"] = west.geometry.apply(lambda g: _overlap_fraction(g, sac_u))
    west["spa_frac"] = west.geometry.apply(lambda g: _overlap_fraction(g, spa_u))
    west["in_sac"] = west["sac_frac"] >= min_overlap
    west["in_spa"] = west["spa_frac"] >= min_overlap
    west["designation"] = [designation_label(a, b)
                           for a, b in zip(west["in_sac"], west["in_spa"])]
    west["source"] = "NHA"
    west["plots_2022"] = None
    west["plots_2021"] = None

    if not include_sac:
        return west.reset_index(drop=True)

    sac = _raised_bog_sacs(sac_max_ha)
    # avoid duplicating a bog we already have as NHA (same footprint)
    nha_u = _union(west)
    sac["nha_frac"] = sac.geometry.apply(lambda g: _overlap_fraction(g, nha_u))
    sac = sac[sac["nha_frac"] < 0.5].copy()
    sac["spa_frac"] = sac.geometry.apply(lambda g: _overlap_fraction(g, spa_u))
    sac["sac_frac"] = 1.0
    sac["in_sac"] = True
    sac["in_spa"] = sac["spa_frac"] >= min_overlap
    sac["designation"] = ["SAC + SPA" if s else "SAC" for s in sac["in_spa"]]
    sac["source"] = "SAC"
    sac["plots_2022"] = sac["SITECODE"].map(DOCUMENTED_PLOTS_2022)
    sac["plots_2021"] = sac["SITECODE"].map(DOCUMENTED_PLOTS_2021)

    cols = ["SITECODE", "SITE_NAME", "COUNTY", "HA", "geometry", "sac_frac",
            "spa_frac", "in_sac", "in_spa", "designation", "source", "plots_2022", "plots_2021"]
    import pandas as pd
    combined = gpd.GeoDataFrame(
        pd.concat([west[cols], sac[cols]], ignore_index=True), crs=west.crs)
    return combined.reset_index(drop=True)


def site_bbox_wgs84(gdf, row_label, buffer_m=300):
    """WGS84 bbox (minx, miny, maxx, maxy) around one site, buffered in metres."""
    geom = gdf.loc[[row_label]].buffer(buffer_m)
    b = gpd.GeoDataFrame(geometry=geom, crs=gdf.crs).to_crs(config.WGS84).total_bounds
    return tuple(b)


def summarise(gdf):
    """Return a small per-county summary DataFrame."""
    g = gdf.copy()
    g["county"] = g["COUNTY"].map(config.TARGET_COUNTIES)
    out = (
        g.groupby("county")
        .agg(sites=("SITE_NAME", "count"), total_ha=("HA", "sum"))
        .sort_values("total_ha", ascending=False)
    )
    return out
