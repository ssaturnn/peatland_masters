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


def build_sites(min_overlap=0.05):
    """Canonical site set: the West-of-Ireland NHA bogs, each tagged with
    whether it also lies within an SAC / SPA (stronger legal status).

    The SAC/SPA boundary files carry no habitat attributes, so they can't
    by themselves say which sites are peat — but the NHA bogs are already
    a verified peat set, and overlaying SAC/SPA adds legal-status tags.
    Returns a GeoDataFrame (ITM) with in_sac/in_spa/sac_frac/spa_frac and
    a 'designation' string.
    """
    west = west_bog_sites(load_nha())
    # Some sites are stored as several polygon rows sharing one SITECODE
    # (multi-part bogs). Dissolve them into one feature per site, unioning
    # geometry and summing the per-part hectares.
    west = west.dissolve(
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
    west["designation"] = [
        designation_label(a, b) for a, b in zip(west["in_sac"], west["in_spa"])
    ]
    return west


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
