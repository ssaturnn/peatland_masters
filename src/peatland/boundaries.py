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
