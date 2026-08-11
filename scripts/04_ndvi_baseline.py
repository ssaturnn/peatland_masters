"""Step 4/5 — NDVI baseline end to end, clipped to a protected site.

Reads red + NIR over a bog site, computes NDVI, flags low-NDVI (bare /
cut peat) pixels, clips them to the actual NPWS polygon, and reports the
exposed area in hectares. This closes the pipeline: imagery in, an
area-of-concern number out.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd

from peatland import boundaries, imagery, detect, geo, config

DATE_RANGE = "2023-05-01/2023-09-30"
NDVI_THRESHOLD = 0.25
PROVIDER = "planetary"


def main():
    west = boundaries.west_bog_sites(boundaries.load_nha())

    # Use a mid-sized site so the whole thing fits a modest window.
    site_row = west.sort_values("HA").iloc[len(west) // 2]
    label = site_row.name
    print(f"Site: {site_row['SITE_NAME']} "
          f"({config.TARGET_COUNTIES[site_row['COUNTY']]}, "
          f"{site_row['HA']:,.0f} ha)")

    bbox = boundaries.site_bbox_wgs84(west, label, buffer_m=200)
    print(f"Search bbox (WGS84): {tuple(round(v, 4) for v in bbox)}")

    item = imagery.search_scene(PROVIDER, bbox, DATE_RANGE, max_cloud=15)
    if item is None:
        print("No scene found."); sys.exit(1)
    print(f"Scene: {item.id}  cloud={item.properties.get('eo:cloud_cover'):.2f}%")

    red, transform, crs = imagery.read_window(item, "red", bbox, PROVIDER)
    nir, _, _ = imagery.read_window(item, "nir", bbox, PROVIDER)
    print(f"Window: {red.shape} px in {crs}")

    ndvi_arr = detect.ndvi(red, nir)
    print(f"NDVI range: {ndvi_arr.min():.2f} .. {ndvi_arr.max():.2f} "
          f"(mean {ndvi_arr.mean():.2f})")

    bare = detect.exposed_peat_mask(ndvi_arr, NDVI_THRESHOLD)

    # Reproject the site polygon into the raster CRS and clip.
    site_geom = west.loc[[label]].to_crs(crs).geometry.iloc[0]
    inside = geo.polygon_mask(site_geom, red.shape, transform)

    bare_in_site = bare & inside

    site_area = geo.mask_area_ha(inside, transform)
    bare_area = geo.mask_area_ha(bare_in_site, transform)
    pct = 100 * bare_area / site_area if site_area else 0

    print(f"\nPixel area: {geo.pixel_area_ha(transform):.4f} ha")
    print(f"Site area in window:   {site_area:8.1f} ha")
    print(f"Low-NDVI (bare) area:  {bare_area:8.1f} ha  ({pct:.1f}% of site)")
    print("\nPipeline closed: imagery -> NDVI -> mask -> clip -> area.")


if __name__ == "__main__":
    main()
