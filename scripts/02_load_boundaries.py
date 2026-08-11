"""Step 2 — load NPWS boundaries, confirm CRS handling and bbox.

Blocker check: does the protected-site vector layer load, reproject to
ITM cleanly, and give a bounding box consistent with our region config?
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peatland import boundaries, config


def main():
    gdf = boundaries.load_nha()
    print(f"Loaded {len(gdf)} NHA polygons")
    print(f"CRS: {gdf.crs}")

    west = boundaries.west_bog_sites(gdf)
    print(f"\nWest-of-Ireland bog sites: {len(west)}")
    print(boundaries.summarise(west).to_string())

    total = west["HA"].sum()
    print(f"\nTotal area: {total:,.0f} ha")

    bounds = west.total_bounds  # (minx, miny, maxx, maxy) in ITM
    print(f"\nBBox (ITM): "
          f"({bounds[0]:,.0f}, {bounds[1]:,.0f}) -> "
          f"({bounds[2]:,.0f}, {bounds[3]:,.0f})")
    print(f"Config bbox: {config.REGION_BBOX_ITM}")

    # Reproject a copy to WGS84 to confirm pyproj transforms work.
    west_wgs = west.to_crs(config.WGS84)
    wb = west_wgs.total_bounds
    print(f"BBox (WGS84): "
          f"({wb[0]:.3f}, {wb[1]:.3f}) -> ({wb[2]:.3f}, {wb[3]:.3f})")


if __name__ == "__main__":
    main()
