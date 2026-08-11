"""Step 3 — Sentinel-2 access (the main blocker check).

Picks the largest West-of-Ireland bog site (Moycullen Bogs, Galway),
searches for a low-cloud Sentinel-2 scene over it, and reads a small
red + NIR window. Prints scene metadata and array shapes. If this runs,
programmatic satellite access is not a blocker.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peatland import boundaries, imagery, config

DATE_RANGE = "2023-05-01/2023-09-30"  # summer, drier, clearer skies


def target_bbox():
    """WGS84 bbox around the largest bog site."""
    west = boundaries.west_bog_sites(boundaries.load_nha())
    site = west.sort_values("HA", ascending=False).iloc[0]
    print(f"Target site: {site['SITE_NAME']} ({site['HA']:,.0f} ha)")
    return boundaries.site_bbox_wgs84(west, site.name, buffer_m=200)


def main():
    bbox = target_bbox()
    print(f"Search bbox (WGS84): {tuple(round(v, 4) for v in bbox)}")

    last_err = None
    for provider in ("planetary", "earthsearch"):
        print(f"\n--- provider: {provider} ---")
        try:
            item = imagery.search_scene(provider, bbox, DATE_RANGE, max_cloud=20)
            if item is None:
                print("  no scenes matched")
                continue
            print(f"  scene:  {item.id}")
            print(f"  date:   {item.properties.get('datetime')}")
            print(f"  cloud:  {item.properties.get('eo:cloud_cover')}%")

            red, tr, crs = imagery.read_window(item, "red", bbox, provider)
            nir, _, _ = imagery.read_window(item, "nir", bbox, provider)
            print(f"  read window CRS: {crs}")
            print(f"  red shape: {red.shape}  nir shape: {nir.shape}")
            print(f"  red range: {int(red.min())}..{int(red.max())}")
            print(f"\n  SUCCESS via {provider}")
            return
        except Exception as e:
            last_err = e
            print(f"  FAIL: {type(e).__name__}: {e}")

    print(f"\nAll providers failed. Last error: {last_err}")
    sys.exit(1)


if __name__ == "__main__":
    main()
