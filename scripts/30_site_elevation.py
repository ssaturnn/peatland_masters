"""Step 30 — median and 90th-percentile elevation of every site.

Reads the Copernicus GLO-30 DEM from the Planetary Computer inside each site
polygon and writes src/peatland/site_elevation.csv, which boundaries.py uses
to tag upland bogs as blanket bog (see UPLAND_M there).

Usage:
    python3 scripts/30_site_elevation.py
"""

import csv
import sys
from pathlib import Path

import numpy as np
import planetary_computer as pc
import rasterio
from pystac_client import Client
from rasterio.mask import mask as rmask

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from peatland import boundaries, config

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def site_elevations(geom, catalog):
    """Elevation values (m) of the DEM pixels inside one WGS84 polygon."""
    values = []
    for item in catalog.search(collections=["cop-dem-glo-30"],
                               intersects=geom.envelope.__geo_interface__).items():
        with rasterio.open(item.assets["data"].href) as src:
            try:
                a, _ = rmask(src, [geom.__geo_interface__], crop=True, filled=False)
            except ValueError:  # polygon outside this tile
                continue
            values.append(a.compressed())
    return np.concatenate(values) if values else np.array([])


def main():
    sites = boundaries.build_sites().to_crs(config.WGS84)
    catalog = Client.open(STAC, modifier=pc.sign_inplace)
    rows = []
    for _, r in sites.iterrows():
        v = site_elevations(r.geometry, catalog)
        rows.append({"SITECODE": r.SITECODE, "SITE_NAME": r.SITE_NAME,
                     "elev_median_m": round(float(np.median(v)), 1) if v.size else "",
                     "elev_p90_m": round(float(np.percentile(v, 90)), 1) if v.size else ""})
        print(f"{r.SITE_NAME[:40]:40} {rows[-1]['elev_median_m']} m")
    with boundaries.ELEVATION_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["SITECODE"]))
    print(f"{len(rows)} sites -> {boundaries.ELEVATION_CSV}")


if __name__ == "__main__":
    main()
