"""Step 15 — cloud calendar: the clearest Sentinel-2 days over a bog.

Sentinel-2 revisits Ireland every ~5 days; most acquisitions are cloudy.
This tool scans EVERY acquisition over a site in a date range and reports,
day by day, how clear the sky actually was over the bog itself (SCL-based
clear fraction, not the whole-tile metadata), marking the days the
pipeline would use.

Usage:
    python3 scripts/15_cloud_calendar.py "Monivea" 2024
    python3 scripts/15_cloud_calendar.py "Namucka" 2022 --range 2022-01-01/2022-12-31

Writes outputs/cloud_calendar_<code>_<label>.csv and prints a table.
"""

import sys
import csv
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import warnings
warnings.filterwarnings("ignore")

import geopandas as gpd

from peatland import boundaries, imagery, geo, config, preprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site", help="site name fragment, e.g. 'Monivea'")
    ap.add_argument("year", nargs="?", type=int, default=None)
    ap.add_argument("--range", dest="rng", default=None,
                    help="explicit date range start/end")
    args = ap.parse_args()
    rng = args.rng or (f"{args.year}-01-01/{args.year}-12-31"
                       if args.year else None)
    if rng is None:
        ap.error("give a year or --range")

    sites = boundaries.build_sites()
    row = sites[sites.SITE_NAME.str.contains(args.site, case=False)].iloc[0]
    print(f"Site: {row['SITE_NAME']} ({row['SITECODE']})  range {rng}\n")

    bbox = boundaries.site_bbox_wgs84(sites, row.name, buffer_m=200)
    ref = imagery.search_scene("planetary", bbox, "2021-05-01/2021-09-15",
                               max_cloud=60)
    red0, transform, crs = imagery.read_window(ref, "red", bbox, "planetary")
    shape = red0.shape
    geom = gpd.GeoSeries([row.geometry], crs=config.ITM).to_crs(crs).iloc[0]
    inside = geo.polygon_mask(geom, shape, transform)

    client = imagery.open_catalog("planetary")
    search = client.search(
        collections=[imagery.PROVIDERS["planetary"]["collection"]],
        bbox=bbox, datetime=rng, max_items=200)
    items = sorted(search.items(),
                   key=lambda it: it.properties.get("datetime", ""))
    print(f"{len(items)} acquisitions found; assessing AOI clear fraction...")
    print(f"{'date':12}{'tile cloud %':>13}{'bog clear %':>13}  verdict")

    rows = []
    for it in items:
        date = it.properties.get("datetime", "")[:10]
        tile_cloud = float(it.properties.get("eo:cloud_cover", -1))
        try:
            rep = preprocess.assess_scene(it, bbox, "planetary", shape, inside)
            clear = rep["aoi_clear"]
        except Exception:
            clear = float("nan")
        verdict = ("CLEAR — usable" if clear >= 0.85 else
                   "partly usable" if clear >= 0.5 else "cloudy")
        star = " *" if clear >= 0.98 else ""
        print(f"{date:12}{tile_cloud:>12.1f} {100*clear:>12.1f}  {verdict}{star}")
        rows.append({"date": date, "scene": it.id,
                     "tile_cloud_pct": round(tile_cloud, 1),
                     "bog_clear_pct": round(100 * clear, 1),
                     "verdict": verdict})

    label = rng.replace("/", "_").replace("-", "")[:17]
    out = Path(f"outputs/cloud_calendar_{row['SITECODE']}_{label}.csv")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    best = sorted((r for r in rows if r["bog_clear_pct"] == r["bog_clear_pct"]),
                  key=lambda r: -r["bog_clear_pct"])[:5]
    print(f"\nBest days: " + ", ".join(f"{r['date']} ({r['bog_clear_pct']}%)"
                                       for r in best))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
