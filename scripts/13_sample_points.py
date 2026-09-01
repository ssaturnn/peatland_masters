"""Step 14 — stratified random points for the formal accuracy assessment.

Implements the sampling half of a Congalton & Green style accuracy
assessment: for each selected site-year, draw random points stratified
by detector output — 'detected' (GNG bare) and 'undetected' (inside the
bog, not flagged) — and export them as CSV + GeoJSON with WGS84
coordinates and the exact scene date.

A human then labels each point by eye against high-resolution imagery
(Tailte Éireann 25 cm ortho / PlanetScope 3 m / Google Earth), filling
the `truth` column with bare_peat / vegetated / water / burn / unsure.
scripts/14_score_points.py (next step) turns the labelled file into a
confusion matrix with precision / recall / F1.
"""

import sys
import csv
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import geopandas as gpd
import rasterio.transform

from peatland import boundaries, imagery, geo, config, preprocess, pipeline, detect

SITES_YEARS = [  # (name fragment, year): active + quiet mix
    ("Monivea", 2022), ("Namucka", 2026), ("Callow", 2026),
    ("Kilnaborris", 2024), ("Derrinlough", 2024), ("Rosroe", 2018),
]
PER_STRATUM = 25          # points per stratum per site-year
SEED = 42
OUT_CSV = Path("outputs/accuracy_points.csv")
OUT_GJ = Path("outputs/accuracy_points.geojson")


def cache_date(code, year):
    r = json.loads((config.OUT_DIR / "cache" / f"{code}.json").read_text())
    for y, d in zip(r["years"], r["dates"]):
        if y == year:
            return d
    return None


def main():
    sites = boundaries.build_sites()
    rng = np.random.default_rng(SEED)
    rows, feats_gj = [], []

    for name, year in SITES_YEARS:
        row = sites[sites.SITE_NAME.str.contains(name)].iloc[0]
        day = cache_date(row["SITECODE"], year)
        if day is None:
            print(f"skip {name} {year}: no scene in cache"); continue
        bbox = boundaries.site_bbox_wgs84(sites, row.name, buffer_m=200)
        ref = imagery.search_scene("planetary", bbox, "2021-05-01/2021-09-15",
                                   max_cloud=40)
        red0, transform, crs = imagery.read_window(ref, "red", bbox, "planetary")
        shape = red0.shape
        geom = gpd.GeoSeries([row.geometry], crs=config.ITM).to_crs(crs).iloc[0]
        inside = geo.polygon_mask(geom, shape, transform)
        item, rep = preprocess.pick_clear_scene(
            "planetary", bbox, f"{day}/{day}", shape, aoi_mask=inside, min_clear=0.5)
        order = list(pipeline.BANDS)
        stack = np.stack([imagery.read_window(item, b, bbox, "planetary",
                                              out_shape=shape)[0].astype("float32")
                          for b in order], -1)
        valid = preprocess.valid_mask(rep["scl"])
        gng = pipeline.gng_bare(stack, order, valid, inside)
        undet = inside & valid & ~gng

        from pyproj import Transformer
        to_wgs = Transformer.from_crs(crs, config.WGS84, always_xy=True)
        for stratum, mask in (("detected", gng), ("undetected", undet)):
            ys, xs = np.nonzero(mask)
            if len(ys) == 0:
                continue
            take = rng.choice(len(ys), size=min(PER_STRATUM, len(ys)),
                              replace=False)
            for i in take:
                # pixel centre -> map -> WGS84
                x_m, y_m = rasterio.transform.xy(transform, ys[i], xs[i])
                lon, lat = to_wgs.transform(x_m, y_m)
                pid = f"{row['SITECODE']}_{year}_{stratum[:3]}_{i}"
                rows.append({
                    "point_id": pid, "site": row["SITE_NAME"], "year": year,
                    "scene_date": day, "stratum": stratum,
                    "lat": round(lat, 6), "lon": round(lon, 6),
                    "truth": "",  # to be filled by the human labeller
                    "notes": "",
                })
                feats_gj.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {"point_id": pid, "site": row["SITE_NAME"],
                                   "year": year, "stratum": stratum,
                                   "scene_date": day},
                })
        print(f"{row['SITE_NAME'][:30]:30} {year}: sampled "
              f"{min(PER_STRATUM, int(gng.sum()))} detected + "
              f"{PER_STRATUM} undetected points")

    OUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    OUT_GJ.write_text(json.dumps({"type": "FeatureCollection",
                                  "features": feats_gj}))
    print(f"\n{len(rows)} points -> {OUT_CSV} and {OUT_GJ}")
    print("Label the `truth` column (bare_peat / vegetated / water / burn / "
          "unsure) against high-res imagery, then run the scorer.")


if __name__ == "__main__":
    main()
