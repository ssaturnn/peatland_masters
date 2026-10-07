"""Step 31 — summer check of spring bare-peat candidates.

Exposed peat stays exposed for months: a cut or stripped surface does not
grow a full green canopy between April and July. Winter-brown vegetation
(Molinia, sedges) does. So for every frozen site-year this script reads the
Sentinel-2 scenes of the same year's summer (15 June - 31 August), builds a
median NDVI from their clear pixels on the frozen 10 m grid, and asks what
became of each method's spring candidates:

- green in summer (NDVI >= 0.50): vegetation that was senescent in spring;
- still bare (NDVI < 0.35): consistent with exposed peat;
- in between, or no clear summer pixel.

The same is done for the sampled points with their reference labels.

Usage:
    python3 scripts/31_summer_check.py outputs/evaluation/2026-09-13-v3-sample \
        outputs/evaluation/2026-10-07-v4-test
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import planetary_computer as pc
import rasterio
from pystac_client import Client
from rasterio.warp import reproject, Resampling, transform_bounds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from peatland import imagery, v4

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
WINDOW = ("06-15", "08-31")
GREEN, BARE = 0.50, 0.35
MAX_SCENES = 4
CLEAR_SCL = (4, 5)  # vegetation, not vegetated


def summer_ndvi(meta, shape, catalog):
    """Median clear-sky summer NDVI on the frozen grid, and the scene dates used."""
    transform = rasterio.Affine(*meta["transform"][:6])
    h, w = shape
    west, south, east, north = rasterio.transform.array_bounds(h, w, transform)
    bbox = transform_bounds(meta["crs"], "EPSG:4326", west, south, east, north)
    y = meta["year"]
    items = list(catalog.search(collections=["sentinel-2-l2a"], bbox=bbox,
                                datetime=f"{y}-{WINDOW[0]}/{y}-{WINDOW[1]}",
                                query={"eo:cloud_cover": {"lt": 40}}).items())
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))

    def on_grid(item, band):
        a, t, c = imagery.read_window(item, band, bbox, "planetary")
        out = np.full(shape, np.nan, np.float32)
        reproject(a.astype(np.float32), out, src_transform=t, src_crs=c,
                  dst_transform=transform, dst_crs=meta["crs"], resampling=Resampling.nearest)
        return out

    layers, dates = [], []
    for item in items:
        scl = on_grid(item, "scl")
        clear = np.isin(scl, CLEAR_SCL)
        if clear.mean() < 0.5:
            continue
        red, nir = on_grid(item, "red"), on_grid(item, "nir")
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = (nir - red) / (nir + red)
        ndvi[~clear] = np.nan
        layers.append(ndvi)
        dates.append(item.properties["datetime"][:10])
        if len(layers) == MAX_SCENES:
            break
    if not layers:
        return None, []
    with np.errstate(all="ignore"):
        return np.nanmedian(np.array(layers), axis=0), sorted(dates)


def fate(ndvi, mask):
    vals = ndvi[mask]
    seen = np.isfinite(vals)
    n = int(mask.sum())
    return {"candidate_px": n, "seen_px": int(seen.sum()),
            "green_px": int((vals[seen] >= GREEN).sum()),
            "bare_px": int((vals[seen] < BARE).sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/evaluation/summer-check.json")
    args = ap.parse_args()
    catalog = Client.open(STAC, modifier=pc.sign_inplace)
    result = {"window": WINDOW, "green_ndvi": GREEN, "bare_ndvi": BARE, "site_years": [], "points": []}
    for run in args.runs:
        labels = {}
        lab = run / "annotation/labels.csv"
        if lab.exists():
            labels = {r["point_id"]: r for r in csv.DictReader(lab.open(encoding="utf-8-sig"))}
        points = list(csv.DictReader((run / "annotation/accuracy_points.csv").open(encoding="utf-8-sig")))
        for npz in sorted(run.glob("*.npz")):
            meta = json.loads(npz.with_suffix(".json").read_text())
            with np.load(npz) as z:
                inside = z["inside"].astype(bool)
                stack = z["stack"]
                masks = {m: z[f"prediction_{m}"].astype(bool) for m in ("ndvi", "rules", "gng")}
            masks["v4"] = v4.filter_mask(masks["gng"], stack, inside, order=tuple(meta["bands"]))
            ndvi, dates = summer_ndvi(meta, inside.shape, catalog)
            entry = {"run": run.name, "key": npz.stem, "site": meta["site"], "year": meta["year"],
                     "spring_date": meta["scene_date"], "role": meta.get("role"), "summer_dates": dates}
            if ndvi is not None:
                entry["methods"] = {m: fate(ndvi, k & inside) for m, k in masks.items()}
                entry["bog_summer_ndvi_median"] = round(float(np.nanmedian(ndvi[inside])), 3)
                for p in points:
                    if not p["point_id"].startswith(npz.stem + "_"):
                        continue
                    v = ndvi[int(p["pixel_row"]), int(p["pixel_col"])]
                    lab_row = labels.get(p["point_id"], {})
                    result["points"].append({
                        "run": run.name, "point_id": p["point_id"], "site": meta["site"],
                        "label": lab_row.get("truth"), "source": lab_row.get("reference_source"),
                        "prediction_gng": p.get("prediction_gng"),
                        "summer_ndvi": None if not np.isfinite(v) else round(float(v), 3)})
            result["site_years"].append(entry)
            g = entry.get("methods", {}).get("gng")
            if g and g["seen_px"]:
                print(f"{meta['site'][:30]:30} {meta['year']} spring {meta['scene_date']}: GNG "
                      f"{g['candidate_px']} px, green in summer {g['green_px'] / g['seen_px']:.0%}, "
                      f"still bare {g['bare_px'] / g['seen_px']:.0%} ({len(dates)} summer scenes)")
            else:
                print(f"{meta['site'][:30]:30} {meta['year']}: no candidates or no clear summer scene")
    args.out.write_text(json.dumps(result, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
