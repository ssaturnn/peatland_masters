"""Step 36 — summer v5 release for the web map.

For every bog of the release and every year 2018-2026 it takes the clearest
summer scene (15 June - 31 August) that passes the shared quality gates on
the release grid, maps exposed peat with the v5 SWIR rule
(src/peatland/v5.py), and renders the card tiles: natural colour with the v5
pixels filled orange, and a clean twin. Site-years within 13 months after a
NASA FIRMS fire inside or within 500 m of the bog are flagged, because v5
also maps peat exposed by fire.

Results are cached per bog in outputs/cache-v5/<code>.json (resumable), the
tiles go to web/data/tiles-v5/, and --write adds the v5 series to
web/data/sites.geojson.

Usage:
    python3 scripts/36_v5_release.py            # build (resumable, 4 workers)
    python3 scripts/36_v5_release.py --write    # merge into sites.geojson
"""

import argparse
import datetime as dt
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from peatland import boundaries, config, geo, imagery, pipeline, preprocess, v5

spec = importlib.util.spec_from_file_location("mc", ROOT / "scripts/12_method_comparison.py")
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)

RELEASE = config.OUT_DIR / "cache-release-v3-uplands"
CACHE = config.OUT_DIR / "cache-v5"
TILES = ROOT / "web/data/tiles-v5"
GEOJSON = ROOT / "web/data/sites.geojson"
FIRMS = config.OUT_DIR / "firms/events.json"
YEARS = range(2018, 2027)
WINDOW = ("06-15", "08-31")
MAXPX = 360
PEAT_RGBA = (255, 140, 0, 190)
# one contiguous patch this large is not domestic turf cutting: in a single summer
# it most likely marks a fire FIRMS did not record (or has not published yet); in
# two or more summers it marks large-scale peat working
LARGE_PATCH_HA = 10


def natural_rgb(stack, order):
    rgb = np.dstack([stack[:, :, order.index(b)] for b in ("red", "green", "blue")]) / 10000.0
    return np.clip(rgb * 3.2, 0, 1) ** (1.0 / 1.4)


def edge(mask):
    m = np.asarray(mask, bool)
    out = np.zeros_like(m)
    out[1:, :] |= m[1:, :] != m[:-1, :]
    out[:, 1:] |= m[:, 1:] != m[:, :-1]
    return out & m


def save_tiles(rgb, peat, inside, stem):
    base = (rgb * 255).astype(np.uint8)
    h, w = peat.shape
    size = (max(1, round(w * MAXPX / max(h, w))), max(1, round(h * MAXPX / max(h, w))))
    Image.fromarray(base).resize(size, Image.LANCZOS).save(TILES / f"{stem}c.jpg", quality=82, optimize=True)
    over = Image.fromarray(base).convert("RGBA")
    layer = np.zeros((h, w, 4), np.uint8)
    layer[peat] = PEAT_RGBA
    layer[edge(inside)] = (255, 255, 255, 170)
    over = Image.alpha_composite(over, Image.fromarray(layer, "RGBA")).convert("RGB")
    # nearest keeps single peat pixels visible after downscaling
    over.resize(size, Image.NEAREST if max(h, w) > MAXPX else Image.LANCZOS).save(
        TILES / f"{stem}.jpg", quality=85, optimize=True)


def fires():
    out = {}
    if FIRMS.exists():
        for e in json.loads(FIRMS.read_text())["events"]:
            if e["min_distance_m"] <= 500:
                out.setdefault(e["code"], []).append(e["start"])
    return out


def build_site(sites, idx, fire_days):
    row = sites.loc[idx]
    code = row["SITECODE"]
    out_f = CACHE / f"{code}.json"
    if out_f.exists():
        return code, "cached"
    record = mc.release_record(code, RELEASE)
    shape = tuple(record["grid"]["shape"])
    transform = rasterio.Affine(*record["grid"]["transform"][:6])
    crs = record["grid"]["crs"]
    bbox = boundaries.site_bbox_wgs84(sites, idx, buffer_m=200)
    geom = gpd.GeoSeries([row.geometry], crs=config.ITM).to_crs(crs).iloc[0]
    site_inside = geo.polygon_mask(geom, shape, transform)
    tidal = (pipeline.mask_from_b64(record["tidal_mask_b64"], shape)
             if record.get("tidal_mask_b64") else np.zeros(shape, bool))
    inside = site_inside & ~tidal
    order = list(pipeline.BANDS)
    years = {}
    for year in YEARS:
        best = None
        try:
            scenes = preprocess.clear_scenes("planetary", bbox, f"{year}-{WINDOW[0]}/{year}-{WINDOW[1]}",
                                             shape, aoi_mask=site_inside, min_clear=0.85, max_scenes=4)
            for item, rep in scenes:
                stack = np.stack([imagery.read_window(item, b, bbox, "planetary", out_shape=shape)[0]
                                  .astype("float32") for b in order], -1)
                q = pipeline.scene_quality(stack, order, rep["scl"], inside, item)
                if pipeline.scene_passes(q) and (best is None or q["clear_fraction"] > best[2]["clear_fraction"]):
                    best = (item, stack, q)
        except Exception as e:  # one failed year must not lose the others
            years[year] = {"error": f"{type(e).__name__}: {e}"}
            continue
        if best is None:
            continue
        item, stack, q = best
        peat = v5.peat_mask(stack, q["valid"], inside, tuple(order))
        labels, n = ndimage.label(peat, structure=np.ones((3, 3), bool))
        largest = int(np.bincount(labels.ravel())[1:].max()) if n else 0
        date = item.properties["datetime"][:10]
        d = dt.date.fromisoformat(date)
        after_fire = [f for f in fire_days.get(code, []) if 0 <= (d - dt.date.fromisoformat(f)).days <= 400]
        save_tiles(natural_rgb(stack, order), peat, site_inside, f"{code}_{year}")
        years[year] = {"date": date, "scene_id": item.id, "clear": round(q["clear_fraction"], 3),
                       "v5_ha": round(geo.mask_area_ha(peat, transform), 2), "after_fire": after_fire,
                       "largest_patch_ha": round(largest * abs(transform.a * transform.e) / 1e4, 2),
                       "patches": int(n)}
    out = {"version": v5.VERSION, "code": code, "site_ha": round(geo.mask_area_ha(inside, transform), 1),
           "window": WINDOW, "years": {str(k): v for k, v in years.items()}}
    out_f.write_text(json.dumps(out, indent=1))
    return code, f"{sum('v5_ha' in v for v in years.values())} years"


def write_geojson():
    data = json.loads(GEOJSON.read_text())
    fire_days = fires()
    for f in data["features"]:
        p = f["properties"]
        c = CACHE / f"{p['code']}.json"
        if not c.exists():
            continue
        r = json.loads(c.read_text())
        ys = sorted((int(y), v) for y, v in r["years"].items() if "v5_ha" in v)
        for _, v in ys:  # recomputed here so a newer FIRMS record needs no imagery rebuild
            d = dt.date.fromisoformat(v["date"])
            v["after_fire"] = [f for f in fire_days.get(p["code"], [])
                               if 0 <= (d - dt.date.fromisoformat(f)).days <= 400]
        p["v5_version"] = r["version"]
        p["v5_years"] = [y for y, _ in ys]
        p["v5_series"] = [v["v5_ha"] for _, v in ys]
        p["v5_dates"] = [v["date"] for _, v in ys]
        p["v5_after_fire"] = [bool(v["after_fire"]) for _, v in ys]
        p["v5_largest_patch"] = [v.get("largest_patch_ha", 0) for _, v in ys]
        large = [v.get("largest_patch_ha", 0) >= LARGE_PATCH_HA for _, v in ys]
        # a large patch in one summer only is most likely a fire; large patches in two
        # or more summers are fields worked year after year (milled or machine-cut peat)
        p["v5_large_working"] = sum(large) >= 2
        p["v5_large_patch"] = [False] * len(large) if p["v5_large_working"] else large
        clean = [v["v5_ha"] for (_, v), big in zip(ys, p["v5_large_patch"])
                 if not v["after_fire"] and not big]
        p["v5_peak_ha"] = max(clean) if clean else 0.0
        p["v5_mean_pct"] = round(100 * float(np.mean(clean)) / max(r["site_ha"], 1), 3) if clean else 0.0
        p["v5_years_with_peat"] = sum(1 for v in clean if v >= 0.1)
    GEOJSON.write_text(json.dumps(data))
    print(f"v5 series written to {GEOJSON}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="merge cached results into sites.geojson")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.write:
        return write_geojson()
    CACHE.mkdir(parents=True, exist_ok=True)
    TILES.mkdir(parents=True, exist_ok=True)
    sites = boundaries.build_sites()
    published = {f["properties"]["code"] for f in json.loads(GEOJSON.read_text())["features"]}
    todo = [i for i in sites.index if sites.loc[i, "SITECODE"] in published]
    fire_days = fires()

    def safe(i):
        try:
            return build_site(sites, i, fire_days)
        except Exception as e:
            return sites.loc[i, "SITECODE"], f"FAILED {type(e).__name__}: {e}"

    with ThreadPoolExecutor(args.workers) as pool:
        for n, (code, status) in enumerate(pool.map(safe, todo), 1):
            print(f"[{n}/{len(todo)}] {code}: {status}", flush=True)


if __name__ == "__main__":
    main()
