"""Step 46 — side-by-side imagery for every bog on the private map.

The card shows, for one summer, the same bog in several images so that the
v5 exposed-peat mask can be checked by eye:

- swir: for every bog-year of the v5 release (outputs/cache-v5/), the exact
  Sentinel-2 scene of the card rendered in short-wave infrared false colour
  (R = SWIR2 2.2 um, G = NIR, B = red, one fixed stretch for all bogs and
  years). Dry exposed peat is pink-brown, vegetation green, water black.
  Tiles: web/data/tiles-v5/<code>_<year>s.jpg (Copernicus data, public).
- planet plan|order|collect|render: one PlanetScope 3 m scene per bog within
  3 days of a v5 summer scene: the year of the bog's largest unflagged v5
  area, or its latest year when v5 finds nothing. Rendered on the extent of
  the Sentinel-2 tile into web/private/planet-v5/ (Planet licence:
  non-commercial research only, never on the public site).

Usage:
    python3 scripts/46_compare_tiles.py swir
    python3 scripts/46_compare_tiles.py planet plan      # search, no quota used
    python3 scripts/46_compare_tiles.py planet order --cap 600
    python3 scripts/46_compare_tiles.py planet collect
    python3 scripts/46_compare_tiles.py planet pilot     # adopt frames already ordered
    python3 scripts/46_compare_tiles.py planet render
"""

import argparse
import datetime as dt
import fcntl
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from pystac_client import Client
import planetary_computer as pc
from shapely.geometry import mapping, shape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from peatland import boundaries, config, geo, imagery, pipeline, planet

spec = importlib.util.spec_from_file_location("mc", ROOT / "scripts/12_method_comparison.py")
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)
spec = importlib.util.spec_from_file_location("pt", ROOT / "scripts/22_planet_web_tiles.py")
pt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pt)

RELEASE = config.OUT_DIR / "cache-release-v3-uplands"
CACHE = config.OUT_DIR / "cache-v5"
TILES = ROOT / "web/data/tiles-v5"
GEOJSON = ROOT / "web/data/sites.geojson"
PS_OUT = ROOT / "outputs/planet_v5"
PS_TILES = ROOT / "web/private/planet-v5"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
MAXPX = 360          # same long side as the other card tiles
# fixed reflectance stretch, so colours compare across bogs and years
SWIR_SCALE = {"swir2": 0.32, "nir": 0.45, "red": 0.16}


def grid(sites, code):
    record = mc.release_record(code, RELEASE)
    shape_ = tuple(record["grid"]["shape"])
    transform = rasterio.Affine(*record["grid"]["transform"][:6])
    crs = record["grid"]["crs"]
    idx = sites.index[sites.SITECODE == code][0]
    bbox = boundaries.site_bbox_wgs84(sites, idx, buffer_m=200)
    geom = gpd.GeoSeries([sites.loc[idx].geometry], crs=config.ITM).to_crs(crs).iloc[0]
    return shape_, transform, crs, bbox, geo.polygon_mask(geom, shape_, transform)


def edge(mask):
    m = np.asarray(mask, bool)
    out = np.zeros_like(m)
    out[1:, :] |= m[1:, :] != m[:-1, :]
    out[:, 1:] |= m[:, 1:] != m[:, :-1]
    return out & m


def swir_rgb(bands):
    rgb = np.dstack([bands[b] / 10000.0 / SWIR_SCALE[b] for b in ("swir2", "nir", "red")])
    return np.clip(rgb, 0, 1) ** (1.0 / 1.2)


def swir_site(sites, catalog, code):
    cache = json.loads((CACHE / f"{code}.json").read_text())
    todo = {y: v for y, v in cache["years"].items()
            if "scene_id" in v and not (TILES / f"{code}_{y}s.jpg").exists()}
    if not todo:
        return code, "done"
    shape_, transform, crs, bbox, inside = grid(sites, code)
    for year, v in todo.items():
        item = next(catalog.search(collections=["sentinel-2-l2a"], ids=[v["scene_id"]]).items())
        bands = {b: imagery.read_window(item, b, bbox, "planetary", out_shape=shape_)[0].astype("float32")
                 for b in ("swir2", "nir", "red")}
        h, w = shape_
        size = (max(1, round(w * MAXPX / max(h, w))), max(1, round(h * MAXPX / max(h, w))))
        img = np.array(Image.fromarray((swir_rgb(bands) * 255).astype(np.uint8)).resize(size, Image.LANCZOS))
        # the boundary is drawn after resizing, so a one-pixel line survives downscaling
        bog = np.asarray(Image.fromarray(inside.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127
        img[edge(bog)] = (255, 255, 255)
        Image.fromarray(img).save(TILES / f"{code}_{year}s.jpg", quality=85, optimize=True)
    return code, f"{len(todo)} tiles"


def run_swir(workers):
    sites = boundaries.build_sites()
    codes = sorted(p.stem for p in CACHE.glob("*.json"))
    catalog = Client.open(STAC, modifier=pc.sign_inplace)

    def safe(code):
        try:
            return swir_site(sites, catalog, code)
        except Exception as e:
            return code, f"FAILED {type(e).__name__}: {e}"

    with ThreadPoolExecutor(workers) as pool:
        for n, (code, status) in enumerate(pool.map(safe, codes), 1):
            print(f"[{n}/{len(codes)}] {code}: {status}", flush=True)


def targets():
    """One summer per bog: its largest unflagged v5 year, else its latest year."""
    out = []
    for f in json.loads(GEOJSON.read_text())["features"]:
        p = f["properties"]
        ys = p.get("v5_years") or []
        if not ys:
            continue
        ok = [i for i in range(len(ys)) if not p["v5_after_fire"][i] and not p["v5_large_patch"][i]]
        best = max(ok, key=lambda i: p["v5_series"][i]) if ok else len(ys) - 1
        i = best if ok and p["v5_series"][best] >= 0.1 else len(ys) - 1
        cache = json.loads((CACHE / f"{p['code']}.json").read_text())["years"][str(ys[i])]
        out.append({"key": f"{p['code']}_{ys[i]}", "code": p["code"], "site": p["name"], "year": ys[i],
                    "s2_date": p["v5_dates"][i], "sentinel2_scene": cache["scene_id"],
                    "v5_ha": p["v5_series"][i]})
    return out


def planet_plan(out):
    path = out / "plan.json"
    result = json.loads(path.read_text()) if path.exists() else None
    if result is None:
        sites = boundaries.build_sites().set_index("SITECODE")
        rows = targets()
        for r in rows:
            aoi, tol = planet.buffered_aoi(sites.loc[r["code"]].geometry)
            r.update(aoi=mapping(aoi), aoi_area_km2=planet.area_km2(aoi), status="unsearched")
        result = {"version": planet.VERSION, "targets": rows, "search_complete": False}
        planet.write_json(path, result)
    s = planet.session()
    for r in result["targets"]:
        if r["status"] != "unsearched":
            continue
        aoi = shape(r["aoi"])
        choice = planet.choose(planet.search(s, aoi, dt.date.fromisoformat(r["s2_date"])), aoi)
        if choice:
            date, frames, cover = choice
            r.update(date=date, delta_days=(dt.date.fromisoformat(date) - dt.date.fromisoformat(r["s2_date"])).days,
                     cover=cover, items=[{k: v for k, v in c.items() if k != "geom"} for c in frames],
                     reserved_km2=r["aoi_area_km2"] * len(frames), status="available")
        else:
            r["status"] = "no_clear_coverage"
        planet.write_json(path, result)
        print(f"{r['key']}: {r['status']} {r.get('date', '')}", flush=True)
    result["search_complete"] = True
    avail = [r for r in result["targets"] if r["status"] == "available"]
    result["available_km2"] = sum(r["reserved_km2"] for r in avail)
    planet.write_json(path, result)
    print(f"{len(avail)} of {len(result['targets'])} bogs available, {result['available_km2']:.1f} km² to order")
    return result


def planet_order(out, cap):
    result = json.loads((out / "plan.json").read_text())
    ledger_path = out / "ledger.json"
    ledger = (json.loads(ledger_path.read_text()) if ledger_path.exists()
              else {"cap_km2": cap, "reserved_km2": 0, "orders": {}})
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    s = planet.session()
    # smallest bogs first, so a cap leaves out the fewest bogs
    for r in sorted(result["targets"], key=lambda r: r.get("reserved_km2", 0)):
        if r["status"] != "available" or r["key"] in ledger["orders"]:
            continue
        if ledger["reserved_km2"] + r["reserved_km2"] > cap:
            print(f"{r['key']}: over the {cap} km² cap, skipped", flush=True)
            continue
        res = planet.reserve_area(ledger, r["key"], r["reserved_km2"], cap=cap)
        name = f"peatland-private-v5-compare-{r['key']}"
        res["name"] = name
        planet.write_json(ledger_path, ledger)
        try:
            oid, bundle = planet.place_order(s, name, [c["id"] for c in r["items"]], shape(r["aoi"]))
        except Exception as exc:
            res.update(status="unresolved_submission", error_type=type(exc).__name__)
            planet.write_json(ledger_path, ledger)
            raise RuntimeError(f"{r['key']}: submission unresolved; reservation retained") from None
        res.update(order_id=oid, bundle=bundle, status="ordered")
        planet.write_json(ledger_path, ledger)
        manifest[r["key"]] = {k: r[k] for k in ("site", "s2_date", "date", "delta_days", "cover",
                                                "aoi_area_km2", "sentinel2_scene")}
        manifest[r["key"]].update(order_id=oid, order_state="ordered", bundle=bundle,
                                  items=[{"id": c["id"], "acquired": c["acquired"], "clear_percent": c["clear"],
                                          "instrument": c["instrument"]} for c in r["items"]])
        planet.write_json(manifest_path, manifest)
        print(f"{r['key']}: ordered; reserved {ledger['reserved_km2']:.1f} km²", flush=True)


def planet_collect(out):
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    s = planet.session()
    pending = 0
    for key, rec in manifest.items():
        if rec.get("mosaic"):
            continue
        try:
            r = s.get(f"{planet.ORDERS}/{rec['order_id']}", timeout=60)
            r.raise_for_status()
            od = r.json()
            rec["order_state"] = od.get("state")
            if rec["order_state"] in ("success", "partial"):
                files = planet.download(s, od, out / key)
                planet.download(s, od, out / key, product="udm2")
                ids = [c["id"] for c in rec["items"]]
                files.sort(key=lambda f: next((i for i, n in enumerate(ids) if f.name.startswith(n)), 999))
                if files:
                    planet.mosaic(files, out / key / "ps_mosaic.tif")
                    rec["mosaic"] = str((out / key / "ps_mosaic.tif").relative_to(ROOT))
            elif rec["order_state"] not in ("failed", "cancelled"):
                pending += 1
        except Exception as exc:
            # never print signed delivery URLs or request headers
            pending += 1
            print(f"{key}: deferred ({type(exc).__name__})", flush=True)
        planet.write_json(manifest_path, manifest)
    print(f"pending: {pending}")


def planet_pilot(out, max_days=5):
    """Adopt PlanetScope already ordered for single-bog studies (outputs/pilot-b/)
    when it lies within a few days of a v5 summer scene: no new order needed."""
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    v5 = {f["properties"]["code"]: f["properties"] for f in json.loads(GEOJSON.read_text())["features"]}
    for folder in sorted((ROOT / "outputs/pilot-b").glob("*/meta.json")):
        meta = json.loads(folder.read_text())
        code = folder.parent.name.split("_")[0]
        p, d = v5.get(code), dt.date.fromisoformat(meta["date"])
        if not p or not (folder.parent / "ps_mosaic.tif").exists():
            continue
        for year, s2 in zip(p["v5_years"], p["v5_dates"]):
            gap = (d - dt.date.fromisoformat(s2)).days
            key = f"{code}_{year}"
            if abs(gap) <= max_days and key not in manifest:
                cache = json.loads((CACHE / f"{code}.json").read_text())["years"][str(year)]
                manifest[key] = {"site": p["name"], "s2_date": s2, "date": meta["date"], "delta_days": gap,
                                 "sentinel2_scene": cache["scene_id"], "order_id": meta["order"],
                                 "mosaic": str((folder.parent / "ps_mosaic.tif").relative_to(ROOT)),
                                 "items": [{"id": meta["item"], "instrument": None}], "source": "pilot"}
                print(f"{key}: adopted {meta['date']} ({gap:+d} d)")
    planet.write_json(manifest_path, manifest)


def planet_render(out):
    manifest = json.loads((out / "manifest.json").read_text())
    sites = boundaries.build_sites()
    tiles_path = PS_TILES / "manifest.json"
    tiles = json.loads(tiles_path.read_text())["tiles"] if tiles_path.exists() else {}
    PS_TILES.mkdir(parents=True, exist_ok=True)
    for key, rec in manifest.items():
        if not rec.get("mosaic"):
            continue
        code, year = key.split("_")
        shape_, transform, crs, _, inside = grid(sites, code)
        t3 = transform * rasterio.Affine.scale(1 / pt.UPSAMPLE)
        shape3 = (shape_[0] * pt.UPSAMPLE, shape_[1] * pt.UPSAMPLE)
        bands = pt.read_on_grid(ROOT / rec["mosaic"], t3, shape3, crs)
        valid = (bands > 0).all(axis=0)
        clear_pct = None
        frames =planet.udm2_clear((ROOT / rec["mosaic"]).parent, [i["id"] for i in rec["items"]], t3, shape3, crs)
        if frames is not None:
            clear, covered = frames
            bog = pt.upsample(inside) & valid & covered
            if bog.any():
                clear_pct = round(100 * float((clear & bog).sum() / bog.sum()), 1)
        size = pt.fit(shape3)
        img = pt.to_image(pt.natural_rgb(bands), valid, size)
        pt.paint(img, pt.edge(pt.resize_mask(pt.upsample(inside), size), 1), pt.BOG_EDGE, 0.7)
        pt.save(img, PS_TILES / f"{key}.jpg")
        tiles[key] = {"site": rec["site"], "date": rec["date"], "s2_date": rec["s2_date"],
                      "delta_days": rec["delta_days"], "clear_pct": clear_pct,
                      "instruments": sorted({i["instrument"] for i in rec["items"] if i["instrument"]}),
                      "items": [i["id"] for i in rec["items"]], "sentinel2_scene": rec["sentinel2_scene"]}
        print(f"{key}: rendered, clear {clear_pct}", flush=True)
    planet.write_json(tiles_path, {"citation": planet.CITATION, "licence": planet.LICENCE, "tiles": tiles})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("swir", "planet"))
    ap.add_argument("action", nargs="?", choices=("plan", "order", "collect", "pilot", "render"))
    ap.add_argument("--cap", type=float, default=600, help="km² cap for every order in this folder")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.what == "swir":
        return run_swir(args.workers)
    out = planet.private_path(PS_OUT)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {"plan": planet_plan, "collect": planet_collect, "pilot": planet_pilot, "render": planet_render}.get(
            args.action, lambda o: planet_order(o, args.cap))(out)


if __name__ == "__main__":
    main()
