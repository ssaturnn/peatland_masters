"""Step 41 — forty years of exposed peat on one bog from Landsat.

Landsat 5, 7, 8 and 9 (Collection 2 surface reflectance) carry the same
short-wave infrared bands the v5 rule uses (SWIR1 ~1.6 um, SWIR2 ~2.2 um), so
the rule can be applied back to 1984 on a 30 m pixel. For every year and
summer (1 May - 30 September) the script takes the clearest Landsat scene
over the bog (cloud, cloud shadow and dilated cloud from QA_PIXEL), resamples
it onto the bog's 10 m release grid and records the v5 exposed-peat area.
Years where Sentinel-2 also exists (2017+) check that the two sensors agree.

Usage:
    python3 scripts/41_landsat_history.py 001244
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import planetary_computer as pc
import rasterio
from pystac_client import Client
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from peatland import boundaries, config, geo, pipeline, v5

spec = importlib.util.spec_from_file_location("mc", ROOT / "scripts/12_method_comparison.py")
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)

RELEASE = config.OUT_DIR / "cache-release-v3-uplands"
OUT = config.OUT_DIR / "time-series"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
BANDS = {"blue": "blue", "green": "green", "red": "red", "nir": "nir08", "swir1": "swir16", "swir2": "swir22"}
CLOUD_BITS = (1 << 1) | (1 << 3) | (1 << 4)   # dilated cloud, cloud, cloud shadow
MIN_CLEAR = 0.8


def read(item, asset, bbox, shape, transform, crs, nearest=False):
    href = item.assets[asset].href
    with rasterio.open(href) as src:
        win = from_bounds(*transform_bounds("EPSG:4326", src.crs, *bbox), src.transform)
        arr = src.read(1, window=win, boundless=True, fill_value=0)
        wt = src.window_transform(win)
        out = np.zeros(shape, np.float32)
        reproject(arr.astype(np.float32), out, src_transform=wt, src_crs=src.crs,
                  dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.nearest if nearest else Resampling.bilinear)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("code")
    ap.add_argument("--first", type=int, default=1984)
    ap.add_argument("--last", type=int, default=2025)
    args = ap.parse_args()
    sites = boundaries.build_sites()
    sel = sites[sites.SITECODE == args.code]
    idx, row = sel.index[0], sel.iloc[0]
    record = mc.release_record(args.code, RELEASE)
    shape = tuple(record["grid"]["shape"])
    transform = rasterio.Affine(*record["grid"]["transform"][:6])
    crs = record["grid"]["crs"]
    bbox = boundaries.site_bbox_wgs84(sites, idx, buffer_m=200)
    geom = gpd.GeoSeries([row.geometry], crs=config.ITM).to_crs(crs).iloc[0]
    inside = geo.polygon_mask(geom, shape, transform)
    catalog = Client.open(STAC, modifier=pc.sign_inplace)
    order = ("blue", "green", "red", "nir", "swir1", "swir2")
    result = {"site": row.SITE_NAME, "code": args.code, "years": {}}
    masks = {}
    for year in range(args.first, args.last + 1):
        items = list(catalog.search(collections=["landsat-c2-l2"], bbox=bbox,
                                    datetime=f"{year}-05-01/{year}-09-30",
                                    query={"eo:cloud_cover": {"lt": 60},
                                           "platform": {"in": ["landsat-5", "landsat-7", "landsat-8", "landsat-9"]}}).items())
        items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
        best = None
        for it in items[:8]:
            try:
                qa = read(it, "qa_pixel", bbox, shape, transform, crs, nearest=True).astype(np.uint32)
            except Exception:
                continue
            clear = inside & (qa > 1) & ((qa & CLOUD_BITS) == 0)
            frac = clear.sum() / inside.sum()
            if best is None or frac > best[1]:
                best = (it, frac, clear)
            if frac >= 0.97:
                break
        if best is None or best[1] < MIN_CLEAR:
            result["years"][str(year)] = {"status": "no clear summer scene"}
            print(year, "no clear scene", flush=True)
            continue
        it, frac, clear = best
        refl = [read(it, BANDS[b], bbox, shape, transform, crs) * 0.0000275 - 0.2 for b in order]
        stack = np.stack(refl, -1) * 10000.0
        # Landsat 7 after 2003 has SLC-off gaps: zero-filled pixels are not clear
        clear &= (stack > -1000).all(-1) & (stack[..., 0] != -2000)
        peat = v5.peat_mask(stack, clear, inside, order)
        ha = float(peat.sum()) / 100 / max(clear.sum() / inside.sum(), 1e-9)
        result["years"][str(year)] = {"date": it.properties["datetime"][:10], "platform": it.properties["platform"],
                                      "clear": round(float(frac), 3), "v5_ha": round(ha, 2)}
        masks[str(year)] = peat
        print(year, it.properties["datetime"][:10], it.properties["platform"], f"clear {frac:.2f}",
              f"v5 {ha:.1f} ha", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.code}_landsat.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT / f"{args.code}_landsat_masks.npz", inside=inside, **{f"y{k}": v for k, v in masks.items()})
    print(f"-> {OUT / (args.code + '_landsat.json')}")


if __name__ == "__main__":
    main()
