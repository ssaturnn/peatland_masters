"""Step 38 — dense Sentinel-2 time series of one bog.

For one bog of the release and a date range, it reads every Sentinel-2 L2A
acquisition (one per day, the least cloudy tile) whose scene classification
shows at least MIN_CLEAR of the bog clear, on the release grid, and stores
per pixel and date: validity (clear, not cloud or shadow), the v5
exposed-peat flag, NBR, SWIR1/SWIR2 and NDVI. The bundle feeds the timing and
fire case studies (scripts/39_cutting_timing.py, scripts/40_fire_signal.py).

Usage:
    python3 scripts/38_time_series.py 001244 2017-03-01 2026-09-30
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from peatland import boundaries, config, geo, imagery, pipeline, preprocess, v5

spec = importlib.util.spec_from_file_location("mc", ROOT / "scripts/12_method_comparison.py")
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)

RELEASE = config.OUT_DIR / "cache-release-v3-uplands"
OUT = config.OUT_DIR / "time-series"
MIN_CLEAR = 0.5
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("code")
    ap.add_argument("start")
    ap.add_argument("end")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sites = boundaries.build_sites()
    sel = sites[sites.SITECODE == args.code]
    idx = sel.index[0]
    row = sel.iloc[0]
    record = mc.release_record(args.code, RELEASE)
    shape = tuple(record["grid"]["shape"])
    transform = rasterio.Affine(*record["grid"]["transform"][:6])
    crs = record["grid"]["crs"]
    bbox = boundaries.site_bbox_wgs84(sites, idx, buffer_m=200)
    geom = gpd.GeoSeries([row.geometry], crs=config.ITM).to_crs(crs).iloc[0]
    site_inside = geo.polygon_mask(geom, shape, transform)
    tidal = (pipeline.mask_from_b64(record["tidal_mask_b64"], shape)
             if record.get("tidal_mask_b64") else np.zeros(shape, bool))
    inside = site_inside & ~tidal

    catalog = Client.open(STAC, modifier=pc.sign_inplace)
    items = list(catalog.search(collections=["sentinel-2-l2a"], bbox=bbox,
                                datetime=f"{args.start}/{args.end}",
                                query={"eo:cloud_cover": {"lt": 80}}).items())
    by_day = {}
    for it in items:   # one acquisition per day: the least cloudy tile
        d = it.properties["datetime"][:10]
        if d not in by_day or it.properties["eo:cloud_cover"] < by_day[d].properties["eo:cloud_cover"]:
            by_day[d] = it
    print(f"{row.SITE_NAME}: {len(by_day)} acquisition days", flush=True)

    order = list(pipeline.BANDS)
    dates, valid, peat, nbr, ratio, ndvi = [], [], [], [], [], []
    for n, (d, it) in enumerate(sorted(by_day.items()), 1):
        try:
            rep = preprocess.assess_scene(it, bbox, "planetary", shape, aoi_mask=site_inside)
            if rep["aoi_clear"] < MIN_CLEAR:
                continue
            stack = np.stack([imagery.read_window(it, b, bbox, "planetary", out_shape=shape)[0]
                              .astype("float32") for b in order], -1)
            q = pipeline.scene_quality(stack, order, rep["scl"], inside, it)
        except Exception as e:
            print(f"  {d}: skipped ({type(e).__name__})", flush=True)
            continue
        ok = q["valid"] & inside
        ix = v5.indices(stack, tuple(order))
        r = stack[:, :, order.index("red")] / 1e4
        nir = stack[:, :, order.index("nir")] / 1e4
        with np.errstate(invalid="ignore", divide="ignore"):
            nd = (nir - r) / (nir + r)
        dates.append(d)
        valid.append(ok)
        peat.append(v5.peat_mask(stack, ok, inside, tuple(order)))
        nbr.append(ix["nbr"].astype(np.float16))
        ratio.append(ix["ratio"].astype(np.float16))
        ndvi.append(nd.astype(np.float16))
        if n % 20 == 0 or True:
            print(f"  {d}: clear {ok.sum() / inside.sum():.2f}, v5 {peat[-1].sum() / 100:.2f} ha", flush=True)
    out = OUT / f"{args.code}_{args.start[:4]}_{args.end[:4]}.npz"
    np.savez_compressed(out, dates=np.array(dates), valid=np.array(valid), peat=np.array(peat),
                        nbr=np.array(nbr), ratio=np.array(ratio), ndvi=np.array(ndvi),
                        inside=inside, site_inside=site_inside, transform=np.array(list(transform)))
    (out.with_suffix(".json")).write_text(json.dumps({
        "code": args.code, "site": row.SITE_NAME, "crs": str(crs), "transform": list(transform),
        "start": args.start, "end": args.end, "min_clear": MIN_CLEAR, "scenes": len(dates),
        "rule": v5.VERSION}, indent=2))
    print(f"{len(dates)} usable scenes -> {out}")


if __name__ == "__main__":
    main()
