"""Step 34 — frozen test run for detector v5.

Selection, written before any v5 prediction on these scenes was looked at:

- industrial test site-years: Mountdillon production bogs on scenes that did
  not calibrate v5 (calibration used the south area in 2018 only): the west
  area in summer 2018 and spring 2020, the south area in summer 2022 and
  spring 2024 (after harvesting stopped);
- protected raised bogs in summer (15 June - 31 August), when cut turf is
  spread and drying: the six SACs with NPWS-documented cutting in the year
  (Callow 2021, Cloonchambers 2021, Corbo 2021, Corliskea 2021, Monivea 2022,
  Barroughter 2022) and two raised-bog SACs with no documented cutting as
  controls (Bellanagare 2022, Carrownagappul 2021).

Protected bogs use the published release grid, site polygon and tidal-water
exclusion; the scene is the clearest summer scene that passes the shared
quality gates. Every bundle stores the v1-v4 predictions and prediction_v5,
so scripts/13_sample_points.py --stratify-by v5 can sample it.

Usage:
    python3 scripts/34_v5_test_run.py outputs/evaluation/2026-10-08-v5-test
"""

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from peatland import boundaries, config, detect, geo, imagery, pipeline, preprocess, v5

spec = importlib.util.spec_from_file_location("mc", ROOT / "scripts/12_method_comparison.py")
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)

INDUSTRIAL = ["BNMW-su_2018", "BNMW-sp_2020", "BNMS-su_2022", "BNMS-sp_2024"]
PROTECTED = [("Callow", 2021, "documented"), ("Cloonchambers", 2021, "documented"),
             ("Corbo", 2021, "documented"), ("Corliskea", 2021, "documented"),
             ("Monivea", 2022, "documented"), ("Barroughter", 2022, "documented"),
             ("Bellanagare", 2022, "control"), ("Carrownagappul", 2021, "control")]
WINDOW = ("06-15", "08-31")


def add_v5(npz_path, meta):
    with np.load(npz_path) as z:
        arrays = dict(z)
    arrays["prediction_v5"] = v5.peat_mask(arrays["stack"], arrays["valid"], arrays["inside"],
                                           tuple(meta["bands"]))
    np.savez_compressed(npz_path, **arrays)
    return arrays


def freeze_protected(sites, name, year, out):
    match = sites[sites.SITE_NAME.str.contains(name, regex=False)]
    row = match.iloc[0]
    record = mc.release_record(row["SITECODE"], config.OUT_DIR / "cache-release-v3-uplands")
    shape = tuple(record["grid"]["shape"])
    transform = rasterio.Affine(*record["grid"]["transform"][:6])
    crs = record["grid"]["crs"]
    bbox = boundaries.site_bbox_wgs84(sites, row.name, buffer_m=200)
    geom = gpd.GeoSeries([row.geometry], crs=config.ITM).to_crs(crs).iloc[0]
    site_inside = geo.polygon_mask(geom, shape, transform)
    tidal = (pipeline.mask_from_b64(record["tidal_mask_b64"], shape)
             if record.get("tidal_mask_b64") else np.zeros(shape, bool))
    inside = site_inside & ~tidal
    scenes = preprocess.clear_scenes("planetary", bbox, f"{year}-{WINDOW[0]}/{year}-{WINDOW[1]}", shape,
                                     aoi_mask=site_inside, min_clear=0.85, max_scenes=6)
    order = list(pipeline.BANDS)
    best = None
    for item, rep in scenes:
        stack = np.stack([imagery.read_window(item, b, bbox, "planetary", out_shape=shape)[0]
                          .astype("float32") for b in order], -1)
        q = pipeline.scene_quality(stack, order, rep["scl"], inside, item)
        if pipeline.scene_passes(q) and (best is None or q["clear_fraction"] > best[2]["clear_fraction"]):
            best = (item, stack, q)
    if best is None:
        print(f"{name} {year}: no summer scene passes the gates")
        return
    item, stack, q = best
    valid = q["valid"]
    ndvi = detect.ndvi(stack[:, :, order.index("red")], stack[:, :, order.index("nir")])
    masks = {"ndvi": pipeline.ndvi_bare(stack, order, ndvi, valid, inside),
             "rules": pipeline.rules_bare(stack, order, valid, inside),
             "kmeans": pipeline.kmeans_bare(stack, order, valid, inside),
             "gng": pipeline.gng_bare(stack, order, valid, inside),
             "v5": v5.peat_mask(stack, valid, inside, tuple(order))}
    meta = {"site_code": row["SITECODE"], "site": row["SITE_NAME"], "year": year,
            "scene_date": item.properties["datetime"][:10], "scene_id": item.id,
            "processing_baseline": item.properties.get("s2:processing_baseline"),
            "detector_version": v5.VERSION, "crs": str(crs), "transform": list(transform),
            "bands": order, "role": "protected", "group": "protected",
            "clear_fraction": q["clear_fraction"],
            "areas_ha": {m: round(geo.mask_area_ha(k, transform), 2) for m, k in masks.items()},
            "scene_selection": "clearest summer scene passing the shared gates, release grid"}
    stem = f"{row['SITECODE']}_{year}"
    np.savez_compressed(out / f"{stem}.npz", stack=stack, valid=valid, inside=inside,
                        transform=np.array(list(transform)),
                        **{f"prediction_{m}": k for m, k in masks.items()})
    (out / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    print(f"{row['SITE_NAME'][:30]:30} {year} {meta['scene_date']}: v5 {meta['areas_ha']['v5']} ha, "
          f"GNG {meta['areas_ha']['gng']} ha", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    src = ROOT / "outputs/positive-control/scenes"
    for key in INDUSTRIAL:
        meta = json.loads((src / f"{key}.json").read_text())
        meta.update(role="industrial", group="industrial", detector_version=v5.VERSION)
        shutil.copy(src / f"{key}.npz", args.out / f"{key}.npz")
        arrays = add_v5(args.out / f"{key}.npz", meta)
        meta["areas_ha"]["v5"] = round(float(arrays["prediction_v5"].sum()) / 100, 2)
        (args.out / f"{key}.json").write_text(json.dumps(meta, indent=2))
        print(f"{key}: v5 {meta['areas_ha']['v5']} ha, GNG {meta['areas_ha']['gng']} ha", flush=True)
    sites = boundaries.build_sites()
    for name, year, _ in PROTECTED:
        freeze_protected(sites, name, year, args.out)


if __name__ == "__main__":
    main()
