"""Step 32 — positive control: industrial milled-peat fields.

The protected bogs gave the detector almost no exposed peat to find: the
spring candidates there green up by summer (scripts/31_summer_check.py). This
script tests the same detector where exposed peat certainly exists: the
Bord na Móna Mountdillon production bogs around Lanesborough (Longford /
Roscommon), milled for peat until industrial harvesting stopped at the end of
2020. Each area is a rectangle drawn by eye around a group of production bogs
on the natural-colour Sentinel-2 scene of 28 June 2018; it includes the
farmland between the bogs, so ploughed and cut fields test precision.

For every area, year and season (spring 1 April - 31 May, summer 15 June -
31 August) it takes the clearest Sentinel-2 scene, runs the same quality gates
and detectors as the release (NDVI threshold, rules, K-means, GNG, v4) on a
fixed grid, records candidate areas, and freezes the arrays in the format of
scripts/12_method_comparison.py so scripts/13_sample_points.py can sample them.

Usage:
    python3 scripts/32_positive_control.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from peatland import detect, geo, imagery, pipeline, preprocess, v4

PROVIDER = "planetary"
OUT = ROOT / "outputs/positive-control"
# (west, south, east, north) in WGS84
AREAS = {
    "BNMS": {"name": "Mountdillon production bogs, south of Lanesborough (Co. Longford)",
             "bbox": (-7.960, 53.600, -7.825, 53.672)},
    "BNMW": {"name": "Mountdillon production bogs, west of the Shannon (Co. Roscommon)",
             "bbox": (-8.060, 53.672, -7.975, 53.742)},
    # added after v5 was frozen, as an external test ~30 km south of Mountdillon
    "BNMX": {"name": "Production bogs south of Athlone (Shannon valley)",
             "bbox": (-7.950, 53.300, -7.800, 53.380)},
}
SEASONS = {"sp": ("04-01", "05-31"), "su": ("06-15", "08-31")}
YEARS = range(2018, 2026)


def grid(bbox):
    """Fixed 10 m grid of an area: the red-band window of one clear scene."""
    item = imagery.search_scene(PROVIDER, bbox, "2018-06-01/2018-08-31", max_cloud=10)
    red, transform, crs = imagery.read_window(item, "red", bbox, PROVIDER)
    return red.shape, transform, crs


def run_scene(area, bbox, shape, transform, crs, year, season):
    a, b = SEASONS[season]
    inside = np.ones(shape, bool)
    item, rep = preprocess.pick_clear_scene(PROVIDER, bbox, f"{year}-{a}/{year}-{b}", shape,
                                            min_clear=0.90, max_cloud=40, limit=16)
    if item is None:
        return None, None
    order = list(pipeline.BANDS)
    stack = np.stack([imagery.read_window(item, band, bbox, PROVIDER, out_shape=shape)[0]
                      .astype("float32") for band in order], -1)
    quality = pipeline.scene_quality(stack, order, rep["scl"], inside, item)
    meta = {"site_code": f"{area}-{season}", "site": f"{AREAS[area]['name']} ({'spring' if season == 'sp' else 'summer'})",
            "year": year, "season": season, "scene_date": rep["datetime"][:10], "scene_id": item.id,
            "processing_baseline": item.properties.get("s2:processing_baseline"),
            "detector_version": pipeline.DETECTOR_VERSION, "crs": str(crs),
            "transform": list(transform), "bands": order, "role": "positive-control",
            "clear_fraction": quality["clear_fraction"], "haze_fraction": quality["haze_frac"],
            "hot_fraction": quality["hot_frac"], "passes_gates": pipeline.scene_passes(quality)}
    if not meta["passes_gates"]:
        return meta, None
    valid = quality["valid"]
    ndvi = detect.ndvi(stack[:, :, order.index("red")], stack[:, :, order.index("nir")])
    masks = {"ndvi": pipeline.ndvi_bare(stack, order, ndvi, valid, inside),
             "rules": pipeline.rules_bare(stack, order, valid, inside),
             "kmeans": pipeline.kmeans_bare(stack, order, valid, inside),
             "gng": pipeline.gng_bare(stack, order, valid, inside)}
    masks["v4"] = v4.filter_mask(masks["gng"], stack, inside, order=tuple(order))
    meta["areas_ha"] = {m: round(geo.mask_area_ha(k, transform), 2) for m, k in masks.items()}
    meta["clear_ha"] = round(geo.mask_area_ha(valid, transform), 1)
    arrays = {"stack": stack, "valid": valid, "inside": inside, "transform": np.array(list(transform)),
              **{f"prediction_{m}": masks[m] for m in ("ndvi", "rules", "kmeans", "gng")}}
    return meta, arrays


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--areas", nargs="+", default=list(AREAS))
    args = ap.parse_args()
    scenes = OUT / "scenes"
    scenes.mkdir(parents=True, exist_ok=True)
    summary_f = OUT / "areas.json"
    summary = json.loads(summary_f.read_text()) if summary_f.exists() else {"areas": AREAS, "runs": {}}
    for area in args.areas:
        bbox = AREAS[area]["bbox"]
        shape, transform, crs = grid(bbox)
        for year in YEARS:
            for season in SEASONS:
                key = f"{area}-{season}_{year}"
                if key in summary["runs"]:
                    continue
                try:
                    meta, arrays = run_scene(area, bbox, shape, transform, crs, year, season)
                except Exception as e:  # one bad scene must not stop the series
                    meta, arrays = {"error": f"{type(e).__name__}: {e}"}, None
                summary["runs"][key] = meta or {"status": "no clear scene"}
                if arrays is not None:
                    np.savez_compressed(scenes / f"{key}.npz", **arrays)
                    (scenes / f"{key}.json").write_text(json.dumps(meta, indent=2))
                summary_f.write_text(json.dumps(summary, indent=2))
                areas = (meta or {}).get("areas_ha", {})
                print(f"{key}: {(meta or {}).get('scene_date', '-')} GNG {areas.get('gng')} ha, "
                      f"v4 {areas.get('v4')} ha, NDVI {areas.get('ndvi')} ha", flush=True)


if __name__ == "__main__":
    main()
