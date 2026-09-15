"""Step 13 — three-method comparison + sensitivity analysis (thesis tables).

Comparison: NDVI threshold vs K-means (fixed k, cluster-labelled) vs GNG
(adaptive prototypes + per-pixel refinement) on control cases where we
know the right answer:

  * Monivea 2022      — documented cutting; reference area is not known
  * Rosroe 2018       — open lough, no cutting (should detect ~0)
  * Moorfield 2025    — spring burn scar (should detect ~0-3, not 60)
  * Namucka 2026      — known active cutting
  * Kilnaborris 2024  — small subtle patches
  * Derrinlough 2024  — quiet site

K-means uses the same 9-D standardized features and the same physical
cluster-labelling rule as GNG's prototypes, so the comparison isolates
the clustering strategy itself (fixed k + batch vs grown prototypes).

Sensitivity: GNG area under NDVI-threshold {0.20,0.25,0.30}, seeds
{7,42,123}, node budgets {40,80,120} on three sites.

Outputs: a new timestamped evaluation directory with frozen scene arrays and JSON reports.
"""

import sys
import argparse
from datetime import datetime, timezone
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import geopandas as gpd
from sklearn.cluster import KMeans  # import before timing either clustering method

from peatland import boundaries, imagery, geo, config, preprocess, pipeline, detect

CASES = [  # (site-name fragment, year, expectation)
    ("Monivea",     2022, "documented heavy cutting"),
    ("Rosroe",      2018, "open lough, no cutting"),
    ("Moorfield",   2025, "spring burn scar"),
    ("Namucka",     2026, "known active cutting"),
    ("Kilnaborris", 2024, "small subtle patches"),
    ("Derrinlough", 2024, "quiet midland site"),
]


RELEASE_CACHE = config.OUT_DIR / "cache-release-v2"


def release_record(code, cache_dir):
    """The published release's cached record for a site: the exact scene IDs,
    grid and tidal-water mask behind the numbers on the map."""
    f = Path(cache_dir) / f"{code}.json"
    if not f.exists():
        raise ValueError(f"No release cache for site {code} in {cache_dir}")
    record = json.loads(f.read_text())
    if record.get("detector_version") != pipeline.DETECTOR_VERSION:
        raise ValueError(f"{f} was built by {record.get('detector_version')!r}; "
                         f"the current detector is {pipeline.DETECTOR_VERSION!r}")
    return record


def kmeans_bare(stack, order, valid, inside, k=6, seed=42):
    """Compatibility wrapper: shared rules and preprocessing with GNG."""
    return pipeline.kmeans_bare(stack, order, valid, inside, k=k, seed=seed)


def load_case(sites, name, year, with_metadata=False, cache_dir=RELEASE_CACHE):
    """Re-read exactly the scene the published release used for this site-year,
    on the release's grid, with the release's tidal-water exclusion applied, so
    the evaluated predictions are the ones behind the published numbers."""
    import rasterio
    matches = sites[sites.SITE_NAME.str.contains(name, regex=False)]
    if len(matches) != 1:
        raise ValueError(f"Expected one site for {name!r}, found {len(matches)}")
    row = matches.iloc[0]
    record = release_record(row["SITECODE"], cache_dir)
    if year not in record["years"]:
        raise ValueError(f"{name}: {year} is not in the release (years {record['years']})")
    k = record["years"].index(year)
    day, scene_id = record["dates"][k], record["scene_ids"][k]
    shape = tuple(record["grid"]["shape"])
    transform = rasterio.Affine(*record["grid"]["transform"][:6])
    crs = record["grid"]["crs"]
    bbox = boundaries.site_bbox_wgs84(sites, row.name, buffer_m=200)
    geom = gpd.GeoSeries([row.geometry], crs=config.ITM).to_crs(crs).iloc[0]
    site_inside = geo.polygon_mask(geom, shape, transform)
    tidal = (pipeline.mask_from_b64(record["tidal_mask_b64"], shape)
             if record.get("tidal_mask_b64") else np.zeros(shape, dtype=bool))
    inside = site_inside & ~tidal
    item = imagery.open_catalog("planetary").get_collection(
        imagery.PROVIDERS["planetary"]["collection"]).get_item(scene_id)
    if item is None:
        raise ValueError(f"Release scene is unavailable: {scene_id}")
    rep = preprocess.assess_scene(item, bbox, "planetary", shape, aoi_mask=site_inside)
    order = list(pipeline.BANDS)
    stack = np.stack([imagery.read_window(item, b, bbox, "planetary", out_shape=shape)[0]
                      .astype("float32") for b in order], -1)
    quality = pipeline.scene_quality(stack, order, rep["scl"], inside, item)
    if not pipeline.scene_passes(quality):
        raise ValueError(f"Scene rejected by shared quality gates: {name} {day}")
    metadata = {"site_code": row["SITECODE"], "site": row["SITE_NAME"],
                "year": year, "scene_date": day, "scene_id": item.id,
                "processing_baseline": item.properties.get("s2:processing_baseline"),
                "detector_version": pipeline.DETECTOR_VERSION,
                "clear_fraction": quality["clear_fraction"],
                "haze_fraction": quality["haze_frac"], "hot_fraction": quality["hot_frac"],
                "cloud_fraction": quality["cloud_frac"], "crs": str(crs),
                "transform": list(transform), "bands": order,
                "tidal_excluded_px": int((tidal & site_inside).sum()),
                "scene_selection": f"Exact scene of the published release "
                                   f"({Path(cache_dir).name}); tidal-water exclusion applied"}
    result = (stack, order, quality["valid"], inside, transform)
    return (*result, metadata) if with_metadata else result


def main():
    parser = argparse.ArgumentParser(description="Four-method comparison on identical scenes; no accuracy claims without labels")
    parser.add_argument("--case", nargs=2, action="append", metavar=("SITE", "YEAR"))
    parser.add_argument("--output-dir", type=Path, default=config.OUT_DIR / "evaluation" /
                        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--from-run", type=Path, help="Reuse frozen scene bundles without network access")
    parser.add_argument("--cache-dir", type=Path, default=RELEASE_CACHE,
                        help="Release cache whose exact scenes, grid and tidal mask are evaluated")
    parser.add_argument("--held-out", action="append", default=[], metavar="SITE",
                        help="Site-name fragment never inspected while tuning thresholds")
    args = parser.parse_args()
    cases = [(n, int(y), "user-selected control") for n, y in args.case] if args.case else CASES
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sites = None if args.from_run else boundaries.build_sites()
    out = {"detector_version": pipeline.DETECTOR_VERSION, "comparison": [], "sensitivity": {},
           "interpretation": "Area and runtime comparison, not accuracy. Frozen dates were selected by an earlier GNG run."}
    if args.from_run and not args.case:
        cases = [(json.loads(f.with_suffix(".json").read_text())["site"],
                  json.loads(f.with_suffix(".json").read_text())["year"], "frozen control")
                 for f in sorted(args.from_run.glob("*.npz"))]
        if not cases:
            parser.error("No frozen scenes found in --from-run")
    for name, year, expect in cases:
        if args.from_run:
            import rasterio
            matches = []
            for f in sorted(args.from_run.glob("*.npz")):
                meta = json.loads(f.with_suffix(".json").read_text())
                if name in meta["site"] and int(meta["year"]) == year:
                    matches.append((f, meta))
            if len(matches) != 1:
                raise ValueError(f"Expected one frozen scene for {name} {year}")
            bundle, metadata = matches[0]
            with np.load(bundle, allow_pickle=False) as arrays:
                stack, valid, inside = arrays["stack"], arrays["valid"], arrays["inside"]
            order = metadata["bands"]
            transform = rasterio.Affine(*metadata["transform"][:6])
            metadata["detector_version"] = pipeline.DETECTOR_VERSION
        else:
            stack, order, valid, inside, transform, metadata = load_case(
                sites, name, year, with_metadata=True, cache_dir=args.cache_dir)
            metadata["role"] = "held-out" if name in args.held_out else "calibration"
        ndvi_arr = detect.ndvi(stack[:, :, order.index("red")], stack[:, :, order.index("nir")])
        methods = {
            "ndvi": lambda: pipeline.ndvi_bare(stack, order, ndvi_arr, valid, inside),
            "rules": lambda: pipeline.rules_bare(stack, order, valid, inside),
            "kmeans": lambda: pipeline.kmeans_bare(stack, order, valid, inside),
            "gng": lambda: pipeline.gng_bare(stack, order, valid, inside),
        }
        row = {"case": f"{name} {year}", "context": expect, **metadata}
        masks = {}
        for method, run in methods.items():
            start = time.perf_counter(); masks[method] = run()
            row[f"t_{method}_s"] = round(time.perf_counter() - start, 4)
            row[f"{method}_ha"] = round(geo.mask_area_ha(masks[method], transform), 4)
        row["gng_rules_disagreement_pixels"] = int((masks["gng"] != masks["rules"]).sum())
        row["gng_kmeans_disagreement_pixels"] = int((masks["gng"] != masks["kmeans"]).sum())
        stem = f"{metadata['site_code']}_{year}"
        np.savez_compressed(args.output_dir / f"{stem}.npz", stack=stack, valid=valid,
                            inside=inside, transform=np.array(list(transform)),
                            **{f"prediction_{k}": v for k, v in masks.items()})
        (args.output_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
        out["comparison"].append(row)
        print(json.dumps(row, indent=2, allow_nan=False), flush=True)
        if not args.skip_sensitivity:
            area = lambda **kw: round(geo.mask_area_ha(
                pipeline.gng_bare(stack, order, valid, inside, **kw), transform), 4)
            out["sensitivity"][row["case"]] = {
                "ndvi_thresh": {str(t): area(thresh=t) for t in (0.20, 0.25, 0.30)},
                "seed": {str(s): area(seed=s) for s in (7, 42, 123)},
                "max_nodes": {str(n): area(max_nodes=n) for n in (40, 80, 120)},
                "adaptive_contrast": {str(a): area(adaptive=a) for a in (False, True)},
            }
        (args.output_dir / "method_comparison.json").write_text(json.dumps(out, indent=2, allow_nan=False))
    print(f"Saved frozen scenes, predictions and report to {args.output_dir}")


if __name__ == "__main__":
    main()
