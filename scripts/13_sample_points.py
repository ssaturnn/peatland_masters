"""Sample frozen evaluation scenes without refetching imagery or changing predictions.

Use --run with the directory produced by 12_method_comparison.py. Outputs
accuracy_points.csv with predictions/design weights, a blind labels.csv for
independent annotation, and locations.geojson. Existing labels are never
replaced: each invocation requires a new output directory.
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio.transform
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from peatland import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--per-stratum", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--strata", choices=("gng", "agreement"), default="gng",
                        help="gng: detected/undetected by GNG. agreement: adds a "
                             "'disagreement' stratum (GNG=0 but another method=1), so the "
                             "pixels where the methods differ are sampled directly")
    parser.add_argument("--stratify-by", default="gng",
                        help="prediction whose detected/undetected split defines the strata "
                             "(gng, or v5 for bundles that store prediction_v5)")
    args = parser.parse_args()
    if args.per_stratum < 1:
        parser.error("--per-stratum must be positive")
    bundles = sorted(args.run.glob("*.npz"))
    if not bundles:
        parser.error("No frozen .npz scenes found in --run")
    out = args.output_dir or args.run / ("annotation-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    if out.exists():
        parser.error(f"Output already exists; choose a new directory: {out}")
    rng = np.random.default_rng(args.seed)
    rows, locations, sources = [], [], []
    for bundle in bundles:
        meta_path = bundle.with_suffix(".json")
        meta = json.loads(meta_path.read_text())
        sources.append({"file": bundle.name, "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                        "metadata_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest()})
        with np.load(bundle, allow_pickle=False) as data:
            valid = data["inside"] & data["valid"]
            predictions = {m: data[f"prediction_{m}"].astype(bool)
                           for m in ("ndvi", "rules", "kmeans", "gng", "v5")
                           if f"prediction_{m}" in data}
            transform = rasterio.Affine(*meta["transform"][:6])
            to_wgs = Transformer.from_crs(meta["crs"], config.WGS84, always_xy=True)
            gng = predictions[args.stratify_by]
            if args.strata == "agreement":
                others = predictions["ndvi"] | predictions["rules"] | predictions["kmeans"]
                strata = (("detected", valid & gng),
                          ("disagreement", valid & ~gng & others),
                          ("undetected", valid & ~gng & ~others))
            else:
                strata = (("detected", valid & gng), ("undetected", valid & ~gng))
            for stratum, mask in strata:
                ys, xs = np.nonzero(mask)
                count = min(args.per_stratum, len(ys))
                for i in rng.choice(len(ys), size=count, replace=False):
                    y, x = int(ys[i]), int(xs[i])
                    map_x, map_y = rasterio.transform.xy(transform, y, x)
                    lon, lat = to_wgs.transform(map_x, map_y)
                    # IDs do not reveal the detector's prediction to the annotator.
                    pid = f"{meta['site_code']}_{meta['year']}_r{y}_c{x}"
                    row = {"point_id": pid, "site": meta["site"], "year": meta["year"],
                           "scene_date": meta["scene_date"], "scene_id": meta["scene_id"],
                           "detector_version": meta["detector_version"],
                           "role": meta.get("role", ""),
                           "stratum": stratum, "stratum_population": len(ys),
                           "stratum_sample_size": count, "pixel_row": y, "pixel_col": x,
                           "lat": round(lat, 7), "lon": round(lon, 7),
                           **{f"prediction_{m}": int(p[y, x]) for m, p in predictions.items()},
                           "truth": "", "reference_source": "", "reference_date": "", "notes": ""}
                    rows.append(row)
                    locations.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
                                      "properties": {k: row[k] for k in ("point_id", "site", "year", "scene_date")}})
    if not rows:
        parser.error("Frozen scenes contain no valid pixels")
    out.mkdir(parents=True, exist_ok=False)
    with (out / "accuracy_points.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    blind_fields = ["point_id", "site", "year", "scene_date", "lat", "lon", "truth", "reference_source", "reference_date", "notes"]
    rng.shuffle(rows)
    with (out / "labels.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=blind_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    (out / "locations.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": locations}))
    (out / "manifest.json").write_text(json.dumps({"seed": args.seed, "per_stratum": args.per_stratum, "strata": args.strata,
                                                  "stratify_by": args.stratify_by,
                                                  "source_run": str(args.run.resolve()), "sources": sources}, indent=2))
    print(f"{len(rows)} points written to {out}")
    print("Label labels.csv independently: bare_peat / vegetated / water / burn / other / unsure.")
    print("Record reference imagery source/date. A bare surface alone does not establish active cutting.")


if __name__ == "__main__":
    main()
