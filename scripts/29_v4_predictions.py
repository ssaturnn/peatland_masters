"""Step 29 — detector v4 predictions on a frozen evaluation run.

Applies the v4 post-filters (src/peatland/v4.py) to the frozen rules, K-means
and GNG masks of every site-year in a run. Writes:

- a copy of annotation/accuracy_points.csv with `prediction_v4_rules`,
  `prediction_v4_kmeans` and `prediction_v4_gng` columns;
- v4_areas.json, the candidate area per site-year under v3 and v4.

The NDVI baseline is left as it is: it is the naive reference, not a detector
under development.

Usage:
    python3 scripts/29_v4_predictions.py outputs/evaluation/2026-09-13-v3-sample
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from peatland import v4

METHODS = ("rules", "kmeans", "gng")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="frozen evaluation run")
    args = ap.parse_args()
    ann = args.run / "annotation"
    with (ann / "accuracy_points.csv").open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields, rows = list(reader.fieldnames), list(reader)

    masks, areas = {}, {}
    for npz in sorted(args.run.glob("*.npz")):
        meta = json.loads(npz.with_suffix(".json").read_text())
        order = tuple(meta["bands"])
        with np.load(npz) as z:
            stack, inside = z["stack"], z["inside"].astype(bool)
            cell_ha = abs(z["transform"][0] * z["transform"][4]) / 10000
            masks[npz.stem] = {}
            areas[npz.stem] = {"site": meta["site"], "year": meta["year"], "role": meta.get("role")}
            for m in METHODS:
                v3 = z[f"prediction_{m}"].astype(bool)
                out = v4.filter_mask(v3, stack, inside, order=order)
                masks[npz.stem][m] = out
                areas[npz.stem][f"{m}_v3_ha"] = round(float(v3.sum() * cell_ha), 2)
                areas[npz.stem][f"{m}_v4_ha"] = round(float(out.sum() * cell_ha), 2)

    for r in rows:
        stem = r["point_id"].rsplit("_r", 1)[0]
        row, col = int(r["pixel_row"]), int(r["pixel_col"])
        for m in METHODS:
            r[f"prediction_v4_{m}"] = str(int(masks[stem][m][row, col]))
    new_fields = fields + [f"prediction_v4_{m}" for m in METHODS if f"prediction_v4_{m}" not in fields]
    out_csv = ann / "accuracy_points_v4.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=new_fields)
        w.writeheader()
        w.writerows(rows)
    (args.run / "v4_areas.json").write_text(json.dumps(
        {"version": v4.VERSION, "boundary_px": v4.BOUNDARY_PX, "mndwi_max": v4.MNDWI_MAX,
         "site_years": areas}, indent=2))
    for stem, a in areas.items():
        print(f"{a['site'][:28]:28} {a['year']} ({a['role']}): GNG {a['gng_v3_ha']} -> {a['gng_v4_ha']} ha")
    print(f"points with v4 predictions -> {out_csv}")


if __name__ == "__main__":
    main()
