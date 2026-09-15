"""Score independently labelled accuracy points; missing labels never count as negatives."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from peatland.evaluation import METHODS, score_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Labelled accuracy CSV")
    parser.add_argument("--output", type=Path, help="JSON report path (optional)")
    parser.add_argument("--labels", type=Path, help="Blind labels.csv to join by point_id")
    parser.add_argument("--methods", nargs="+", default=list(METHODS),
                        help="prediction_<method> columns to score (default: the four frozen methods)")
    parser.add_argument("--common-support", action="store_true",
                        help="after the label join, keep only points with a 0/1 prediction for "
                             "every scored method; population weights then no longer apply")
    args = parser.parse_args()
    try:
        with args.input.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if args.labels:
            with args.labels.open(newline="", encoding="utf-8-sig") as f:
                labels = list(csv.DictReader(f))
            by_id = {}
            for row in labels:
                pid = row.get("point_id", "")
                if not pid or pid in by_id:
                    raise ValueError(f"Missing or duplicate label point_id: {pid!r}")
                by_id[pid] = row
            if set(by_id) != {r.get("point_id") for r in rows}:
                raise ValueError("Label IDs must exactly match the sampled point IDs")
            for row in rows:
                label = by_id[row["point_id"]]
                for key in ("truth", "reference_source", "reference_date", "notes"):
                    row[key] = label.get(key, "")
        dropped = None
        if args.common_support:
            kept = [r for r in rows if all(str(r.get(f"prediction_{m}", "")) in {"0", "1"}
                                           for m in args.methods)]
            dropped, rows = len(rows) - len(kept), kept
        report = score_points(rows, method_names=tuple(args.methods))
        if dropped is not None:
            report["common_support"] = {
                "kept_points": len(rows), "dropped_points": dropped,
                "note": "Points without a prediction from every scored method were removed; "
                        "weighted estimates do not apply to this subset."}
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
