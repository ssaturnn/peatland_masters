"""Step 19 — agreement between two annotators on the same blind sample.

Compares the primary labels (the blind labels.csv) with a second,
independent annotation of the same points, given as JSON Lines with
{"n" or "point_id", "label", "confidence", "reason"} per point, where n is
the 1-based row number in labels.csv (the tool's "point n of N").

Reports raw agreement, Cohen's kappa over all classes and for the binary
bare-peat decision the accuracy metrics depend on, the confusion table, and
every disagreement for adjudication. Points either annotator marked unsure
are listed separately and left out of kappa, as in the accuracy scoring.
The primary labels are never modified: adjudication is done in the tool.

Usage:
    python3 scripts/19_compare_annotators.py <annotation>/labels.csv second.jsonl --output report.md
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

CLASSES = ["bare_peat", "vegetated", "water", "burn", "other", "unsure"]


def cohen_kappa(pairs):
    """Cohen's kappa for a list of (a, b) label pairs; None if undefined."""
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def load_second(path, rows):
    by_id = {r["point_id"]: r for r in rows}
    out = {}
    decoder = json.JSONDecoder()
    records = []
    for line in Path(path).read_text().splitlines():
        # tolerate several objects on one line (files joined without newlines)
        pos, line = 0, line.strip()
        while (start := line.find("{", pos)) != -1:
            rec, pos = decoder.raw_decode(line, start)
            records.append(rec)
    for rec in records:
        pid = rec.get("point_id")
        if pid is None and "n" in rec:
            n = int(rec["n"])
            pid = rows[n - 1]["point_id"] if 1 <= n <= len(rows) else None
        label = str(rec.get("label", "")).strip().lower()
        if pid in by_id and label in CLASSES:
            out[pid] = rec | {"label": label}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", type=Path, help="primary blind labels.csv")
    ap.add_argument("second", type=Path, help="second annotator, JSON Lines")
    ap.add_argument("--output", type=Path, help="markdown report (optional)")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.labels.open(newline="", encoding="utf-8-sig")))
    number = {r["point_id"]: i for i, r in enumerate(rows, 1)}
    second = load_second(args.second, rows)
    both = [(r, second[r["point_id"]]) for r in rows
            if r["truth"].strip() and r["point_id"] in second]
    firm = [(r, s) for r, s in both if "unsure" not in (r["truth"], s["label"])]
    pairs = [(r["truth"], s["label"]) for r, s in firm]
    binary = [(a == "bare_peat", b == "bare_peat") for a, b in pairs]

    k_all, k_bin = cohen_kappa(pairs), cohen_kappa(binary)
    agree = sum(a == b for a, b in pairs)
    lines = [f"# Annotator agreement ({len(both)} points labelled by both)", "",
             f"- decisive points (neither unsure): **{len(firm)}**",
             f"- raw agreement: **{agree}/{len(firm)}**"
             + (f" ({100 * agree / len(firm):.0f}%)" if firm else ""),
             f"- Cohen's kappa, all classes: **{'n/a' if k_all is None else f'{k_all:.2f}'}**",
             f"- Cohen's kappa, bare peat vs not: **{'n/a' if k_bin is None else f'{k_bin:.2f}'}**",
             f"- unsure by either annotator: {len(both) - len(firm)}", "",
             "## Confusion (rows = primary, columns = second)", "",
             "| primary \\ second | " + " | ".join(CLASSES) + " |",
             "|---|" + "---|" * len(CLASSES)]
    table = Counter((r["truth"], s["label"]) for r, s in both)
    for a in CLASSES:
        lines.append(f"| {a} | " + " | ".join(str(table[(a, b)] or "") for b in CLASSES) + " |")
    lines += ["", "## Disagreements to re-check in the tool", "",
              "| point | site | date | primary | second | confidence | second's reason |",
              "|---|---|---|---|---|---|---|"]
    for r, s in both:
        if r["truth"] != s["label"]:
            lines.append(f"| {number[r['point_id']]} | {r['site']} | {r['scene_date']} | "
                         f"{r['truth']} | {s['label']} | {s.get('confidence', '')} | "
                         f"{str(s.get('reason', '')).replace('|', '/')} |")
    report = "\n".join(lines) + "\n"
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)


if __name__ == "__main__":
    main()
