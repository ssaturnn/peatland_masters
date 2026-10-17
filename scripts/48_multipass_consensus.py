"""Step 48 — merge three or more blind annotation passes into reference labels.

Generalises step 27 from two passes to any number. Every pass labels every
point of the reference sample from the same image chips, without seeing the
other passes or any detector prediction. This script:

- takes the majority label of each point. A point is "agree" when at least
  `--agree` passes give the same label, "majority" when a plurality of at least
  half the passes does, and "disagree" otherwise (proposed as unsure). An unsure
  vote counts as an abstention: the label is taken among the other votes;
- writes the result into labels.csv for every point the author has not
  confirmed, keeping the pending mark, so step 28 still reports provenance.
  The notes start with "[model agree", "[model majority" or "[model disagree",
  and step 28 repeats its scoring without the disagreements;
- reports agreement: Fleiss' kappa over all passes, Cohen's kappa for every pair
  of passes, both on all classes and on bare peat against the rest, and how many
  points the new consensus changes against the earlier two-pass proposal;
- writes model-pass/review-multipass.md with the contested points.

Detector predictions are never read here.

Usage:
    python3 scripts/48_multipass_consensus.py outputs/evaluation/2026-10-08-v5-test
"""

import argparse
import csv
import importlib.util
import json
import shutil
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("step27", ROOT / "scripts" / "27_model_prelabels.py")
step27 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step27)
load_jsonl, cohen_kappa, write_csv_atomic, fmt = (step27.load_jsonl, step27.cohen_kappa,
                                                  step27.write_csv_atomic, step27.fmt)
PENDING_MARK = step27.PENDING_MARK


def fleiss_kappa(table):
    """Fleiss' kappa for rows of per-class counts with equal raters per row."""
    rows = [r for r in table if sum(r.values()) > 1]
    if not rows:
        return None
    raters = sum(rows[0].values())
    rows = [r for r in rows if sum(r.values()) == raters]
    n = len(rows)
    classes = set().union(*rows)
    p_j = {c: sum(r.get(c, 0) for r in rows) / (n * raters) for c in classes}
    p_i = [(sum(v * v for v in r.values()) - raters) / (raters * (raters - 1)) for r in rows]
    p_bar, p_e = sum(p_i) / n, sum(v * v for v in p_j.values())
    return None if p_e == 1 else (p_bar - p_e) / (1 - p_e)


def binary(label):
    return "bare" if label == "bare_peat" else "not"


def vote(records, agree_at):
    """(label, status, counts) from one point's pass records."""
    counts = Counter(r["label"] for r in records)
    decided = Counter({k: v for k, v in counts.items() if k != "unsure"})
    if not decided:
        return "unsure", "disagree", counts
    (top, top_n), *rest = decided.most_common()
    tied = rest and rest[0][1] == top_n
    if not tied and top_n >= agree_at:
        return top, "agree", counts
    if not tied and top_n * 2 >= len(records):
        return top, "majority", counts
    return "unsure", "disagree", counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="frozen evaluation run with annotation/")
    ap.add_argument("--passes", default="ABCDE", help="pass letters to merge")
    ap.add_argument("--agree", type=int, default=4, help="votes needed for 'agree'")
    ap.add_argument("--dry-run", action="store_true", help="report only, leave labels.csv")
    args = ap.parse_args()
    ann = args.run / "annotation"
    folder = ann / "model-pass"
    labels_path = ann / "labels.csv"
    with labels_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields, rows = list(reader.fieldnames), list(reader)
    if any(k.startswith("prediction_") for k in fields):
        raise SystemExit("labels.csv exposes predictions: use the blind file")

    passes = {p: load_jsonl(sorted(folder.glob(f"pass{p}_batch*.jsonl"))) for p in args.passes}
    for p, recs in passes.items():
        missing = len(rows) - len(recs)
        if missing:
            raise SystemExit(f"pass {p} is missing {missing} of {len(rows)} points")
    source = (f"model consensus, {len(passes)} blind passes "
              f"({PENDING_MARK}, pending review)")

    table, table_bin, results = [], [], []
    statuses, proposed, changed = Counter(), Counter(), []
    for n, row in enumerate(rows, 1):
        recs = [passes[p][n] for p in passes]
        if any(r["point_id"] != row["point_id"] for r in recs):
            raise SystemExit(f"point {n}: pass records do not match labels.csv")
        label, status, counts = vote(recs, args.agree)
        table.append(counts)
        table_bin.append(Counter(binary(r["label"]) for r in recs))
        confirmed = bool(row["truth"]) and PENDING_MARK not in (row.get("reference_source") or "")
        results.append((n, row, label, status, counts, recs))
        if confirmed:
            continue
        statuses[status] += 1
        proposed[label] += 1
        before = row["truth"]
        if binary(before) != binary(label):
            changed.append((n, row, before, label, counts))
        votes = " ".join(f"{p}:{r['label']}" for p, r in zip(passes, recs))
        tally = ", ".join(f"{k} {v}" for k, v in counts.most_common())
        reasons = " | ".join(f"{p}: {str(r.get('reason', '')).strip()}"
                             for p, r in zip(passes, recs))
        row.update(truth=label, reference_source=source, reference_date=row["scene_date"],
                   notes=f"[model {status}, {tally}] {votes} || {reasons}"[:900])

    pair_stats = {}
    for p, q in combinations(passes, 2):
        pairs = [(passes[p][n]["label"], passes[q][n]["label"]) for n in range(1, len(rows) + 1)]
        pair_stats[f"{p}-{q}"] = {
            "raw": sum(x == y for x, y in pairs) / len(pairs),
            "kappa": cohen_kappa(pairs),
            "kappa_bare_vs_not": cohen_kappa([(binary(x), binary(y)) for x, y in pairs])}
    stats = {
        "passes": list(passes), "points": len(rows),
        "fleiss_kappa": fleiss_kappa(table),
        "fleiss_kappa_bare_vs_not": fleiss_kappa(table_bin),
        "unanimous": sum(1 for c in table if len(c) == 1),
        "status": dict(statuses), "proposed": dict(proposed),
        "changed_bare_vs_not_against_previous": len(changed),
        "pairs": pair_stats}

    names = list(passes)
    lines = ["# Multi-pass consensus", "",
             f"Generated {datetime.now():%Y-%m-%d %H:%M}. {len(passes)} blind passes "
             f"({', '.join(names)}) over {len(rows)} points. 'Agree' needs {args.agree} "
             f"matching votes; unsure votes abstain.", "",
             f"Fleiss' kappa {fmt(stats['fleiss_kappa'])} (all classes), "
             f"{fmt(stats['fleiss_kappa_bare_vs_not'])} (bare peat against the rest). "
             f"Unanimous points: {stats['unanimous']} of {len(rows)}.", "",
             "Status: " + ", ".join(f"{k} {v}" for k, v in statuses.most_common())
             + ". Proposed: " + ", ".join(f"{k} {v}" for k, v in proposed.most_common()) + ".", "",
             "## Pairwise agreement", "",
             "| Pair | Raw | Cohen's kappa | Kappa, bare vs not |", "|---|---:|---:|---:|"]
    lines += [f"| {k} | {fmt(v['raw'])} | {fmt(v['kappa'])} | {fmt(v['kappa_bare_vs_not'])} |"
              for k, v in pair_stats.items()]
    lines += ["", "## Points whose bare-peat status changed against the previous proposal", "",
              "| # | Site | Date | Before | Now | Votes |", "|---:|---|---|---|---|---|"]
    lines += [f"| {n} | {r['site']} | {r['scene_date']} | {b} | {a} | "
              f"{', '.join(f'{k} {v}' for k, v in c.most_common())} |"
              for n, r, b, a, c in changed] or ["| – | none | | | | |"]
    lines += ["", "## Contested points (majority or disagree)", "",
              "| # | Site | Date | Proposed | Status | " + " | ".join(names) + " |",
              "|---:|---|---|---|---|" + "---|" * len(names)]
    lines += [f"| {n} | {r['site']} | {r['scene_date']} | {lab} | {st} | "
              + " | ".join(x["label"] for x in recs) + " |"
              for n, r, lab, st, c, recs in results if st != "agree"] or ["| – | none |"]
    (folder / "review-multipass.md").write_text("\n".join(lines) + "\n")
    (folder / "agreement-multipass.json").write_text(json.dumps(stats, indent=2))

    if not args.dry_run:
        backup = labels_path.with_name(
            f"labels.backup-{datetime.now():%Y%m%d-%H%M%S}-before-multipass.csv")
        shutil.copy2(labels_path, backup)
        write_csv_atomic(labels_path, fields, rows)
        print(f"labels.csv updated; backup {backup.name}")
    print(json.dumps({k: v for k, v in stats.items() if k != "pairs"}, indent=2))
    print(f"review sheet {folder / 'review-multipass.md'}")


if __name__ == "__main__":
    main()
