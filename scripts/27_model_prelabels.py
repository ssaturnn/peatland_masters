"""Step 27 — merge independent model annotation passes into review pre-labels.

Two blind model passes label every point of the reference sample from the same
image chips the annotator sees (PlanetScope 3 m first where it exists). This
script:

- writes the agreed proposal into labels.csv for every point still awaiting
  review, marked as a second-annotator pre-label, so the labelling tool shows it
  as "to review". Labels the author has confirmed are never changed;
- writes a review sheet: points where the passes disagree, and confirmed labels
  the passes contradict, so the author can re-check them first;
- reports agreement (raw and Cohen's kappa) between the passes, with the
  author's confirmed labels and with the earlier second annotation.

Pre-labels are proposals. Only labels the author confirms in the tool are scored.
Detector predictions are never read here.

Usage:
    python3 scripts/27_model_prelabels.py outputs/evaluation/2026-09-13-v3-sample
"""

import argparse
import csv
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

CLASSES = ("bare_peat", "vegetated", "water", "burn", "other", "unsure")
CONFIDENCE = {"high": 3, "medium": 2, "low": 1}
PENDING_MARK = "second annotator"   # the labelling tool shows these as "to review"
SOURCE = "model pre-label, two blind passes (second annotator, pending review)"


def load_jsonl(paths):
    """Records keyed by 1-based point number; tolerates objects joined on one line."""
    decoder, out = json.JSONDecoder(), {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            pos, line = 0, line.strip()
            while (start := line.find("{", pos)) != -1:
                rec, pos = decoder.raw_decode(line, start)
                label = str(rec.get("label", "")).strip().lower()
                if label in CLASSES and "n" in rec:
                    out[int(rec["n"])] = rec | {"label": label}
    return out


def consensus(a, b, earlier=None):
    """Proposal from two passes: (label, confidence, status).

    Agreement keeps the label at the lower of the two confidences. On a
    disagreement the more confident pass wins, an unsure pass defers to the
    other, and a tie is broken by the earlier annotation when it sides with
    one pass; otherwise the point is proposed as unsure. Every disagreement
    is reported for review.
    """
    if a is None and b is None:
        return None, None, "missing"
    if a is None or b is None:
        one = a or b
        return one["label"], "low", "single pass"
    conf = lambda r: CONFIDENCE.get(str(r.get("confidence", "")).lower(), 1)
    if a["label"] == b["label"]:
        level = min(conf(a), conf(b))
        return a["label"], next(k for k, v in CONFIDENCE.items() if v == level), "agree"
    if "unsure" in (a["label"], b["label"]):
        other = b if a["label"] == "unsure" else a
        return other["label"], "low", "disagree"
    if conf(a) != conf(b):
        return max(a, b, key=conf)["label"], "low", "disagree"
    if earlier and earlier["label"] in (a["label"], b["label"]):
        return earlier["label"], "low", "disagree"
    return "unsure", "low", "disagree"


def cohen_kappa(pairs):
    """Cohen's kappa for (label_1, label_2) pairs; None when undefined."""
    pairs = [(x, y) for x, y in pairs if x and y]
    n = len(pairs)
    if not n:
        return None
    observed = sum(x == y for x, y in pairs) / n
    first, second = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    expected = sum(first[c] * second[c] for c in set(first) | set(second)) / n ** 2
    return None if expected == 1 else (observed - expected) / (1 - expected)


def agreement(pairs):
    pairs = [(x, y) for x, y in pairs if x and y]
    raw = sum(x == y for x, y in pairs) / len(pairs) if pairs else None
    bare = [("bare" if x == "bare_peat" else "not", "bare" if y == "bare_peat" else "not")
            for x, y in pairs]
    return {"n": len(pairs), "raw": raw, "kappa": cohen_kappa(pairs),
            "kappa_bare_vs_not": cohen_kappa(bare)}


def note(status, conf, a, b, earlier):
    """Compact review note: status first, so it survives truncation."""
    part = lambda tag, r: (f"{tag}: {r['label']}/{r.get('confidence', '?')} — "
                           f"{str(r.get('reason', '')).strip()}") if r else f"{tag}: none"
    text = f"[model {status}, {conf}] {part('A', a)} | {part('B', b)}"
    if earlier:
        text += f" | earlier annotator: {earlier['label']}"
    return text[:480]


def write_csv_atomic(path, fields, rows):
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def fmt(x, digits=2):
    return "–" if x is None else f"{x:.{digits}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="frozen evaluation run with annotation/")
    ap.add_argument("--dry-run", action="store_true", help="write the review sheet only")
    args = ap.parse_args()
    ann = args.run / "annotation"
    passes = ann / "model-pass"
    labels_path = ann / "labels.csv"
    with labels_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields, rows = list(reader.fieldnames), list(reader)
    if any(k.startswith("prediction_") for k in fields):
        raise SystemExit("labels.csv exposes predictions: use the blind file")

    pass_a = load_jsonl(sorted(passes.glob("passA_batch*.jsonl")))
    pass_b = load_jsonl(sorted(passes.glob("passB_batch*.jsonl")))
    earlier = load_jsonl(sorted((args.run / "second-annotation").glob("labels-*.jsonl")))

    sheet, rechecks, disagreements = [], [], []
    proposed, statuses = Counter(), Counter()
    pairs_ab, pairs_author, pairs_earlier = [], [], []
    for n, row in enumerate(rows, 1):
        a, b, e = pass_a.get(n), pass_b.get(n), earlier.get(n)
        label, conf, status = consensus(a, b, e)
        pairs_ab.append((a and a["label"], b and b["label"]))
        pairs_earlier.append((label if status == "agree" else None, e and e["label"]))
        confirmed = bool(row["truth"]) and PENDING_MARK not in (row.get("reference_source") or "")
        if confirmed:
            if status == "agree":
                pairs_author.append((row["truth"], label))
                if label != row["truth"]:
                    rechecks.append((n, row, label, a, b))
            continue
        if label is None:
            continue
        statuses[status] += 1
        proposed[label] += 1
        if status != "agree":
            disagreements.append((n, row, label, a, b, e))
        row.update(truth=label, reference_source=SOURCE, reference_date=row["scene_date"],
                   notes=note(status, conf, a, b, e))

    stats = {"passes_A_vs_B": agreement(pairs_ab),
             "model_consensus_vs_author_confirmed": agreement(pairs_author),
             "model_consensus_vs_earlier_annotator": agreement(pairs_earlier)}
    lines = [
        "# Model pre-labels for review",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}. Pass A: {len(pass_a)} points, pass B: "
        f"{len(pass_b)} points, earlier annotator: {len(earlier)} points.",
        "",
        "Pre-labels are proposals for the points still awaiting review. Labels the author "
        "confirmed are unchanged. Only author-confirmed labels are scored.",
        "",
        "## Agreement",
        "",
        "| Comparison | Points | Raw | Cohen's kappa | Kappa, bare peat vs not |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, s in stats.items():
        lines.append(f"| {name.replace('_', ' ')} | {s['n']} | {fmt(s['raw'])} | "
                     f"{fmt(s['kappa'])} | {fmt(s['kappa_bare_vs_not'])} |")
    lines += ["", f"Proposals written: {sum(proposed.values())} — "
              + ", ".join(f"{k} {v}" for k, v in proposed.most_common())
              + " · " + ", ".join(f"{k} {v}" for k, v in statuses.most_common()), ""]
    lines += ["## 1. Re-check first: your confirmed labels that both passes contradict", "",
              "Both passes agree with each other and differ from your label. Open with `G` and the number.", "",
              "| # | Site | Date | Your label | Both passes | Pass A says |", "|---:|---|---|---|---|---|"]
    lines += [f"| {n} | {r['site']} | {r['scene_date']} | {r['truth']} | {lab} | "
              f"{str(a.get('reason', '')).replace('|', '/')} |" for n, r, lab, a, b in rechecks] or ["| – | none | | | | |"]
    lines += ["", "## 2. Pending points where the passes disagree", "",
              "| # | Site | Date | Proposed | Pass A | Pass B | Earlier |", "|---:|---|---|---|---|---|---|"]
    lines += [f"| {n} | {r['site']} | {r['scene_date']} | {lab} | {a and a['label']} | "
              f"{b and b['label']} | {e and e['label']} |" for n, r, lab, a, b, e in disagreements] \
        or ["| – | none | | | | | |"]
    (passes / "review.md").write_text("\n".join(lines) + "\n")
    (passes / "agreement.json").write_text(json.dumps(stats, indent=2))

    if not args.dry_run:
        backup = labels_path.with_name(f"labels.backup-{datetime.now():%Y%m%d-%H%M%S}-before-prelabels.csv")
        shutil.copy2(labels_path, backup)
        write_csv_atomic(labels_path, fields, rows)
        print(f"labels.csv updated; backup {backup.name}")
    print(json.dumps(stats, indent=2))
    print(f"proposals {sum(proposed.values())}: {dict(proposed)} | {dict(statuses)} | "
          f"re-check {len(rechecks)} confirmed labels | review sheet {passes / 'review.md'}")


if __name__ == "__main__":
    main()
