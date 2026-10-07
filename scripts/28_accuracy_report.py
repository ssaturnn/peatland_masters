"""Step 28 — accuracy report for the reference sample, with label provenance.

Joins the frozen sample (annotation/accuracy_points.csv, which carries the
predictions and strata) with the reference labels (annotation/labels.csv) and
scores NDVI, rules, K-means and GNG on all site-years, the held-out site-years
and the calibration site-years. For each it gives:

- sample metrics (precision, recall, F1 for the bare-peat class);
- population-weighted metrics (Olofsson et al., 2014);
- stratified bootstrap 95% intervals.

Points labelled unsure are left out, and each stratum's sample size is recounted
from the points that remain. This assumes the unsure points are missing at random
within their stratum.

The report states how many reference labels the author has confirmed and how
many are still model pre-labels awaiting review (scripts/27). While any remain,
the report is provisional. It repeats the scoring without the points the model
passes disagreed on, and on the author-confirmed points alone.

Usage:
    python3 scripts/28_accuracy_report.py outputs/evaluation/2026-09-13-v3-sample
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from peatland.evaluation import METHODS, score_points
from peatland.uncertainty import bootstrap_intervals

PENDING_MARK = "second annotator"
DISAGREE_MARK = "[model disagree"
SCOPES = {"all site-years": None, "held-out": "held-out", "calibration": "calibration",
          "test": "test", "industrial": "industrial", "protected": "protected", "external": "external"}
NAMES = {"ndvi": "NDVI threshold", "rules": "Spectral rules", "kmeans": "K-means", "gng": "GNG",
         "v4_rules": "Spectral rules v4", "v4_kmeans": "K-means v4", "v4_gng": "GNG v4",
         "v5": "SWIR peat v5"}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def join(points, labels):
    """Sample rows with the reference label, its provenance and the review note."""
    by_id = {r["point_id"]: r for r in labels}
    if set(by_id) != {p["point_id"] for p in points}:
        raise ValueError("labels.csv and accuracy_points.csv list different points")
    rows = []
    for p in points:
        label = by_id[p["point_id"]]
        truth = label.get("truth", "").strip().lower()
        pending = PENDING_MARK in (label.get("reference_source") or "")
        rows.append(p | {"truth": truth, "reference_source": label.get("reference_source", ""),
                         "notes": label.get("notes", ""),
                         "provenance": "none" if not truth else "model" if pending else "author"})
    return rows


def usable(rows):
    """Drop unsure or unlabelled points and recount each stratum's sample size."""
    kept = [dict(r) for r in rows if r["truth"] not in ("", "unsure")]
    sizes = Counter((r["site"], r["year"], r["stratum"]) for r in kept)
    for r in kept:
        r["stratum_sample_size"] = str(sizes[(r["site"], r["year"], r["stratum"])])
    return kept


def score(rows, replicates, methods=METHODS):
    report = score_points(rows, method_names=methods)
    intervals = (bootstrap_intervals(rows, method_names=methods, replicates=replicates)
                 if report["weighting_available"] else {"status": "unavailable"})
    return report, intervals


def error_breakdown(rows, method):
    """What the method's false positives are, and where its misses sit."""
    pred = lambda r: str(r.get(f"prediction_{method}")) == "1"
    fp = Counter(r["truth"] for r in rows if pred(r) and r["truth"] != "bare_peat")
    fn = Counter(f"{r['site']} {r['year']}" for r in rows if not pred(r) and r["truth"] == "bare_peat")
    per_site = defaultdict(Counter)
    for r in rows:
        key = "tp" if pred(r) and r["truth"] == "bare_peat" else "fp" if pred(r) else \
              "fn" if r["truth"] == "bare_peat" else "tn"
        per_site[(r["site"], r["year"], r["role"])][key] += 1
    return {"false_positive_truth": dict(fp), "misses_by_site_year": dict(fn),
            "per_site_year": {f"{s} {y} ({role})": dict(c) for (s, y, role), c in sorted(per_site.items())}}


def strict(rows, contested):
    """The strict reading: contested interior bands count as not bare peat."""
    return [r | {"truth": "vegetated"} if r["point_id"] in contested and r["truth"] == "bare_peat"
            else r for r in rows]


SUMMER_CHECK = ROOT / "outputs/evaluation/summer-check.json"


def summer_checked(rows, path=SUMMER_CHECK):
    """Bare-peat labels whose pixel is green the following summer count as vegetated.

    Exposed peat does not grow a full canopy within a season; winter-brown
    vegetation does (scripts/31_summer_check.py). Returns the relabelled rows
    and how many labels changed, or (None, 0) without a summer check.
    """
    if not path.exists():
        return None, 0
    data = json.loads(path.read_text())
    green = data["green_ndvi"]
    ndvi = {p["point_id"]: p["summer_ndvi"] for p in data["points"]}
    out, changed = [], 0
    for r in rows:
        v = ndvi.get(r["point_id"])
        if r["truth"] == "bare_peat" and v is not None and v >= green:
            out.append(r | {"truth": "vegetated"})
            changed += 1
        else:
            out.append(r)
    return out, changed


def sample_metrics(rows, method):
    """Precision, recall and F1 on the points where the method has a 0/1 value."""
    c = Counter()
    for r in rows:
        value = str(r.get(f"prediction_{method}", ""))
        if value not in ("0", "1") or r["truth"] in ("", "unsure"):
            continue
        pred, actual = value == "1", r["truth"] == "bare_peat"
        c["tp" if pred and actual else "fp" if pred else "fn" if actual else "tn"] += 1
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None
    rc = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None
    f1 = 2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"]) if c["tp"] + c["fp"] + c["fn"] else None
    return {"n": sum(c.values()), "precision": p, "recall": rc, "f1": f1, "confusion": dict(c)}


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def ci(intervals, method, metric):
    m = intervals.get("methods", {}).get(method, {}).get(metric)
    if not m or m.get("lower") is None:
        return ""
    return f" ({100 * m['lower']:.0f}–{100 * m['upper']:.0f})"


def markdown(result):
    prov = result["provenance"]
    lines = [f"# Accuracy of candidate bare-peat detection — {result['status']}", ""]
    if result["status"] == "provisional":
        lines += [f"**Provisional.** {prov.get('model', 0)} of the reference labels are model "
                  f"pre-labels that the author has not yet reviewed; {prov.get('author', 0)} are "
                  "author-confirmed. The figures will change after review and must not be "
                  "reported as final.", ""]
    lines += ["Bare peat is the positive class. Sample figures count points; weighted "
              "figures scale each stratum to its population in the sampled site-years "
              "(Olofsson et al., 2014), with stratified bootstrap 95% intervals in brackets. "
              "Unsure points are excluded and strata recounted.", ""]
    for scope, block in result["scopes"].items():
        lines += [f"## {scope.capitalize()} ({block['points']} usable points, "
                  f"{block['bare_peat']} bare peat)", "",
                  "| Method | Precision | Recall | F1 | Weighted precision | Weighted recall | Weighted F1 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for m in result["methods"]:
            s = block["report"]["methods"].get(m, {})
            sm, wm = s.get("sample") or {}, s.get("weighted") or {}
            iv = block["intervals"]
            lines.append(f"| {NAMES[m]} | {pct(sm.get('precision'))} | {pct(sm.get('recall'))} | "
                         f"{pct(sm.get('f1'))} | {pct(wm.get('precision'))}{ci(iv, m, 'precision')} | "
                         f"{pct(wm.get('recall'))}{ci(iv, m, 'recall')} | "
                         f"{pct(wm.get('f1'))}{ci(iv, m, 'f1')} |")
        lines.append("")
    gng = result["errors"]["gng"]
    lines += ["## What GNG gets wrong (all site-years)", "",
              "False positives by reference class: " + (", ".join(
                  f"{k} {v}" for k, v in sorted(gng["false_positive_truth"].items(), key=lambda kv: -kv[1]))
                  or "none") + ".", "",
              "Misses by site-year: " + (", ".join(
                  f"{k} ({v})" for k, v in sorted(gng["misses_by_site_year"].items(), key=lambda kv: -kv[1]))
                  or "none") + ".", "",
              "| Site-year | TP | FP | FN | TN |", "|---|---:|---:|---:|---:|"]
    for key, c in gng["per_site_year"].items():
        lines.append(f"| {key} | {c.get('tp', 0)} | {c.get('fp', 0)} | {c.get('fn', 0)} | {c.get('tn', 0)} |")
    lines += ["", "## Sensitivity", "",
              "| Reference set | Points | GNG precision | GNG recall | GNG F1 |", "|---|---:|---:|---:|---:|"]
    for name, s in result["sensitivity"].items():
        lines.append(f"| {name} | {s['points']} | {pct(s['precision'])} | {pct(s['recall'])} | {pct(s['f1'])} |")
    if result.get("readings"):
        lines += ["", "## Inclusive and strict readings of the contested points", "",
                  f"{len(result['contested'])} bare-peat proposals are pale sinuous or branching bands "
                  "inside the bog, without cut-bank geometry. The inclusive reading counts them as "
                  "bare peat; the strict reading does not. The truth lies between.", "",
                  "| Scope | Method | Inclusive P / R / F1 | Strict P / R / F1 |", "|---|---|---|---|"]
        for scope, methods in result["readings"].items():
            for m, both in methods.items():
                a, b = both["inclusive"], both["strict"]
                lines.append(f"| {scope} | {NAMES.get(m, m)} | {pct(a['precision'])} / {pct(a['recall'])} / "
                             f"{pct(a['f1'])} | {pct(b['precision'])} / {pct(b['recall'])} / {pct(b['f1'])} |")
    if result.get("summer"):
        sm = result["summer"]
        lines += ["", "## Summer-checked reading", "",
                  f"{sm['changed']} bare-peat labels sit on pixels that are green (NDVI >= 0.50) "
                  "in the same year's summer Sentinel-2 median (scripts/31_summer_check.py). Exposed "
                  "peat does not grow a canopy within a season, so this reading counts them as "
                  "vegetation that was winter-brown on the spring date.", "",
                  "| Scope | Method | As labelled P / R / F1 | Summer-checked P / R / F1 |", "|---|---|---|---|"]
        for scope, methods in sm["scopes"].items():
            for m, both in methods.items():
                a, b = both["as_labelled"], both["summer_checked"]
                lines.append(f"| {scope} | {NAMES.get(m, m)} | {pct(a['precision'])} / {pct(a['recall'])} / "
                             f"{pct(a['f1'])} | {pct(b['precision'])} / {pct(b['recall'])} / {pct(b['f1'])} |")
    if result.get("two_date"):
        td = result["two_date"]
        lines += ["", f"## Two-date detector on its support ({td['points']} points, "
                  f"{', '.join(td['site_years'])})", "",
                  "| Method | Inclusive P / R / F1 | Strict P / R / F1 |", "|---|---|---|"]
        for m, both in td["methods"].items():
            a, b = both["inclusive"], both["strict"]
            lines.append(f"| {NAMES.get(m, m)} | {pct(a['precision'])} / {pct(a['recall'])} / {pct(a['f1'])} | "
                         f"{pct(b['precision'])} / {pct(b['recall'])} / {pct(b['f1'])} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="frozen evaluation run with annotation/")
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--points", type=Path,
                    help="points CSV with prediction columns (default: annotation/accuracy_points.csv)")
    ap.add_argument("--methods", nargs="+", default=list(METHODS),
                    help="prediction_<method> columns to score (keep gng for the error breakdown)")
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    args = ap.parse_args()
    ann = args.run / "annotation"
    methods = tuple(args.methods)
    rows = join(read_csv(args.points or ann / "accuracy_points.csv"), read_csv(ann / "labels.csv"))
    provenance = Counter(r["provenance"] for r in rows)
    status = "provisional" if provenance.get("model") or provenance.get("none") else "final"

    scopes = {}
    for name, role in SCOPES.items():
        subset = usable([r for r in rows if role is None or r["role"] == role])
        if not subset:
            continue
        report, intervals = score(subset, args.replicates, methods)
        scopes[name] = {"points": len(subset),
                        "bare_peat": sum(r["truth"] == "bare_peat" for r in subset),
                        "provenance": dict(Counter(r["provenance"] for r in subset)),
                        "report": report, "intervals": intervals}

    all_usable = usable(rows)
    errors = {m: error_breakdown(all_usable, m) for m in methods}

    def sample_gng(subset):
        s = score_points(subset)["methods"].get("gng", {}).get("sample") or {}
        return {"points": len(subset), "precision": s.get("precision"),
                "recall": s.get("recall"), "f1": s.get("f1")}
    sensitivity = {
        "all labels (as reported above)": sample_gng(all_usable),
        "without points the model passes disagreed on": sample_gng(
            usable([r for r in rows if DISAGREE_MARK not in r["notes"]])),
        "author-confirmed labels only": sample_gng(
            usable([r for r in rows if r["provenance"] == "author"])),
    }
    contested_path = ann / "model-pass" / "contested.json"
    contested = ({p["point_id"] for p in json.loads(contested_path.read_text())["points"]}
                 if contested_path.exists() else set())
    readings = {}
    if contested:
        for name, role in SCOPES.items():
            subset = [r for r in rows if role is None or r["role"] == role]
            if not subset:
                continue
            readings[name] = {m: {"inclusive": sample_metrics(subset, m),
                                  "strict": sample_metrics(strict(subset, contested), m)}
                              for m in methods}
    summer_rows, summer_changed = summer_checked(rows)
    summer = None
    if summer_rows is not None:
        summer = {"changed": summer_changed, "scopes": {}}
        for name, role in SCOPES.items():
            pair = [(a, b) for a, b in zip(rows, summer_rows) if role is None or a["role"] == role]
            if pair:
                summer["scopes"][name] = {m: {"as_labelled": sample_metrics([a for a, _ in pair], m),
                                              "summer_checked": sample_metrics([b for _, b in pair], m)}
                                          for m in methods}
    two_date = None
    trajectory_csv = ROOT / "outputs/multitemporal_gng/accuracy_points_trajectory.csv"
    if trajectory_csv.exists():
        extra = {r["point_id"]: r for r in read_csv(trajectory_csv)}
        support = [r | {k: v for k, v in extra[r["point_id"]].items() if k.startswith("prediction_traj")}
                   for r in rows if str(extra.get(r["point_id"], {}).get("prediction_trajectory_gng", "")) in ("0", "1")]
        two_date = None if not support else {"points": len(support),
                    "site_years": sorted({f"{r['site'].split(' Bog')[0]} {r['year']}" for r in support}),
                    "methods": {m: {"inclusive": sample_metrics(support, m),
                                    "strict": sample_metrics(strict(support, contested), m)}
                                for m in ("gng", "trajectory_gng", "trajectory_rules")}}
    NAMES.update(trajectory_gng="Two-date GNG", trajectory_rules="Two-date rules")
    result = {"status": status, "methods": list(methods),
              "provenance": dict(provenance), "scopes": scopes,
              "errors": errors, "sensitivity": sensitivity,
              "contested": sorted(contested), "readings": readings, "two_date": two_date, "summer": summer,
              "assumptions": ["Unsure points are excluded and each stratum's sample size is "
                              "recounted, assuming they are missing at random within the stratum.",
                              "Weighted estimates apply to the sampled site-years only."]}
    suffix = ("provisional" if status == "provisional" else "final") + args.tag
    (ann / f"accuracy.{suffix}.json").write_text(json.dumps(result, indent=2, default=str))
    (ann / f"accuracy.{suffix}.md").write_text(markdown(result))
    print(markdown(result))


if __name__ == "__main__":
    main()
