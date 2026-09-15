"""Accuracy reports from independent reference labels on sampled pixels."""

from collections import Counter, defaultdict
import math

TRUTH_CLASSES = {"bare_peat", "vegetated", "water", "burn", "other", "unsure"}
METHODS = ("ndvi", "rules", "kmeans", "gng")


def _metrics(counts):
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    ratio = lambda a, b: a / b if b else None
    return {"confusion": dict(counts), "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn), "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "accuracy": ratio(tp + tn, tp + fp + fn + tn)}


def _prediction(row, method):
    value = row.get(f"prediction_{method}", "")
    if value == "" and method == "gng":
        value = {"detected": "1", "undetected": "0"}.get(row.get("stratum"), "")
    if str(value) not in {"0", "1"}:
        return None
    return int(value)


def score_points(rows, method_names=METHODS):
    """Score complete or partial labels without inventing missing truth.

    Unweighted scores describe the labelled sample only. Population-weighted
    estimates require complete labels and known N/n for every sampled stratum,
    and apply only to the sampled site-years, not all West-of-Ireland bogs.
    """
    ids, groups = set(), defaultdict(list)
    truth_counts = Counter()
    for row in rows:
        pid = row.get("point_id", "").strip()
        if not pid or pid in ids:
            raise ValueError(f"Missing or duplicate point_id: {pid!r}")
        ids.add(pid)
        truth = row.get("truth", "").strip().lower()
        if truth and truth not in TRUTH_CLASSES:
            raise ValueError(f"{pid}: unknown truth label {truth!r}")
        truth_counts[truth or "unlabelled"] += 1
        groups[(row.get("site"), row.get("year"), row.get("stratum"))].append(row)
        for method in method_names:
            value = row.get(f"prediction_{method}", "")
            if value != "" and str(value) not in {"0", "1"}:
                raise ValueError(f"{pid}: invalid {method} prediction {value!r}")

    weights = {}
    complete = bool(rows) and not truth_counts["unlabelled"] and not truth_counts["unsure"]
    weighting_available = complete
    for group in groups.values():
        populations = {str(r.get("stratum_population", "")) for r in group}
        sizes = {str(r.get("stratum_sample_size", "")) for r in group}
        try:
            if len(populations) != 1 or len(sizes) != 1:
                raise ValueError
            population, size = float(next(iter(populations))), float(next(iter(sizes)))
            if (not math.isfinite(population) or not math.isfinite(size)
                    or size != len(group) or population < size or size <= 0
                    or not population.is_integer()):
                raise ValueError
        except (ValueError, TypeError):
            weighting_available = False
            continue
        for row in group:
            weights[row["point_id"]] = population / size

    labelled = [r for r in rows if r.get("truth", "").strip().lower()
                not in {"", "unsure"}]
    methods = {}
    for method in method_names:
        if not labelled or any(_prediction(r, method) is None for r in labelled):
            continue
        counts = dict.fromkeys(("tp", "fp", "fn", "tn"), 0)
        weighted = dict.fromkeys(counts, 0.0)
        for row in labelled:
            actual = row["truth"].strip().lower() == "bare_peat"
            pred = bool(_prediction(row, method))
            key = ("tp" if actual else "fp") if pred else ("fn" if actual else "tn")
            counts[key] += 1
            if weighting_available:
                weighted[key] += weights[row["point_id"]]
        methods[method] = {"sample": _metrics(counts),
                           "weighted": _metrics(weighted) if weighting_available else None}
    versions = sorted({r["detector_version"] for r in rows if r.get("detector_version")})
    if len(versions) > 1:
        raise ValueError("Mixed detector versions: evaluate each frozen run separately")
    return {
        "status": "complete" if complete else "awaiting_labels",
        "total_points": len(rows), "labelled_points": len(labelled),
        "truth_counts": dict(truth_counts), "detector_versions": versions,
        "weighting_available": weighting_available,
        "scope": "Sampled site-years only; bare surface, not confirmed cutting or legality.",
        "notes": ["Sample metrics are descriptive and are not regional accuracy estimates.",
                  "Weighted metrics require complete labels and recorded stratum population/sample sizes.",
                  "Uncertainty intervals and independent test-site validation remain separate evaluation tasks."],
        "methods": methods,
    }
