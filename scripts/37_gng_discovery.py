"""Step 37 — can GNG discover exposed peat without thresholds?

In v1-v4 GNG sat under fixed spectral rules, so its masks matched the rules.
This experiment gives it a different role: unsupervised discovery.

1. Train a Growing Neural Gas on standardised six-band summer reflectance
   (blue, green, red, NIR, SWIR1, SWIR2: no indices, no thresholds) pooled
   from the 14 frozen summer scenes of the v5 test and external runs.
2. Name each node "peat" or "not peat" from the labelled points of one set
   only: the industrial points of the v5 test (a node is peat when most of its
   labelled points are bare peat; nodes without labels are not peat).
3. Score the named map on the other labelled points, the external industrial
   area and the protected bogs in summer, and compare with v5 on the same
   points. K-means with the same number of clusters, named the same way, is
   the control.
4. Describe the peat nodes: does the network separate the SWIR2 > NIR
   signature by itself?

Usage:
    python3 scripts/37_gng_discovery.py
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from peatland import gng, v5

RUNS = [ROOT / "outputs/evaluation/2026-10-08-v5-test", ROOT / "outputs/evaluation/2026-10-08-v5-external"]
OUT = ROOT / "outputs/evaluation/gng-discovery.json"
PER_SCENE = 6000
SEEDS = (7, 42, 123)
MAX_NODES = 40


def scenes():
    for run in RUNS:
        for npz in sorted(run.glob("*.npz")):
            meta = json.loads(npz.with_suffix(".json").read_text())
            with np.load(npz) as z:
                yield run, npz.stem, meta, z["stack"].astype(np.float32), (z["valid"] & z["inside"]).astype(bool)


def points():
    out = []
    for run in RUNS:
        ann = run / "annotation"
        labels = {r["point_id"]: r["truth"] for r in csv.DictReader((ann / "labels.csv").open(encoding="utf-8-sig"))}
        for p in csv.DictReader((ann / "accuracy_points.csv").open(encoding="utf-8-sig")):
            t = labels.get(p["point_id"], "")
            if t in ("", "unsure"):
                continue
            out.append({"run": run.name, "stem": p["point_id"].rsplit("_r", 1)[0], "role": p["role"],
                        "row": int(p["pixel_row"]), "col": int(p["pixel_col"]),
                        "bare": t == "bare_peat", "v5": p.get("prediction_v5") == "1"})
    return out


def metrics(pairs):
    c = Counter(("tp" if p and t else "fp") if p else ("fn" if t else "tn") for p, t in pairs)
    prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None
    rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None
    f1 = 2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"]) if c["tp"] + c["fp"] + c["fn"] else None
    return {"n": len(pairs), "precision": prec, "recall": rec, "f1": f1, **c}


def main():
    rng = np.random.default_rng(0)
    data, feats = {}, []
    for run, stem, meta, stack, ok in scenes():
        refl = stack / 10000.0
        data[(run.name, stem)] = refl
        idx = np.flatnonzero(ok.ravel() & np.isfinite(refl.reshape(-1, 6)).all(1))
        pick = rng.choice(idx, size=min(PER_SCENE, len(idx)), replace=False)
        feats.append(refl.reshape(-1, 6)[pick])
    X = np.concatenate(feats)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    pts = points()
    for p in pts:
        p["x"] = (data[(p["run"], p["stem"])][p["row"], p["col"]] - mu) / sd
    train = [p for p in pts if p["role"] == "industrial"]
    test = {"external industrial": [p for p in pts if p["role"] == "external"],
            "protected bogs (summer)": [p for p in pts if p["role"] == "protected"]}
    result = {"features": "standardised blue, green, red, NIR, SWIR1, SWIR2 reflectance",
              "training_pixels": int(len(Z)), "naming_points": len(train),
              "test_points": {k: len(v) for k, v in test.items()}, "runs": []}

    def evaluate(name, seed, assign, centres):
        votes = defaultdict(list)
        for p in train:
            votes[int(assign(p["x"][None])[0])].append(p["bare"])
        peat_nodes = {k for k, v in votes.items() if sum(v) > len(v) / 2}
        row = {"method": name, "seed": seed, "clusters": int(len(centres)),
               "peat_clusters": sorted(peat_nodes), "naming": metrics(
                   [(int(assign(p["x"][None])[0]) in peat_nodes, p["bare"]) for p in train])}
        for scope, ps in test.items():
            row[scope] = metrics([(int(assign(p["x"][None])[0]) in peat_nodes, p["bare"]) for p in ps])
        # what the peat clusters look like, in reflectance
        proto = centres * sd + mu
        nir, s1, s2 = proto[:, 3], proto[:, 4], proto[:, 5]
        row["peat_prototypes"] = [{"cluster": int(k), "nbr": round(float((nir[k] - s2[k]) / (nir[k] + s2[k])), 3),
                                   "swir1": round(float(s1[k]), 3),
                                   "ndvi": round(float((nir[k] - proto[k, 2]) / (nir[k] + proto[k, 2])), 3)}
                                  for k in sorted(peat_nodes)]
        result["runs"].append(row)
        print(f"{name:8} seed {seed:3}: {len(centres)} clusters, {len(peat_nodes)} peat | " + " | ".join(
            f"{s}: F1 {row[s]['f1']:.2f} (P {row[s]['precision'] or 0:.2f} R {row[s]['recall'] or 0:.2f})"
            for s in test), flush=True)

    for seed in SEEDS:
        net = gng.GrowingNeuralGas(max_nodes=MAX_NODES, rng=np.random.default_rng(seed))
        net.fit(Z[np.random.default_rng(seed).choice(len(Z), size=min(30000, len(Z)), replace=False)], n_steps=40000)
        evaluate("GNG", seed, net.predict, net.weights)
        k = len(net.weights)
        km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(Z[np.random.default_rng(seed).choice(len(Z), size=30000, replace=False)])
        evaluate("K-means", seed, km.predict, km.cluster_centers_)
    result["v5"] = {s: metrics([(p["v5"], p["bare"]) for p in ps]) for s, ps in test.items()}
    for s, m in result["v5"].items():
        print(f"v5       on {s}: F1 {m['f1']:.2f} (P {m['precision']:.2f} R {m['recall']:.2f})")
    OUT.write_text(json.dumps(result, indent=2, default=float))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
