"""Step 49 — wet peat that v5 misses: a supplementary blind sample.

On the industrial test fields almost all area-weighted omission came from five
missed peat points (four of them on the south area in spring 2024, after
harvesting stopped). They are dark in every band: SWIR1 just under 0.22 and NBR
above zero, which v5's dry-peat rule rejects. v5.wet_peat_mask describes that
signature; its thresholds were read off those five points, so the original
sample cannot test it. This step draws a new sample instead:

- sample: on every frozen scene of the v5 test and external runs, split the
  valid site area into three strata (v5 detection, wet-peat signature,
  the rest) and draw points from each (industrial 4 / 10 / 4, protected
  0 / 5 / 2), skipping points already in the original sample. The blind
  labels.csv and batches carry opaque ids (W001...) and no site name, so
  annotators see neither the site, the season nor the stratum.
- evaluate: after five blind passes (instructions as for step 48), pool the
  original five-pass sample with the new one and estimate area-weighted
  precision, recall and F1, post-stratified by the finer map, for v5, for
  v5 plus every wet-peat pixel, and for v5 plus wet-peat pixels within
  v5.WET_NEAR_PX pixels of a v5 detection. Intervals come from a stratified
  bootstrap.

The frozen sample (sample_points.csv) was drawn with float32 band ratios;
v5.wet_peat_mask works in float64, so a re-run of `sample` can pick different
points where a borderline pixel changes stratum. `evaluate` always reads the
frozen sample and recomputes the strata from the mask.

Usage:
    python3 scripts/49_wet_peat.py sample
    python3 scripts/49_wet_peat.py evaluate
"""

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from peatland import v5

EVAL = ROOT / "outputs" / "evaluation"
RUNS = ["2026-10-08-v5-test", "2026-10-08-v5-external"]
OUT = EVAL / "2026-10-17-wet-supplement"
PLAN = {"industrial": {"v5": 4, "wet": 10, "rest": 4},
        "protected": {"v5": 0, "wet": 5, "rest": 2}}
SEED = 1017
CLASSES = {1: "v5", 2: "wet far", 3: "wet near"}     # 0: rest


def step(name, f):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def role(stem):
    return "protected" if stem[0].isdigit() else "industrial"


def sample():
    ann = OUT / "annotation"
    if (ann / "labels.csv").exists():
        raise SystemExit(f"{ann / 'labels.csv'} exists; the sample is frozen")
    (ann / "model-pass").mkdir(parents=True, exist_ok=True)
    s18, s20 = step("s18", "18_label_tool.py"), step("s20", "20_planet_reference.py")
    rng = np.random.default_rng(SEED)
    rows, strata = [], []
    for run in RUNS:
        taken = set(pd.read_csv(EVAL / run / "annotation/accuracy_points.csv").point_id)
        for npz in sorted((EVAL / run).glob("*.npz")):
            stem, meta = npz.stem, json.loads(npz.with_suffix(".json").read_text())
            z = np.load(npz)
            order = tuple(meta["bands"])
            ok = z["valid"] & z["inside"]
            dry = z["prediction_v5"] & ok
            wet = v5.wet_peat_mask(z["stack"], z["valid"], z["inside"], order)
            tr = rasterio.Affine(*meta["transform"][:6])
            to_ll = Transformer.from_crs(meta["crs"], "EPSG:4326", always_xy=True)
            for name, mask in (("v5", dry), ("wet", wet), ("rest", ok & ~dry & ~wet)):
                strata.append(dict(run=run, stem=stem, site=meta["site"], year=meta["year"],
                                   stratum=name, population=int(mask.sum())))
                k = PLAN[role(stem)][name]
                if k == 0:
                    continue
                idx = np.argwhere(mask)
                n = 0
                for rr, cc in idx[rng.permutation(len(idx))]:
                    pid = f"{stem}_r{rr}_c{cc}"
                    if pid in taken:
                        continue
                    lon, lat = to_ll.transform(*(tr * (cc + 0.5, rr + 0.5)))
                    rows.append(dict(point_id=pid, run=run, stem=stem, site=meta["site"],
                                     year=meta["year"], scene_date=meta["scene_date"],
                                     role=role(stem), stratum=name, pixel_row=int(rr),
                                     pixel_col=int(cc), lat=round(lat, 7), lon=round(lon, 7)))
                    n += 1
                    if n == k:
                        break
    df = pd.DataFrame(rows)
    blind = df.sample(frac=1, random_state=7).reset_index(drop=True)
    ids = {p: f"W{n + 1:03d}" for n, p in enumerate(blind.point_id)}
    df["opaque_id"] = df.point_id.map(ids)
    df.to_csv(OUT / "sample_points.csv", index=False)     # strata stay outside annotation/
    pd.DataFrame(strata).to_csv(OUT / "strata.csv", index=False)

    chips = ann / "chips"
    manifests = {}
    for run in RUNS:
        sub = [dict(point_id=ids[r.point_id], lat=r.lat, lon=r.lon,
                    _real=r.point_id) for r in df[df.run == run].itertuples()]
        # step 18 reads the pixel from the id, so render under the real id and rename
        s18.render_chips([dict(point_id=p["_real"]) for p in sub], EVAL / run, chips)
        man = json.loads((EVAL / run / "planet/manifest.json").read_text())
        manifests.update(man)
        for stem, m in man.items():
            pts = [p for p in sub if p["_real"].startswith(stem + "_")]
            d = m["delta_days"]
            when = "same day" if d == 0 else f"{d:+d} day" + ("s" if abs(d) > 1 else "")
            s20.chips(m["mosaic"], pts, chips, f"PlanetScope 3 m · {m['date']} · {when}")
        for p in sub:
            for f in chips.glob(f"{p['_real']}_*.png"):
                f.rename(f.with_name(p["point_id"] + f.name[len(p["_real"]):]))
    (ann / "key.png").write_bytes((EVAL / RUNS[0] / "annotation/key.png").read_bytes())

    with open(ann / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["point_id", "site", "year", "scene_date", "lat", "lon", "truth",
                               "reference_source", "reference_date", "notes"])
        w.writeheader()
        for r in blind.itertuples():
            w.writerow(dict(point_id=ids[r.point_id], site="withheld", year=r.year,
                            scene_date=r.scene_date, lat=r.lat, lon=r.lon, truth="",
                            reference_source="", reference_date="", notes=""))
    recs = []
    for n, r in enumerate(blind.itertuples(), 1):
        m, has = manifests[r.stem], (chips / f"{ids[r.point_id]}_ps_tc.png").exists()
        recs.append(dict(n=n, point_id=ids[r.point_id], site="withheld", scene_date=r.scene_date,
                         lat=str(r.lat), lon=str(r.lon),
                         planetscope=dict(date=m["date"], delta_days=m["delta_days"]) if has else None))
    for b in range(4):
        part = recs[b * len(recs) // 4:(b + 1) * len(recs) // 4]
        (ann / "model-pass" / f"batch{b + 1}.json").write_text(json.dumps(part, indent=1))
    print(df.groupby(["role", "stratum"]).size().to_string(), f"\n{len(df)} points")


def vote(labels):
    """Majority label among decided votes; unsure abstains; a tie or < half is unsure."""
    c = Counter(x for x in labels if x != "unsure")
    if not c:
        return "unsure"
    (top, n), *rest = c.most_common()
    if (rest and rest[0][1] == n) or 2 * n < len(labels):
        return "unsure"
    return top


def supplement_labels():
    votes = {}
    for f in sorted((OUT / "annotation/model-pass").glob("pass*_batch*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                votes.setdefault(r["point_id"], []).append(r["label"])
    sp = pd.read_csv(OUT / "sample_points.csv")
    sp["passes"] = sp.opaque_id.map(lambda i: len(votes.get(i, [])))
    sp["truth"] = sp.opaque_id.map(lambda i: vote(votes.get(i, [])))
    return sp


def original_labels(run):
    pts = pd.read_csv(EVAL / run / "annotation/accuracy_points.csv").drop(columns=["truth"])
    lab = pd.read_csv(EVAL / run / "annotation/labels.csv")[["point_id", "truth"]]
    pts = pts.merge(lab, on="point_id")
    pts["stem"] = pts.point_id.str.rsplit("_r", n=1).str[0]
    pts["run"] = run
    return pts


def estimate(d, pop, positive):
    tp = fp = fn = 0.0
    for (stem, c), g in d.groupby(["stem", "c"]):
        w = pop[(stem, c)] / len(g)
        bare = g.bare.sum()
        if c in positive:
            tp, fp = tp + w * bare, fp + w * (len(g) - bare)
        else:
            fn += w * bare
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    return p, r, 2 * p * r / (p + r) if p + r else float("nan")


def evaluate(reps=1000, seed=1):
    sp = supplement_labels()
    pts = pd.concat([original_labels(r) for r in RUNS]
                    + [sp[["run", "stem", "pixel_row", "pixel_col", "truth"]]], ignore_index=True)
    pts = pts[pts.truth != "unsure"].reset_index(drop=True)
    pts["bare"] = (pts.truth == "bare_peat").astype(int)
    pop, cls = {}, pd.Series(0, index=pts.index)
    for (run, stem), g in pts.groupby(["run", "stem"]):
        meta = json.loads((EVAL / run / f"{stem}.json").read_text())
        z = np.load(EVAL / run / f"{stem}.npz")
        ok = z["valid"] & z["inside"]
        dry = z["prediction_v5"] & ok
        wet = v5.wet_peat_mask(z["stack"], z["valid"], z["inside"], tuple(meta["bands"]))
        close = wet & v5.near(dry)
        c = np.zeros(ok.shape, np.int8)
        c[dry], c[wet & ~close], c[close] = 1, 2, 3
        for k in range(4):
            pop[(stem, k)] = int(((c == k) & ok).sum())
        cls[g.index] = c[g.pixel_row, g.pixel_col]
    pts["c"] = cls
    maps = {"v5": {1}, "v5 + wet peat": {1, 2, 3},
            f"v5 + wet peat within {v5.WET_NEAR_PX} px": {1, 3}}
    scopes = {"industrial test": pts.stem.str.match("BNM[SW]"),
              "industrial external": pts.stem.str.startswith("BNMX"),
              "protected": pts.stem.str[0].str.isdigit()}
    rng = np.random.default_rng(seed)
    report = {"supplement": {"points": len(sp), "passes": sp.passes.value_counts().to_dict(),
                             "labels": sp.groupby(["role", "stratum", "truth"]).size()
                             .rename(lambda x: x).reset_index(name="n").to_dict("records")},
              "scopes": {}}
    for scope, sel in scopes.items():
        d = pts[sel]
        groups = [g for _, g in d.groupby(["stem", "c"])]
        res = {"points": d.groupby("c").size().rename(lambda k: CLASSES.get(k, "rest")).to_dict()}
        for name, positive in maps.items():
            p, r, f1 = estimate(d, pop, positive)
            boot = np.array([estimate(pd.concat([g.iloc[rng.integers(0, len(g), len(g))]
                                                 for g in groups]), pop, positive)
                             for _ in range(reps)])
            lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
            res[name] = {k: {"estimate": round(v, 3), "ci95": [round(a, 3), round(b, 3)]}
                         for k, v, a, b in zip(("precision", "recall", "f1"), (p, r, f1), lo, hi)}
            print(f"{scope:20} {name:28} P {p:.2f} [{lo[0]:.2f}-{hi[0]:.2f}]  "
                  f"R {r:.2f} [{lo[1]:.2f}-{hi[1]:.2f}]  F1 {f1:.2f} [{lo[2]:.2f}-{hi[2]:.2f}]")
        report["scopes"][scope] = res
    (OUT / "accuracy.json").write_text(json.dumps(report, indent=2, default=int))
    print(f"report {OUT / 'accuracy.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("sample", "evaluate"))
    ap.add_argument("--reps", type=int, default=1000, help="bootstrap replicates")
    args = ap.parse_args()
    sample() if args.action == "sample" else evaluate(args.reps)


if __name__ == "__main__":
    main()
