"""Step 50 — are v5's protected-bog false positives next to real cutting?

On protected bogs 37 of the 80 detected test points were labelled vegetated,
yet they are spectrally the same as the true detections (median SWIR1 0.25,
NBR -0.04 against -0.10) and the annotators' notes keep placing them "at the
edge of a cut face" or "between turf strips". v5 relies on SWIR bands sampled
at 20 m, so a pixel beside a cut face can carry its signal. This step relabels
every protected test point blind with a 30 m question: is there exposed peat
covering about a tenth or more of the 3 x 3-pixel block around the point?

- prepare: chips with the 30 m block (cyan) and the centre pixel (yellow) on
  PlanetScope 3 m and Sentinel-2, opaque ids (B001...), four batches. The
  map from ids to points stays outside annotation/.
- evaluate: five-pass majority (unsure abstains), pairwise kappa, how many
  pixel-level false positives have peat in their block, and the share of
  detected area with peat in the pixel or within the block. The detected
  stratum is a simple random sample of v5 pixels per site, so the share is
  weighted by the detected area of each site, with a stratified bootstrap.

Usage:
    python3 scripts/50_block_check.py prepare
    python3 scripts/50_block_check.py evaluate
"""

import argparse
import importlib.util
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs" / "evaluation"
RUN = EVAL / "2026-10-08-v5-test"
OUT = EVAL / "2026-10-17-block-protected"
YELLOW, CYAN = (253, 224, 71), (0, 230, 255)


def _box(d, a, b, colour, width):
    d.rectangle([a - 1, a - 1, b + 1, b + 1], outline=(0, 0, 0), width=1)
    d.rectangle([a, a, b, b], outline=colour, width=width)


def prepare():
    ann = OUT / "annotation"
    if (OUT / "id_map.csv").exists():
        raise SystemExit(f"{OUT / 'id_map.csv'} exists; the set is frozen")
    chips = ann / "chips"
    chips.mkdir(parents=True, exist_ok=True)
    (ann / "model-pass").mkdir(exist_ok=True)
    spec = importlib.util.spec_from_file_location("s18", ROOT / "scripts/18_label_tool.py")
    s18 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s18)
    pts = pd.read_csv(RUN / "annotation/accuracy_points.csv")
    pts = pts[pts.role == "protected"].sample(frac=1, random_state=11).reset_index(drop=True)
    pts["opaque_id"] = [f"B{n + 1:03d}" for n in range(len(pts))]
    pts["stem"] = pts.point_id.str.rsplit("_r", n=1).str[0]
    pts[["point_id", "opaque_id"]].to_csv(OUT / "id_map.csv", index=False)
    manifest = json.loads((RUN / "planet/manifest.json").read_text())
    font = ImageFont.load_default(size=13)
    for stem, g in pts.groupby("stem"):
        order = json.loads((RUN / f"{stem}.json").read_text())["bands"]
        stack = np.load(RUN / f"{stem}.npz")["stack"]
        m = manifest[stem]
        for r in g.itertuples():
            half, scale = 16, 10                       # 330 m of 10 m pixels
            win = s18._window(stack, r.pixel_row, r.pixel_col, half)
            for kind in ("tc", "fc"):
                im = Image.fromarray(s18._composite(win, order, kind)).resize((330, 330), Image.NEAREST)
                d = ImageDraw.Draw(im)
                _box(d, (half - 1) * scale, (half + 2) * scale - 1, CYAN, 2)
                _box(d, half * scale, (half + 1) * scale - 1, YELLOW, 1)
                im.save(chips / f"{r.opaque_id}_{kind}_near.png")
            with rasterio.open(m["mosaic"]) as ds:
                bands = ({"blue": 1, "green": 2, "red": 3, "nir": 4} if ds.count == 4 else
                         {"blue": 2, "green": 4, "red": 6, "nir": 8})
                x, y = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True).transform(r.lon, r.lat)
                row, col = ds.index(x, y)
                h = 37                                 # 75 PlanetScope pixels, about 225 m
                a = ds.read([bands[b] for b in ("blue", "green", "red", "nir")],
                            window=Window(col - h, row - h, 2 * h + 1, 2 * h + 1),
                            boundless=True, fill_value=0).astype("float32") / 1e4
                res = abs(ds.transform.a)
            empty = (a == 0).all(0)
            blue, green, red, nir = a
            for kind, rgb, gamma in (("tc", np.dstack([red, green, blue]) * 3.2, 1.4),
                                     ("fc", np.dstack([nir * 2.2, red * 3.2, green * 3.2]), 1.2)):
                img = np.clip(rgb, 0, 1) ** (1 / gamma)
                img[empty] = 0.13
                size = (2 * h + 1) * 4
                im = Image.fromarray((img * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
                d = ImageDraw.Draw(im)
                centre, per_m = (h + 0.5) * 4, 4 / res
                for metres, colour, width in ((30, CYAN, 2), (10, YELLOW, 1)):
                    k = metres * per_m / 2
                    _box(d, centre - k, centre + k - 1, colour, width)
                d.rectangle([0, 0, size, 18], fill=(0, 0, 0))
                d.text((4, 2), f"PlanetScope 3 m · {m['date']} · same day", font=font, fill=(240, 240, 240))
                im.save(chips / f"{r.opaque_id}_ps_{kind}.png")
    (ann / "key.png").write_bytes((RUN / "annotation/key.png").read_bytes())
    recs = [dict(n=n + 1, point_id=r.opaque_id, scene_date=r.scene_date, lat=str(r.lat), lon=str(r.lon))
            for n, r in enumerate(pts.itertuples())]
    for b in range(4):
        part = recs[b * len(recs) // 4:(b + 1) * len(recs) // 4]
        (ann / "model-pass" / f"batch{b + 1}.json").write_text(json.dumps(part, indent=1))
    print(f"{len(pts)} points, chips in {chips}")


def vote(labels):
    c = Counter(x for x in labels if x != "unsure")
    if not c:
        return None
    (top, n), *rest = c.most_common()
    if (rest and rest[0][1] == n) or 2 * n < len(labels):
        return None
    return top


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe)


def evaluate(reps=2000, seed=3):
    votes = {}
    for f in sorted((OUT / "annotation/model-pass").glob("pass*_batch*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                votes.setdefault(r["point_id"], {})[f.name[4]] = r["label"]
    pts = pd.read_csv(RUN / "annotation/accuracy_points.csv").drop(columns=["truth"])
    pts = pts.merge(pd.read_csv(RUN / "annotation/labels.csv")[["point_id", "truth"]], on="point_id")
    pts = pts.merge(pd.read_csv(OUT / "id_map.csv"), on="point_id")
    pts["stem"] = pts.point_id.str.rsplit("_r", n=1).str[0]
    pts["votes"] = pts.opaque_id.map(lambda i: votes.get(i, {}))
    pts["block"] = pts.votes.map(lambda v: vote(list(v.values())))
    d = pts[pts.block.notna()].copy()
    d["bare"] = (d.truth == "bare_peat").astype(int)
    d["peat_block"] = (d.block == "peat_in_block").astype(int)
    passes = sorted({p for v in d.votes for p in v})
    full = d[d.votes.map(len) == len(passes)]
    ks = [kappa(full.votes.map(lambda v: v[p] == "peat_in_block"),
                full.votes.map(lambda v: v[q] == "peat_in_block"))
          for p, q in itertools.combinations(passes, 2)]
    det = d[d.prediction_v5 == 1]
    fp = det[det.bare == 0]

    def share(s):
        g = s.groupby("stem").agg(N=("stratum_population", "first"), pix=("bare", "mean"),
                                  blk=("peat_block", "mean"))
        return (g.N * g.pix).sum() / g.N.sum(), (g.N * g.blk).sum() / g.N.sum()

    rng = np.random.default_rng(seed)
    groups = [g for _, g in det.groupby("stem")]
    boot = np.array([share(pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups]))
                     for _ in range(reps)])
    est, (lo, hi) = share(det), np.percentile(boot, [2.5, 97.5], axis=0)
    per_site = det.groupby("site").agg(points=("bare", "size"), peat_in_pixel=("bare", "sum"),
                                       peat_in_block=("peat_block", "sum"))
    und = d[d.prediction_v5 == 0]
    report = {
        "points": len(pts), "decided": len(d),
        "votes_for_peat": d.votes.map(lambda v: sum(x == "peat_in_block" for x in v.values()))
                           .value_counts().sort_index().to_dict(),
        "pairwise_kappa": {"min": round(min(ks), 3), "max": round(max(ks), 3), "mean": round(float(np.mean(ks)), 3)},
        "detected_points": len(det),
        "pixel_false_positives": len(fp),
        "pixel_false_positives_with_peat_in_block": int(fp.peat_block.sum()),
        "per_point_precision": {"pixel": round(det.bare.mean(), 3), "block": round(det.peat_block.mean(), 3)},
        "area_weighted_precision": {"pixel": [round(est[0], 3), round(lo[0], 3), round(hi[0], 3)],
                                    "block": [round(est[1], 3), round(lo[1], 3), round(hi[1], 3)]},
        "undetected_points_with_peat_in_block": f"{int(und.peat_block.sum())} of {len(und)}",
        "per_site": per_site.to_dict("index"),
        "pixel_vs_block": pd.crosstab(d.bare, d.peat_block).to_dict()}
    (OUT / "block-check.json").write_text(json.dumps(report, indent=2, default=int))
    print(json.dumps(report, indent=2, default=int))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("prepare", "evaluate"))
    args = ap.parse_args()
    prepare() if args.action == "prepare" else evaluate()


if __name__ == "__main__":
    main()
