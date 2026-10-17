"""Step 51 — do v5 false positives on protected bogs persist through the summer?

If the protected-bog false positives were short-lived (dry heath on the one
driest day), requiring the signature on more than one summer scene would
remove them. This step runs v5 on every clear Sentinel-2 scene from 15 May to
15 September of each protected test site-year, on the frozen grid, and counts
for every labelled point on how many other scenes it is detected.

- fetch: search and read the scenes (Planetary Computer), compute v5 per
  scene and store the masks; the frozen test scene must be reproduced exactly.
- evaluate: per-point and area-weighted accuracy when a detection must also
  appear on at least k other scenes (k = 1, 2, 3).

Usage:
    python3 scripts/51_summer_persistence.py fetch
    python3 scripts/51_summer_persistence.py evaluate
"""

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RUN = ROOT / "outputs/evaluation/2026-10-08-v5-test"
OUT = ROOT / "outputs/evaluation/2026-10-17-persistence"


def fetch():
    from peatland import boundaries, config, geo, imagery, pipeline, preprocess, v5
    OUT.mkdir(parents=True, exist_ok=True)
    sites = boundaries.build_sites()
    order = list(pipeline.BANDS)
    for jf in sorted(RUN.glob("0*.json")):
        meta, stem = json.loads(jf.read_text()), jf.stem
        if (OUT / f"{stem}.npz").exists():
            continue
        z = np.load(RUN / f"{stem}.npz")
        shape, inside = z["valid"].shape, z["inside"]
        row = sites[sites.SITECODE == meta["site_code"]].iloc[0]
        bbox = boundaries.site_bbox_wgs84(sites, row.name, buffer_m=200)
        transform = rasterio.Affine(*meta["transform"][:6])
        geom = gpd.GeoSeries([row.geometry], crs=config.ITM).to_crs(meta["crs"]).iloc[0]
        site_inside = geo.polygon_mask(geom, shape, transform)
        y = meta["year"]
        scenes = preprocess.clear_scenes("planetary", bbox, f"{y}-05-15/{y}-09-15", shape,
                                         aoi_mask=site_inside, min_clear=0.5, max_scenes=12, limit=40)
        dates, masks, valids = [], [], []
        for item, rep in scenes:
            stack = np.stack([imagery.read_window(item, b, bbox, "planetary", out_shape=shape)[0]
                              .astype("float32") for b in order], -1)
            valid = pipeline.scene_quality(stack, order, rep["scl"], inside, item)["valid"]
            dates.append(item.properties["datetime"][:10])
            masks.append(v5.peat_mask(stack, valid, inside, tuple(order)))
            valids.append(valid)
            print(stem, dates[-1], "v5 px", int(masks[-1].sum()), flush=True)
        if meta["scene_date"] not in dates:
            raise SystemExit(f"{stem}: frozen scene {meta['scene_date']} not found")
        same = (masks[dates.index(meta["scene_date"])] == z["prediction_v5"]).mean()
        if same < 1:
            raise SystemExit(f"{stem}: frozen scene not reproduced ({same:.5f})")
        np.savez_compressed(OUT / f"{stem}.npz", dates=np.array(dates), v5=np.array(masks),
                            valid=np.array(valids))


def evaluate():
    pts = pd.read_csv(RUN / "annotation/accuracy_points.csv").drop(columns=["truth"])
    pts = pts.merge(pd.read_csv(RUN / "annotation/labels.csv")[["point_id", "truth"]], on="point_id")
    pts = pts[(pts.role == "protected") & (pts.truth != "unsure")].copy()
    pts["stem"] = pts.point_id.str.rsplit("_r", n=1).str[0]
    pts["bare"] = (pts.truth == "bare_peat").astype(int)
    other = []
    for r in pts.itertuples():
        z = np.load(OUT / f"{r.stem}.npz")
        dates = list(z["dates"])
        keep = [i for i in range(len(dates)) if dates[i] != r.scene_date]
        v = z["valid"][keep, r.pixel_row, r.pixel_col]
        other.append(int((z["v5"][keep, r.pixel_row, r.pixel_col] & v).sum()))
    pts["other_dates"] = other

    def scores(pred):
        tp = int(((pred == 1) & (pts.bare == 1)).sum())
        fp = int(((pred == 1) & (pts.bare == 0)).sum())
        fn = int(((pred == 0) & (pts.bare == 1)).sum())
        wtp = wfp = wfn = 0.0
        for _, g in pts.assign(pred=pred).groupby(["stem", "stratum"]):
            w = g.stratum_population.iloc[0] / len(g)
            wtp += w * ((g.pred == 1) & (g.bare == 1)).sum()
            wfp += w * ((g.pred == 1) & (g.bare == 0)).sum()
            wfn += w * ((g.pred == 0) & (g.bare == 1)).sum()
        p, rc = wtp / (wtp + wfp), wtp / (wtp + wfn)
        return {"tp": tp, "fp": fp, "fn": fn, "precision": round(tp / (tp + fp), 3),
                "weighted": {"precision": round(p, 3), "recall": round(rc, 3),
                             "f1": round(2 * p * rc / (p + rc), 3)}}

    det = pts[pts.prediction_v5 == 1]
    report = {"scenes_per_site": {s: len(np.load(OUT / f"{s}.npz")["dates"]) for s in sorted(pts.stem.unique())},
              "detected_points_other_dates": det.groupby("truth").other_dates.describe().round(2).to_dict("index"),
              "rules": {"v5": scores(pts.prediction_v5)}}
    for k in (1, 2, 3):
        report["rules"][f"v5 and on >= {k} other scenes"] = scores(
            ((pts.prediction_v5 == 1) & (pts.other_dates >= k)).astype(int))
    (OUT / "persistence.json").write_text(json.dumps(report, indent=2, default=int))
    print(json.dumps(report, indent=2, default=int))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("fetch", "evaluate"))
    args = ap.parse_args()
    fetch() if args.action == "fetch" else evaluate()


if __name__ == "__main__":
    main()
