"""Step 45 — does v5 mistake dry (winter-dead) grass for cut peat?

The summer check (scripts/31_summer_check.py) showed that the spring
"bare peat" candidates of the old NDVI detector turn green by summer: they are
winter-dead Molinia and sedge, i.e. dry grass. This script uses those pixels
as a sample of dry grass and asks three things:

1. On the same spring scenes (April-May frozen site-years of the v3 sample and
   the v4 test), which share of the dry-grass pixels does v5 flag as exposed
   peat? Dry grass = old NDVI-rule candidate whose median summer NDVI
   (15 June - 31 August, same year) is >= 0.50.
2. How do dry grass and exposed peat differ spectrally? Medians of NDVI, NBR,
   SWIR1 and SWIR1/SWIR2 for the dry-grass pixels and for v5 exposed peat on
   the Bord na Móna positive-control scenes (spring and summer 2018).
3. In the labelled v5 test (outputs/evaluation/2026-10-08-v5-test), which
   reference classes do the v5 false positives belong to, and does any
   annotator note describe dry or dead vegetation?

Usage:
    python3 scripts/45_dry_grass_check.py
"""

import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import planetary_computer as pc
from pystac_client import Client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from peatland import config, v5

spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts/31_summer_check.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)

EVAL = config.OUT_DIR / "evaluation"
RUNS = (EVAL / "2026-09-13-v3-sample", EVAL / "2026-10-07-v4-test")
TEST = EVAL / "2026-10-08-v5-test/annotation"
PC_SCENES = config.OUT_DIR / "positive-control/scenes"
PC_RUNS = ("BNMS-sp_2018", "BNMS-su_2018", "BNMW-sp_2018", "BNMW-su_2018")
OUT = EVAL / "dry-grass-check.json"
GREEN = 0.50

# keyword classes for the annotator notes of v5 false positives, tested in this order
NOTE_CLASSES = (
    ("dry or dead vegetation", ("dead", "dry grass", "senescent", "molinia", "straw", "bleached", "withered")),
    ("track or road", ("track", "road", "gravel")),
    ("revegetating cutaway", ("revegetat",)),
    ("mixed pixel at a cut face or strip edge", ("edge", "margin", "beside", "adjacent", "nearby", "just ",
                                                  "between", "next to", "mixed", "boundary", "border")),
)


def spectra(stack, order, mask):
    ix = v5.indices(stack, order)
    red = stack[..., order.index("red")].astype(float)
    nir = stack[..., order.index("nir")].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / (nir + red)
    out = {"n_px": int(mask.sum())}
    for k, a in (("ndvi", ndvi), ("nbr", ix["nbr"]), ("swir1", ix["swir1"]), ("swir1_swir2", ix["ratio"])):
        v = a[mask]
        v = v[np.isfinite(v)]
        out[k] = round(float(np.median(v)), 3) if v.size else None
    return out, ix


def dry_grass(catalog):
    rows, pooled = [], {"flagged": 0, "n": 0, "fails": Counter()}
    spec_parts = []
    for run in RUNS:
        for npz in sorted(run.glob("*.npz")):
            meta = json.loads(npz.with_suffix(".json").read_text())
            if int(meta["scene_date"][5:7]) not in (4, 5):
                continue
            order = tuple(meta["bands"])
            with np.load(npz) as z:
                stack, valid, inside = z["stack"], z["valid"].astype(bool), z["inside"].astype(bool)
                old = z["prediction_ndvi"].astype(bool) & inside
            if not old.any():
                continue
            summer, dates = sc.summer_ndvi(meta, inside.shape, catalog)
            if summer is None:
                continue
            grass = old & valid & np.isfinite(summer) & (summer >= GREEN)
            if not grass.any():
                continue
            flagged = v5.peat_mask(stack, valid, inside, order) & grass
            s, ix = spectra(stack, order, grass)
            with np.errstate(invalid="ignore"):
                fails = {"nbr_not_negative": int((grass & ~(ix["nbr"] < 0)).sum()),
                         "swir1_below_0_22": int((grass & ~(ix["swir1"] >= v5.SWIR1_MIN)).sum()),
                         "ratio_below_1_25": int((grass & ~(ix["ratio"] >= v5.SWIR_RATIO_MIN)).sum())}
            rows.append({"key": npz.stem, "site": meta["site"], "spring_date": meta["scene_date"],
                         "summer_dates": dates, "dry_grass_px": int(grass.sum()),
                         "v5_flagged_px": int(flagged.sum()),
                         "v5_flagged_share": round(float(flagged.sum() / grass.sum()), 4),
                         "fails_test": fails, "spectra": s})
            pooled["flagged"] += int(flagged.sum())
            pooled["n"] += int(grass.sum())
            pooled["fails"].update(fails)
            spec_parts.append((stack, order, grass))
            print(f"{meta['site'][:32]:32} {meta['scene_date']}: dry grass {grass.sum():5d} px, "
                  f"v5 flags {flagged.sum():4d} ({flagged.sum() / grass.sum():.1%})", flush=True)
    # pooled spectra over all dry-grass pixels
    vals = {k: [] for k in ("ndvi", "nbr", "swir1", "swir1_swir2")}
    for stack, order, grass in spec_parts:
        ix = v5.indices(stack, order)
        red = stack[..., order.index("red")].astype(float)
        nir = stack[..., order.index("nir")].astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = (nir - red) / (nir + red)
        for k, a in (("ndvi", ndvi), ("nbr", ix["nbr"]), ("swir1", ix["swir1"]), ("swir1_swir2", ix["ratio"])):
            vals[k].append(a[grass])
    pooled_spec = {k: round(float(np.nanmedian(np.concatenate(v))), 3) for k, v in vals.items() if v}
    return {"site_years": rows, "dry_grass_px": pooled["n"], "v5_flagged_px": pooled["flagged"],
            "v5_flagged_share": round(pooled["flagged"] / max(pooled["n"], 1), 4),
            "fails_test_px": dict(pooled["fails"]), "spectra_dry_grass": pooled_spec}


def peat_spectra():
    out = {}
    vals = {k: [] for k in ("ndvi", "nbr", "swir1", "swir1_swir2")}
    for name in PC_RUNS:
        npz = PC_SCENES / f"{name}.npz"
        if not npz.exists():
            continue
        meta = json.loads((config.OUT_DIR / "positive-control/areas.json").read_text())["runs"][name]
        order = tuple(meta["bands"])
        with np.load(npz) as z:
            stack, valid = z["stack"], z["valid"].astype(bool)
        peat = v5.peat_mask(stack, valid, np.ones(valid.shape, bool), order)
        s, ix = spectra(stack, order, peat)
        out[name] = {"scene_date": meta["scene_date"], **s}
        red = stack[..., order.index("red")].astype(float)
        nir = stack[..., order.index("nir")].astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = (nir - red) / (nir + red)
        for k, a in (("ndvi", ndvi), ("nbr", ix["nbr"]), ("swir1", ix["swir1"]), ("swir1_swir2", ix["ratio"])):
            vals[k].append(a[peat])
    pooled = {k: round(float(np.nanmedian(np.concatenate(v))), 3) for k, v in vals.items() if v}
    return {"scenes": out, "pooled": pooled}


def false_positives():
    labels = {r["point_id"]: r for r in csv.DictReader((TEST / "labels.csv").open(encoding="utf-8-sig"))}
    points = list(csv.DictReader((TEST / "accuracy_points.csv").open(encoding="utf-8-sig")))
    flagged = [p for p in points if p.get("prediction_v5") == "1" and p["point_id"] in labels]
    by_truth = Counter(labels[p["point_id"]]["truth"] for p in flagged)
    fps = [p for p in flagged if labels[p["point_id"]]["truth"] == "vegetated"]
    classes = Counter()
    examples = {}
    for p in fps:
        note = labels[p["point_id"]]["notes"].lower()
        cls = next((c for c, keys in NOTE_CLASSES if any(k in note for k in keys)), "other vegetation")
        classes[cls] += 1
        examples.setdefault(cls, p["point_id"])
    scored = by_truth["bare_peat"] + by_truth["vegetated"]
    return {"v5_flagged_points": len(flagged), "by_reference": dict(by_truth),
            "precision_bare_vs_vegetated": round(by_truth["bare_peat"] / scored, 3) if scored else None,
            "false_positives_vegetated": len(fps), "false_positive_note_classes": dict(classes),
            "examples": examples,
            "labels_note": "reference labels are model pre-labels from two blind passes, pending the author's review"}


def main():
    catalog = Client.open(STAC, modifier=pc.sign_inplace) if (STAC := sc.STAC) else None
    grass = dry_grass(catalog)
    peat = peat_spectra()
    fp = false_positives()
    out = {"green_ndvi": GREEN, "dry_grass": grass, "industrial_peat": peat, "v5_test_false_positives": fp}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({"dry_grass_px": grass["dry_grass_px"], "v5_flagged_px": grass["v5_flagged_px"],
                      "v5_flagged_share": grass["v5_flagged_share"], "fails_test_px": grass["fails_test_px"],
                      "spectra_dry_grass": grass["spectra_dry_grass"], "spectra_peat": peat["pooled"],
                      "false_positives": fp}, indent=2))


if __name__ == "__main__":
    main()
