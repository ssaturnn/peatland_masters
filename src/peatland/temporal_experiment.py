"""Calibrate, freeze and apply the two-date seasonal detector (scripts/24).

Development uses the four calibration site-years only. `freeze` records the
parameters, the scene rules and the code hashes before any held-out imagery
is fetched; `apply` refuses to run without that record or after any of them
has changed. Nothing here reads reference labels: the outputs are masks,
areas and agreement with the frozen v3 masks, not accuracy.
"""

import csv
import datetime as dt
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import multitemporal as mt
from . import pipeline
from .planet import write_json

ROOT = Path(__file__).resolve().parents[2]
PIXEL_HA = 0.01
FROZEN = "frozen_params.json"
GREENUP_SWEEP = (0.05, 0.10, 0.15)
TIDAL_KEY = "001567_2026"
METHODS = ("trajectory_gng", "trajectory_rules")


def _ha(mask):
    return round(int(np.asarray(mask, bool).sum()) * PIXEL_HA, 2)


def _rounded(values, digits=4):
    return {k: round(v, digits) if isinstance(v, float) else v for k, v in values.items()}


def _keys(metas, role):
    return sorted(k for k, m in metas.items() if m["role"] == role)


def detector_sha256(params):
    """Detector code plus parameters: what `apply` must reproduce exactly."""
    h = hashlib.sha256((ROOT / "src/peatland/multitemporal.py").read_bytes())
    h.update(json.dumps(params.as_dict(), sort_keys=True).encode())
    return h.hexdigest()


def load_pair(out, key):
    """Early and late stacks, or None with the records that explain why not."""
    folder = Path(out) / "pairs" / key
    records = {}
    for season in ("early", "late"):
        path = folder / f"{season}.json"
        records[season] = json.loads(path.read_text()) if path.exists() else {"status": "not_fetched"}
    if not all(records[s].get("status") == "available" and (folder / f"{s}.npz").exists()
               for s in ("early", "late")):
        return None, records
    data = {}
    for season in ("early", "late"):
        with np.load(folder / f"{season}.npz") as z:
            data[season] = (z["stack"].astype(np.float32), z["valid"].astype(bool))
    return data, records


def _scene(record):
    return {k: record.get(k) for k in ("status", "date", "scene_id", "source", "acceptance",
                                       "fallback", "clear_fraction")}


def evaluate(key, meta, run, out, params):
    """Run the detector on one site-year; a missing pair is unavailable, never zero."""
    data, records = load_pair(out, key)
    with np.load(Path(run) / f"{key}.npz") as z:
        inside, v3 = z["inside"].astype(bool), z["prediction_gng"].astype(bool)
    row = {"key": key, "site": meta["site"], "year": meta["year"], "role": meta["role"],
           "v3_scene_date": meta["scene_date"], "v3_gng_ha": _ha(v3),
           "early": _scene(records["early"]), "late": _scene(records["late"])}
    if data is None:
        missing = [s for s in ("early", "late") if records[s].get("status") != "available"]
        row.update(status="unavailable", reason=f"no usable {' or '.join(missing)} scene")
        early_path = Path(out) / "pairs" / key / "early.npz"
        if records["early"].get("status") == "available" and early_path.exists():
            with np.load(early_path) as z:
                cand, _ = mt.early_candidates(z["stack"], z["valid"], inside, params)
            row["early_candidates_ha"] = _ha(cand)
            if not cand.any():
                # persistent candidates are a subset of early ones: empty by construction
                row.update(status="no_early_candidates", persistent_gng_ha=0.0,
                           persistent_rules_ha=0.0,
                           reason="no early candidates, so the result is empty whatever a "
                                  "late scene would show")
        return row, None
    (early, valid_early), (late, valid_late) = data["early"], data["late"]
    result = mt.detect_trajectory(early, late, valid_early, valid_late, inside, params)
    support, cand = result["support"], result["early_rule"]
    seen = cand & support
    d = result["diagnostics"]
    row.update(
        status="available",
        support_fraction=round(float(support.sum() / inside.sum()), 4),
        early_candidates_ha=_ha(cand),
        early_candidates_seen_late=round(float(seen.sum() / cand.sum()), 4) if cand.any() else None,
        persistent_rules_ha=_ha(result["rules"]),
        persistent_gng_ha=_ha(result["gng"]),
        dropped_by_late_date_ha=_ha(seen & ~result["rules"]),
        thresholds={"early": round(d["early_threshold"], 4), "late": round(d["late_threshold"], 4)},
        gng_nodes=d["nodes"], bare_nodes=d["bare_nodes"],
        v3_outside_support_ha=_ha(v3 & ~support),
        vs_v3_gng=_rounded(mt.mask_comparison(v3, result["gng"], support)),
        vs_v3_rules=_rounded(mt.mask_comparison(v3, result["rules"], support)))
    return row, result


def _rgb(stack):
    rgb = np.nan_to_num(np.asarray(stack, float)[..., [2, 1, 0]]) / 10000
    return np.clip(rgb * 3.2, 0, 1) ** (1 / 1.4)


def figure(path, row, early, late, frozen, v3, result):
    """Early, late, frozen v3 and persistence panels on one grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dropped = result["early_rule"] & result["support"] & ~result["gng"]
    panels = (
        (early, f"Early {row['early']['date']}", ()),
        (late, f"Late {row['late']['date']}", ()),
        (frozen, f"Frozen v3 GNG, {row['v3_scene_date']}", ((v3, (0.16, 0.5, 1.0)),)),
        (early, "Persistent (amber), dropped by the late date (magenta)",
         ((dropped, (0.9, 0.2, 0.8)), (result["gng"], (1.0, 0.77, 0.24)))),
    )
    h, w = v3.shape
    fig, axes = plt.subplots(1, 4, figsize=(16, min(8.0, 4.0 * h / w + 0.8)))
    for ax, (stack, title, overlays) in zip(axes, panels):
        ax.imshow(_rgb(stack))
        for mask, color in overlays:
            if mask.any():
                layer = np.zeros((*mask.shape, 4))
                layer[mask] = (*color, 0.85)
                ax.imshow(layer, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
    fig.suptitle(f"{row['site']} {row['year']} ({row['role']}), {mt.VERSION}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _figure_for(key, row, result, run, out, path):
    data, _ = load_pair(out, key)
    with np.load(Path(run) / f"{key}.npz") as z:
        frozen, v3 = z["stack"], z["prediction_gng"].astype(bool)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure(path, row, data["early"][0], data["late"][0], frozen, v3, result)


def calibrate(metas, run, out, fetch_pair, rules, **_):
    keys = _keys(metas, "calibration")
    for key in keys:
        fetch_pair(key, metas[key], run, out)
    params, folder, rows = mt.Parameters(), out / "calibration", []
    for key in keys:
        row, result = evaluate(key, metas[key], run, out, params)
        if result is not None:
            _figure_for(key, row, result, run, out, folder / "figures" / f"{key}.png")
            (early, ve), (late, vl) = load_pair(out, key)[0].values()
            with np.load(Path(run) / f"{key}.npz") as z:
                inside = z["inside"].astype(bool)
            row["greenup_sensitivity"] = {}
            for g in GREENUP_SWEEP:
                swept = mt.detect_trajectory(early, late, ve, vl, inside, replace(params, max_greenup=g))
                row["greenup_sensitivity"][f"{g:.2f}"] = {"rules_ha": _ha(swept["rules"]),
                                                          "gng_ha": _ha(swept["gng"])}
        rows.append(row)
        print(f"{key}: {row['status']} persistent GNG {row.get('persistent_gng_ha')} ha, "
              f"v3 {row['v3_gng_ha']} ha", flush=True)
    write_json(folder / "summary.json", {
        "version": mt.VERSION, "parameters": params.as_dict(), "scene_rules": rules,
        "site_years": rows,
        "note": "Calibration site-years only. Areas and agreement with the frozen v3 masks; "
                "no reference labels were used."})


def freeze(metas, run, out, fetch_pair, rules, **_):
    held = _keys(metas, "held-out")
    fetched = [k for k in held if (out / "pairs" / k).exists()]
    if fetched:
        raise ValueError(f"Held-out pairs already exist ({', '.join(fetched)}): "
                         "freezing now would not be blind")
    params = mt.Parameters()
    record = {"version": mt.VERSION, "parameters": params.as_dict(),
              "detector_sha256": detector_sha256(params), "scene_rules": rules,
              "calibration_site_years": _keys(metas, "calibration"), "held_out_site_years": held,
              "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "note": "Written before any held-out imagery was fetched. apply refuses to run "
                      "with different detector code, parameters or scene rules."}
    path = out / FROZEN
    if path.exists():
        old = json.loads(path.read_text())
        if (old.get("detector_sha256"), old.get("scene_rules")) == (record["detector_sha256"], rules):
            print("Already frozen with the same detector and scene rules")
            return
        raise ValueError("A different frozen record exists; remove it deliberately to re-freeze")
    write_json(path, record)
    print(f"Frozen {mt.VERSION} -> {path}")


def check_frozen(out, rules):
    """The frozen parameters, or an error if anything changed since freeze."""
    path = out / FROZEN
    if not path.exists():
        raise ValueError("Run freeze first: apply needs frozen_params.json")
    record = json.loads(path.read_text())
    params = mt.Parameters(**record["parameters"])
    if record["detector_sha256"] != detector_sha256(params):
        raise ValueError("Detector code or parameters changed since freeze")
    if record["scene_rules"] != rules:
        raise ValueError("Scene rules or scene-selection code changed since freeze")
    return params, record


def apply(metas, run, out, fetch_pair, rules, **_):
    params, record = check_frozen(out, rules)
    for key in sorted(metas):
        fetch_pair(key, metas[key], run, out)
    rows = []
    for key in sorted(metas):
        row, result = evaluate(key, metas[key], run, out, params)
        if result is not None:
            with np.load(Path(run) / f"{key}.npz") as z:
                inside, transform = z["inside"], z["transform"]
            (out / "masks").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / "masks" / f"{key}.npz",
                                prediction_trajectory_gng=result["gng"],
                                prediction_trajectory_rules=result["rules"],
                                early_candidates=result["early_rule"], support=result["support"],
                                inside=inside, transform=transform)
            _figure_for(key, row, result, run, out, out / "figures" / f"{key}.png")
        rows.append(row)
        print(f"{key} ({row['role']}): {row['status']} persistent GNG "
              f"{row.get('persistent_gng_ha')} ha, v3 {row['v3_gng_ha']} ha", flush=True)
    write_json(out / "apply_summary.json", {
        "version": mt.VERSION, "detector_sha256": record["detector_sha256"],
        "frozen_at": record["frozen_at"], "site_years": rows,
        "note": "Frozen parameters on all eight site-years. Agreement with the frozen v3 masks "
                "on common clear support; no reference labels were used."})


def tidal(metas, run, out, **_):
    """0/1/2-pixel growth of the ever-water exclusion, calibration scene only."""
    key = TIDAL_KEY
    if metas[key]["role"] != "calibration":
        raise ValueError("The tidal buffer is evaluated on a calibration scene only")
    with np.load(Path(run) / f"{key}.npz") as z:
        stack, valid = z["stack"], z["valid"].astype(bool)
        inside, v3 = z["inside"].astype(bool), z["prediction_gng"].astype(bool)
    record = json.loads((ROOT / "outputs/cache-release-v3" / f"{key.split('_')[0]}.json").read_text())
    water = pipeline.mask_from_b64(record["tidal_mask_b64"], inside.shape)
    rows, masks = [], {}
    for px in (0, 1, 2):
        keep = mt.buffered_inside(inside, water, px)
        gng = pipeline.gng_bare(stack, pipeline.BANDS, valid, keep)
        rules = pipeline.rules_bare(stack, pipeline.BANDS, valid, keep)
        ring = inside & ~keep
        masks[px] = gng
        rows.append({"buffer_px": px, "buffer_m": 10 * px, "excluded_ring_ha": _ha(ring),
                     "gng_ha": _ha(gng), "rules_ha": _ha(rules),
                     "gng_change_vs_v3_ha": round(_ha(gng) - _ha(v3), 2),
                     "v3_gng_inside_ring_ha": _ha(v3 & ring),
                     "gng_changed_outside_ring_ha": _ha((gng ^ v3) & ~ring)})
        print(rows[-1], flush=True)
    write_json(out / "tidal_buffer.json", {
        "version": mt.TIDAL_VERSION, "site_year": key, "scene_date": metas[key]["scene_date"],
        "reproduces_frozen_v3": bool(np.array_equal(masks[0], v3)), "rows": rows,
        "note": "Detection area = frozen site area minus the ever-water mask grown by N pixels "
                "(8-neighbour). Candidate areas only; no reference labels."})


def add_predictions(rows, masks_dir):
    """Frozen trajectory values at each sampled pixel; '' where unobserved, never 0."""
    cache = {}
    for row in rows:
        key = row["point_id"].split("_r")[0]
        if key not in cache:
            path = Path(masks_dir) / f"{key}.npz"
            cache[key] = dict(np.load(path)) if path.exists() else None
        masks = cache[key]
        y, x = int(row["pixel_row"]), int(row["pixel_col"])
        for method in METHODS:
            value = ""
            if masks is not None and masks["support"][y, x]:
                value = str(int(masks[f"prediction_{method}"][y, x]))
            row[f"prediction_{method}"] = value
    return rows


def predictions(metas, run, out, points=None, output=None, **_):
    if points is None or output is None:
        raise ValueError("predictions needs --points and --output")
    with Path(points).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0]) + [f"prediction_{m}" for m in METHODS if f"prediction_{m}" not in rows[0]]
    add_predictions(rows, out / "masks")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    observed = sum(r["prediction_trajectory_gng"] != "" for r in rows)
    print(f"{observed} of {len(rows)} points have a trajectory value -> {output}")


def run_action(action, metas, run, out, fetch_pair, rules, points=None, output=None):
    actions = {"calibrate": calibrate, "freeze": freeze, "apply": apply,
               "tidal": tidal, "predictions": predictions}
    if action not in actions:
        raise ValueError(f"Unknown action: {action}")
    actions[action](metas, Path(run), Path(out), fetch_pair=fetch_pair, rules=rules,
                    points=points, output=output)
