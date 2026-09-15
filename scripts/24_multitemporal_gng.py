"""Two-date seasonal candidates developed on calibration scenes only.

Actions:
  fetch-calibration  early and late scenes for the four calibration site-years
  calibrate          diagnostics, a green-up sensitivity table and figures
  freeze             record parameters, scene rules and hashes (before any
                     held-out imagery is fetched)
  apply              fetch held-out pairs, run the frozen detector on all
                     eight site-years, compare with the frozen v3 masks
  tidal              0/1/2-pixel shoreline buffer on Tullaghan Bay 2026
  predictions        add the frozen masks' values to a sampled points CSV so
                     scripts/14_score_points.py can score them
"""

import argparse
import datetime as dt
import hashlib
import inspect
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import reproject, transform_bounds, Resampling
from peatland import imagery, pipeline
from peatland import multitemporal as mt
from peatland.planet import private_path, write_json
from peatland.preprocess import assess_scene

RUN = ROOT / "outputs/evaluation/2026-09-13-v3-sample"
OUT = ROOT / "outputs/multitemporal_gng"
CALIBRATION = {"002352_2022", "002381_2023", "000221_2025", "001567_2026"}
FETCH_VERSION = "2026-09-15-season-pairs-v5"
WINDOWS = {"early": ("04-01", "05-15", "04-25"), "late": ("06-01", "06-30", "06-15")}
LATE_FALLBACK = ("07-01", "07-15")
QUICK_CLEAR = 0.85
RULES = {
    "fetch_version": FETCH_VERSION,
    "windows": {season: list(w) for season, w in WINDOWS.items()},
    "late_fallback_window": list(LATE_FALLBACK),
    "order": "June before the July fallback, then nearest to the target date, then scene cloud cover",
    "early_acceptance": "the frozen scene when it falls in the window, else the v3 scene gates on the whole bog",
    "late_acceptance": ("the v3 scene gates on the whole bog; a candidate-pixel variant was tested on "
                        "the calibration scenes and rejected (docs/multitemporal-gng.md)"),
    "quick_screen": f"SCL clear fraction >= {QUICK_CLEAR} on the bog before a full read",
}


def metadata(run):
    return {p.stem: json.loads(p.read_text()) for p in sorted(run.glob("*.json"))
            if p.with_suffix(".npz").exists()}


def current_params(out):
    """Frozen parameters once recorded, else the defaults under calibration."""
    path = out / "frozen_params.json"
    if path.exists():
        return mt.Parameters(**json.loads(path.read_text())["parameters"])
    return mt.Parameters()


def read_aligned(item, shape, transform, crs):
    """Read small windows and reproject each band onto the exact frozen grid."""
    bbox = transform_bounds(crs, "EPSG:4326", *array_bounds(*shape, transform))
    def read(band):
        categorical = band == "scl"
        method = Resampling.nearest if categorical else Resampling.bilinear
        arr, src_t, src_crs = imagery.read_window(item, band, bbox, "planetary",
                                                out_shape=shape, resampling=method)
        result = np.full(shape, 0 if categorical else np.nan, dtype=np.float32)
        reproject(arr, result, src_transform=src_t, src_crs=src_crs,
                  dst_transform=transform, dst_crs=crs, src_nodata=0 if categorical else np.nan,
                  dst_nodata=0 if categorical else np.nan, resampling=method)
        return result
    with ThreadPoolExecutor(max_workers=3) as pool:
        arrays = list(pool.map(read, pipeline.BANDS + ["scl"]))
    return np.stack(arrays[:6], axis=-1), arrays[-1]


def search_items(catalog, bbox, year, season):
    """All candidate scenes of a season window in the declared preference order."""
    start, end, target = WINDOWS[season]
    items = list(catalog.search(collections=["sentinel-2-l2a"], bbox=bbox,
                 datetime=f"{year}-{start}/{year}-{end}T23:59:59Z",
                 query={"eo:cloud_cover": {"lte": 100}}, max_items=300).items())
    if season == "late":
        a, b = LATE_FALLBACK
        items += list(catalog.search(collections=["sentinel-2-l2a"], bbox=bbox,
                      datetime=f"{year}-{a}/{year}-{b}T23:59:59Z",
                      query={"eo:cloud_cover": {"lte": 100}}, max_items=100).items())
    day0 = dt.date.fromisoformat(f"{year}-{target}")
    items.sort(key=lambda it: (season == "late" and it.properties["datetime"][5:7] != "06",
                              abs((dt.date.fromisoformat(it.properties["datetime"][:10]) - day0).days),
                              it.properties.get("eo:cloud_cover", 100), it.id))
    return items


def fetch_pair(key, meta, run, out):
    folder = out / "pairs" / key
    folder.mkdir(parents=True, exist_ok=True)
    with np.load(run / f"{key}.npz") as z:
        inside, transform = z["inside"], rasterio.Affine(*z["transform"][:6])
        frozen_stack, frozen_valid = z["stack"], z["valid"]
    bbox = transform_bounds(meta["crs"], "EPSG:4326", *array_bounds(*inside.shape, transform))
    year = meta["year"]
    for season, (start, end, target) in WINDOWS.items():
        path = folder / f"{season}.npz"
        record_path = path.with_suffix(".json")
        if record_path.exists():
            old = json.loads(record_path.read_text())
            if old.get("version") == FETCH_VERSION and old.get("status") in (
                    "available", "no_usable_scene", "no_early_scene"):
                continue
        path.unlink(missing_ok=True)
        record = {"version": FETCH_VERSION, "key": key, "role": meta["role"],
                  "season": season, "window": [f"{year}-{start}", f"{year}-{end}"],
                  "target_date": f"{year}-{target}", "attempts": []}
        if season == "late":
            record["fallback_window"] = [f"{year}-{d}" for d in LATE_FALLBACK]
        if season == "early" and start <= meta["scene_date"][5:] <= end:
            np.savez_compressed(path, stack=frozen_stack, valid=frozen_valid)
            record.update(status="available", scene_id=meta["scene_id"], date=meta["scene_date"],
                          source="frozen scene", acceptance="frozen scene",
                          clear_fraction=float((inside & frozen_valid).sum() / inside.sum()))
            write_json(record_path, record)
            print(f"{key} early: reused {meta['scene_date']}", flush=True)
            continue
        candidates = None
        if season == "late":
            if not (folder / "early.npz").exists():
                record["status"] = "no_early_scene"
                write_json(record_path, record)
                print(f"{key} late: skipped, no early scene", flush=True)
                continue
            with np.load(folder / "early.npz") as e:
                candidates, _ = mt.early_candidates(e["stack"], e["valid"], inside, current_params(out))
            record["early_candidate_pixels"] = int(candidates.sum())
        record["acceptance"] = "whole bog"
        catalog = imagery.open_catalog("planetary")
        seen = set()
        for item in search_items(catalog, bbox, year, season):
            # Distinct tiles remain candidates; only repeated processing of a tile/day is skipped.
            identity = (item.properties["datetime"][:10], item.properties.get("s2:mgrs_tile"))
            if identity in seen:
                continue
            seen.add(identity)
            attempt = {"scene_id": item.id, "date": item.properties["datetime"][:10]}
            try:
                quick = assess_scene(item, bbox, "planetary", inside.shape, aoi_mask=inside)
                if quick["aoi_clear"] < QUICK_CLEAR:
                    attempt.update(scl_clear_fraction=quick["aoi_clear"], passes=False)
                    record["attempts"].append(attempt)
                    write_json(record_path, record)
                    continue
                stack, scl = read_aligned(item, inside.shape, transform, meta["crs"])
                quality = pipeline.scene_quality(stack, pipeline.BANDS, scl, inside, item)
                attempt.update({k: v for k, v in quality.items() if k != "valid"})
                attempt["passes"] = bool(pipeline.scene_passes(quality))
                if candidates is not None:
                    attempt["candidate_cover"] = mt.candidate_cover(candidates, quality["valid"])
                record["attempts"].append(attempt)
                write_json(record_path, record)
                if attempt["passes"]:
                    np.savez_compressed(path, stack=stack, valid=quality["valid"], scl=scl)
                    record.update(status="available", scene_id=item.id, date=attempt["date"],
                                  source="Planetary Computer, reprojected to frozen grid",
                                  fallback=season == "late" and attempt["date"][5:7] != "06",
                                  clear_fraction=quality["clear_fraction"])
                    break
            except Exception as exc:
                attempt["error_type"] = type(exc).__name__
                record["attempts"].append(attempt)
                write_json(record_path, record)
            print(f"{key} {season}: checked {attempt['date']} passes={attempt.get('passes', False)}",
                  flush=True)
        if not path.exists():
            record["status"] = "no_usable_scene"
        write_json(record_path, record)
        print(f"{key} {season}: {record['status']} {record.get('date', '')}", flush=True)


def fetch_code_sha256():
    """Hash of the scene-selection code, so apply can prove it is unchanged."""
    source = "".join(inspect.getsource(f) for f in (fetch_pair, search_items, read_aligned))
    return hashlib.sha256(source.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("fetch-calibration", "calibrate", "freeze", "apply",
                                       "tidal", "predictions"))
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--points", type=Path, help="sampled points CSV (predictions action)")
    ap.add_argument("--output", type=Path, help="CSV to write (predictions action)")
    args = ap.parse_args()
    out = private_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metas = metadata(args.run)
    calibration = {k for k, m in metas.items() if m["role"] == "calibration"}
    if calibration != CALIBRATION:
        raise ValueError("Unexpected calibration split")
    if args.action == "fetch-calibration":
        for key in sorted(calibration):
            fetch_pair(key, metas[key], args.run, out)
        return
    from peatland.temporal_experiment import run_action
    rules = dict(RULES, fetch_code_sha256=fetch_code_sha256())
    run_action(args.action, metas, args.run, out, fetch_pair, rules=rules,
               points=args.points, output=args.output)


if __name__ == "__main__":
    main()
