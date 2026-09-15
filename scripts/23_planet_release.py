"""Plan, order and collect clipped PlanetScope for the private release map."""

import argparse
import datetime as dt
import fcntl
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shapely.geometry import shape, mapping
from peatland import boundaries, planet

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/planet_release"


def targets(features, existing):
    """Latest card years first, then each site's documented survey years."""
    latest, documented = [], []
    for feature in features:
        p = feature["properties"]
        for year in [p["years"][-1]] + [y for y in (2021, 2022)
                if p.get(f"plots_{y}") is not None and y != p["years"][-1]]:
            key = f"{p['code']}_{year}"
            if key in existing or year not in p["years"]:
                continue
            i = p["years"].index(year)
            target = {"key": key, "code": p["code"], "site": p["name"], "year": year,
                      "s2_date": p["dates"][i], "sentinel2_scene": p["scene_ids"][i],
                      "priority": "latest" if year == p["years"][-1] else "documented"}
            (latest if target["priority"] == "latest" else documented).append(target)
    return latest + documented


def plan(out):
    path = out / "plan.json"
    if path.exists():
        return json.loads(path.read_text())
    features = json.loads((ROOT / "web/data/sites.geojson").read_text())["features"]
    existing = json.loads((ROOT / "web/private/planet/manifest.json").read_text())["tiles"]
    sites = boundaries.build_sites().set_index("SITECODE")
    rows = targets(features, existing)
    for r in rows:
        card = json.loads((ROOT / f"web/data/tiles/{r['key']}.json").read_text())
        if card["scene_id"] != r["sentinel2_scene"] or card["scene_date"] != r["s2_date"]:
            raise ValueError(f"{r['key']}: release/card identity mismatch")
        aoi, tolerance = planet.buffered_aoi(sites.loc[r["code"]].geometry)
        r.update(aoi=mapping(aoi), aoi_area_km2=planet.area_km2(aoi),
                 vertices=planet.vertex_count(aoi), simplify_m=tolerance, status="unsearched")
    result = {"version": planet.VERSION, "cap_km2": 1300, "buffer_m": 250,
              "area_basis": "Full geodesic clip AOI per ordered frame, including overlaps",
              "harmonized": False, "existing_keys": sorted(existing), "targets": rows,
              "target_aoi_km2": sum(r["aoi_area_km2"] for r in rows), "search_complete": False}
    planet.write_json(path, result)
    planet.write_json(out / "ledger.json", {"cap_km2": 1300, "reserved_km2": 0, "orders": {}})
    return result


def search_all(out, result):
    s = planet.session()
    for r in result["targets"]:
        if r["status"] != "unsearched":
            continue
        aoi = shape(r["aoi"])
        candidates = planet.search(s, aoi, dt.date.fromisoformat(r["s2_date"]))
        choice = planet.choose(candidates, aoi)
        r["search_count"] = len(candidates)
        r["downloadable_count"] = sum(c["download"] for c in candidates)
        if choice:
            date, frames, cover = choice
            r.update(date=date, delta_days=(dt.date.fromisoformat(date) - dt.date.fromisoformat(r["s2_date"])).days,
                     cover=cover, items=[{k: v for k, v in c.items() if k != "geom"} for c in frames],
                     reserved_km2=r["aoi_area_km2"] * len(frames), status="available")
        else:
            r["status"] = "no_clear_coverage"
        planet.write_json(out / "plan.json", result)
        print(f"{r['key']}: {r['status']} {r.get('date', '')}", flush=True)
    used = 0
    for r in result["targets"]:
        if r["status"] in ("available", "budget_excluded"):
            if used + r["reserved_km2"] <= result["cap_km2"]:
                used += r["reserved_km2"]
                r["status"] = "available"
            else:
                r["status"] = "budget_excluded"
    result.update(search_complete=True, planned_order_km2=used)
    planet.write_json(out / "plan.json", result)
    print(f"Complete plan: {used:.6f} km² of {result['cap_km2']} km²", flush=True)


def order_all(out, result):
    if not result["search_complete"]:
        raise ValueError("Complete the entire search and quota plan before ordering")
    ledger_path = out / "ledger.json"
    ledger = json.loads(ledger_path.read_text())
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    s = planet.session()
    for r in result["targets"]:
        key = r["key"]
        if key in ledger["orders"]:
            continue
        if r["status"] != "available":
            manifest[key] = {"site": r["site"], "s2_date": r["s2_date"],
                             "status": r["status"], "aoi_area_km2": r["aoi_area_km2"]}
            planet.write_json(manifest_path, manifest)
            continue
        reservation = planet.reserve_area(ledger, key, r["reserved_km2"])
        name = f"peatland-private-release-20260915-{key}"
        reservation["name"] = name
        planet.write_json(ledger_path, ledger)
        try:
            oid, bundle = planet.place_order(s, name, [c["id"] for c in r["items"]], shape(r["aoi"]))
        except Exception as exc:
            reservation["status"] = "unresolved_submission"
            reservation["error_type"] = type(exc).__name__
            planet.write_json(ledger_path, ledger)
            # No automatic resubmission after a possibly accepted POST.
            raise RuntimeError(f"{key}: submission unresolved; reservation retained") from None
        reservation.update(order_id=oid, bundle=bundle, status="ordered")
        planet.write_json(ledger_path, ledger)
        manifest[key] = {k: r[k] for k in ("site", "s2_date", "date", "delta_days", "cover",
                                          "aoi_area_km2", "sentinel2_scene")}
        manifest[key].update(order_id=oid, order_state="ordered", bundle=bundle,
                            items=[{"id": c["id"], "acquired": c["acquired"],
                                    "clear_percent": c["clear"], "instrument": c["instrument"]}
                                   for c in r["items"]], citation=planet.CITATION,
                            licence=planet.LICENCE, harmonized=False, version=planet.VERSION)
        planet.write_json(manifest_path, manifest)
        print(f"{key}: ordered; cumulative reserved area {ledger['reserved_km2']:.6f} km²", flush=True)


def collect(out):
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    ledger = json.loads((out / "ledger.json").read_text())
    s = planet.session()
    pending = 0
    for key, rec in manifest.items():
        if not rec.get("order_id") or rec.get("mosaic"):
            continue
        try:
            response = s.get(f"{planet.ORDERS}/{rec['order_id']}", timeout=60)
            response.raise_for_status()
            od = response.json()
            state = od.get("state")
            rec["order_state"] = state
            ledger["orders"][key]["status"] = state
            if state in ("success", "partial"):
                files = planet.download(s, od, out / key)
                ids = [c["id"] for c in rec["items"]]
                files.sort(key=lambda f: next((i for i, name in enumerate(ids) if f.name.startswith(name)), 999))
                if files:
                    destination = out / key / "ps_mosaic.tif"
                    planet.mosaic(files, destination)
                    rec["mosaic"] = str(destination.relative_to(ROOT))
                    rec["downloaded_frames"] = len(files)
            elif state not in ("failed", "cancelled"):
                pending += 1
            planet.write_json(manifest_path, manifest)
            planet.write_json(out / "ledger.json", ledger)
            print(f"{key}: {state}; mosaic={bool(rec.get('mosaic'))}", flush=True)
        except Exception as exc:
            # Never print signed delivery URLs or request headers.
            pending += 1
            print(f"{key}: collection deferred ({type(exc).__name__})", flush=True)
    return pending


def fetch_udm2(out):
    """Download Planet's UDM2 quality masks for collected orders; places no orders."""
    manifest = json.loads((out / "manifest.json").read_text())
    s = planet.session()
    missing = 0
    for key, rec in manifest.items():
        if not rec.get("order_id") or not rec.get("mosaic"):
            continue
        if any((out / key).glob("*_udm2*.tif")):
            continue
        try:
            response = s.get(f"{planet.ORDERS}/{rec['order_id']}", timeout=60)
            response.raise_for_status()
            files = planet.download(s, response.json(), out / key, product="udm2")
            print(f"{key}: {len(files)} UDM2 file(s)", flush=True)
        except Exception as exc:
            # Never print signed delivery URLs or request headers.
            missing += 1
            print(f"{key}: UDM2 deferred ({type(exc).__name__})", flush=True)
    return missing


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("plan", "order", "collect", "udm2"))
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    out = planet.private_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "collect":
            print(f"Pending: {collect(out)}")
        elif args.action == "udm2":
            print(f"Missing: {fetch_udm2(out)}")
        else:
            result = plan(out)
            if not result["search_complete"]:
                search_all(out, result)
            if args.action == "order":
                order_all(out, result)


if __name__ == "__main__":
    main()
