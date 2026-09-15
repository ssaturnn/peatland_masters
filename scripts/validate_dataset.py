"""Data-integrity validation for the generated web dataset.

Reads web/data/sites.geojson and checks invariants that must hold for
the map to be trustworthy. Also cross-checks each site's imagery-derived
area against the NPWS register (a georeferencing sanity check). Exits
non-zero on any failure, so it can gate a deploy.
"""

import sys
import argparse
import math
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peatland import boundaries, config, pipeline

GEOJSON = Path(__file__).resolve().parents[1] / "web" / "data" / "sites.geojson"
REQUIRED = [
    "name", "county", "designation", "site_ha", "years", "ndvi_series",
    "gng_series", "bare_now_ndvi_ha", "bare_now_gng_ha", "newly_ndvi_ha",
    "newly_gng_ha", "rate_ndvi_ha_yr", "rate_gng_ha_yr", "both_cov_pct",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=GEOJSON)
    parser.add_argument("--tiles", type=Path, help="Require both images and matching provenance for each site-year")
    parser.add_argument("--require-current-version", action="store_true")
    args = parser.parse_args()
    fc = json.loads(args.dataset.read_text())
    feats = fc["features"]
    problems = []

    def check(cond, msg):
        if not cond:
            problems.append(msg)

    # NPWS register areas for the area cross-check
    reg = {r["SITE_NAME"]: r["HA"] for _, r in boundaries.build_sites().iterrows()}

    for f in feats:
        p = f["properties"]
        name = p.get("name", "?")
        for k in REQUIRED:
            check(k in p, f"{name}: missing property {k}")
        if any(k not in p for k in REQUIRED):
            continue
        if args.require_current_version:
            check(p.get("detector_version") == pipeline.DETECTOR_VERSION,
                  f"{name}: stale or unversioned detector results")
        # series length matches years
        check(len(p["ndvi_series"]) == len(p["years"]),
              f"{name}: ndvi_series length != years")
        check(len(p["gng_series"]) == len(p["years"]),
              f"{name}: gng_series length != years")
        check(bool(p["years"]), f"{name}: no usable survey years")
        check(p["years"] == sorted(set(p["years"])), f"{name}: years must be unique and ordered")
        for key in ("ndvi_series", "gng_series"):
            check(all(isinstance(v, (int, float)) and math.isfinite(v)
                      and 0 <= v <= p["site_ha"] + 0.5 for v in p[key]),
                  f"{name}: invalid area in {key}")
        if args.tiles:
            dates, scene_ids = p.get("dates", []), p.get("scene_ids") or []
            check(len(dates) == len(p["years"]), f"{name}: dates missing or misaligned")
            check(len(scene_ids) == len(p["years"]), f"{name}: exact scene IDs missing or misaligned")
            for i, year in enumerate(p["years"]):
                stem = f"{p.get('code', '')}_{year}"
                for suffix in (".jpg", "c.jpg"):
                    check((args.tiles / (stem + suffix)).exists(), f"{name}: missing image {stem + suffix}")
                manifest = args.tiles / (stem + ".json")
                check(manifest.exists(), f"{name}: missing image provenance for {year}")
                if manifest.exists() and i < len(scene_ids) and i < len(dates):
                    try:
                        identity = json.loads(manifest.read_text())
                        check(identity == {"scene_id": scene_ids[i], "scene_date": dates[i],
                                           "detector_version": p.get("detector_version")},
                              f"{name}: image provenance mismatch for {year}")
                    except (ValueError, OSError):
                        check(False, f"{name}: unreadable image provenance for {year}")
        # bare area cannot exceed the site area
        for k in ("bare_now_ndvi_ha", "bare_now_gng_ha"):
            check(p[k] <= p["site_ha"] + 0.5,
                  f"{name}: {k}={p[k]} exceeds site_ha={p['site_ha']}")
        # coverage is a percentage
        check(0 <= p["both_cov_pct"] <= 100,
              f"{name}: both_cov_pct out of range: {p['both_cov_pct']}")
        # areas non-negative
        for k in ("newly_ndvi_ha", "newly_gng_ha", "reveg_ndvi_ha",
                  "reveg_gng_ha"):
            check(p.get(k, 0) >= 0, f"{name}: {k} negative")
        # area cross-check vs NPWS register (±8%)
        if name in reg and reg[name] > 0:
            rel = abs(p["site_ha"] - reg[name]) / reg[name]
            check(rel <= 0.08,
                  f"{name}: site_ha {p['site_ha']} vs NPWS {reg[name]:.0f} "
                  f"({100*rel:.1f}% off)")

    print(f"Validated {len(feats)} features.")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for m in problems[:40]:
            print("  -", m)
        sys.exit(1)
    print("All invariants hold; areas match the NPWS register within 8%.")


if __name__ == "__main__":
    main()
