"""Data-integrity validation for the generated web dataset.

Reads web/data/sites.geojson and checks invariants that must hold for
the map to be trustworthy. Also cross-checks each site's imagery-derived
area against the NPWS register (a georeferencing sanity check). Exits
non-zero on any failure, so it can gate a deploy.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peatland import boundaries, config

GEOJSON = Path(__file__).resolve().parents[1] / "web" / "data" / "sites.geojson"
REQUIRED = [
    "name", "county", "designation", "site_ha", "years", "ndvi_series",
    "gng_series", "bare_now_ndvi_ha", "bare_now_gng_ha", "newly_ndvi_ha",
    "newly_gng_ha", "rate_ndvi_ha_yr", "rate_gng_ha_yr", "both_cov_pct",
]


def main():
    fc = json.loads(GEOJSON.read_text())
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
        # series length matches years
        check(len(p["ndvi_series"]) == len(p["years"]),
              f"{name}: ndvi_series length != years")
        check(len(p["gng_series"]) == len(p["years"]),
              f"{name}: gng_series length != years")
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
