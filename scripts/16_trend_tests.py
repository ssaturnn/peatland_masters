"""Step 16 — Mann-Kendall trend test per bog, written into the web GeoJSON.

For every site the non-parametric Mann-Kendall test asks whether the
GNG bare-area series shows a monotonic trend, without assuming linearity
or normality — appropriate for short annual series with outlier years.
Adds mk_z / mk_p / mk_trend ("increasing" / "decreasing" / "none",
two-sided p < 0.05) to each feature's properties in place.

Run after 10_build_dataset.py:
    python3 scripts/16_trend_tests.py
"""

import argparse
import json
import math
from pathlib import Path

GEOJSON = Path(__file__).resolve().parents[1] / "web" / "data" / "sites.geojson"
ALPHA = 0.05


def mann_kendall(xs):
    """Return (z, p, trend) for a numeric series (normal approximation,
    tie-corrected variance; series shorter than 4 points -> no trend)."""
    n = len(xs)
    if n < 4:
        return 0.0, 1.0, "none"
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = xs[j] - xs[i]
            s += (d > 0) - (d < 0)
    # tie correction
    counts = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    tie_term = sum(t * (t - 1) * (2 * t + 5) for t in counts.values() if t > 1)
    var = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var <= 0:
        return 0.0, 1.0, "none"
    z = (s - math.copysign(1, s)) / math.sqrt(var) if s != 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))          # two-sided
    trend = "none"
    if p < ALPHA:
        trend = "increasing" if z > 0 else "decreasing"
    return round(z, 2), round(p, 4), trend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=GEOJSON,
                    help="GeoJSON to annotate in place (default: web dataset)")
    args = ap.parse_args()
    geojson = args.dataset
    fc = json.loads(geojson.read_text())
    n_up = n_down = 0
    for f in fc["features"]:
        p = f["properties"]
        z, pv, trend = mann_kendall([float(v) for v in p.get("gng_series", [])])
        p["mk_z"], p["mk_p"], p["mk_trend"] = z, pv, trend
        n_up += trend == "increasing"
        n_down += trend == "decreasing"
        if trend != "none":
            print(f"  {trend:10}  z={z:5.2f} p={pv:6.4f}  {p['name']}")
    geojson.write_text(json.dumps(fc))
    print(f"\n{len(fc['features'])} sites: {n_up} increasing, "
          f"{n_down} decreasing (p<{ALPHA}) -> {geojson}")


if __name__ == "__main__":
    main()
