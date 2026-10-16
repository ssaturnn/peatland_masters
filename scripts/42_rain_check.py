"""Step 42 — is the scene-to-scene jumping of exposed peat rain or working?

For a bog with a field-exposure series (outputs/time-series/{code}_fields.json,
share of the worked fields flagged v5 exposed peat on every clear scene), it
takes daily rainfall and reference evapotranspiration from the ERA5 reanalysis
(Open-Meteo archive API) at the bog and compares, on April-September scenes:

- the share against rain in the 3 and 7 days before the acquisition;
- the share on dry scenes (< 2 mm in 3 days) and wet scenes (>= 5 mm) in two
  periods, so that a decline in working can be told apart from wetter years.

Usage:
    python3 scripts/42_rain_check.py 001244 --split 2024
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import requests
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from peatland import boundaries, config

OUT = config.OUT_DIR / "time-series"
API = "https://archive-api.open-meteo.com/v1/archive"
DRY_MM, WET_MM = 2.0, 5.0


def window(series, d, days):
    """Sum over the `days` days before d (the acquisition day is excluded)."""
    return sum(series.get(str(d - dt.timedelta(days=i))) or 0.0 for i in range(1, days + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("code")
    ap.add_argument("--split", type=int, default=2024, help="first year of the second period")
    args = ap.parse_args()
    rows = json.loads((OUT / f"{args.code}_fields.json").read_text())["rows"]
    sites = boundaries.build_sites()
    geom = sites[sites.SITECODE == args.code].geometry.iloc[0]
    pt = gpd.GeoSeries([geom.centroid], crs=config.ITM).to_crs("EPSG:4326").iloc[0]
    first, last = rows[0][0][:4] + "-01-01", rows[-1][0]
    w = requests.get(API, params={"latitude": round(pt.y, 4), "longitude": round(pt.x, 4),
                                  "start_date": first, "end_date": last,
                                  "daily": "precipitation_sum,et0_fao_evapotranspiration",
                                  "timezone": "Europe/Dublin"}, timeout=120).json()["daily"]
    rain = dict(zip(w["time"], w["precipitation_sum"]))
    et0 = dict(zip(w["time"], w["et0_fao_evapotranspiration"]))

    scenes = []
    for day, share, _ in rows:
        d = dt.date.fromisoformat(day)
        if 4 <= d.month <= 9:
            scenes.append({"date": day, "share": share, "rain_3d": round(window(rain, d, 3), 1),
                           "rain_7d": round(window(rain, d, 7), 1),
                           "balance_7d": round(window(rain, d, 7) - window(et0, d, 7), 1)})
    share = np.array([s["share"] for s in scenes])
    out = {"code": args.code, "scenes": len(scenes), "rain_source": "ERA5 via Open-Meteo archive"}
    for key in ("rain_3d", "rain_7d", "balance_7d"):
        r = spearmanr(share, [s[key] for s in scenes])
        out[f"spearman_{key}"] = {"rho": round(float(r.statistic), 3), "p": float(r.pvalue)}
    r3 = np.array([s["rain_3d"] for s in scenes])
    late = np.array([int(s["date"][:4]) >= args.split for s in scenes])
    for name, m in ((f"before_{args.split}", ~late), (f"from_{args.split}", late)):
        dry, wet = m & (r3 < DRY_MM), m & (r3 >= WET_MM)
        out[name] = {"dry_n": int(dry.sum()), "dry_mean_share": round(float(share[dry].mean()), 3),
                     "wet_n": int(wet.sum()), "wet_mean_share": round(float(share[wet].mean()), 3)}
    t = mannwhitneyu(share[~late & (r3 < DRY_MM)], share[late & (r3 < DRY_MM)])
    out["dry_scenes_before_vs_after"] = {"U": float(t.statistic), "p": float(t.pvalue)}
    out["scene_rows"] = scenes
    (OUT / f"{args.code}_rain.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "scene_rows"}, indent=2))


if __name__ == "__main__":
    main()
