"""Step 47 — does exposed peat behave like a fire, and does FIRMS see one?

Two checks against the question "could this be an unrecorded (smouldering)
peat fire?", from data already on disk:

1. Behaviour at Castle Ffrench East (001244). A fire scar stays dark and
   charred until it regrows; worked dry peat goes dark when it rains and comes
   back when it dries. From the field-exposure series
   (outputs/time-series/001244_fields.json) and the rain at each scene
   (outputs/time-series/001244_rain.json, scripts/42_rain_check.py), it counts
   April-September "off" scenes (< 20% of the fields exposed) after an "on"
   scene (>= 60%) earlier in the same season, how many come back on later
   that season (and within 10 days), and the rain before them. It also reads the SWIR1/SWIR2 ratio of the field pixels on dry
   scenes (>= 50% exposed) against the fresh-burn ratio at Cloonmoylan
   (outputs/time-series/000248_2019_2022_fire.json).

2. FIRMS for every bog-year of the v5 release with exposed peat (>= 0.1 ha):
   any VIIRS (375 m) or MODIS (1 km) detection of any confidence within 1 km
   of the boundary, from two years before to two years after. The keyless
   FIRMS country archives on disk cover 2018-2024.

Usage:
    python3 scripts/47_fire_behaviour_firms.py
"""

import csv
import datetime as dt
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from peatland import boundaries, config

TS = config.OUT_DIR / "time-series"
RAW = config.OUT_DIR / "firms/raw"
SITES = ROOT / "web/data/sites.geojson"
OUT = config.OUT_DIR / "evaluation/fire-behaviour-firms.json"
ON, OFF, RECOVER_DAYS = 0.6, 0.2, 10
NEAR_M, YEARS_AROUND, MIN_HA = 1000, 2, 0.1
NON_VEGETATION_TYPES = {"1", "2", "3"}
FIELD_FREQ = 0.4


def behaviour():
    rows = json.loads((TS / "001244_fields.json").read_text())["rows"]
    rain = {r["date"]: r for r in json.loads((TS / "001244_rain.json").read_text())["scene_rows"]}
    season = [(dt.date.fromisoformat(d), s) for d, s, _ in rows if 4 <= int(d[5:7]) <= 9]
    offs = []
    for i in range(1, len(season)):
        d, s = season[i]
        earlier = [(d0, s0) for d0, s0 in season[:i] if d0.year == d.year and s0 >= ON]
        if s < OFF and earlier:
            prev_d, prev_s = earlier[-1]
            back = next(((d2 - d).days for d2, s2 in season[i + 1:] if s2 >= ON and (d2 - d).days <= RECOVER_DAYS), None)
            later = next(((d2 - d).days for d2, s2 in season[i + 1:] if s2 >= ON and d2.year == d.year), None)
            offs.append({"date": str(d), "share": s, "previous": str(prev_d), "previous_share": prev_s,
                         "rain_3d": rain.get(str(d), {}).get("rain_3d"),
                         "rain_7d": rain.get(str(d), {}).get("rain_7d"),
                         "back_on_within_10d": back, "back_on_same_season_days": later})
    r3_off = [o["rain_3d"] for o in offs if o["rain_3d"] is not None]
    on_rain = [rain[str(d)]["rain_3d"] for d, s in season if s >= ON and str(d) in rain]

    z = np.load(TS / "001244_2017_2026.npz")
    dates = z["dates"]
    keep = np.array([4 <= int(x[5:7]) <= 9 and int(x[:4]) <= 2023 for x in dates])
    freq = (z["peat"][keep] & z["valid"][keep]).sum(0) / np.maximum(z["valid"][keep].sum(0), 1)
    fields = z["inside"] & (freq >= FIELD_FREQ)
    shares = dict((d, s) for d, s, _ in rows)
    ratios = []
    for i, d in enumerate(dates):
        if 4 <= int(d[5:7]) <= 9 and shares.get(str(d), 0) >= 0.5:
            m = fields & z["valid"][i]
            ratios.append(z["ratio"][i][m].astype(np.float32))
    ratios = np.concatenate(ratios)
    ratios = ratios[np.isfinite(ratios)]
    fire = json.loads((TS / "000248_2019_2022_fire.json").read_text())
    first = min((s for s in fire["scenes"] if s["days_from_fire"] > 0), key=lambda s: s["days_from_fire"])
    return {
        "off_after_on_events": len(offs),
        "back_on_within_10d": sum(o["back_on_within_10d"] is not None for o in offs),
        "back_on_same_season": sum(o["back_on_same_season_days"] is not None for o in offs),
        "median_days_back_on": float(np.median([o["back_on_same_season_days"] for o in offs
                                                if o["back_on_same_season_days"] is not None])) if offs else None,
        "rain_3d_off_median": float(np.median(r3_off)) if r3_off else None,
        "rain_3d_on_median": float(np.median(on_rain)) if on_rain else None,
        "off_events": offs,
        "field_ratio_dry_scenes": {"n_obs": int(ratios.size), "median": round(float(np.median(ratios)), 3),
                                   "p5": round(float(np.percentile(ratios, 5)), 3),
                                   "share_below_1_2": round(float((ratios < 1.2).mean()), 4)},
        "cloonmoylan_fresh_burn": {"date": first["date"], "days_from_fire": first["days_from_fire"],
                                   "ratio": first["burn_ratio"]},
    }


def firms():
    dets = []
    for f in sorted(RAW.glob("*.csv")):
        with f.open() as fh:
            for r in csv.DictReader(fh):
                if str(r.get("type", "")) in NON_VEGETATION_TYPES:
                    continue
                dets.append({"date": r["acq_date"], "lat": float(r["latitude"]), "lon": float(r["longitude"]),
                             "instrument": r["instrument"], "confidence": r["confidence"], "frp": r["frp"]})
    years_on_disk = sorted({int(d["date"][:4]) for d in dets})
    pts = gpd.GeoSeries([Point(d["lon"], d["lat"]) for d in dets], crs="EPSG:4326").to_crs(config.ITM)
    sites = boundaries.build_sites()
    geo = json.loads(SITES.read_text())
    out = []
    for f in geo["features"]:
        p = f["properties"]
        row = sites[sites.SITECODE == p["code"]]
        if row.empty:
            continue
        geom = row.geometry.iloc[0]
        near = pts.distance(geom) <= NEAR_M
        idx = np.flatnonzero(near.values)
        for i, (y, ha) in enumerate(zip(p.get("v5_years") or [], p.get("v5_series") or [])):
            if ha is None or ha < MIN_HA:
                continue
            hits = [dets[k] for k in idx if abs(int(dets[k]["date"][:4]) - y) <= YEARS_AROUND]
            covered = [yy for yy in range(y - YEARS_AROUND, y + YEARS_AROUND + 1) if yy in years_on_disk]
            out.append({"code": p["code"], "name": p["name"], "year": y, "v5_ha": ha,
                        "after_fire_flag": bool((p.get("v5_after_fire") or [False] * 99)[i]),
                        "large_patch_flag": bool((p.get("v5_large_patch") or [False] * 99)[i]),
                        "firms_years_covered": covered,
                        "firms_hits": len(hits),
                        "firms_dates": sorted({h["date"] for h in hits})})
    for r in out:
        r["firms_same_or_previous_year"] = any(int(x[:4]) in (r["year"], r["year"] - 1) for x in r["firms_dates"])
    silent = [r for r in out if r["firms_hits"] == 0]
    close = [r for r in out if r["firms_same_or_previous_year"]]
    covered = [r for r in out if r["year"] <= max(years_on_disk)]
    seen = [r for r in out if r["firms_hits"] > 0]
    return {"firms_years_on_disk": years_on_disk, "near_m": NEAR_M, "years_around": YEARS_AROUND,
            "min_v5_ha": MIN_HA, "bog_years_with_peat": len(out),
            "bogs_with_peat": len({r["code"] for r in out}),
            "bog_years_firms_silent": len(silent), "bog_years_firms_seen": len(seen),
            "bogs_firms_seen": sorted({r["name"] for r in seen}),
            "v5_ha_firms_silent": round(sum(r["v5_ha"] for r in silent), 1),
            "v5_ha_firms_seen": round(sum(r["v5_ha"] for r in seen), 1),
            "bog_years_firms_same_or_previous_year": len(close),
            "v5_ha_firms_same_or_previous_year": round(sum(r["v5_ha"] for r in close), 1),
            "bog_years_up_to_last_firms_year": len(covered),
            "bog_years_up_to_last_firms_year_silent": sum(r["firms_hits"] == 0 for r in covered),
            "total_v5_ha": round(sum(r["v5_ha"] for r in out), 1),
            "rows": out}


def main():
    b = behaviour()
    f = firms()
    OUT.write_text(json.dumps({"behaviour_001244": b, "firms": f}, indent=2))
    print(json.dumps({k: v for k, v in b.items() if k != "off_events"}, indent=2))
    print(json.dumps({k: v for k, v in f.items() if k != "rows"}, indent=2))
    for r in f["rows"]:
        if r["firms_hits"]:
            print(f"  {r['name'][:34]:34s} {r['year']} v5 {r['v5_ha']:6.2f} ha  FIRMS {r['firms_hits']:3d}  "
                  f"{','.join(r['firms_dates'][:4])}  fire_flag={r['after_fire_flag']}")


if __name__ == "__main__":
    main()
