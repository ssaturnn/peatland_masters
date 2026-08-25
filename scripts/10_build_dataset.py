"""Step 11 — build the full web dataset: all 57 bogs, multi-year, both methods.

Runs the multi-year both-detector pipeline for every West-of-Ireland NHA
bog, tagged with SAC/SPA legal status. Results are cached per site code
(outputs/cache/<code>.json) so the run is resumable and incremental —
the web GeoJSON is rewritten after every site, so a partial run still
produces a usable map. Delete a cache file to recompute that site.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd

from peatland import boundaries, config, pipeline

PROVIDER = "planetary"
CACHE_DIR = config.OUT_DIR / "cache"
OUT = Path(__file__).resolve().parents[1] / "web" / "data" / "sites.geojson"


def properties(row, r):
    """Assemble GeoJSON properties from a pipeline result `r`."""
    ndvi_now = r["ndvi_series"][-1] if r["ndvi_series"] else 0.0
    gng_now = r["gng_series"][-1] if r["gng_series"] else 0.0
    site_ha = r["site_ha"]
    plots = row.get("plots_2022")
    return {
        "code": row["SITECODE"],
        "name": row["SITE_NAME"],
        "county": config.TARGET_COUNTIES[row["COUNTY"]],
        "designation": row["designation"],
        "source": row.get("source", "NHA"),
        "plots_2022": None if plots is None or (isinstance(plots, float)
                      and plots != plots) else int(plots),
        "plots_2021": (lambda v: None if v is None or (isinstance(v, float)
                       and v != v) else int(v))(row.get("plots_2021")),
        "site_ha": site_ha,
        "years": r["years"],
        "ndvi_series": r["ndvi_series"],
        "gng_series": r["gng_series"],
        "dates": r["dates"],
        "span": r["span"],
        "bare_now_ndvi_ha": ndvi_now,
        "bare_now_gng_ha": gng_now,
        "bare_pct_now": round(100 * gng_now / max(site_ha, 1), 1),
        "newly_ndvi_ha": r["newly_ndvi_ha"],
        "newly_gng_ha": r["newly_gng_ha"],
        "reveg_ndvi_ha": r["reveg_ndvi_ha"],
        "reveg_gng_ha": r["reveg_gng_ha"],
        "rate_ndvi_ha_yr": r["rate_ndvi_ha_yr"],
        "rate_gng_ha_yr": r["rate_gng_ha_yr"],
        "both_cov_pct": r["both_cov_pct"],
    }


def write_geojson(features):
    fc = {"type": "FeatureCollection",
          "years": pipeline.YEARS,
          "features": features}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fc))


def main():
    sites = boundaries.build_sites()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(sites)} sites to process\n")

    features = []
    for pos, (idx, row) in enumerate(sites.iterrows(), 1):
        code = row["SITECODE"]
        cache_f = CACHE_DIR / f"{code}.json"
        label = row["SITE_NAME"]

        if cache_f.exists():
            r = json.loads(cache_f.read_text())
            src = "cache"
        else:
            bbox = boundaries.site_bbox_wgs84(sites, idx, buffer_m=200)
            row = dict(row)
            row["_geom_wgs"] = (gpd.GeoSeries([row["geometry"]], crs=config.ITM)
                                .to_crs(config.WGS84).iloc[0])
            try:
                t0 = time.time()
                r = pipeline.process_site(row, bbox, PROVIDER)
            except Exception as e:
                print(f"[{pos}/{len(sites)}] FAIL {label}: "
                      f"{type(e).__name__}: {e}")
                continue
            if r is None:
                print(f"[{pos}/{len(sites)}] skip {label}: no imagery")
                continue
            r["_took_s"] = round(time.time() - t0, 1)
            cache_f.write_text(json.dumps(r))
            src = f"{r['_took_s']}s"

        # coverage guard: a bog straddling two Sentinel-2 tiles gets only a
        # partial window from a single scene. If the imagery-derived area is
        # far from the NPWS register, coverage is incomplete — drop it rather
        # than report a wrong area. (Full fix would mosaic across tiles.)
        reg_ha = float(row["HA"])
        if r.get("site_ha") and reg_ha and abs(r["site_ha"] - reg_ha) / reg_ha > 0.15:
            print(f"[{pos}/{len(sites)}] drop {label}: partial tile coverage "
                  f"(imagery {r['site_ha']} ha vs register {reg_ha:.0f} ha)")
            continue

        # simplified polygon for the map
        poly = (sites.loc[[idx]].to_crs(config.WGS84)
                .geometry.simplify(0.0007).iloc[0])
        features.append({"type": "Feature",
                         "geometry": poly.__geo_interface__,
                         "properties": properties(row if isinstance(row, dict)
                                                  else dict(row), r)})
        write_geojson(features)  # incremental: always have a usable file

        print(f"[{pos}/{len(sites)}] {label[:34]:34} "
              f"ndvi {r['ndvi_series']} gng {r['gng_series']} "
              f"new(gng) {r.get('newly_gng_ha')} [{src}]")

    print(f"\nDone. {len(features)} sites written -> {OUT}")


if __name__ == "__main__":
    main()
