"""Step 10 — export per-site results as GeoJSON for the web map.

For a spread of bog sites, runs the cloud-aware two-epoch pipeline and
writes web/data/sites.geojson: one feature per site (simplified polygon
in WGS84) carrying bare-peat area now, area then, and newly-cut area.
The web map reads this file directly.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from peatland import boundaries, imagery, detect, geo, config, preprocess, change

PROVIDER = "planetary"
RANGE_A = ("2018", "2018-05-01/2018-09-15")
RANGE_B = ("2024", "2024-05-01/2024-09-15")
RANGE_REF = "2022-05-01/2022-09-15"
NDVI_BARE = 0.25
N_SITES = 5
OUT = Path(__file__).resolve().parents[1] / "web" / "data" / "sites.geojson"


def bare_for(item, bbox, shape, inside, scl):
    red, _, _ = imagery.read_window(item, "red", bbox, PROVIDER, out_shape=shape)
    nir, _, _ = imagery.read_window(item, "nir", bbox, PROVIDER, out_shape=shape)
    ndvi = detect.ndvi(red, nir)
    valid = preprocess.valid_mask(scl)
    bare = detect.exposed_peat_mask(ndvi, NDVI_BARE) & valid & inside
    return bare, valid


def process(west, label):
    row = west.loc[label]
    bbox = boundaries.site_bbox_wgs84(west, label, buffer_m=200)

    ref = imagery.search_scene(PROVIDER, bbox, RANGE_REF, max_cloud=30)
    if ref is None:
        return None
    red0, transform, crs = imagery.read_window(ref, "red", bbox, PROVIDER)
    shape = red0.shape
    site_geom = west.loc[[label]].to_crs(crs).geometry.iloc[0]
    inside = geo.polygon_mask(site_geom, shape, transform)

    item_a, rep_a = preprocess.pick_clear_scene(
        PROVIDER, bbox, RANGE_A[1], shape, aoi_mask=inside, min_clear=0.85)
    item_b, rep_b = preprocess.pick_clear_scene(
        PROVIDER, bbox, RANGE_B[1], shape, aoi_mask=inside, min_clear=0.85)
    if item_a is None or item_b is None:
        return None

    bare_a, valid_a = bare_for(item_a, bbox, shape, inside, rep_a["scl"])
    bare_b, valid_b = bare_for(item_b, bbox, shape, inside, rep_b["scl"])
    both_valid = valid_a & valid_b & inside
    masks = change.compare(bare_a, bare_b, both_valid)

    ha = lambda m: round(geo.mask_area_ha(m, transform), 1)
    site_ha = ha(inside)
    props = {
        "name": row["SITE_NAME"],
        "county": config.TARGET_COUNTIES[row["COUNTY"]],
        "site_ha": site_ha,
        "year_a": RANGE_A[0], "year_b": RANGE_B[0],
        "date_a": rep_a["datetime"][:10], "date_b": rep_b["datetime"][:10],
        "bare_a_ha": ha(bare_a), "bare_b_ha": ha(bare_b),
        "newly_bare_ha": ha(masks["newly_bare"]),
        "revegetated_ha": ha(masks["revegetated"]),
        "net_change_ha": round(ha(bare_b) - ha(bare_a), 1),
        "bare_pct_now": round(100 * ha(bare_b) / max(site_ha, 1), 1),
        "valid_coverage_pct": round(100 * both_valid.sum() / max(inside.sum(), 1), 1),
    }

    # simplified polygon in WGS84 for the map
    poly_wgs = (west.loc[[label]].to_crs(config.WGS84)
                .geometry.simplify(0.0008).iloc[0])
    return {"type": "Feature",
            "geometry": poly_wgs.__geo_interface__,
            "properties": props}


def main():
    west = boundaries.west_bog_sites(boundaries.load_nha())
    ranked = west.sort_values("HA")
    n = len(ranked)
    picks = [ranked.iloc[int(n * f)].name
             for f in (0.4, 0.55, 0.7, 0.82, 0.92)][:N_SITES]

    feats = []
    for label in picks:
        try:
            t0 = time.time()
            f = process(west, label)
            if f:
                feats.append(f)
                p = f["properties"]
                print(f"OK  {p['name']} ({p['county']}): "
                      f"bare {p['bare_a_ha']}→{p['bare_b_ha']} ha, "
                      f"new {p['newly_bare_ha']} ha  [{time.time()-t0:.0f}s]")
            else:
                print(f"skip {label}")
        except Exception as e:
            print(f"FAIL {label}: {type(e).__name__}: {e}")

    fc = {"type": "FeatureCollection",
          "generated_epochs": {"a": RANGE_A[0], "b": RANGE_B[0]},
          "features": feats}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fc))
    print(f"\nSaved {len(feats)} sites -> {OUT}")


if __name__ == "__main__":
    main()
