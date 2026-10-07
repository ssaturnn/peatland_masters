"""Step 35 — summer v5 survey of the raised-bog SACs in the documented years.

For every raised-bog SAC of the release and each year with NPWS plot counts
(2021, 2022), take the clearest summer scene (15 June - 31 August) that passes
the shared gates on the release grid, and record the v5 exposed-peat area and,
for comparison, the v3 GNG area. Then compare SACs with documented cutting in
that year against the others, and the v5 area against the plot counts.

Usage:
    python3 scripts/35_v5_survey.py
"""

import importlib.util
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from peatland import boundaries, config, geo, imagery, pipeline, preprocess, v5

spec = importlib.util.spec_from_file_location("mc", ROOT / "scripts/12_method_comparison.py")
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)

CACHE = config.OUT_DIR / "cache-release-v3-uplands"
OUT = ROOT / "outputs/evaluation/v5-survey.json"
YEARS = (2021, 2022)
WINDOW = ("06-15", "08-31")


def survey(sites, idx, row, year):
    record = mc.release_record(row["SITECODE"], CACHE)
    shape = tuple(record["grid"]["shape"])
    transform = rasterio.Affine(*record["grid"]["transform"][:6])
    crs = record["grid"]["crs"]
    bbox = boundaries.site_bbox_wgs84(sites, idx, buffer_m=200)
    geom = gpd.GeoSeries([row.geometry], crs=config.ITM).to_crs(crs).iloc[0]
    site_inside = geo.polygon_mask(geom, shape, transform)
    tidal = (pipeline.mask_from_b64(record["tidal_mask_b64"], shape)
             if record.get("tidal_mask_b64") else np.zeros(shape, bool))
    inside = site_inside & ~tidal
    order = list(pipeline.BANDS)
    best = None
    for item, rep in preprocess.clear_scenes("planetary", bbox, f"{year}-{WINDOW[0]}/{year}-{WINDOW[1]}",
                                             shape, aoi_mask=site_inside, min_clear=0.85, max_scenes=4):
        stack = np.stack([imagery.read_window(item, b, bbox, "planetary", out_shape=shape)[0]
                          .astype("float32") for b in order], -1)
        q = pipeline.scene_quality(stack, order, rep["scl"], inside, item)
        if pipeline.scene_passes(q) and (best is None or q["clear_fraction"] > best[2]["clear_fraction"]):
            best = (item, stack, q)
    if best is None:
        return None
    item, stack, q = best
    peat = v5.peat_mask(stack, q["valid"], inside, tuple(order))
    gng = pipeline.gng_bare(stack, order, q["valid"], inside)
    site_ha = geo.mask_area_ha(inside, transform)
    return {"scene_date": item.properties["datetime"][:10], "clear_fraction": round(q["clear_fraction"], 3),
            "site_ha": round(site_ha, 1), "v5_ha": round(geo.mask_area_ha(peat, transform), 2),
            "gng_ha": round(geo.mask_area_ha(gng, transform), 2)}


def main():
    sites = boundaries.build_sites()
    sacs = sites[(sites.source == "SAC") & (sites.bog_type == "raised")]
    result = json.loads(OUT.read_text()) if OUT.exists() else {"window": WINDOW, "rows": {}}
    for idx, row in sacs.iterrows():
        for year in YEARS:
            key = f"{row['SITECODE']}_{year}"
            if key in result["rows"]:
                continue
            plots = row.get(f"plots_{year}")
            plots = None if plots is None or plots != plots else int(plots)
            try:
                r = survey(sites, idx, row, year)
            except Exception as e:
                r = {"error": f"{type(e).__name__}: {e}"}
            result["rows"][key] = {"site": row["SITE_NAME"], "year": year, "plots": plots,
                                   "documented": plots is not None or row["SITECODE"] == "002110",
                                   **(r or {"status": "no summer scene passes"})}
            OUT.write_text(json.dumps(result, indent=2))
            print(key, result["rows"][key], flush=True)
    rows = [r for r in result["rows"].values() if "v5_ha" in r]
    for metric in ("v5", "gng"):
        doc = [100 * r[f"{metric}_ha"] / r["site_ha"] for r in rows if r["documented"]]
        oth = [100 * r[f"{metric}_ha"] / r["site_ha"] for r in rows if not r["documented"]]
        if doc and oth:
            t = mannwhitneyu(doc, oth, alternative="greater")
            print(f"{metric}: documented median {np.median(doc):.3f}% (n={len(doc)}) vs others "
                  f"{np.median(oth):.3f}% (n={len(oth)}), Mann-Whitney one-sided p={t.pvalue:.4f}")
        counted = [(r["plots"], r[f"{metric}_ha"]) for r in rows if r["plots"] is not None]
        if len(counted) >= 4:
            rho = spearmanr(*zip(*counted))
            print(f"{metric}: Spearman plots vs area over {len(counted)} site-years rho={rho.statistic:.2f} p={rho.pvalue:.3f}")


if __name__ == "__main__":
    main()
