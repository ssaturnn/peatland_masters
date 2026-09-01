"""Step 13 — three-method comparison + sensitivity analysis (thesis tables).

Comparison: NDVI threshold vs K-means (fixed k, cluster-labelled) vs GNG
(adaptive prototypes + per-pixel refinement) on control cases where we
know the right answer:

  * Monivea 2022      — documented heavy cutting (should detect ~40 ha)
  * Rosroe 2018       — open lough, no cutting (should detect ~0)
  * Moorfield 2025    — spring burn scar (should detect ~0-3, not 60)
  * Namucka 2026      — known active cutting
  * Kilnaborris 2024  — small subtle patches
  * Derrinlough 2024  — quiet site

K-means uses the same 9-D standardized features and the same physical
cluster-labelling rule as GNG's prototypes, so the comparison isolates
the clustering strategy itself (fixed k + batch vs grown prototypes).

Sensitivity: GNG area under NDVI-threshold {0.20,0.25,0.30}, seeds
{7,42,123}, node budgets {40,80,120} on three sites.

Outputs: outputs/method_comparison.json + printed markdown tables.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import geopandas as gpd
from sklearn.cluster import KMeans

from peatland import boundaries, imagery, geo, config, preprocess, pipeline, detect

CASES = [  # (site-name fragment, year, expectation)
    ("Monivea",     2022, "documented heavy cutting"),
    ("Rosroe",      2018, "open lough, no cutting"),
    ("Moorfield",   2025, "spring burn scar"),
    ("Namucka",     2026, "known active cutting"),
    ("Kilnaborris", 2024, "small subtle patches"),
    ("Derrinlough", 2024, "quiet midland site"),
]


def cache_date(code, year):
    """The scene date the dataset chose for this site-year."""
    f = config.OUT_DIR / "cache" / f"{code}.json"
    r = json.loads(f.read_text())
    for y, d in zip(r["years"], r["dates"]):
        if y == year:
            return d
    return None


def kmeans_bare(stack, order, valid, inside, k=6, seed=42):
    """K-means analogue of the GNG detector: same features, same
    physical labelling of cluster centroids, fixed k."""
    h, w, b = stack.shape
    refl = stack / 10000.0
    raw = pipeline._gng_feature_stack(refl, order).reshape(-1, len(pipeline.GNG_FEATURES))
    mu, sd = raw.mean(axis=0), raw.std(axis=0) + 1e-9
    feats = (raw - mu) / sd
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(
        feats[np.random.default_rng(seed).choice(len(feats), min(8000, len(feats)), replace=False)])
    lab = km.predict(feats)
    cent_raw = km.cluster_centers_ * sd + mu
    nonpeat = pipeline._nonpeat_nodes(cent_raw)
    peat_px = ~nonpeat[lab].reshape(h, w)
    ndvi = raw[:, pipeline.GNG_FEATURES.index("ndvi")].reshape(h, w)
    nir = refl[:, :, order.index("nir")]
    swir2 = refl[:, :, order.index("swir2")]
    nbr = (nir - swir2) / (nir + swir2 + 1e-9)
    return ((ndvi < pipeline.GNG_BARE_NDVI) & peat_px
            & (nbr > pipeline.BURN_NBR_FLOOR) & valid & inside
            & (pipeline.brightness(stack, order) < pipeline.BRIGHT_CAP))


def load_case(sites, name, year):
    row = sites[sites.SITE_NAME.str.contains(name)].iloc[0]
    day = cache_date(row["SITECODE"], year)
    bbox = boundaries.site_bbox_wgs84(sites, row.name, buffer_m=200)
    ref = imagery.search_scene("planetary", bbox, "2021-05-01/2021-09-15", max_cloud=40)
    red0, transform, crs = imagery.read_window(ref, "red", bbox, "planetary")
    shape = red0.shape
    geom = gpd.GeoSeries([row.geometry], crs=config.ITM).to_crs(crs).iloc[0]
    inside = geo.polygon_mask(geom, shape, transform)
    item, rep = preprocess.pick_clear_scene(
        "planetary", bbox, f"{day}/{day}", shape, aoi_mask=inside, min_clear=0.5)
    order = list(pipeline.BANDS)
    stack = np.stack([imagery.read_window(item, b, bbox, "planetary", out_shape=shape)[0]
                      .astype("float32") for b in order], -1)
    valid = preprocess.valid_mask(rep["scl"])
    return stack, order, valid, inside, transform


def main():
    sites = boundaries.build_sites()
    out = {"comparison": [], "sensitivity": {}}

    print("=== THREE-METHOD COMPARISON (control cases) ===")
    print(f"{'case':28} {'NDVI ha':>8} {'KMeans ha':>10} {'GNG ha':>8}  expected")
    for name, year, expect in CASES:
        stack, order, valid, inside, transform = load_case(sites, name, year)
        refl = stack / 10000.0
        red = refl[:, :, order.index("red")]
        nir = refl[:, :, order.index("nir")]
        ndvi_arr = detect.ndvi(red, nir)

        nd = pipeline.ndvi_bare(stack, order, ndvi_arr, valid, inside)
        t0 = time.time(); km = kmeans_bare(stack, order, valid, inside); t_km = time.time() - t0
        t0 = time.time(); gg = pipeline.gng_bare(stack, order, valid, inside); t_gg = time.time() - t0

        ha = lambda m: round(geo.mask_area_ha(m, transform), 1)
        row = {"case": f"{name} {year}", "expected": expect,
               "ndvi_ha": ha(nd), "kmeans_ha": ha(km), "gng_ha": ha(gg),
               "t_kmeans_s": round(t_km, 1), "t_gng_s": round(t_gg, 1)}
        out["comparison"].append(row)
        print(f"{row['case']:28} {row['ndvi_ha']:>8} {row['kmeans_ha']:>10} "
              f"{row['gng_ha']:>8}  {expect}")

    print("\n=== SENSITIVITY: GNG under parameter sweeps ===")
    for name, year, _ in CASES[:3]:
        stack, order, valid, inside, transform = load_case(sites, name, year)
        ha = lambda m: round(geo.mask_area_ha(m, transform), 1)
        sweep = {}
        sweep["ndvi_thresh"] = {
            str(t): ha(pipeline.gng_bare(stack, order, valid, inside, thresh=t))
            for t in (0.20, 0.25, 0.30)}
        sweep["seed"] = {
            str(s): ha(pipeline.gng_bare(stack, order, valid, inside, seed=s))
            for s in (7, 42, 123)}
        sweep["max_nodes"] = {
            str(n): ha(pipeline.gng_bare(stack, order, valid, inside, max_nodes=n))
            for n in (40, 80, 120)}
        out["sensitivity"][f"{name} {year}"] = sweep
        print(f"{name} {year}: thresh {sweep['ndvi_thresh']} | "
              f"seeds {sweep['seed']} | nodes {sweep['max_nodes']}")

    Path("outputs/method_comparison.json").write_text(json.dumps(out, indent=1))
    print("\nsaved outputs/method_comparison.json")


if __name__ == "__main__":
    main()
