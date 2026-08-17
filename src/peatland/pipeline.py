"""End-to-end per-site pipeline: multi-year, both detectors.

For one bog it establishes a fixed 10 m grid, then for each survey year
picks the scene clearest over the bog (SCL), reads a 6-band stack, and
detects bare peat two ways:

  * NDVI   — simple per-pixel threshold (stable baseline for the trend)
  * GNG    — unsupervised multi-spectral clustering, labelled by physics

It returns per-year areas for both methods, plus change metrics
(newly-bare = new cutting, re-vegetated) and a linear cutting rate.
The heavy work is per (site, year); callers cache on the site code.
"""

import numpy as np

from . import imagery, detect, geo, gng

BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]
YEARS = [2018, 2020, 2022, 2024]
NDVI_BARE = 0.25
# Clouds/haze that the SCL mask misses are bright and low-NDVI, so they can
# masquerade as bare peat. Real exposed peat is dark brown (low reflectance),
# so we refuse to call a pixel "bare" if it is too bright in the visible bands.
BRIGHT_CAP = 0.30


def _year_range(year):
    return f"{year}-05-01/{year}-09-15"


def brightness(stack, order):
    """Mean visible-band reflectance (0-1) — high for cloud, low for peat."""
    return (stack[:, :, order.index("blue")]
            + stack[:, :, order.index("green")]
            + stack[:, :, order.index("red")]) / 3.0 / 10000.0


def ndvi_bare(stack, order, ndvi, valid, inside, thresh=NDVI_BARE):
    """NDVI-threshold bare mask, cloud-masked and brightness-guarded."""
    return (detect.exposed_peat_mask(ndvi, thresh) & valid & inside
            & (brightness(stack, order) < BRIGHT_CAP))


def gng_bare(stack, order, ndvi, valid, inside, thresh=NDVI_BARE, seed=42):
    """Bare-peat mask from GNG clusters, refined by per-pixel NDVI.

    GNG gives spatially coherent clusters; we keep the low-vigour ones
    (mean NDVI below the scene mean) and require each pixel to be
    genuinely bare (NDVI < thresh). Deterministic via a fixed seed.
    """
    h, w, b = stack.shape
    feats = stack.reshape(-1, b) / 10000.0
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(feats), size=min(8000, len(feats)), replace=False)
    net = gng.GrowingNeuralGas(max_nodes=40, rng=rng)
    net.fit(feats[idx], n_steps=15000)
    net.prune_long_edges(factor=1.0)
    cluster = np.array([net.components()[n] for n in net.predict(feats)])

    ndvi_flat = ndvi.reshape(-1)
    scene_mean = float(ndvi_flat.mean())
    low = [c for c in set(cluster)
           if ndvi_flat[cluster == c].mean() < scene_mean]
    mask = (np.isin(cluster, low) & (ndvi_flat < thresh)).reshape(h, w)
    return mask & valid & inside & (brightness(stack, order) < BRIGHT_CAP)


def detect_year(item, bbox, provider, shape, inside, scl):
    """Read a scene onto the FIXED reference grid and return both masks.

    Every band is resampled to `shape` so all years, the SCL mask and the
    site polygon share one grid — different scenes over the same bbox can
    otherwise land on slightly different pixel windows.
    """
    from . import preprocess
    order = list(BANDS)
    layers = [imagery.read_window(item, b, bbox, provider, out_shape=shape)[0]
              .astype("float32") for b in order]
    stack = np.stack(layers, axis=-1)
    red = stack[:, :, order.index("red")]
    nir = stack[:, :, order.index("nir")]
    ndvi = detect.ndvi(red, nir)
    valid = preprocess.valid_mask(scl)
    ndvi_mask = ndvi_bare(stack, order, ndvi, valid, inside)
    gng_mask = gng_bare(stack, order, ndvi, valid, inside)
    return {"ndvi": ndvi_mask, "gng": gng_mask, "valid": valid}


def _rate_ha_per_yr(years, areas):
    """Linear slope (ha/year) of bare area over time; 0 if <2 points."""
    ys = np.array(years, dtype=float)
    xs = np.array(areas, dtype=float)
    if len(ys) < 2 or np.allclose(ys, ys[0]):
        return 0.0
    slope = np.polyfit(ys, xs, 1)[0]
    return float(slope)


def process_site(row, bbox, provider, years=YEARS):
    """Run the full multi-year pipeline for one site row.

    Returns a dict of per-year areas (both methods) and change metrics,
    or None if imagery could not be obtained. `row` must expose geometry
    in the raster-alignment CRS handling done by the caller via `bbox`.
    """
    from . import preprocess

    # reference grid
    ref = imagery.search_scene(provider, bbox, "2021-05-01/2021-09-15",
                               max_cloud=40)
    if ref is None:
        return None
    red0, transform, crs = imagery.read_window(ref, "red", bbox, provider)
    shape = red0.shape

    site_geom = row["_geom_wgs"]  # provided by caller, reproject here
    import geopandas as gpd
    from . import config
    inside = geo.polygon_mask(
        gpd.GeoSeries([site_geom], crs=config.WGS84).to_crs(crs).iloc[0],
        shape, transform)

    per_year = {}
    masks_first = masks_last = None
    for i, yr in enumerate(years):
        item, rep = preprocess.pick_clear_scene(
            provider, bbox, _year_range(yr), shape,
            aoi_mask=inside, min_clear=0.92, limit=12)
        if item is None:
            per_year[yr] = None
            continue
        d = detect_year(item, bbox, provider, shape, inside, rep["scl"])
        per_year[yr] = {
            "date": rep["datetime"][:10],
            "ndvi_ha": round(geo.mask_area_ha(d["ndvi"], transform), 2),
            "gng_ha": round(geo.mask_area_ha(d["gng"], transform), 2),
            "clear": round(rep["aoi_clear"], 3),
            "_masks": d,
        }
        if masks_first is None:
            masks_first = (yr, d, inside)
        masks_last = (yr, d, inside)

    site_ha = round(geo.mask_area_ha(inside, transform), 1)
    valid_years = [y for y in years if per_year.get(y)]

    out = {"site_ha": site_ha, "years": valid_years,
           "ndvi_series": [per_year[y]["ndvi_ha"] for y in valid_years],
           "gng_series": [per_year[y]["gng_ha"] for y in valid_years],
           "dates": [per_year[y]["date"] for y in valid_years],
           "clear_min": min((per_year[y]["clear"] for y in valid_years),
                            default=0.0)}

    # change between first and last valid year (both methods)
    if masks_first and masks_last and masks_first[0] != masks_last[0]:
        (_, df, _), (_, dl, _) = masks_first, masks_last
        both = df["valid"] & dl["valid"] & inside
        for m in ("ndvi", "gng"):
            newly = dl[m] & ~df[m] & both
            reveg = df[m] & ~dl[m] & both
            out[f"newly_{m}_ha"] = round(geo.mask_area_ha(newly, transform), 2)
            out[f"reveg_{m}_ha"] = round(geo.mask_area_ha(reveg, transform), 2)
        out["both_cov_pct"] = round(100 * both.sum() / max(inside.sum(), 1), 1)
        out["span"] = f"{masks_first[0]}–{masks_last[0]}"
    else:
        for m in ("ndvi", "gng"):
            out[f"newly_{m}_ha"] = 0.0
            out[f"reveg_{m}_ha"] = 0.0
        out["both_cov_pct"] = 0.0
        out["span"] = ""

    out["rate_ndvi_ha_yr"] = round(_rate_ha_per_yr(valid_years, out["ndvi_series"]), 2)
    out["rate_gng_ha_yr"] = round(_rate_ha_per_yr(valid_years, out["gng_series"]), 2)
    return out
