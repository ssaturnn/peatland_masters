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
# Cluster-level thresholds for labelling a GNG cluster as bare/cut peat by its
# spectral centroid (reflectance 0-1), calibrated on real signatures:
#   vegetation  NDVI~0.63  SWIR1<<NIR
#   bare peat   NDVI~0.10  SWIR1~=NIR  SWIR1~0.16
#   water/shadow                        SWIR1~0
# Per-pixel NDVI cut-off used inside the GNG detector — the same sensitivity
# as the 0.25 baseline, so GNG finds all the real bare peat NDVI does, while
# the SWIR test below removes the water/shadow that fools a bare NDVI.
GNG_BARE_NDVI = 0.25
SWIR_WATER_FLOOR = 0.07    # below this a prototype is water / deep shadow


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


def _node_is_nonpeat(weight, order):
    """A GNG prototype that is water or deep shadow — very low SWIR — i.e.
    a low-NDVI surface that is NOT exposed peat."""
    return weight[order.index("swir1")] < SWIR_WATER_FLOOR


def gng_bare(stack, order, valid, inside, seed=42, max_nodes=80,
             thresh=GNG_BARE_NDVI):
    """Bare-peat mask: NDVI sensitivity refined by GNG multispectral clustering.

    GNG learns prototype spectra (nodes) over the full 6-band signature via
    Competitive Hebbian Learning. Its job here is discrimination the NDVI
    baseline cannot do: any pixel whose nearest prototype is water or deep
    shadow (very low SWIR) is rejected, even when its NDVI is low. Within
    the remaining (peat-plausible) prototypes, a pixel is bare if its own
    NDVI is below `thresh` — so small cut patches are still found, unlike a
    pure node-label rule.

    So the detector keeps the sensitivity of the NDVI baseline (it finds the
    same real bare peat) but adds spectral specificity: it removes the
    low-NDVI water and shadow that a vegetation index alone mistakes for bare
    peat. On a bog with open water this is a visible correction; on a clean
    bog the two agree. Deterministic via a fixed seed.
    """
    h, w, b = stack.shape
    refl = stack / 10000.0
    feats = refl.reshape(-1, b)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(feats), size=min(8000, len(feats)), replace=False)
    net = gng.GrowingNeuralGas(max_nodes=max_nodes, rng=rng)
    net.fit(feats[idx], n_steps=15000)

    node_nonpeat = np.array([_node_is_nonpeat(net.weights[i], order)
                             for i in range(len(net.weights))])
    peat_node = ~node_nonpeat[net.predict(feats)].reshape(h, w)

    red = refl[:, :, order.index("red")]
    nir = refl[:, :, order.index("nir")]
    ndvi = (nir - red) / (nir + red + 1e-9)
    return ((ndvi < thresh) & peat_node & valid & inside
            & (brightness(stack, order) < BRIGHT_CAP))


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
    gng_mask = gng_bare(stack, order, valid, inside)
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
