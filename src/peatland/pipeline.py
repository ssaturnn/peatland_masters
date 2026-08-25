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
# Annual survey points, each sampled in a FIXED seasonal window (May 1 –
# July 15): the turf-cutting season, when freshly cut banks and spread turf
# are most visible, and a constant season keeps years comparable (a May
# scene and a September scene of the same bog can differ several-fold from
# phenology alone). Per year the detector takes the MAXIMUM bare area over
# every clear scene in the window — the peak visible extraction state —
# which removes the single-acquisition-date lottery.
YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
SEASON = ("05-01", "07-15")
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
# Burn discrimination: spring gorse/bog fires leave low-NDVI scars that mimic
# cut peat. NBR = (NIR-SWIR2)/(NIR+SWIR2) separates them cleanly — measured
# on our own data: real cut peat at Monivea NBR ≈ +0.09 (p10 +0.02), a burn
# scar at Moorfield NBR ≈ −0.13. Pixels at or below the floor are burns.
BURN_NBR_FLOOR = 0.0
# Scene-level haze gate: the SCL badly under-reports broken cumulus (a scene
# over Doogort East scored "88% clear" while visually solid cloud). Fraction
# of AOI pixels brighter than 0.25 reflectance: measured 0.60 on that cloudy
# scene vs <=0.005 on genuinely clear scenes. Scenes above the gate are
# dropped from the season-max pool.
HAZE_BRIGHT = 0.25
HAZE_MAX_FRAC = 0.15


def _year_range(year):
    return f"{year}-{SEASON[0]}/{year}-{SEASON[1]}"


def brightness(stack, order):
    """Mean visible-band reflectance (0-1) — high for cloud, low for peat."""
    return (stack[:, :, order.index("blue")]
            + stack[:, :, order.index("green")]
            + stack[:, :, order.index("red")]) / 3.0 / 10000.0


def ndvi_bare(stack, order, ndvi, valid, inside, thresh=NDVI_BARE):
    """NDVI-threshold bare mask, cloud-masked and brightness-guarded."""
    return (detect.exposed_peat_mask(ndvi, thresh) & valid & inside
            & (brightness(stack, order) < BRIGHT_CAP))


# Feature vector fed to the GNG: the 6 bands plus three physically meaningful
# indices, all z-scored per scene. Standardization stops the high-variance
# NIR/SWIR bands from dominating the Euclidean BMU step (Wongoutong 2024);
# ratio indices add illumination-invariant structure (NDVI vegetation vigour,
# MNDWI water/shadow, NBR2 dry-bare-surface).
GNG_FEATURES = ["blue", "green", "red", "nir", "swir1", "swir2",
                "ndvi", "mndwi", "nbr2"]


def _gng_feature_stack(refl, order):
    """(H,W,9) raw feature stack: reflectance bands + NDVI, MNDWI, NBR2."""
    red = refl[:, :, order.index("red")]
    nir = refl[:, :, order.index("nir")]
    green = refl[:, :, order.index("green")]
    swir1 = refl[:, :, order.index("swir1")]
    swir2 = refl[:, :, order.index("swir2")]
    ndvi = (nir - red) / (nir + red + 1e-9)
    mndwi = (green - swir1) / (green + swir1 + 1e-9)
    nbr2 = (swir1 - swir2) / (swir1 + swir2 + 1e-9)
    return np.dstack([refl, ndvi, mndwi, nbr2])


def _nonpeat_nodes(weights_raw):
    """Prototypes that are water or deep shadow, from de-standardized
    feature values: MNDWI > 0 is the standard water signal (McFeeters/Xu),
    and a very low SWIR1 catches dark shadow that MNDWI can miss."""
    mndwi = weights_raw[:, GNG_FEATURES.index("mndwi")]
    swir1 = weights_raw[:, GNG_FEATURES.index("swir1")]
    return (mndwi > 0.0) | (swir1 < SWIR_WATER_FLOOR)


def gng_bare(stack, order, valid, inside, seed=42, max_nodes=80,
             thresh=GNG_BARE_NDVI):
    """Bare-peat mask: NDVI sensitivity refined by GNG multispectral clustering.

    GNG learns prototype spectra over a 9-D standardized feature space
    (6 bands + NDVI/MNDWI/NBR2, z-scored per scene) via Competitive Hebbian
    Learning. Its job is discrimination the NDVI baseline cannot do: any
    pixel whose nearest prototype is water or deep shadow (MNDWI > 0 or very
    low SWIR1) is rejected even when its NDVI is low. Within the remaining
    peat-plausible prototypes, a pixel is bare if its own NDVI is below
    `thresh` — so small cut patches are still found.

    Net effect: the sensitivity of the NDVI baseline on real bare peat, plus
    spectral specificity against the water/shadow a vegetation index alone
    mistakes for bare peat. Deterministic via a fixed seed.
    """
    h, w, b = stack.shape
    refl = stack / 10000.0
    raw = _gng_feature_stack(refl, order).reshape(-1, len(GNG_FEATURES))
    mu = raw.mean(axis=0)
    sd = raw.std(axis=0) + 1e-9
    feats = (raw - mu) / sd

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(feats), size=min(8000, len(feats)), replace=False)
    net = gng.GrowingNeuralGas(max_nodes=max_nodes, rng=rng)
    net.fit(feats[idx], n_steps=15000)

    weights_raw = net.weights * sd + mu          # back to physical units
    node_nonpeat = _nonpeat_nodes(weights_raw)
    peat_node = ~node_nonpeat[net.predict(feats)].reshape(h, w)

    ndvi = raw[:, GNG_FEATURES.index("ndvi")].reshape(h, w)
    nir = refl[:, :, order.index("nir")]
    swir2 = refl[:, :, order.index("swir2")]
    nbr = (nir - swir2) / (nir + swir2 + 1e-9)   # burn scars: NBR <= 0
    return ((ndvi < thresh) & peat_node & (nbr > BURN_NBR_FLOOR)
            & valid & inside & (brightness(stack, order) < BRIGHT_CAP))


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
    br = brightness(stack, order)
    haze_frac = float(((br > HAZE_BRIGHT) & inside).sum() / max(inside.sum(), 1))
    ndvi_mask = ndvi_bare(stack, order, ndvi, valid, inside)
    gng_mask = gng_bare(stack, order, valid, inside)
    return {"ndvi": ndvi_mask, "gng": gng_mask, "valid": valid,
            "haze_frac": haze_frac}


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
        # season-max: every clear scene in the fixed window; the year's
        # figure is the peak visible bare area (see YEARS comment above)
        scenes = preprocess.clear_scenes(
            provider, bbox, _year_range(yr), shape,
            aoi_mask=inside, min_clear=0.85)
        if not scenes:
            per_year[yr] = None
            continue
        best = None
        for item, rep in scenes:
            d = detect_year(item, bbox, provider, shape, inside, rep["scl"])
            if d["haze_frac"] > HAZE_MAX_FRAC:
                continue  # broken cloud the SCL missed — unusable scene
            g = geo.mask_area_ha(d["gng"], transform)
            if best is None or g > best["gng_ha"]:
                best = {
                    "date": rep["datetime"][:10],
                    "ndvi_ha": round(geo.mask_area_ha(d["ndvi"], transform), 2),
                    "gng_ha": round(g, 2),
                    "clear": round(rep["aoi_clear"], 3),
                    "n_scenes": len(scenes),
                    "_masks": d,
                }
        per_year[yr] = best
        if best is None:
            continue  # every scene in the window was hazy
        if masks_first is None:
            masks_first = (yr, best["_masks"], inside)
        masks_last = (yr, best["_masks"], inside)

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
