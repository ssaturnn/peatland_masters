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

import base64

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
# Season windows differ by bog type. Turf cutting on the raised bogs opens
# in April, so their window starts Apr 15 to catch the freshest faces.
# Blanket bogs keep the later start: their vegetation (Molinia, Calluna) is
# still winter-brown in April, and the phenology false-positive risk of an
# earlier window outweighs the extra scenes.
SEASONS = {"raised": ("04-15", "07-15"), "blanket": SEASON}
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
# Adaptive contrast floor: senescent bog vegetation (winter-brown Molinia on
# the blanket bogs in May, April dead grass on the raised bogs) can sit just
# under the fixed NDVI threshold across a large area without any peat being
# exposed. Fresh cut peat, by contrast, is always an ANOMALY against the
# bog's own vegetated matrix. So the effective GNG threshold on each scene is
# capped at (75th-percentile NDVI over the bog − CONTRAST_DROP): on a green
# scene (matrix ≈ 0.6) the cap is inactive and the fixed 0.25 applies; on a
# uniformly brown scene (matrix ≈ 0.35) it tightens to ≈ 0.17, rejecting the
# in-distribution brown pixels while keeping genuinely dark cut faces
# (fresh peat NDVI ≈ 0.10). Floored so the detector never goes blind.
CONTRAST_DROP = 0.18
GNG_NDVI_FLOOR = 0.10
# Cloud-shadow projection (SCL class 3 misses many shadows): project the
# scene's cloud pixels along the anti-solar azimuth for a range of plausible
# cloud heights and mark dark-NIR pixels under the projection as shadow.
SHADOW_CLOUD_HEIGHTS_M = (400.0, 800.0, 1200.0, 1600.0, 2000.0)
SHADOW_NIR_DARK = 0.15


def _year_range(year, bog_type="raised"):
    a, b = SEASONS.get(bog_type, SEASON)
    return f"{year}-{a}/{year}-{b}"


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


DETECTOR_VERSION = "2026-09-13-cloud-screen-v3"
# Bright-cloud screen (v3). The SCL misses broken cumulus, and after the
# radiometric harmonisation the brightness haze gate lets cloud fields
# through. Cloud is several times brighter in blue than any bog surface
# (vegetation ~0.03-0.05, bare peat mostly <=0.10), so blue reflectance
# above CLOUD_BLUE marks cloud; the mask is dilated by CLOUD_BUFFER_PX to
# take out the darker, parallax-shifted cloud edges, which otherwise read
# as low-NDVI "bare" pixels. Calibrated on the cloud-affected calibration
# scenes (Doogort East 2023, Moorfield 2025) only.
CLOUD_BLUE = 0.16
CLOUD_BUFFER_PX = 3
# Scene haze gate (v3): HOT-style haze index blue - 0.5*red (after Zhang et
# al., 2002). A scene where more than HOT_MAX_FRAC of the bog exceeds
# HOT_HAZE is a haze/cloud field and leaves the season pool (measured: 0.57
# on the hazy Moorfield 2025 scene vs <=0.063 on usable scenes).
HOT_HAZE = 0.02
HOT_MAX_FRAC = 0.15


def ever_water_mask(scene_reports, shape):
    """Union of SCL open-water pixels over every clear scene of a site.

    Intertidal flats defeat single-scene spectral rules: at high tide they
    are open water, at low tide dark wet sediment with the same low-NDVI,
    low-brightness, moisture-bearing signature as fresh bare peat — the
    estuarine NHAs (Tullaghan Bay) otherwise report hundreds of hectares
    of "cutting" that track the tide state at acquisition. A pixel that is
    open water on ANY clear scene in the record is tidal or lacustrine and
    is excluded from detection in every year."""
    water = np.zeros(shape, dtype=bool)
    for rep in scene_reports:
        water |= np.asarray(rep["scl"]) == 6
    return water


def mask_to_b64(mask):
    """Compact JSON-safe encoding of a bool mask (bit-packed, base64)."""
    return base64.b64encode(np.packbits(mask.astype(np.uint8))).decode()


def mask_from_b64(text, shape):
    bits = np.unpackbits(np.frombuffer(base64.b64decode(text), dtype=np.uint8))
    return bits[:shape[0] * shape[1]].reshape(shape).astype(bool)


def spectral_bare(stack, order, valid, inside, method="gng", seed=42,
                  max_nodes=80, k=6, thresh=GNG_BARE_NDVI,
                  adaptive=True):
    """Comparable GNG, K-means and rules-only bare-surface detectors.

    All three use the same valid AOI, contrast threshold, burn and brightness
    guards. Only water/shadow assignment differs: nearest learned prototype
    for GNG/K-means, or the pixel's own spectrum for rules-only. This permits
    an ablation of clustering without attributing rule improvements to GNG.
    Scaling and training exclude cloud, no-data and pixels outside the bog.
    Outputs are candidate bare surfaces, not confirmed extraction activity.
    """
    if method not in {"gng", "kmeans", "rules"}:
        raise ValueError(f"Unknown spectral method: {method}")
    if list(order) != BANDS:
        raise ValueError(f"Expected band order {BANDS}, received {list(order)}")
    stack = np.asarray(stack, dtype=np.float64)
    shape = stack.shape[:2]
    if stack.ndim != 3 or stack.shape[2] != len(BANDS):
        raise ValueError("Expected a six-band H x W x B stack")
    if valid.shape != shape or inside.shape != shape:
        raise ValueError("Validity and site masks must match the image grid")
    sel = (np.asarray(valid, dtype=bool) & np.asarray(inside, dtype=bool)
           & np.isfinite(stack).all(axis=2))
    sel &= brightness(stack, order) < BRIGHT_CAP
    result = np.zeros(shape, dtype=bool)
    if not sel.any():
        return result

    # Form features only from trusted pixels: invalid values cannot influence
    # normalization, learned prototypes or the adaptive contrast percentile.
    refl = stack[sel] / 10000.0
    raw = _gng_feature_stack(refl[:, None, :], order)[:, 0, :]
    if method == "rules" or len(raw) < 2:
        nonpeat = _nonpeat_nodes(raw)
    else:
        mu, sd = raw.mean(axis=0), raw.std(axis=0) + 1e-9
        feats = (raw - mu) / sd
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(feats), size=min(8000, len(feats)), replace=False)
        if method == "gng":
            net = gng.GrowingNeuralGas(max_nodes=max_nodes, rng=rng)
            net.fit(feats[idx], n_steps=15000)
            prototypes, labels = net.weights, net.predict(feats)
        else:
            from sklearn.cluster import KMeans
            if k < 1:
                raise ValueError("k must be positive")
            clusters = min(k, len(np.unique(feats[idx], axis=0)))
            net = KMeans(n_clusters=clusters, n_init=4, random_state=seed)
            net.fit(feats[idx])
            prototypes, labels = net.cluster_centers_, net.predict(feats)
        nonpeat = _nonpeat_nodes(prototypes * sd + mu)[labels]

    ndvi = raw[:, GNG_FEATURES.index("ndvi")]
    eff = thresh
    if adaptive and len(raw) >= 100:
        matrix = float(np.quantile(ndvi, 0.75))
        eff = min(thresh, max(matrix - CONTRAST_DROP, GNG_NDVI_FLOOR))
    nir, swir2 = refl[:, order.index("nir")], refl[:, order.index("swir2")]
    nbr = (nir - swir2) / (nir + swir2 + 1e-9)
    result[sel] = (ndvi < eff) & ~nonpeat & (nbr > BURN_NBR_FLOOR)
    return result


def gng_bare(stack, order, valid, inside, seed=42, max_nodes=80,
             thresh=GNG_BARE_NDVI, adaptive=True):
    return spectral_bare(stack, order, valid, inside, method="gng",
                         seed=seed, max_nodes=max_nodes, thresh=thresh,
                         adaptive=adaptive)


def kmeans_bare(stack, order, valid, inside, k=6, seed=42,
                thresh=GNG_BARE_NDVI, adaptive=True):
    return spectral_bare(stack, order, valid, inside, method="kmeans",
                         seed=seed, k=k, thresh=thresh, adaptive=adaptive)


def rules_bare(stack, order, valid, inside, thresh=GNG_BARE_NDVI,
               adaptive=True):
    return spectral_bare(stack, order, valid, inside, method="rules",
                         thresh=thresh, adaptive=adaptive)


def _shift2d(m, dr, dc):
    """Shift a bool mask by (dr, dc) pixels, zero-filling the edges."""
    out = np.zeros_like(m)
    h, w = m.shape
    r0, r1 = max(dr, 0), min(h + dr, h)
    c0, c1 = max(dc, 0), min(w + dc, w)
    if r0 < r1 and c0 < c1:
        out[r0:r1, c0:c1] = m[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return out


def _sun_geometry(item):
    """(azimuth_deg, zenith_deg) from STAC metadata, or (None, None)."""
    p = item.properties
    az = p.get("view:sun_azimuth", p.get("s2:mean_solar_azimuth"))
    zen = p.get("s2:mean_solar_zenith")
    if zen is None and p.get("view:sun_elevation") is not None:
        zen = 90.0 - float(p["view:sun_elevation"])
    if az is None or zen is None:
        return None, None
    return float(az), float(zen)


def cloud_shadow_mask(scl, br, nir_refl, item, px_m=10.0, cloud=None):
    """Geometric cloud-shadow mask the SCL misses.

    SCL's own shadow class (3) badly under-detects. Instead: take the
    scene's cloud pixels (SCL 8/9/10 plus bright unclassified pixels),
    project them along the anti-solar azimuth for a range of plausible
    cloud-base heights (offset = h·tan(solar zenith)), and call a pixel
    shadow when it sits under a projection AND is dark in NIR — vegetation
    and peat in shadow both go very dark at 842 nm. Clouds outside the
    read window cannot be projected; that residual risk stays with the
    scene-level haze gate.
    """
    az, zen = _sun_geometry(item)
    if cloud is None:
        cloud = np.isin(scl, (8, 9, 10)) | (br > HAZE_BRIGHT)
    if az is None or not cloud.any():
        return np.zeros_like(cloud, dtype=bool)
    shadow_az = np.deg2rad((az + 180.0) % 360.0)
    tanz = np.tan(np.deg2rad(zen))
    proj = np.zeros_like(cloud, dtype=bool)
    for h_m in SHADOW_CLOUD_HEIGHTS_M:
        d = h_m * tanz
        dc = int(round(d * np.sin(shadow_az) / px_m))
        dr = int(round(-d * np.cos(shadow_az) / px_m))
        proj |= _shift2d(cloud, dr, dc)
    return proj & (nir_refl < SHADOW_NIR_DARK) & ~cloud


def scene_quality(stack, order, scl, inside, item):
    """Shared quality masks for dataset, comparison, sampling and rendering."""
    from scipy import ndimage
    from . import preprocess
    valid = preprocess.valid_mask(scl) & np.isfinite(stack).all(axis=2)
    br = brightness(stack, order)
    blue = np.nan_to_num(stack[:, :, order.index("blue")] / 10000.0)
    red = np.nan_to_num(stack[:, :, order.index("red")] / 10000.0)
    cloud = np.isin(scl, (8, 9, 10)) | (blue > CLOUD_BLUE)
    if CLOUD_BUFFER_PX and cloud.any():
        size = 2 * CLOUD_BUFFER_PX + 1
        cloud = ndimage.binary_dilation(cloud, structure=np.ones((size, size), bool))
    nir = stack[:, :, order.index("nir")]
    shadow = cloud_shadow_mask(scl, br, nir / 10000.0, item, cloud=cloud)
    valid &= ~cloud & ~shadow
    denom = max(int(inside.sum()), 1)
    return {
        "valid": valid,
        "haze_frac": float(((br > HAZE_BRIGHT) & inside).sum() / denom),
        "hot_frac": float(((blue - 0.5 * red > HOT_HAZE) & inside).sum() / denom),
        "cloud_frac": float((cloud & inside).sum() / denom),
        "shadow_frac": float((shadow & inside).sum() / denom),
        "clear_fraction": float((valid & inside).sum() / denom),
    }


def scene_passes(quality):
    """Shared scene gates: bright haze, HOT haze field, too little clear bog."""
    return (quality["haze_frac"] <= HAZE_MAX_FRAC
            and quality.get("hot_frac", 0.0) <= HOT_MAX_FRAC
            and quality["clear_fraction"] >= 0.85)


def detect_year(item, bbox, provider, shape, inside, scl):
    """Read a scene onto the FIXED reference grid and return both masks.

    Every band is resampled to `shape` so all years, the SCL mask and the
    site polygon share one grid — different scenes over the same bbox can
    otherwise land on slightly different pixel windows.
    """
    order = list(BANDS)
    layers = [imagery.read_window(item, b, bbox, provider, out_shape=shape)[0]
              .astype("float32") for b in order]
    stack = np.stack(layers, axis=-1)
    red = stack[:, :, order.index("red")]
    nir = stack[:, :, order.index("nir")]
    ndvi = detect.ndvi(red, nir)
    quality = scene_quality(stack, order, scl, inside, item)
    valid = quality["valid"]
    # vegetated-matrix greenness (p75 NDVI over the bog): scenes where the
    # whole matrix is senescent give the detector little contrast to work
    # with, so callers can flag years measured only on such scenes
    sel = valid & inside
    matrix = float(np.quantile(ndvi[sel], 0.75)) if sel.sum() >= 100 else float("nan")
    ndvi_mask = ndvi_bare(stack, order, ndvi, valid, inside)
    gng_mask = gng_bare(stack, order, valid, inside)
    return {"ndvi": ndvi_mask, "gng": gng_mask, **quality,
            "matrix_p75": matrix, "stack": stack}


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

    bog_type = row.get("bog_type", "raised")

    # Phase 1: gather every year's clear scenes once, and build the site's
    # ever-water mask across ALL of them, so early years are screened with
    # water knowledge from later scenes too (see ever_water_mask).
    year_scenes = {}
    for yr in years:
        year_scenes[yr] = preprocess.clear_scenes(
            provider, bbox, _year_range(yr, bog_type), shape,
            aoi_mask=inside, min_clear=0.85)
    tidal = ever_water_mask(
        [rep for scenes in year_scenes.values() for _, rep in scenes], shape)
    det_inside = inside & ~tidal

    per_year = {}
    masks_first = masks_last = None
    for i, yr in enumerate(years):
        # season-max: every clear scene in the fixed window; the year's
        # figure is the peak visible bare area (see YEARS comment above)
        scenes = year_scenes[yr]
        if not scenes:
            per_year[yr] = None
            continue
        best = None
        for item, rep in scenes:
            d = detect_year(item, bbox, provider, shape, det_inside,
                            rep["scl"])
            if not scene_passes(d):
                continue  # cloud/haze the SCL missed, or too little clear bog
            g = geo.mask_area_ha(d["gng"], transform)
            if best is None or g > best["gng_ha"]:
                best = {
                    "scene_id": item.id,
                    "processing_baseline": item.properties.get("s2:processing_baseline"),
                    "date": rep["datetime"][:10],
                    "ndvi_ha": round(geo.mask_area_ha(d["ndvi"], transform), 2),
                    "gng_ha": round(g, 2),
                    "clear": round(d["clear_fraction"], 3),
                    "n_scenes": len(scenes),
                    "matrix": round(d["matrix_p75"], 2),
                    "_masks": {k: v for k, v in d.items() if k != "stack"},
                }
        per_year[yr] = best
        if best is None:
            continue  # every scene in the window was hazy
        if masks_first is None:
            masks_first = (yr, best["_masks"], inside)
        masks_last = (yr, best["_masks"], inside)

    site_ha = round(geo.mask_area_ha(inside, transform), 1)
    valid_years = [y for y in years if per_year.get(y)]

    out = {"detector_version": DETECTOR_VERSION,
           "water_excl_pct": round(
               100 * (tidal & inside).sum() / max(inside.sum(), 1), 1),
           "tidal_mask_b64": mask_to_b64(tidal),
           "site_ha": site_ha, "years": valid_years,
           "ndvi_series": [per_year[y]["ndvi_ha"] for y in valid_years],
           "gng_series": [per_year[y]["gng_ha"] for y in valid_years],
           "dates": [per_year[y]["date"] for y in valid_years],
           "scene_ids": [per_year[y]["scene_id"] for y in valid_years],
           "processing_baselines": [per_year[y]["processing_baseline"] for y in valid_years],
           "grid": {"shape": list(shape), "transform": list(transform), "crs": str(crs)},
           "matrix_series": [per_year[y].get("matrix") for y in valid_years],
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
