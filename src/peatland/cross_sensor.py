"""Cross-sensor comparison of PlanetScope 3 m with Sentinel-2 10 m candidates.

Pure calculations shared by the single-site case study (scripts/21) and the
multi-site resolution study (scripts/25): the adaptive NDVI threshold, the
cross-sensor NDVI calibration, exact area-weighted regridding between the two
pixel sizes, patch and narrow-feature statistics, and overlap metrics.

PlanetScope has no SWIR, so the Sentinel-2 burn and water/deep-shadow guards
cannot be reproduced here; everything below describes candidate bare surface,
never confirmed turf cutting.
"""

import numpy as np
from rasterio.warp import reproject, Resampling
from scipy import ndimage, sparse

from .gng import GrowingNeuralGas

NDVI_CAP = 0.25
CONTRAST_DROP = 0.18       # same physical contrast rule as pipeline.py v3
NDVI_FLOOR = 0.10
MIN_COVERAGE = 0.95        # assessed Planet area / complete Sentinel-2 cell


def adaptive_threshold(ndvi, analysis=None):
    """max(0.10, min(0.25, site NDVI p75 - 0.18)); ignore nonfinite data."""
    values = np.asarray(ndvi, dtype=float)
    if analysis is not None:
        values = values[np.asarray(analysis, dtype=bool)]
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError("No finite site NDVI values for the adaptive threshold")
    return float(max(NDVI_FLOOR, min(NDVI_CAP, np.quantile(values, 0.75) - CONTRAST_DROP)))


def indices(reflectance):
    """NDVI and green/NIR NDWI from an H x W x 4 blue/green/red/NIR stack."""
    blue, green, red, nir = np.moveaxis(reflectance, -1, 0)
    ndvi = (nir - red) / (nir + red + 1e-9)
    ndwi = (green - nir) / (green + nir + 1e-9)
    return ndvi, ndwi


def candidate_mask(ndvi, ndwi, analysis, threshold):
    """Low NDVI, with NDWI > 0 water excluded; strict NDVI inequality."""
    return (np.asarray(analysis, dtype=bool) & np.isfinite(ndvi) & np.isfinite(ndwi)
            & (ndvi < threshold) & (ndwi <= 0.0))


def fit_ndvi_calibration(planet_ndvi, sentinel2_ndvi, planet_ndwi, sentinel2_ndwi, support):
    """OLS S2 = a * PS10 + b on every finite, nonwater supported cell.

    Inputs are matching arrays on the common 10 m grid. Planet indices must
    be ratios of area-mean reflectance bands, not averages of native ratios.
    No candidate labels enter the fit. Return metrics and the selected-cell
    mask; RMSE uses n as denominator and r is Pearson correlation (undefined
    for a constant response). Inputs are never modified.
    """
    x, y, xw, yw = [np.asarray(values, dtype=float) for values in
                    (planet_ndvi, sentinel2_ndvi, planet_ndwi, sentinel2_ndwi)]
    selected = np.asarray(support, dtype=bool).copy()
    if any(values.shape != selected.shape for values in (x, y, xw, yw)):
        raise ValueError("Calibration inputs must have matching shapes")
    selected &= (np.isfinite(x) & np.isfinite(y) & np.isfinite(xw) & np.isfinite(yw)
                 & (xw <= 0) & (yw <= 0))
    x, y = x[selected], y[selected]
    if x.size < 2:
        raise ValueError("Calibration requires at least two finite nonwater paired cells")
    xc, yc = x - x.mean(), y - y.mean()
    xx, yy = float(xc @ xc), float(yc @ yc)
    if np.ptp(x) == 0:
        raise ValueError("Calibration requires varying PlanetScope NDVI")
    xy = float(xc @ yc)
    a = xy / xx
    b = float(y.mean() - a * x.mean())
    r = float(np.clip(xy / np.sqrt(xx * yy), -1, 1)) if np.ptp(y) > 0 else None
    return {"a": a, "b": b, "r": r,
            "rmse": float(np.sqrt(np.mean((y - (a * x + b)) ** 2))),
            "n": int(x.size)}, selected


def gng_candidates(reflectance, analysis, threshold, seed=42, ndvi=None):
    """Pixel NDVI gate plus physical water labelling of learned prototypes.

    Like pipeline.spectral_bare, only the water decision is learned: a
    pixel's nearest de-standardized prototype is water if its NDWI > 0.
    The direct pixel NDWI exclusion is also retained, so the variant can
    only remove candidates. No centroid-NDVI gate or spatial cleanup is added.
    """
    native_ndvi, ndwi = indices(reflectance)
    ndvi = native_ndvi if ndvi is None else np.asarray(ndvi)
    if ndvi.shape != analysis.shape:
        raise ValueError("GNG NDVI and analysis must have matching shapes")
    raw = np.column_stack([reflectance[analysis], ndvi[analysis], ndwi[analysis]])
    result = candidate_mask(ndvi, ndwi, analysis, threshold)
    info = {"seed": seed, "max_nodes": 80, "training_steps": 15000,
            "features": ["blue", "green", "red", "nir", "ndvi", "ndwi"],
            "labelling": "pixel NDVI < threshold, pixel NDWI <= 0, prototype NDWI <= 0",
            "sample_pixels": min(8000, len(raw)), "site_pixels": len(raw)}
    if len(raw) < 2:
        info.update({"fitted": False, "nodes": 0, "water_nodes": 0})
        return result, info
    mu, sd = raw.mean(axis=0), raw.std(axis=0) + 1e-9
    features = (raw - mu) / sd
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(features), size=info["sample_pixels"], replace=False)
    net = GrowingNeuralGas(max_nodes=info["max_nodes"], rng=rng)
    net.fit(features[sample], n_steps=info["training_steps"])
    weights_raw = net.weights * sd + mu
    water_nodes = weights_raw[:, 5] > 0.0
    result[analysis] &= ~water_nodes[net.predict(features)]
    info.update({"fitted": True, "nodes": len(net.weights),
                 "water_nodes": int(water_nodes.sum()),
                 "feature_mean": mu.tolist(), "feature_std_plus_epsilon": sd.tolist()})
    return result, info


def north_up(transform):
    if transform.b != 0 or transform.d != 0 or transform.a <= 0 or transform.e >= 0:
        raise ValueError("Expected a north-up, unrotated raster grid")


def _axis_overlap(src_start, src_step, src_size, dst_start, dst_step, dst_size):
    """Sparse destination-by-source matrix of exact interval overlap lengths."""
    src = src_start + np.arange(src_size) * src_step
    dst = dst_start + np.arange(dst_size) * dst_step
    overlap = np.maximum(0.0, np.minimum(dst[:, None] + dst_step, src + src_step)
                         - np.maximum(dst[:, None], src))
    return sparse.csr_matrix(overlap)


def _grid_weights(src_shape, src_transform, dst_shape, dst_transform):
    north_up(src_transform)
    north_up(dst_transform)
    cols = _axis_overlap(src_transform.c, src_transform.a, src_shape[1],
                         dst_transform.c, dst_transform.a, dst_shape[1])
    rows = _axis_overlap(-src_transform.f, -src_transform.e, src_shape[0],
                         -dst_transform.f, -dst_transform.e, dst_shape[0])
    return rows, cols


def _weighted_area(values, rows, cols):
    return cols.dot(rows.dot(np.asarray(values, dtype=float)).T).T


def aggregate_to_grid(mask, analysis, src_transform, dst_shape, dst_transform):
    """Exact area-weighted bare fraction and assessed coverage of each cell.

    Both north-up grids must use the same projected CRS. Intersect source
    pixel rectangles with target cells; do not round the 10/3 scale ratio.
    Bare fraction is bare area / observed assessed area. Coverage is observed
    assessed area / full target-cell area, including raster-edge gaps.
    An unsupported cell has fraction NaN and coverage zero, not a negative label.
    """
    mask, analysis = np.asarray(mask, dtype=bool), np.asarray(analysis, dtype=bool)
    if mask.ndim != 2 or mask.shape != analysis.shape:
        raise ValueError("Candidate and analysis masks must be matching 2D arrays")
    rows, cols = _grid_weights(mask.shape, src_transform, dst_shape, dst_transform)
    support = _weighted_area(analysis, rows, cols)
    bare = _weighted_area(mask & analysis, rows, cols)
    fraction = np.full(dst_shape, np.nan, dtype=float)
    np.divide(bare, support, out=fraction, where=support > 0)
    cell_area = abs(dst_transform.a * dst_transform.e)
    return np.clip(fraction, 0, 1), np.clip(support / cell_area, 0, 1)


def aggregate_labels(fraction, coverage, minimum_coverage=MIN_COVERAGE):
    """A cell is bare at fraction >= 0.5, provided it has sufficient support."""
    supported = np.isfinite(fraction) & (coverage > 0) & (coverage >= minimum_coverage)
    return supported & (fraction >= 0.5), supported


def area_mean_reflectance(reflectance, analysis, src_transform, dst_shape, dst_transform):
    """Exact area-mean bands and assessed coverage, before computing indices."""
    reflectance = np.asarray(reflectance, dtype=float)
    analysis = np.asarray(analysis, dtype=bool)
    if reflectance.ndim != 3 or reflectance.shape[:2] != analysis.shape:
        raise ValueError("Reflectance must be H x W x bands with a matching analysis mask")
    valid = analysis & np.isfinite(reflectance).all(axis=2)
    rows, cols = _grid_weights(analysis.shape, src_transform, dst_shape, dst_transform)
    support = _weighted_area(valid, rows, cols)
    mean = np.full((*dst_shape, reflectance.shape[2]), np.nan)
    for band in range(reflectance.shape[2]):
        total = _weighted_area(np.where(valid, reflectance[:, :, band], 0), rows, cols)
        np.divide(total, support, out=mean[:, :, band], where=support > 0)
    return mean, np.clip(support / abs(dst_transform.a * dst_transform.e), 0, 1)


def patch_statistics(mask, pixel_area_m2):
    """Eight-connected patches, with no sieve; areas and histogram in hectares."""
    if pixel_area_m2 <= 0:
        raise ValueError("Pixel area must be positive")
    labels, count = ndimage.label(np.asarray(mask, dtype=bool), structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())[1:] * pixel_area_m2 / 10000.0
    edges = [0, 0.01, 0.05, 0.1, 0.5, 1, float("inf")]
    histogram = np.histogram(sizes, bins=edges)[0]
    return {"count": int(count), "connectivity": 8,
            "area_ha": float(sizes.sum()), "sizes_ha": np.sort(sizes).tolist(),
            "min_ha": float(sizes.min()) if count else None,
            "median_ha": float(np.median(sizes)) if count else None,
            "p90_ha": float(np.quantile(sizes, 0.9)) if count else None,
            "max_ha": float(sizes.max()) if count else None,
            "histogram_bins_ha": ["[0,0.01)", "[0.01,0.05)", "[0.05,0.1)",
                                  "[0.1,0.5)", "[0.5,1)", "[1,infinity)"],
            "histogram_counts": histogram.tolist()}


def narrow_features(mask, transform, width_m=10.0):
    """Area removed by opening with a 10 m diameter disk sampled on the grid.

    This is a morphological proxy: native-pixel discretization makes widths
    near 10 m ambiguous and boundary protrusions can be removed too.
    """
    north_up(transform)
    dx, dy = transform.a, -transform.e
    radius = width_m / 2
    xs = np.arange(-int(np.ceil(radius / dx)), int(np.ceil(radius / dx)) + 1) * dx
    ys = np.arange(-int(np.ceil(radius / dy)), int(np.ceil(radius / dy)) + 1) * dy
    disk = ys[:, None] ** 2 + xs[None, :] ** 2 <= radius ** 2
    opened = ndimage.binary_opening(mask, structure=disk, border_value=0)
    removed = mask & ~opened
    total = int(mask.sum())
    info = {"method": "binary opening; disk footprint uses pixel-centre distances",
            "nominal_diameter_m": width_m, "footprint": disk.astype(int).tolist(),
            "removed_pixels": int(removed.sum()),
            "removed_area_ha": float(removed.sum() * dx * dy / 10000),
            "share_of_bare_area": float(removed.sum() / total) if total else None}
    return removed, info


def overlap_statistics(first, second, support):
    a, b = first & support, second & support
    intersection, union = int((a & b).sum()), int((a | b).sum())
    return {"support_cells": int(support.sum()), "first_cells": int(a.sum()),
            "second_cells": int(b.sum()), "intersection_cells": intersection,
            "union_cells": union, "iou": intersection / union if union else None,
            "only_first_cells": int((a & ~b).sum()),
            "only_second_cells": int((b & ~a).sum())}


def distribution(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return {"pixels": len(values), **{
        name: float(np.quantile(values, q)) if values.size else None
        for name, q in [("p10", 0.1), ("p25", 0.25), ("median", 0.5),
                        ("p75", 0.75), ("p90", 0.9)]}}


def project_mask(mask, src_transform, src_crs, dst_shape, dst_transform, dst_crs):
    result = np.zeros(dst_shape, dtype=np.uint8)
    reproject(mask.astype(np.uint8), result, src_transform=src_transform,
              src_crs=src_crs, dst_transform=dst_transform, dst_crs=dst_crs,
              resampling=Resampling.nearest, dst_nodata=0)
    return result.astype(bool)


def zoom_bounds(mask, transform, width_m=400):
    """Densest full 400 m window in the frozen mask; ties use row then column."""
    size = min(int(round(width_m / transform.a)), *mask.shape)
    integral = np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    counts = (integral[size:, size:] - integral[:-size, size:]
              - integral[size:, :-size] + integral[:-size, :-size])
    row, col = np.unravel_index(np.argmax(counts), counts.shape)
    west, north = transform * (int(col), int(row))
    east, south = transform * (int(col + size), int(row + size))
    return (west, south, east, north), int(counts[row, col])


def true_colour(stack, valid):
    rgb = np.clip(stack[:, :, [2, 1, 0]] / 10000.0 * 3.2, 0, 1) ** (1 / 1.4)
    rgb[~valid] = 0.13
    return rgb


def compare_masks(planet, planet_analysis, planet_transform,
                  sentinel2, sentinel2_transform, common):
    """Area, aggregation, patch and narrow-feature metrics for one site-year.

    `planet` is the 3 m candidate mask on its own grid, `sentinel2` the frozen
    10 m mask, and `common` the 10 m cells where both sensors are assessed.
    IoU is None when neither sensor has a candidate on the common support: an
    empty union is undefined, never a perfect or a zero score.
    """
    ps_area = abs(planet_transform.a * planet_transform.e)
    s2_area = abs(sentinel2_transform.a * sentinel2_transform.e)
    fraction, coverage = aggregate_to_grid(planet, planet_analysis, planet_transform,
                                           sentinel2.shape, sentinel2_transform)
    aggregated, _ = aggregate_labels(fraction, coverage)
    overlap = overlap_statistics(aggregated, sentinel2, common)
    _, narrow = narrow_features(planet, planet_transform)
    planet_ha = float(planet.sum() * ps_area / 10000)
    sentinel2_ha = float((sentinel2 & common).sum() * s2_area / 10000)
    planet_common_ha = float((aggregated & common).sum() * s2_area / 10000)
    return {
        "planet_area_ha": planet_ha,
        "planet_area_common_support_ha": planet_common_ha,
        "sentinel2_area_common_support_ha": sentinel2_ha,
        "area_ratio_common_support": planet_common_ha / sentinel2_ha if sentinel2_ha else None,
        "iou": overlap["iou"],
        "only_3m_cells": overlap["only_first_cells"],
        "only_10m_cells": overlap["only_second_cells"],
        "shared_cells": overlap["intersection_cells"],
        "overlap": overlap,
        "narrow_share": narrow["share_of_bare_area"],
        "narrow_area_ha": narrow["removed_area_ha"],
        "patches_3m": patch_statistics(planet, ps_area)["count"],
        "patches_10m": patch_statistics(sentinel2 & common, s2_area)["count"],
    }, aggregated


def summarise(values):
    """Median, quartiles and range of a per-site metric, ignoring undefined ones."""
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not values.size:
        return {"n": 0, "median": None, "p25": None, "p75": None, "min": None, "max": None}
    return {"n": int(values.size), "median": float(np.median(values)),
            "p25": float(np.quantile(values, 0.25)), "p75": float(np.quantile(values, 0.75)),
            "min": float(values.min()), "max": float(values.max())}
