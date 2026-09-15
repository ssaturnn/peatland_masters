"""Step 21 — same-day, 3 m PlanetScope case study at Monivea Bog SAC.

Calibrate PlanetScope NDVI on paired site cells, then compare an adaptive
NDVI/NDWI mask and a four-band GNG variant with the saved Sentinel-2 v3
GNG prediction. Retain naive threshold transfer as a baseline. These are bare-surface
candidates, not confirmed turf cutting. PlanetScope has no SWIR: the
Sentinel-2 burn and water/deep-shadow guards cannot be reproduced.

Usage from the repository root:
    python3 scripts/21_planet_case_study.py
    python3 scripts/21_planet_case_study.py --run outputs/evaluation/<run> \
        --out outputs/planet_case_study

No imagery is downloaded and the frozen detector is never re-run.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import reproject, Resampling
from scipy import ndimage, sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True

from peatland.gng import GrowingNeuralGas

STEM = "002352_2022"
DATE = "2022-04-23"
DEFAULT_RUN = Path("outputs/evaluation/2026-09-13-v3-sample")
DEFAULT_OUT = Path("outputs/planet_case_study")
NDVI_CAP = 0.25
CONTRAST_DROP = 0.18       # same physical contrast rule as pipeline.py v3
NDVI_FLOOR = 0.10
MIN_COVERAGE = 0.95       # assessed Planet area / complete Sentinel-2 cell
CITATION = ("Planet Team (2026). Planet Application Program Interface: In Space for Life "
            "on Earth. San Francisco, CA. https://api.planet.com")
LIMITATIONS = [
    "Single date and one calibration site; candidates are not confirmed turf cutting.",
    "PlanetScope has no SWIR: the Sentinel-2 NBR burn and SWIR water/deep-shadow "
    "guards are unavailable. No SCL or UDM2 cloud/shadow mask is used here.",
    "A possible 1-2 pixel co-registration error is not corrected or measured; "
    "3-6 m on the Planet grid can materially change narrow-feature overlap.",
    "Site/date OLS harmonisation reduces cross-sensor NDVI differences but does not "
    "correct co-registration, bandpass or spatial response. Applying a 10 m fit at "
    "3 m assumes the relationship transfers across scale; residuals are in-sample.",
    "Patch counts depend on 8-connectivity, pixel size and the assessed-area boundary; "
    "no minimum mapping unit is imposed.",
    "Opening loss is a narrow-feature proxy, including small objects and protrusions, "
    "not a measured width or cutting-area estimate; widths are quantized at 3 m.",
]


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


def _north_up(transform):
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
    _north_up(src_transform)
    _north_up(dst_transform)
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
    _north_up(transform)
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


def compare_variant(rules, gng, analysis, ps_transform, s2_mask, s2_transform,
                    common, frozen_on_planet):
    """The same native-grid, aggregation and morphology metrics for each variant."""
    ps_area = abs(ps_transform.a * ps_transform.e)
    s2_area = abs(s2_transform.a * s2_transform.e)
    fraction, coverage = aggregate_to_grid(rules, analysis, ps_transform, s2_mask.shape, s2_transform)
    gng_fraction, _ = aggregate_to_grid(gng, analysis, ps_transform, s2_mask.shape, s2_transform)
    aggregated, _ = aggregate_labels(fraction, coverage)
    aggregated_gng, _ = aggregate_labels(gng_fraction, coverage)
    overlap = overlap_statistics(aggregated, s2_mask, common)
    overlap_gng = overlap_statistics(aggregated_gng, s2_mask, common)
    ablation = overlap_statistics(gng, rules, analysis)
    narrow, narrow_info = narrow_features(rules, ps_transform)
    narrow_gng, narrow_gng_info = narrow_features(gng, ps_transform)
    metrics = {
        "candidate_area_ha": {"planet_rules": float(rules.sum() * ps_area / 10000),
                              "planet_gng": float(gng.sum() * ps_area / 10000),
                              "sentinel2_gng": float(s2_mask.sum() * s2_area / 10000),
                              "sentinel2_gng_common_support": float((s2_mask & common).sum() * s2_area / 10000),
                              "planet_minus_sentinel2": float((rules.sum() * ps_area - s2_mask.sum() * s2_area) / 10000)},
        "aggregation": {"method": "exact source/target rectangle intersections in the shared metre CRS",
                        "fraction_denominator": "observed assessed area in each 10 m cell",
                        "minimum_assessed_coverage": MIN_COVERAGE, "bare_fraction_threshold": 0.5,
                        "unsupported_fraction": "NaN in masks.npz",
                        "comparison_domain": "Sentinel-2 valid & inside & Planet assessed coverage >= 0.95",
                        "only_3m_cells": overlap["only_first_cells"],
                        "only_10m_cells": overlap["only_second_cells"],
                        "iou": overlap["iou"], "rules_vs_sentinel2": overlap,
                        "gng_vs_sentinel2": overlap_gng,
                        "planet_rules_10m_area_ha": float((aggregated & common).sum() * s2_area / 10000),
                        "planet_gng_10m_area_ha": float((aggregated_gng & common).sum() * s2_area / 10000),
                        "max_bare_fraction_on_common_support": float(fraction[common].max()) if common.any() else None,
                        "planet_bare_area_on_common_support_ha": float(np.nansum(
                            fraction[common] * coverage[common]) * s2_area / 10000),
                        "planet_gng_bare_area_on_common_support_ha": float(np.nansum(
                            gng_fraction[common] * coverage[common]) * s2_area / 10000)},
        "gng_ablation": {"comparison": "first=Planet GNG, second=Planet pixel rules",
                         **ablation, "changed_pixels": int((gng ^ rules).sum()),
                         "changed_area_ha": float((gng ^ rules).sum() * ps_area / 10000)},
        "native_3m_overlap": {
            "reference": "frozen Sentinel-2 GNG mask reprojected nearest onto Planet grid",
            "planet_candidate_pixels_inside_reference": int((rules & frozen_on_planet).sum()),
            "planet_candidate_pixels_outside_reference": int((rules & ~frozen_on_planet).sum()),
            "planet_candidate_area_inside_reference_ha": float((rules & frozen_on_planet).sum() * ps_area / 10000),
            "planet_candidate_area_outside_reference_ha": float((rules & ~frozen_on_planet).sum() * ps_area / 10000)},
        "patches": {"planet_rules_3m": patch_statistics(rules, ps_area),
                    "planet_gng_3m": patch_statistics(gng, ps_area),
                    "sentinel2_gng_10m": patch_statistics(s2_mask, s2_area),
                    "planet_rules_aggregated_10m_common": patch_statistics(aggregated & common, s2_area),
                    "planet_gng_aggregated_10m_common": patch_statistics(aggregated_gng & common, s2_area),
                    "sentinel2_gng_10m_common": patch_statistics(s2_mask & common, s2_area)},
        "narrow_features": {"planet_rules": narrow_info, "planet_gng": narrow_gng_info},
    }
    arrays = {"planet_rules": rules, "planet_gng": gng, "planet_narrow": narrow,
              "planet_gng_narrow": narrow_gng,
              "planet_bare_fraction_10m": fraction, "planet_gng_fraction_10m": gng_fraction,
              "planet_rules_aggregated_10m": aggregated & common,
              "planet_gng_aggregated_10m": aggregated_gng & common}
    return metrics, arrays


def _distribution(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return {"pixels": len(values), **{
        name: float(np.quantile(values, q)) if values.size else None
        for name, q in [("p10", 0.1), ("p25", 0.25), ("median", 0.5),
                        ("p75", 0.75), ("p90", 0.9)]}}


def _project_mask(mask, src_transform, src_crs, dst_shape, dst_transform, dst_crs):
    result = np.zeros(dst_shape, dtype=np.uint8)
    reproject(mask.astype(np.uint8), result, src_transform=src_transform,
              src_crs=src_crs, dst_transform=dst_transform, dst_crs=dst_crs,
              resampling=Resampling.nearest, dst_nodata=0)
    return result.astype(bool)


def _zoom_bounds(mask, transform, width_m=400):
    """Densest full 400 m window in the frozen mask; ties use row then column."""
    size = min(int(round(width_m / transform.a)), *mask.shape)
    integral = np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    counts = (integral[size:, size:] - integral[:-size, size:]
              - integral[size:, :-size] + integral[:-size, :-size])
    row, col = np.unravel_index(np.argmax(counts), counts.shape)
    west, north = transform * (int(col), int(row))
    east, south = transform * (int(col + size), int(row + size))
    return (west, south, east, north), int(counts[row, col])


def _true_colour(stack, valid):
    rgb = np.clip(stack[:, :, [2, 1, 0]] / 10000.0 * 3.2, 0, 1) ** (1 / 1.4)
    rgb[~valid] = 0.13
    return rgb


def write_figures(out, s2, ps, bounds, zoom, summary, calibration_pairs):
    # Keep the font cache local to the repository.
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".mplcache")
    os.environ["XDG_CACHE_HOME"] = str(ROOT / ".mplcache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    def panel(ax, data, window, title):
        stack, valid, analysis, mask, transform = data
        west, south, east, north = array_bounds(*mask.shape, transform)
        ax.imshow(_true_colour(stack, valid), extent=(west, east, south, north),
                  interpolation="nearest", origin="upper")
        xs = transform.c + (np.arange(mask.shape[1]) + 0.5) * transform.a
        ys = transform.f + (np.arange(mask.shape[0]) + 0.5) * transform.e
        for layer, colour, width in [(analysis, "#f3f3f3", 0.65), (mask, "#ffd447", 0.9)]:
            if layer.any() and not layer.all():
                ax.contour(xs, ys, layer.astype(float), levels=[0.5],
                           colors=[colour], linewidths=width)
        ax.set_xlim(window[0], window[2])
        ax.set_ylim(window[1], window[3])
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11, loc="left", pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        length = 100 if window == zoom else 500
        x = window[0] + (window[2] - window[0]) * 0.06
        y = window[1] + (window[3] - window[1]) * 0.06
        ax.plot([x, x + length], [y, y], color="black", linewidth=5)
        ax.plot([x, x + length], [y, y], color="white", linewidth=2)
        ax.text(x + length / 2, y + (window[3] - window[1]) * 0.02,
                f"{length} m", ha="center", color="white", fontsize=9,
                bbox={"facecolor": "black", "alpha": 0.65, "pad": 2, "edgecolor": "none"})
        ax.text(0.95, 0.95, "N ↑", transform=ax.transAxes, ha="right", va="top",
                color="white", fontsize=11)

    areas = summary["candidate_area_ha"]
    for filename, window, subtitle in [
        ("whole_site.png", bounds, "Whole assessed site; cyan box marks the zoom"),
        ("cutting_margin_zoom.png", zoom,
         "Cutting-margin zoom: densest 400 m window in the frozen Sentinel-2 mask"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(13.6, 8.3))
        fig.suptitle("Monivea Bog SAC · 23 April 2022", fontsize=18, x=0.06, ha="left", y=0.975)
        fig.text(0.06, 0.924, subtitle, fontsize=11)
        panel(axes[0], s2, window,
              f"Sentinel-2 · 10 m · frozen v3 GNG\nWhole-site candidates: {areas['sentinel2_gng']:.4f} ha")
        panel(axes[1], ps, window,
              f"PlanetScope · 3 m · calibrated NDVI + NDWI\nWhole-site candidates: {areas['planet_rules']:.4f} ha")
        if filename == "whole_site.png":
            for ax in axes:
                ax.add_patch(Rectangle((zoom[0], zoom[1]), zoom[2] - zoom[0], zoom[3] - zoom[1],
                                       fill=False, edgecolor="#4fe4ff", linewidth=1.3))
        fig.legend(handles=[Line2D([0], [0], color="#b89500", lw=2, label="Candidate outline"),
                            Line2D([0], [0], color="#999999", lw=1, label="Assessed-area boundary")],
                   loc="lower left", bbox_to_anchor=(0.055, 0.093), ncol=2, frameon=False, fontsize=10)
        fig.text(0.06, 0.083, "Candidate bare surface; no confirmed-cutting labels. "
                 "PlanetScope has no SWIR burn or deep-shadow guards.", fontsize=9)
        fig.text(0.06, 0.059, "Common RGB display: reflectance × 3.2, gamma 1.4; "
                 "native grids, nearest-neighbour display. NDVI calibrated on paired site cells.", fontsize=9)
        fig.text(0.06, 0.025, CITATION, fontsize=8)
        fig.subplots_adjust(left=0.055, right=0.965, bottom=0.16, top=0.83, wspace=0.08)
        fig.savefig(out / filename, dpi=150, facecolor="white")
        plt.close(fig)

    x, y = calibration_pairs
    fit = summary["calibration"]
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    density = ax.hexbin(x, y, gridsize=65, mincnt=1, bins="log", cmap="viridis")
    limits = [float(min(x.min(), y.min()) - 0.02), float(max(x.max(), y.max()) + 0.02)]
    line_x = np.array(limits)
    ax.plot(line_x, line_x, color="#777777", ls="--", lw=1, label="1:1")
    ax.plot(line_x, fit["a"] * line_x + fit["b"], color="#df542b", lw=1.7, label="OLS fit")
    ax.set(xlim=limits, ylim=limits, xlabel="PlanetScope NDVI at 10 m (NDVI of mean bands)",
           ylabel="Sentinel-2 NDVI at 10 m")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", frameon=False)
    ax.text(0.04, 0.96, f"S2 = {fit['a']:.6f} × PS {fit['b']:+.6f}\n"
            f"r = {fit['r']:.6f} · RMSE = {fit['rmse']:.6f}\nn = {fit['n']:,}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    fig.colorbar(density, ax=ax, shrink=0.8, label="Paired cells per hexagon (log scale)")
    fig.suptitle("Monivea Bog SAC · cross-sensor calibration", fontsize=12, y=0.96)
    fig.text(0.08, 0.895, "23 April 2022 · all finite nonwater paired site cells", fontsize=9)
    fig.text(0.04, 0.045, "Planet Team (2026). Planet Application Program Interface: In Space for Life on Earth.\n"
             "San Francisco, CA. https://api.planet.com", fontsize=6.8)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.18, top=0.86)
    fig.savefig(out / "ndvi_calibration.png", dpi=150, facecolor="white")
    plt.close(fig)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_case(run, out):
    files = {"sentinel2_npz": run / f"{STEM}.npz", "sentinel2_json": run / f"{STEM}.json",
             "planet_mosaic": run / "planet" / STEM / "ps_mosaic.tif",
             "planet_manifest": run / "planet/manifest.json"}
    meta = json.loads(files["sentinel2_json"].read_text())
    provenance = json.loads(files["planet_manifest"].read_text())[STEM]
    if meta["scene_date"] != DATE or provenance["date"] != DATE:
        raise ValueError(f"Both acquisitions must be {DATE}")
    if meta.get("detector_version") != "2026-09-13-cloud-screen-v3":
        raise ValueError("Expected the frozen Sentinel-2 v3 detector")
    with np.load(files["sentinel2_npz"], allow_pickle=False) as data:
        s2_stack = data["stack"].copy()
        s2_valid = data["valid"].astype(bool)
        s2_inside = data["inside"].astype(bool)
        frozen = data["prediction_gng"].astype(bool)
    s2_transform = rasterio.Affine(*meta["transform"][:6])
    s2_crs = rasterio.crs.CRS.from_user_input(meta["crs"])
    with rasterio.open(files["planet_mosaic"]) as ds:
        if ds.count not in (4, 8):
            raise ValueError("Expected a four-band or eight-band PlanetScope mosaic")
        bands = [1, 2, 3, 4] if ds.count == 4 else [2, 4, 6, 8]
        ps_stack = np.moveaxis(ds.read(bands), 0, -1).astype(float)
        ps_valid = ((ds.read_masks(bands) > 0).all(axis=0)
                    & np.isfinite(ps_stack).all(axis=2) & (ps_stack != 0).all(axis=2))
        ps_transform, ps_crs = ds.transform, ds.crs
    if ps_crs != s2_crs or not s2_crs.is_projected or s2_crs.linear_units != "metre":
        raise ValueError("Exact area aggregation requires both grids in the same projected metre CRS")
    _north_up(ps_transform)
    _north_up(s2_transform)
    ps_inside = _project_mask(s2_inside, s2_transform, s2_crs, ps_valid.shape, ps_transform, ps_crs)
    analysis = ps_inside & ps_valid
    s2_analysis = s2_inside & s2_valid
    s2_mask = frozen & s2_analysis
    ps_area = abs(ps_transform.a * ps_transform.e)
    s2_area = abs(s2_transform.a * s2_transform.e)
    ndvi, ndwi = indices(ps_stack / 10000)
    s2_ndvi, s2_ndwi = indices(s2_stack[:, :, :4] / 10000)
    ps_mean, coverage = area_mean_reflectance(
        ps_stack / 10000, analysis, ps_transform, s2_mask.shape, s2_transform)
    ps_10_ndvi, ps_10_ndwi = indices(ps_mean)
    paired = s2_analysis & (coverage > 0)
    fit, calibration_support = fit_ndvi_calibration(
        ps_10_ndvi, s2_ndvi, ps_10_ndwi, s2_ndwi, paired)
    # Include all paired site cells in OLS; the stricter coverage cutoff below
    # only controls the majority-label comparison at assessed-area boundaries.
    supported = (coverage > 0) & (coverage >= MIN_COVERAGE)
    common = s2_analysis & supported
    sensitivity, _ = fit_ndvi_calibration(ps_10_ndvi, s2_ndvi, ps_10_ndwi, s2_ndwi, common)
    calibrated_ndvi = fit["a"] * ndvi + fit["b"]
    calibrated_10_ndvi = fit["a"] * ps_10_ndvi + fit["b"]
    threshold = adaptive_threshold(calibrated_ndvi, analysis)
    rules = candidate_mask(calibrated_ndvi, ndwi, analysis, threshold)
    gng, gng_info = gng_candidates(ps_stack / 10000, analysis, threshold, ndvi=calibrated_ndvi)
    naive_threshold = adaptive_threshold(ndvi, analysis)
    naive_rules = candidate_mask(ndvi, ndwi, analysis, naive_threshold)
    naive_gng, naive_gng_info = gng_candidates(ps_stack / 10000, analysis, naive_threshold)
    print(f"Calibration: a={fit['a']:.9f}, b={fit['b']:.9f}, r={fit['r']:.9f}, "
          f"RMSE={fit['rmse']:.9f}, n={fit['n']}", flush=True)
    print(f"Calibrated site NDVI p75={np.quantile(calibrated_ndvi[analysis], 0.75):.6f}; "
          f"threshold={threshold:.6f}; candidates={rules.sum()} pixels", flush=True)
    frozen_on_planet = _project_mask(s2_mask, s2_transform, s2_crs,
                                     analysis.shape, ps_transform, ps_crs) & analysis
    metrics, arrays = compare_variant(rules, gng, analysis, ps_transform, s2_mask,
                                      s2_transform, common, frozen_on_planet)
    naive_metrics, naive_arrays = compare_variant(
        naive_rules, naive_gng, analysis, ps_transform, s2_mask,
        s2_transform, common, frozen_on_planet)
    collocated = s2_mask & common
    diagnostics = {"planet_site_ndvi": _distribution(ndvi[analysis]),
                   "planet_calibrated_site_ndvi": _distribution(calibrated_ndvi[analysis]),
                   "sentinel2_site_ndvi": _distribution(s2_ndvi[s2_analysis]),
                   "planet_ndvi_within_frozen_sentinel2_candidates": _distribution(ndvi[frozen_on_planet]),
                   "planet_calibrated_ndvi_within_frozen_sentinel2_candidates": _distribution(
                       calibrated_ndvi[frozen_on_planet]),
                   "sentinel2_ndvi_on_common_frozen_candidates": _distribution(s2_ndvi[collocated]),
                   "planet_ndvi_of_mean_bands_on_common_frozen_candidates": _distribution(ps_10_ndvi[collocated]),
                   "planet_calibrated_ndvi_of_mean_bands_on_common_frozen_candidates": _distribution(
                       calibrated_10_ndvi[collocated]),
                   "paired_planet_minus_sentinel2_ndvi_on_frozen_candidates": _distribution(
                       ps_10_ndvi[collocated] - s2_ndvi[collocated]),
                   "paired_calibrated_planet_minus_sentinel2_ndvi_on_frozen_candidates": _distribution(
                       calibrated_10_ndvi[collocated] - s2_ndvi[collocated]),
                   "planet_pixels_below_threshold_within_frozen_candidates": int((rules & frozen_on_planet).sum()),
                   "planet_water_pixels_excluded": int((analysis & (ndwi > 0)).sum())}
    zoom, zoom_count = _zoom_bounds(s2_mask, s2_transform)
    rr, cc = np.where(s2_inside)
    west, north = s2_transform * (max(0, int(cc.min()) - 5), max(0, int(rr.min()) - 5))
    east, south = s2_transform * (min(s2_mask.shape[1], int(cc.max()) + 6),
                                 min(s2_mask.shape[0], int(rr.max()) + 6))
    bounds = (west, south, east, north)
    summary = {
        "site": meta["site"], "site_code": "002352", "scene_date": DATE,
        "sentinel2_scene_id": meta["scene_id"], "detector_version": meta["detector_version"],
        "inputs": {name: {"file": path.relative_to(run).as_posix(), "sha256": _sha256(path)}
                   for name, path in files.items()},
        "planet_provenance": {**provenance, "mosaic": files["planet_mosaic"].relative_to(run).as_posix()},
        "citation": CITATION,
        "grids": {"crs": s2_crs.to_string(), "sentinel2_transform": list(s2_transform)[:6],
                  "planet_transform": list(ps_transform)[:6], "sentinel2_shape": list(s2_mask.shape),
                  "planet_shape": list(analysis.shape), "planet_band_indices_1_based": bands,
                  "sentinel2_pixel_area_m2": s2_area, "planet_pixel_area_m2": ps_area},
        "analysis": {"definition": "nearest-reprojected Sentinel-2 inside intersected with Planet valid data",
                     "sentinel2_inside_area_ha": float(s2_inside.sum() * s2_area / 10000),
                     "sentinel2_valid_inside_area_ha": float(s2_analysis.sum() * s2_area / 10000),
                     "planet_reprojected_inside_area_ha": float(ps_inside.sum() * ps_area / 10000),
                     "planet_valid_inside_area_ha": float(analysis.sum() * ps_area / 10000),
                     "planet_invalid_inside_pixels": int((ps_inside & ~ps_valid).sum()),
                     "common_10m_cells": int(common.sum()),
                     "common_10m_area_ha": float(common.sum() * s2_area / 10000),
                     "sentinel2_valid_cells_excluded_by_planet_coverage": int((s2_analysis & ~supported).sum())},
        "calibration": {**fit, "equation": "NDVI_S2 = a * NDVI_PS10 + b",
                        "estimator": "ordinary least squares with intercept; each paired cell has equal weight",
                        "planet_10m_indices": "NDVI and NDWI of exact area-mean reflectance bands over assessed support",
                        "domain": "all Sentinel-2 valid & inside cells with Planet assessed coverage > 0; finite paired indices; NDWI <= 0 on both sensors",
                        "candidate_masks_used": False, "paired_site_cells_before_exclusion": int(paired.sum()),
                        "water_cells_excluded": int((paired & ((ps_10_ndwi > 0) | (s2_ndwi > 0))).sum()),
                        "nonfinite_cells_excluded": int((paired & ~(
                            np.isfinite(ps_10_ndvi) & np.isfinite(s2_ndvi)
                            & np.isfinite(ps_10_ndwi) & np.isfinite(s2_ndwi))).sum()),
                        "application": "NDVI_cal = a * native PlanetScope NDVI + b; no clipping; NDWI unchanged",
                        "rmse_definition": "sqrt(mean((NDVI_S2 - fitted NDVI_S2)^2)); in-sample NDVI units",
                        "comparison_coverage_sensitivity": {"minimum_assessed_coverage": MIN_COVERAGE, **sensitivity}},
        "method": {"primary_mask": "planet_rules", "variant": "calibrated",
                   "ndvi_scale": "Sentinel-2 scale from all-paired-cell OLS", "threshold": threshold,
                   "site_ndvi_p75": float(np.quantile(calibrated_ndvi[analysis], 0.75)),
                   "threshold_formula": "max(0.10, min(0.25, site NDVI p75 - 0.18))",
                   "water_exclusion": "(green - nir) / (green + nir) > 0",
                   "swir_available": False, "burn_guard_available": False,
                   "swir_shadow_guard_available": False, "cloud_shadow_quality_mask_used": False,
                   "gng": gng_info},
        **metrics,
        "naive_transfer": {"description": "uncalibrated PlanetScope NDVI with the unchanged threshold formula",
                           "method": {"threshold": naive_threshold,
                                      "site_ndvi_p75": float(np.quantile(ndvi[analysis], 0.75)),
                                      "ndvi_scale": "native PlanetScope", "gng": naive_gng_info},
                           **naive_metrics},
        "spectral_diagnostics": diagnostics,
        "zoom": {"selection": "maximum frozen Sentinel-2 candidate count in a full 400 m square; first row/column breaks ties",
                 "bounds_west_south_east_north": list(zoom), "sentinel2_bare_cells": zoom_count},
        "limitations": LIMITATIONS,
        "artifacts": ["summary.json", "masks.npz", "whole_site.png", "cutting_margin_zoom.png",
                      "ndvi_calibration.png"],
    }
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "masks.npz", **arrays,
                        **{f"naive_{name}": values for name, values in naive_arrays.items()},
                        planet_analysis=analysis, sentinel2_analysis=s2_analysis,
                        sentinel2_gng=s2_mask, comparison_support=common,
                        calibration_support=calibration_support,
                        planet_ndvi=ndvi, planet_calibrated_ndvi=calibrated_ndvi,
                        planet_ndvi_10m=ps_10_ndvi, planet_ndwi_10m=ps_10_ndwi,
                        sentinel2_ndvi=s2_ndvi, sentinel2_ndwi=s2_ndwi,
                        planet_coverage_10m=coverage,
                        planet_transform=np.array(ps_transform), sentinel2_transform=np.array(s2_transform),
                        crs=np.array(s2_crs.to_string()))
    write_figures(out, (s2_stack, s2_valid, s2_analysis, s2_mask, s2_transform),
                  (ps_stack, ps_valid, analysis, rules, ps_transform), bounds, zoom, summary,
                  (ps_10_ndvi[calibration_support], s2_ndvi[calibration_support]))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"calibration": fit, "threshold": threshold,
                      "candidate_area_ha": summary["candidate_area_ha"],
                      "iou": metrics["aggregation"]["iou"],
                      "only_3m_cells": metrics["aggregation"]["only_3m_cells"],
                      "only_10m_cells": metrics["aggregation"]["only_10m_cells"],
                      "patches_3m": summary["patches"]["planet_rules_3m"]["count"],
                      "patches_10m": summary["patches"]["sentinel2_gng_10m"]["count"],
                      "narrow_share": metrics["narrow_features"]["planet_rules"]["share_of_bare_area"],
                      "gng_changed_pixels": metrics["gng_ablation"]["changed_pixels"]}, indent=2))
    for filename in summary["artifacts"]:
        print(f"Saved {(out / filename).relative_to(ROOT)}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN, help="frozen evaluation run")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="case-study output directory")
    args = ap.parse_args()
    if args.run.is_absolute() or args.out.is_absolute():
        ap.error("--run and --out must be repository-relative paths")
    run = ROOT / args.run
    out = ROOT / args.out
    if not out.resolve().is_relative_to(ROOT) or out.resolve() == ROOT:
        ap.error("--out must be a directory inside the repository")
    if out.resolve() == run.resolve() or run.resolve() in out.resolve().parents:
        ap.error("--out must be separate from the read-only input run")
    run_case(run, out)


if __name__ == "__main__":
    main()
