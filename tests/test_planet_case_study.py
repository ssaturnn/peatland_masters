"""Synthetic checks for the offline PlanetScope case-study calculations."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from rasterio import Affine

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/21_planet_case_study.py"
SPEC = importlib.util.spec_from_file_location("planet_case_study", SCRIPT)
case = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(case)


@pytest.mark.parametrize("ndvi, expected", [
    ([0.1, 0.2, 0.3, 0.4], 0.145),
    ([0.6, 0.7, 0.8], 0.25),
    ([-0.1, 0.0, 0.1], 0.10),
])
def test_adaptive_threshold_percentile_cap_and_floor(ndvi, expected):
    assert case.adaptive_threshold(ndvi) == pytest.approx(expected)


def test_adaptive_threshold_uses_only_finite_assessed_values():
    ndvi = np.array([[0.3, 0.4, 100.0], [np.nan, np.inf, -100.0]])
    analysis = np.array([[True, True, False], [True, True, False]])
    assert case.adaptive_threshold(ndvi, analysis) == pytest.approx(0.195)


def test_adaptive_threshold_empty_site_is_explicit():
    with pytest.raises(ValueError, match="No finite site"):
        case.adaptive_threshold([np.nan])


def test_candidate_threshold_and_water_boundaries():
    ndvi = np.array([[0.249, 0.25, 0.1, 0.1, np.nan, 0.1]])
    ndwi = np.array([[0.0, -0.1, 0.01, -0.1, -0.1, np.nan]])
    analysis = np.array([[True, True, True, False, True, True]])
    actual = case.candidate_mask(ndvi, ndwi, analysis, 0.25)
    np.testing.assert_array_equal(actual, [[True, False, False, False, False, False]])


def test_calibration_recovers_known_synthetic_line_over_entire_site():
    x = np.linspace(-0.1, 0.9, 30).reshape(5, 6)
    y = 1.18 * x - 0.12
    water = np.zeros_like(x)
    support = np.ones_like(x, dtype=bool)
    original = [values.copy() for values in (x, y, water, support)]
    fit, selected = case.fit_ndvi_calibration(x, y, water, water, support)
    assert fit == pytest.approx({"a": 1.18, "b": -0.12, "r": 1, "rmse": 0, "n": x.size})
    assert selected.all()
    for values, expected in zip((x, y, water, support), original):
        np.testing.assert_array_equal(values, expected)


def test_calibration_excludes_either_sensor_water_nonfinite_and_unsupported_cells():
    x = np.linspace(0.1, 0.9, 9)
    y = 1.3 * x - 0.15
    xw, yw = np.zeros(9), np.zeros(9)
    support = np.ones(9, dtype=bool)
    xw[0], yw[1] = 0.01, 0.01
    y[:2] = 100  # Water must not influence the fitted line.
    x[2], y[3], xw[4], yw[5] = np.nan, np.inf, np.nan, np.inf
    support[6] = False
    y[6] = -100
    fit, selected = case.fit_ndvi_calibration(x, y, xw, yw, support)
    np.testing.assert_array_equal(selected, [False] * 7 + [True, True])
    assert fit == pytest.approx({"a": 1.3, "b": -0.15, "r": 1, "rmse": 0, "n": 2})
    assert support.sum() == 8


def test_calibration_reports_ols_rmse_and_pearson_r_with_residuals():
    x = np.arange(5) * 0.2
    residual = np.array([1, -2, 2, -2, 1]) * 0.01  # Orthogonal to x and the intercept.
    y = 1.2 * x - 0.12 + residual
    fit, _ = case.fit_ndvi_calibration(x, y, np.zeros(5), np.zeros(5), np.ones(5, bool))
    assert fit["a"] == pytest.approx(1.2)
    assert fit["b"] == pytest.approx(-0.12)
    assert fit["rmse"] == pytest.approx(np.sqrt(14 / 5) * 0.01)
    assert fit["r"] == pytest.approx(np.sqrt(1.2 ** 2 * 0.08 / (1.2 ** 2 * 0.08 + 0.00028)))


@pytest.mark.parametrize("x, support, message", [
    ([0.1, 0.2], [False, False], "at least two"),
    ([0.1, 0.2], [True, False], "at least two"),
    ([0.2, 0.2], [True, True], "varying PlanetScope"),
])
def test_calibration_degenerate_samples_are_explicit(x, support, message):
    with pytest.raises(ValueError, match=message):
        case.fit_ndvi_calibration(x, x, [0, 0], [0, 0], support)


def test_calibration_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="matching shapes"):
        case.fit_ndvi_calibration([0.1, 0.2], [0.1], [0, 0], [0, 0], [True, True])


def test_calibration_constant_response_has_undefined_correlation():
    fit, _ = case.fit_ndvi_calibration([0.1, 0.2], [0.3, 0.3], [0, 0], [0, 0], [True, True])
    assert fit["a"] == pytest.approx(0)
    assert fit["b"] == pytest.approx(0.3)
    assert fit["rmse"] == pytest.approx(0)
    assert fit["r"] is None


def test_10m_reflectance_averaging_precedes_ndvi_and_weights_partial_pixels():
    reflectance = np.ones((4, 4, 4)) * 0.1
    reflectance[:, :2, 3] = 0.3
    mean, coverage = case.area_mean_reflectance(
        reflectance, np.ones((4, 4), bool), Affine(3, 0, 0, 0, -3, 12),
        (1, 1), Affine(10, 0, 0, 0, -10, 12))
    assert coverage[0, 0] == pytest.approx(1)
    np.testing.assert_allclose(mean[0, 0], [0.1, 0.1, 0.1, 0.22])
    ndvi, _ = case.indices(mean)
    # Mean of native NDVI would be 0.6 * 0.5 = 0.3, not 0.375.
    assert ndvi[0, 0] == pytest.approx(0.375)


def test_reflectance_averaging_excludes_invalid_bands_and_preserves_empty_cells():
    reflectance = np.ones((2, 2, 4)) * 0.2
    reflectance[0, 0, 0] = np.nan
    analysis = np.array([[True, True], [False, False]])
    mean, coverage = case.area_mean_reflectance(
        reflectance, analysis, Affine(5, 0, 0, 0, -5, 10),
        (1, 2), Affine(10, 0, 0, 0, -10, 10))
    np.testing.assert_allclose(mean[0, 0], 0.2)
    assert np.isnan(mean[0, 1]).all()
    np.testing.assert_allclose(coverage, [[0.25, 0]])


def test_gng_uses_calibrated_pixel_ndvi_and_retains_native_water_exclusion():
    # Native NDVI = 0.3; calibration moves it below 0.25.
    reflectance = np.array([[[0.1, 0.1, 0.14, 0.26]]])
    analysis = np.ones((1, 1), bool)
    native_ndvi, _ = case.indices(reflectance)
    calibrated_ndvi = 1.18 * native_ndvi - 0.12
    naive, _ = case.gng_candidates(reflectance, analysis, 0.25)
    calibrated, _ = case.gng_candidates(reflectance, analysis, 0.25, ndvi=calibrated_ndvi)
    assert not naive.any() and calibrated.all()
    reflectance[:, :, 1] = 0.3  # Positive NDWI remains water after NDVI calibration.
    water, _ = case.gng_candidates(reflectance, analysis, 0.25, ndvi=calibrated_ndvi)
    assert not water.any()


def test_10m_aggregation_counts_exactly_half_as_bare():
    mask = np.zeros((10, 20), dtype=bool)
    mask[:5, :10] = True
    mask[:4, 10:] = True
    fraction, coverage = case.aggregate_to_grid(
        mask, np.ones_like(mask), Affine(1, 0, 0, 0, -1, 10),
        (1, 2), Affine(10, 0, 0, 0, -10, 10))
    np.testing.assert_allclose(fraction, [[0.5, 0.4]])
    np.testing.assert_allclose(coverage, 1)
    labels, support = case.aggregate_labels(fraction, coverage)
    np.testing.assert_array_equal(labels, [[True, False]])
    assert support.all()


def test_3m_to_10m_uses_fractional_edge_pixels():
    mask = np.zeros((4, 4), dtype=bool)
    mask[:, :2] = True
    fraction, coverage = case.aggregate_to_grid(
        mask, np.ones_like(mask), Affine(3, 0, 0, 0, -3, 12),
        (1, 1), Affine(10, 0, 0, 0, -10, 12))
    # The first two columns occupy 6 m of the 10 m cell, not 2/3 or 2/4.
    assert fraction[0, 0] == pytest.approx(0.6)
    assert coverage[0, 0] == pytest.approx(1.0)


def test_shifted_grid_and_raster_edge_coverage():
    mask = np.zeros((4, 4), dtype=bool)
    mask[:, 0] = True
    fraction, coverage = case.aggregate_to_grid(
        mask, np.ones_like(mask), Affine(3, 0, 1, 0, -3, 12),
        (1, 1), Affine(10, 0, 0, 0, -10, 12))
    # Raster begins at x=1: 90 m2 observed, 30 m2 bare, 10 m2 unsupported.
    assert fraction[0, 0] == pytest.approx(1 / 3)
    assert coverage[0, 0] == pytest.approx(0.9)
    _, support = case.aggregate_labels(fraction, coverage)
    assert not support.any()


def test_nodata_is_excluded_from_fraction_denominator():
    mask = np.array([[True, True], [False, False]])
    analysis = np.array([[True, True], [False, False]])
    fraction, coverage = case.aggregate_to_grid(
        mask, analysis, Affine(5, 0, 0, 0, -5, 10),
        (1, 1), Affine(10, 0, 0, 0, -10, 10))
    assert fraction[0, 0] == pytest.approx(1)
    assert coverage[0, 0] == pytest.approx(0.5)
    labels, support = case.aggregate_labels(fraction, coverage)
    assert not labels.any() and not support.any()


def test_no_support_is_nan_and_unassessed_not_nonbare():
    mask = np.ones((2, 2), dtype=bool)
    fraction, coverage = case.aggregate_to_grid(
        mask, ~mask, Affine(5, 0, 0, 0, -5, 10),
        (1, 2), Affine(10, 0, 0, 0, -10, 10))
    assert np.isnan(fraction).all()
    assert not coverage.any()
    labels, support = case.aggregate_labels(fraction, coverage)
    assert not labels.any() and not support.any()


def test_aggregation_conserves_bare_area_across_offsets():
    mask = np.array([[True, False, True], [True, True, False]])
    fraction, coverage = case.aggregate_to_grid(
        mask, np.ones_like(mask), Affine(3, 0, 1.25, 0, -3, 8.75),
        (2, 2), Affine(10, 0, 0, 0, -10, 20))
    assert np.nansum(fraction * coverage * 100) == pytest.approx(mask.sum() * 9)
    assert (coverage * 100).sum() == pytest.approx(mask.size * 9)


def test_coverage_boundary_is_inclusive():
    labels, support = case.aggregate_labels(np.array([0.5, 0.8]), np.array([0.95, 0.949]))
    np.testing.assert_array_equal(labels, [True, False])
    np.testing.assert_array_equal(support, [True, False])


def test_patch_statistics_diagonals_are_connected():
    result = case.patch_statistics(np.eye(3, dtype=bool), 9)
    assert result["count"] == 1
    assert result["sizes_ha"] == pytest.approx([0.0027])
    assert result["area_ha"] == pytest.approx(0.0027)


def test_patch_statistics_sizes_quantiles_and_histogram():
    mask = np.array([[True, True, False, True], [False, False, False, False]])
    result = case.patch_statistics(mask, 100)
    assert result["count"] == 2
    assert result["sizes_ha"] == pytest.approx([0.01, 0.02])
    assert result["area_ha"] == pytest.approx(0.03)
    assert result["median_ha"] == pytest.approx(0.015)
    assert result["p90_ha"] == pytest.approx(0.019)
    assert result["histogram_counts"] == [0, 2, 0, 0, 0, 0]


def test_patch_statistics_empty_mask():
    result = case.patch_statistics(np.zeros((2, 2), dtype=bool), 9)
    assert result["count"] == 0 and result["area_ha"] == 0
    assert result["sizes_ha"] == []
    assert result["median_ha"] is None and result["max_ha"] is None
    assert sum(result["histogram_counts"]) == 0


def test_narrow_opening_removes_thin_strip_keeps_square():
    mask = np.zeros((12, 14), dtype=bool)
    mask[1:6, 1:6] = True
    mask[9, 1:11] = True
    removed, result = case.narrow_features(mask, Affine(3, 0, 0, 0, -3, 36))
    assert removed[9, 1:11].all()
    assert not removed[1:6, 1:6].any()
    assert result["removed_area_ha"] == pytest.approx(0.009)
    assert result["share_of_bare_area"] == pytest.approx(10 / 35)


def test_overlap_uses_common_support_and_empty_union_is_undefined():
    first = np.array([[True, False, True]])
    second = np.array([[True, True, False]])
    support = np.array([[True, True, False]])
    result = case.overlap_statistics(first, second, support)
    assert result["iou"] == pytest.approx(0.5)
    assert result["only_first_cells"] == 0 and result["only_second_cells"] == 1
    empty = np.zeros((2, 2), dtype=bool)
    assert case.overlap_statistics(empty, empty, ~empty)["iou"] is None
