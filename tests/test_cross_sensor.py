"""Synthetic checks for the shared cross-sensor comparison functions."""

import numpy as np
import pytest
from rasterio import Affine

from peatland.cross_sensor import compare_masks, summarise

PLANET = Affine(5, 0, 0, 0, -5, 20)     # 4 x 4 pixels of 5 m
SENTINEL2 = Affine(10, 0, 0, 0, -10, 20)  # 2 x 2 cells of 10 m


def planet_mask(cells):
    mask = np.zeros((4, 4), dtype=bool)
    for row, col in cells:
        mask[2 * row:2 * row + 2, 2 * col:2 * col + 2] = True
    return mask


def test_compare_masks_areas_overlap_and_ratio():
    planet = planet_mask([(0, 0)])                 # fills exactly one 10 m cell
    sentinel2 = np.array([[True, True], [False, False]])
    result, aggregated = compare_masks(planet, np.ones((4, 4), bool), PLANET,
                                       sentinel2, SENTINEL2, np.ones((2, 2), bool))
    np.testing.assert_array_equal(aggregated, [[True, False], [False, False]])
    assert result["planet_area_ha"] == pytest.approx(0.01)
    assert result["planet_area_common_support_ha"] == pytest.approx(0.01)
    assert result["sentinel2_area_common_support_ha"] == pytest.approx(0.02)
    assert result["area_ratio_common_support"] == pytest.approx(0.5)
    assert result["shared_cells"] == 1
    assert result["only_3m_cells"] == 0
    assert result["only_10m_cells"] == 1
    assert result["iou"] == pytest.approx(0.5)


def test_compare_masks_counts_cells_found_only_at_3m():
    planet = planet_mask([(0, 0), (1, 1)])
    sentinel2 = np.array([[True, False], [False, False]])
    result, _ = compare_masks(planet, np.ones((4, 4), bool), PLANET,
                              sentinel2, SENTINEL2, np.ones((2, 2), bool))
    assert result["only_3m_cells"] == 1
    assert result["only_10m_cells"] == 0
    assert result["iou"] == pytest.approx(0.5)
    assert result["area_ratio_common_support"] == pytest.approx(2)


def test_compare_masks_empty_sentinel2_keeps_planet_area_and_leaves_ratio_undefined():
    planet = planet_mask([(0, 0)])
    empty = np.zeros((2, 2), dtype=bool)
    result, _ = compare_masks(planet, np.ones((4, 4), bool), PLANET,
                              empty, SENTINEL2, np.ones((2, 2), bool))
    assert result["planet_area_ha"] == pytest.approx(0.01)
    assert result["sentinel2_area_common_support_ha"] == 0
    assert result["area_ratio_common_support"] is None
    assert result["iou"] == 0            # a real disagreement, not an undefined one
    assert result["only_3m_cells"] == 1


def test_compare_masks_both_empty_is_undefined_not_perfect():
    empty_planet, empty_s2 = np.zeros((4, 4), bool), np.zeros((2, 2), bool)
    result, _ = compare_masks(empty_planet, np.ones((4, 4), bool), PLANET,
                              empty_s2, SENTINEL2, np.ones((2, 2), bool))
    assert result["iou"] is None
    assert result["narrow_share"] is None
    assert result["planet_area_ha"] == 0


def test_compare_masks_narrow_share_removes_features_below_10m():
    narrow = np.zeros((4, 4), dtype=bool)
    narrow[0, :] = True                  # a 5 m wide strip cannot hold a 10 m disk
    result, _ = compare_masks(narrow, np.ones((4, 4), bool), PLANET,
                              np.zeros((2, 2), bool), SENTINEL2, np.ones((2, 2), bool))
    assert result["narrow_share"] == pytest.approx(1.0)
    assert result["narrow_area_ha"] == pytest.approx(4 * 25 / 10000)


def test_compare_masks_respects_the_common_support():
    planet = planet_mask([(0, 0), (1, 1)])
    sentinel2 = np.array([[True, False], [False, True]])
    support = np.array([[True, False], [False, False]])
    result, _ = compare_masks(planet, np.ones((4, 4), bool), PLANET,
                              sentinel2, SENTINEL2, support)
    assert result["shared_cells"] == 1
    assert result["iou"] == pytest.approx(1.0)
    assert result["sentinel2_area_common_support_ha"] == pytest.approx(0.01)


def test_summarise_reports_median_quartiles_and_range():
    assert summarise([1, 2, 3, 4, 5]) == {"n": 5, "median": 3, "p25": 2, "p75": 4,
                                          "min": 1, "max": 5}


def test_summarise_ignores_undefined_values():
    assert summarise([None, 2, np.nan, 4, np.inf]) == {"n": 2, "median": 3, "p25": 2.5,
                                                       "p75": 3.5, "min": 2, "max": 4}
    assert summarise([None, np.nan])["n"] == 0
    assert summarise([])["median"] is None
