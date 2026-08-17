import numpy as np

from peatland import change, preprocess, boundaries, pipeline


def test_change_compare_classes():
    # 2x2: pixel states A(bare?), B(bare?)
    bare_a = np.array([[True, False], [True, False]])
    bare_b = np.array([[True, True], [False, False]])
    both = np.ones((2, 2), dtype=bool)
    m = change.compare(bare_a, bare_b, both)
    assert m["newly_bare"].tolist() == [[False, True], [False, False]]
    assert m["revegetated"].tolist() == [[False, False], [True, False]]
    assert m["stable_bare"].tolist() == [[True, False], [False, False]]


def test_change_respects_both_valid():
    bare_a = np.array([[False, False]])
    bare_b = np.array([[True, True]])
    both = np.array([[True, False]])  # second pixel invalid at one date
    m = change.compare(bare_a, bare_b, both)
    # only the valid pixel can be counted as newly bare
    assert m["newly_bare"].tolist() == [[True, False]]


def test_valid_mask_excludes_cloud_classes():
    scl = np.array([[4, 5, 8], [9, 3, 6]])  # 4,5,6 clear; 8,9 cloud; 3 shadow
    v = preprocess.valid_mask(scl)
    assert v.tolist() == [[True, True, False], [False, False, True]]


def test_clear_fraction_over_aoi():
    scl = np.array([[4, 8], [4, 4]])            # 3 clear of 4
    aoi = np.array([[True, True], [True, False]])  # aoi = 3 cells, 2 clear
    assert abs(preprocess.clear_fraction(scl, aoi) - 2 / 3) < 1e-9


def test_designation_label():
    assert boundaries.designation_label(False, False) == "NHA"
    assert boundaries.designation_label(True, False) == "NHA + SAC"
    assert boundaries.designation_label(True, True) == "NHA + SAC + SPA"


def test_rate_slope():
    # bare area rising 2 ha/yr
    assert abs(pipeline._rate_ha_per_yr([2018, 2020], [1.0, 5.0]) - 2.0) < 1e-9
    assert pipeline._rate_ha_per_yr([2020], [3.0]) == 0.0  # single point


def test_brightness_guard_rejects_cloud():
    import numpy as np
    order = pipeline.BANDS
    # two pixels, both low-NDVI (bare-like): one dark peat, one bright cloud
    #   bands: blue green red nir swir1 swir2  (reflectance*10000)
    dark = [800, 900, 1000, 1100, 1200, 1000]     # brightness ~0.09
    cloud = [3500, 3600, 3700, 3800, 3000, 2500]  # brightness ~0.36
    stack = np.array([[dark, cloud]], dtype="float32")     # shape (1,2,6)
    ndvi = np.array([[0.1, 0.1]])                          # both low NDVI
    valid = np.ones((1, 2), dtype=bool)
    inside = np.ones((1, 2), dtype=bool)
    mask = pipeline.ndvi_bare(stack, order, ndvi, valid, inside)
    assert mask.tolist() == [[True, False]]  # cloud pixel rejected as too bright
