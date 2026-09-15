from types import SimpleNamespace

import numpy as np
import pytest

from peatland.imagery import harmonize_band


def item(baseline="04.00", meta=None, **properties):
    return SimpleNamespace(id="test-scene", properties={"s2:processing_baseline": baseline, **properties},
                           assets={"B04": SimpleNamespace(extra_fields=meta or {})})


def test_new_and_old_baseline_have_same_reflectance():
    old = harmonize_band(np.array([1000, 2000], dtype="uint16"), item("03.00"), "red", "planetary")
    new = harmonize_band(np.array([2000, 3000], dtype="uint16"), item("05.10"), "red", "planetary")
    np.testing.assert_array_equal(old, new)


def test_reprocessed_old_acquisition_uses_baseline_not_year():
    it = item("05.10", datetime="2018-05-15T12:00:00Z")
    assert harmonize_band(np.array([2000]), it, "red", "planetary")[0] == 1000


def test_nodata_is_nan_and_negative_reflectance_clamped():
    result = harmonize_band(np.array([0, 500, 1000, 1500], dtype="uint16"), item(), "red", "planetary")
    assert np.isnan(result[0])
    np.testing.assert_array_equal(result[1:], [0, 0, 500])


def test_categorical_scl_not_shifted():
    scl = np.array([0, 3, 4, 8], dtype="uint8")
    np.testing.assert_array_equal(harmonize_band(scl, item(), "scl", "planetary"), scl)


def test_explicit_scale_offset_prevents_double_correction():
    it = item(meta={"raster:bands": [{"scale": 0.0001, "offset": 0}]})
    assert harmonize_band(np.array([2000]), it, "red", "earthsearch")[0] == pytest.approx(2000)


def test_earthsearch_harmonized_flag_prevents_double_offset():
    it = item(**{"earthsearch:boa_offset_applied": True})
    assert harmonize_band(np.array([2000]), it, "red", "earthsearch")[0] == 2000


def test_missing_baseline_cannot_silently_mix_radiometry():
    with pytest.raises(ValueError, match="Missing processing baseline"):
        harmonize_band(np.array([2000]), item(None), "red", "planetary")
