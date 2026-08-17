import numpy as np
from rasterio.transform import Affine

from peatland import geo


def transform_10m():
    # 10 m pixels, north-up: a=10, e=-10
    return Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 700000.0)


def test_pixel_area_is_one_hundredth_ha():
    # a 10 m x 10 m pixel = 100 m^2 = 0.01 ha
    assert abs(geo.pixel_area_ha(transform_10m()) - 0.01) < 1e-12


def test_mask_area_counts_true_cells():
    t = transform_10m()
    mask = np.zeros((10, 10), dtype=bool)
    mask[:5, :2] = True  # 10 cells
    assert abs(geo.mask_area_ha(mask, t) - 0.10) < 1e-9  # 10 * 0.01


def test_empty_mask_is_zero():
    assert geo.mask_area_ha(np.zeros((4, 4), dtype=bool), transform_10m()) == 0.0
