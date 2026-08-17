import numpy as np

from peatland import detect


def test_ndvi_known_values():
    red = np.array([[100.0, 0.0]])
    nir = np.array([[300.0, 0.0]])
    out = detect.ndvi(red, nir)
    # (300-100)/(300+100) = 0.5 ; divide-by-zero guarded to 0
    assert abs(out[0, 0] - 0.5) < 1e-6
    assert out[0, 1] == 0.0


def test_ndvi_range():
    rng = np.random.default_rng(0)
    red = rng.uniform(1, 5000, (50, 50))
    nir = rng.uniform(1, 5000, (50, 50))
    out = detect.ndvi(red, nir)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_exposed_peat_mask_threshold():
    ndvi = np.array([[0.1, 0.25, 0.4]])
    m = detect.exposed_peat_mask(ndvi, threshold=0.25)
    assert m.tolist() == [[True, False, False]]  # strict <
