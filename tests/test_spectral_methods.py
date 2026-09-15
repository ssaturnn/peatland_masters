import numpy as np
import pytest

from peatland import pipeline, gng


def scene():
    # Vegetation, bare candidate, water, burn: six bands in pipeline order.
    spectra = np.array([[600, 800, 1000, 4000, 1800, 1400],
                        [600, 800, 1600, 2000, 1800, 1400],
                        [300, 500, 400, 450, 100, 80],
                        [600, 800, 1600, 1800, 2000, 2400]], dtype=float)
    stack = np.tile(spectra[0], (20, 20, 1))
    for i in range(1, 4):
        stack[i, :5] = spectra[i]
    return stack, np.ones((20, 20), dtype=bool)


@pytest.mark.parametrize("method", ["gng", "kmeans", "rules"])
def test_valid_aoi_prediction_ignores_invalid_surroundings(method):
    stack, valid = scene()
    original = pipeline.spectral_bare(stack, pipeline.BANDS, valid, valid, method=method, max_nodes=8)
    padded = np.full((22, 22, 6), np.nan)
    padded[1:-1, 1:-1] = stack
    inside = np.zeros((22, 22), dtype=bool); inside[1:-1, 1:-1] = True
    # Even finite extreme values outside the AOI must not affect the fit.
    padded[0] = 1e9
    actual = pipeline.spectral_bare(padded, pipeline.BANDS, np.ones_like(inside), inside,
                                    method=method, max_nodes=8)
    np.testing.assert_array_equal(original, actual[1:-1, 1:-1])
    assert not actual[~inside].any()


def test_rules_ablation_rejects_water_and_burn_keeps_bare():
    stack, valid = scene()
    result = pipeline.rules_bare(stack, pipeline.BANDS, valid, valid)
    assert result[1, :5].all()
    assert not result[2:4].any()


@pytest.mark.parametrize("method", ["gng", "kmeans", "rules"])
def test_no_valid_pixels_yield_empty_mask(method):
    stack, valid = scene()
    result = pipeline.spectral_bare(stack, pipeline.BANDS, ~valid, valid, method=method)
    assert not result.any()


def test_requested_threshold_below_floor_is_respected():
    stack, valid = scene()
    stack[:] = [600, 800, 1800, 2000, 1800, 1400]  # NDVI about .053
    assert not pipeline.rules_bare(stack, pipeline.BANDS, valid, valid, thresh=.04).any()


def test_batched_nearest_node_matches_direct_distances():
    rng = np.random.default_rng(3)
    net = gng.GrowingNeuralGas()
    net.weights = rng.normal(size=(12, 9))
    data = rng.normal(size=(5000, 9))
    expected = np.argmin(((data[:, None, :] - net.weights[None, :, :]) ** 2).sum(axis=2), axis=1)
    np.testing.assert_array_equal(net.predict(data), expected)
