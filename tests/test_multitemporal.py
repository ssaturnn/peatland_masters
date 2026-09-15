"""Synthetic persistence, support and shoreline checks."""

import numpy as np
import pytest
from peatland.multitemporal import (Parameters, trajectory_features, detect_trajectory,
                                    tidal_buffer, mask_comparison, early_candidates,
                                    candidate_cover, buffered_inside)


def spectrum(ndvi):
    nir = .18
    red = nir * (1 - ndvi) / (1 + ndvi)
    return np.array([.04, .05, red, nir, .15, .12]) * 10000


def stack_of(ndvi):
    """Peat-like six-band stack whose NDVI follows the given array."""
    ndvi = np.asarray(ndvi, float)
    const = lambda v: np.full(ndvi.shape, v)
    nir = const(.18)
    red = nir * (1 - ndvi) / (1 + ndvi)
    return np.stack([const(.04), const(.05), red, nir, const(.15), const(.12)], axis=-1) * 10000


def test_early_candidates_use_only_clear_bog_pixels():
    ndvi = np.full((4, 4), .6); ndvi[0, :2] = .1; ndvi[3, 0] = .1
    valid = np.ones((4, 4), bool); valid[0, 1] = False
    inside = np.ones((4, 4), bool); inside[3] = False
    mask, threshold = early_candidates(stack_of(ndvi), valid, inside)
    assert threshold == Parameters().ndvi_cap
    assert mask[0, 0] and mask.sum() == 1


def test_candidate_cover_counts_seen_candidates():
    cand = np.zeros((10, 10), bool); cand[:5] = True
    valid = np.ones((10, 10), bool); valid[:2] = False
    assert candidate_cover(cand, valid) == {'candidate_pixels': 50, 'seen_pixels': 30, 'cover': .6}
    assert candidate_cover(np.zeros((10, 10), bool), valid)['cover'] is None


def test_late_clouds_do_not_move_the_early_threshold():
    rng = np.random.default_rng(0)
    early = stack_of(rng.uniform(.05, .7, (20, 20)))
    late = stack_of(rng.uniform(.05, .7, (20, 20)))
    valid = np.ones((20, 20), bool)
    half = valid.copy(); half[:10] = False
    p = Parameters(training_steps=100, max_nodes=4)
    full = detect_trajectory(early, late, valid, valid, valid, p)['diagnostics']
    clouded = detect_trajectory(early, late, valid, half, valid, p)['diagnostics']
    assert clouded['early_threshold'] == full['early_threshold']
    assert full['early_threshold'] == early_candidates(early, valid, valid, p)[1]
    assert clouded['early_candidates_seen_late'] < clouded['early_candidate_pixels']


def test_buffered_inside_removes_only_the_grown_ring():
    water = np.zeros((7, 7), bool); water[3, 3] = True
    inside = ~water
    assert buffered_inside(inside, water, 0).sum() == 48
    assert buffered_inside(inside, water, 1).sum() == 40
    assert buffered_inside(inside, water, 2).sum() == 24


def test_features_have_physical_changes():
    a = np.tile(spectrum(.1), (2, 2, 1))
    b = np.tile(spectrum(.6), (2, 2, 1))
    f = trajectory_features(a, b)
    assert f.shape == (2, 2, 12)
    assert np.allclose(f[..., 0], .1)
    assert np.allclose(f[..., 8], .5)
    assert np.allclose(f[..., 9:], 0)


def test_persistent_rule_rejects_greenup_and_invalid_support():
    a = np.tile(spectrum(.1), (3, 3, 1))
    b = a.copy(); b[0, 0] = spectrum(.6)
    valid = np.ones((3, 3), bool); valid[1, 1] = False
    r = detect_trajectory(a, b, valid, valid, np.ones_like(valid),
                          Parameters(training_steps=100, max_nodes=4))
    assert r['rules'].sum() == 7
    assert not r['support'][1, 1]
    assert not r['rules'][0, 0]
    assert not np.any(r['gng'] & ~r['rules'])


def test_invalid_surroundings_do_not_change_training():
    a = np.tile(spectrum(.1), (3, 3, 1))
    valid = np.ones((3, 3), bool); valid[0, 0] = False
    p = Parameters(training_steps=100, max_nodes=4)
    first = detect_trajectory(a, a, valid, valid, valid, p)
    changed = a.copy(); changed[0, 0] = np.nan
    second = detect_trajectory(changed, changed, valid, valid, valid, p)
    assert np.array_equal(first['gng'], second['gng'])
    assert first['diagnostics'] == second['diagnostics']


def test_empty_support_is_explicit_and_not_agreement_one():
    a = np.tile(spectrum(.1), (2, 2, 1)); empty = np.zeros((2, 2), bool)
    result = detect_trajectory(a, a, empty, empty, ~empty)
    assert result['diagnostics']['support_pixels'] == 0
    assert mask_comparison(empty, empty, empty)['agreement'] is None


def test_tidal_buffer_has_finite_extent_and_zero_is_noop():
    water = np.zeros((7, 7), bool); water[3, 3] = True
    assert tidal_buffer(water, 0).sum() == 1
    assert tidal_buffer(water, 1).sum() == 9
    assert tidal_buffer(water, 2).sum() == 25
    with pytest.raises(ValueError): tidal_buffer(water, -1)


def test_area_agreement_and_support():
    a = np.array([[1, 1], [0, 0]], bool)
    b = np.array([[0, 1], [1, 1]], bool)
    support = np.array([[1, 1], [1, 0]], bool)
    result = mask_comparison(a, b, support)
    assert result['iou'] == pytest.approx(1/3)
    assert result['removed_ha'] == .01
    assert result['added_ha'] == .01
