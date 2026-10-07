"""Detector v4 post-filters."""

import numpy as np

from peatland import v4


def stack_with(green, swir1, shape=(5, 5)):
    s = np.zeros(shape + (6,))
    s[..., 1], s[..., 4] = green, swir1
    return s


def test_mndwi_separates_dry_peat_from_wet_sediment():
    assert v4.mndwi(stack_with(0.124, 0.373))[0, 0] < -0.45      # dry peat
    assert v4.mndwi(stack_with(0.139, 0.226))[0, 0] > -0.30      # wet sand or mud


def test_interior_drops_only_the_outer_ring():
    inside = np.zeros((7, 7), bool); inside[1:6, 1:6] = True
    core = v4.interior(inside)
    assert core.sum() == 9 and core[2:5, 2:5].all()
    assert (v4.interior(inside, 0) == inside).all()


def test_filter_never_adds_and_removes_edge_and_wet_pixels():
    inside = np.ones((5, 5), bool)
    mask = np.ones((5, 5), bool)
    stack = stack_with(0.124, 0.373)
    stack[2, 2, 4] = 0.15                      # one wet pixel in the interior
    out = v4.filter_mask(mask, stack, inside)
    assert not out[0].any() and not out[:, 0].any()          # boundary ring gone
    assert not out[2, 2]                                       # wet pixel gone
    assert out[1, 1] and out.sum() == 8
    assert not v4.filter_mask(np.zeros((5, 5), bool), stack, inside).any()


def test_nan_reflectance_is_never_kept():
    stack = stack_with(np.nan, 0.3)
    assert not v4.filter_mask(np.ones((5, 5), bool), stack, np.ones((5, 5), bool)).any()
