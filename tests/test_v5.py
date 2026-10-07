"""Detector v5: the SWIR signature of exposed peat."""

import numpy as np

from peatland import v5

# median spectra (x10000) measured in this project
MILLED_PEAT = [225, 304, 450, 1224, 3452, 2556]     # Mountdillon, summer 2018
BURN = [350, 380, 570, 1050, 1980, 1740]            # Moorfield, spring 2025
BOG = [380, 520, 600, 1900, 1700, 900]              # vegetated bog, spring
WATER = [400, 500, 350, 200, 120, 80]


def px(spectrum):
    return np.array(spectrum, float)[None, None, :]


def flag(spectrum):
    one = np.ones((1, 1), bool)
    return bool(v5.peat_mask(px(spectrum), one, one)[0, 0])


def test_exposed_peat_is_flagged_and_burn_bog_water_are_not():
    assert flag(MILLED_PEAT)
    assert not flag(BURN)
    assert not flag(BOG)
    assert not flag(WATER)


def test_exposed_peat_has_moderate_ndvi_and_negative_nbr():
    b, g, r, n, s1, s2 = MILLED_PEAT
    assert 0.4 < (n - r) / (n + r) < 0.5          # above the old 0.25 threshold
    assert (n - s2) / (n + s2) < -0.3             # the old NBR > 0 guard removed it


def test_mask_respects_validity_and_site():
    stack = np.repeat(px(MILLED_PEAT), 3, axis=1)
    valid = np.array([[True, False, True]])
    inside = np.array([[True, True, False]])
    assert v5.peat_mask(stack, valid, inside).tolist() == [[True, False, False]]


def test_nan_is_never_flagged():
    one = np.ones((1, 1), bool)
    assert not v5.peat_mask(np.full((1, 1, 6), np.nan), one, one).any()
