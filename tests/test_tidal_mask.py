"""The ever-water union that screens intertidal flats out of detection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from peatland import pipeline


def test_union_over_scenes():
    a = np.zeros((3, 3), dtype=np.uint8)
    b = np.zeros((3, 3), dtype=np.uint8)
    a[0, 0] = 6          # water at high tide only
    b[2, 2] = 6          # water in a later year only
    b[1, 1] = 4          # vegetation is not water
    mask = pipeline.ever_water_mask([{"scl": a}, {"scl": b}], (3, 3))
    assert mask[0, 0] and mask[2, 2]
    assert mask.sum() == 2


def test_no_scenes_means_no_exclusion():
    mask = pipeline.ever_water_mask([], (2, 2))
    assert not mask.any()
