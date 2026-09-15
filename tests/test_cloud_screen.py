"""Bright-cloud screen and the shared scene gates."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from peatland import pipeline

ORDER = list(pipeline.BANDS)
SCL_VEGETATION = 4


class _NoSun:
    properties = {}  # no solar geometry: shadow projection is skipped


def _bog(h=20, w=20, blue=300.0):
    """Dark vegetated bog: reflectance 0.04, NIR 0.25, blue as given (x10000)."""
    stack = np.full((h, w, len(ORDER)), 400.0, dtype="float32")
    stack[:, :, ORDER.index("nir")] = 2500.0
    stack[:, :, ORDER.index("blue")] = blue
    return stack


def _quality(stack):
    h, w = stack.shape[:2]
    scl = np.full((h, w), SCL_VEGETATION, dtype=np.uint8)
    return pipeline.scene_quality(stack, ORDER, scl, np.ones((h, w), bool), _NoSun())


def test_bright_pixel_is_cloud_and_its_edge_is_buffered():
    stack = _bog()
    stack[10, 10, ORDER.index("blue")] = 3000.0  # cumulus, blue 0.30
    q = _quality(stack)
    b = pipeline.CLOUD_BUFFER_PX
    assert not q["valid"][10, 10]
    assert not q["valid"][10, 10 + b] and not q["valid"][10 - b, 10]
    assert q["valid"][10, 10 + b + 1] and q["valid"][0, 0]


def test_clear_dark_bog_passes_every_gate():
    q = _quality(_bog())
    assert q["clear_fraction"] == 1.0 and q["cloud_frac"] == 0.0
    assert pipeline.scene_passes(q)


def test_haze_field_fails_the_hot_gate_without_being_cloud():
    q = _quality(_bog(blue=900.0))  # blue 0.09: hazy, but below CLOUD_BLUE
    assert q["cloud_frac"] == 0.0
    assert q["hot_frac"] > pipeline.HOT_MAX_FRAC
    assert not pipeline.scene_passes(q)
