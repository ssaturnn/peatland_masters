"""Detector v4: two physical post-filters on the frozen v3 candidates.

v4 keeps every v3 decision and removes two kinds of pixel that the v3 error
analysis traced most false positives to, on the calibration site-years only:

- **boundary pixels**: the outermost ring of the assessed area. A 10 m pixel on
  the site boundary mixes the bog with whatever lies outside it (field, road,
  shore, water), so its spectrum is not the bog's;
- **wet surfaces**: pixels whose modified normalised difference water index,
  (green - SWIR1) / (green + SWIR1), is above -0.30. Wet sand, tidal mud and
  pool edges are dark in SWIR1 relative to green; dry exposed peat is bright
  in SWIR1 (MNDWI about -0.4 to -0.55 on the calibration sample).

On the calibration site-years the two filters together removed 15 of 20 false
positives and none of 10 bare-peat detections. The threshold keeps a margin
of 0.09 from the wettest bare peat observed there (-0.39). Wet, freshly cut peat
on other dates may sit closer to the threshold, which is why v4 is judged on a
fresh test sample.
"""

import numpy as np
from scipy import ndimage

VERSION = "2026-10-07-edge-wetness-v4"
BOUNDARY_PX = 1          # remove pixels within this distance of the assessed-area edge
MNDWI_MAX = -0.30        # keep only surfaces drier than this
BANDS = ("blue", "green", "red", "nir", "swir1", "swir2")


def mndwi(stack, order=BANDS):
    """(green - SWIR1) / (green + SWIR1) of a reflectance or DN stack."""
    green = np.asarray(stack[..., order.index("green")], float)
    swir1 = np.asarray(stack[..., order.index("swir1")], float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (green - swir1) / (green + swir1)


def interior(inside, boundary_px=BOUNDARY_PX):
    """The assessed area without its outermost `boundary_px` ring of pixels."""
    inside = np.asarray(inside, bool)
    if boundary_px <= 0:
        return inside.copy()
    # the array edge counts as outside: a site cut by the window still has a boundary there
    padded = np.pad(inside, 1, constant_values=False)
    return ndimage.distance_transform_edt(padded)[1:-1, 1:-1] > boundary_px


def filter_mask(mask, stack, inside, order=BANDS, boundary_px=BOUNDARY_PX, mndwi_max=MNDWI_MAX):
    """v4 candidates: v3 candidates that are interior and dry. Never adds a pixel."""
    mask = np.asarray(mask, bool)
    wet = mndwi(stack, order)
    dry = np.isfinite(wet) & (wet <= mndwi_max)
    return mask & interior(inside, boundary_px) & dry
