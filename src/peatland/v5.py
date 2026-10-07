"""Detector v5: exposed peat from its short-wave infrared signature.

Industrial milled peat, the one place in the region where large areas of
exposed peat are certain, contradicts the v1-v4 definition of bare peat
(NDVI below about 0.25, NBR above zero). On the Mountdillon production fields
(scripts/32_positive_control.py) dry exposed peat has a moderate NDVI
(median 0.42; peat is dark in red as well as in near infrared) and is bright
in short-wave infrared: SWIR2 exceeds NIR, so NBR is negative (median -0.4).
The v3 burn guard (NBR > 0) therefore removed real peat, and the NDVI
threshold kept winter-brown vegetation instead (scripts/31_summer_check.py).

v5 tests the SWIR signature directly:

- NBR < 0: SWIR2 above NIR, as on exposed peat and on burn scars;
- SWIR1 >= 0.22: dry peat is bright in SWIR1 (median 0.33); char, shadow
  and wet surfaces are dark;
- SWIR1 / SWIR2 >= 1.25: fresh burn scars have a flatter SWIR (median 1.14)
  than peat (1.35);
- MNDWI <= -0.30: not water or saturated sediment (as in v4).

Thresholds were set on two calibration scenes only, the south Mountdillon
area in 2018 (spring and summer) and the Moorfield 2025 spring burn: they keep
82-85% of the peat pixels and 4% of the burn pixels. Everything else is test.
"""

import numpy as np

VERSION = "2026-10-08-swir-peat-v5"
SWIR1_MIN = 0.22
SWIR_RATIO_MIN = 1.25
MNDWI_MAX = -0.30
BANDS = ("blue", "green", "red", "nir", "swir1", "swir2")


def indices(stack, order=BANDS):
    """NBR, MNDWI, SWIR1 reflectance and SWIR1/SWIR2 of a DN (x10000) stack."""
    b = {name: np.asarray(stack[..., order.index(name)], float) / 10000.0
         for name in ("green", "nir", "swir1", "swir2")}
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"nbr": (b["nir"] - b["swir2"]) / (b["nir"] + b["swir2"]),
                "mndwi": (b["green"] - b["swir1"]) / (b["green"] + b["swir1"]),
                "swir1": b["swir1"],
                "ratio": b["swir1"] / b["swir2"]}


def peat_mask(stack, valid, inside, order=BANDS, swir1_min=SWIR1_MIN,
              ratio_min=SWIR_RATIO_MIN, mndwi_max=MNDWI_MAX):
    """Exposed-peat candidates: clear pixels inside the site with the SWIR signature."""
    ix = indices(stack, order)
    with np.errstate(invalid="ignore"):
        signature = ((ix["nbr"] < 0) & (ix["swir1"] >= swir1_min)
                     & (ix["ratio"] >= ratio_min) & (ix["mndwi"] <= mndwi_max))
    return np.asarray(valid, bool) & np.asarray(inside, bool) & np.nan_to_num(signature).astype(bool)
