"""Change detection between two epochs.

Given a bare-peat mask for an earlier date (A) and a later date (B),
plus the pixels that are cloud-clear in both, classify every pixel of
the bog into:

  * newly bare   — vegetated at A, bare at B  → candidate NEW cutting
  * re-vegetated — bare at A, vegetated at B   → recovery / restoration
  * stable bare  — bare at both dates
  * stable veg   — vegetated at both dates

Only pixels valid (cloud-clear) at BOTH dates are classified; the rest
are 'no-data' and excluded so cloud does not masquerade as change.
"""

import numpy as np


def compare(bare_a, bare_b, both_valid):
    """Return per-pixel change masks, restricted to both-valid pixels."""
    ba = bare_a & both_valid
    bb = bare_b & both_valid
    return {
        "newly_bare": bb & ~ba & both_valid,
        "revegetated": ba & ~bb & both_valid,
        "stable_bare": ba & bb,
        "both_valid": both_valid,
    }


def change_class_raster(masks, shape):
    """Encode change masks as a single small-int raster for display.

    0 = no-data (not clear at both dates), 1 = stable vegetation,
    2 = stable bare, 3 = re-vegetated, 4 = newly bare.
    """
    out = np.zeros(shape, dtype="uint8")
    out[masks["both_valid"]] = 1              # baseline: clear & vegetated
    out[masks["stable_bare"]] = 2
    out[masks["revegetated"]] = 3
    out[masks["newly_bare"]] = 4
    return out
