"""Spectral detection of exposed / cut peat surfaces."""

import numpy as np


def ndvi(red, nir):
    """Normalised Difference Vegetation Index, safe against divide-by-zero."""
    red = red.astype("float32")
    nir = nir.astype("float32")
    denom = nir + red
    out = np.zeros_like(denom)
    valid = denom != 0
    out[valid] = (nir[valid] - red[valid]) / denom[valid]
    return out


def exposed_peat_mask(ndvi_arr, threshold=0.25):
    """Boolean mask of likely bare / cut peat: low NDVI = little vegetation."""
    return ndvi_arr < threshold
