"""Georeferencing helpers: polygon rasterisation and area from pixels."""

import numpy as np
from rasterio.features import geometry_mask


def polygon_mask(geom_in_raster_crs, out_shape, transform):
    """Boolean mask (True = inside polygon) for a raster window."""
    return ~geometry_mask(
        [geom_in_raster_crs],
        out_shape=out_shape,
        transform=transform,
        invert=False,
    )


def pixel_area_ha(transform):
    """Area of a single pixel in hectares, from the affine transform."""
    px_m2 = abs(transform.a * transform.e)  # |width * height| in CRS metres
    return px_m2 / 10_000.0


def mask_area_ha(mask, transform):
    """Total area of True cells in a boolean mask, in hectares."""
    return int(np.count_nonzero(mask)) * pixel_area_ha(transform)
