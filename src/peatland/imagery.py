"""Sentinel-2 access via STAC + cloud-optimised GeoTIFF window reads.

Two providers are supported, tried in order:
  1. Microsoft Planetary Computer  (needs asset signing, anonymous search)
  2. Element84 earth-search / AWS   (fully public COGs, no signing)

Only the pixel window we ask for is read — full tiles are never
downloaded.
"""

import rasterio
import numpy as np
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from pystac_client import Client

from . import config

PROVIDERS = {
    "planetary": {
        "url": "https://planetarycomputer.microsoft.com/api/stac/v1",
        "collection": "sentinel-2-l2a",
        "sign": True,
    },
    "earthsearch": {
        "url": "https://earth-search.aws.element84.com/v1",
        "collection": "sentinel-2-l2a",
        "sign": False,
    },
}


def open_catalog(provider):
    p = PROVIDERS[provider]
    if p["sign"]:
        import planetary_computer
        return Client.open(p["url"], modifier=planetary_computer.sign_inplace)
    return Client.open(p["url"])


def search_scene(provider, bbox_wgs84, date_range, max_cloud=20, limit=10):
    """Return the least-cloudy Sentinel-2 L2A item over bbox in date_range."""
    client = open_catalog(provider)
    search = client.search(
        collections=[PROVIDERS[provider]["collection"]],
        bbox=bbox_wgs84,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
        max_items=limit,
    )
    items = list(search.items())
    if not items:
        return None
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
    return items[0]


def _band_asset(item, band, provider):
    """Resolve the asset key for a band name across provider conventions."""
    # Planetary Computer uses B02.., earth-search uses lowercase 'red' etc.
    candidates = [band, band.upper(), band.lower()]
    name_map = {
        "B02": "blue", "B03": "green", "B04": "red",
        "B08": "nir", "B11": "swir16", "B12": "swir22", "SCL": "scl",
    }
    if band in name_map:
        candidates.append(name_map[band])
    for key in candidates:
        if key in item.assets:
            return item.assets[key]
    raise KeyError(f"band {band!r} not in assets: {list(item.assets)[:12]}")


def _asset_href(item, band, provider):
    return _band_asset(item, band, provider).href


def harmonize_band(array, item, band, provider):
    """Return spectral values as reflectance * 10000; leave SCL unchanged.

    Processing baseline >= 04.00 introduced BOA_ADD_OFFSET=-1000 DN.
    Baseline metadata, rather than acquisition year, also covers reprocessed
    historical scenes. Explicit STAC scale/offset takes precedence. Zero DN
    denotes no-data and becomes NaN, excluded by the detector's validity mask.
    Negative reflectance is clamped to zero for the existing detector contract.
    """
    band = config.S2_BANDS.get(band, band)
    if band == "SCL":
        return array
    asset = _band_asset(item, band, provider)
    raster_bands = asset.extra_fields.get("raster:bands", [])
    meta = raster_bands[0] if raster_bands else {}
    values = np.asarray(array, dtype=np.float32)
    nodata = meta.get("nodata", 0)
    invalid = ~np.isfinite(values) | (values == nodata)
    if "scale" in meta and "offset" in meta:
        result = (values * float(meta["scale"]) + float(meta["offset"])) * 10000.0
    elif provider == "earthsearch" and item.properties.get("earthsearch:boa_offset_applied") is True:
        result = values.copy()
    else:
        baseline = item.properties.get("s2:processing_baseline")
        if baseline is None:
            raise ValueError(f"Missing processing baseline for {item.id}: cannot harmonize {band}")
        try:
            baseline = float(baseline)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid processing baseline for {item.id}") from exc
        if not np.isfinite(baseline):
            raise ValueError(f"Invalid processing baseline for {item.id}")
        result = values - (1000.0 if baseline >= 4.0 else 0.0)
    result = np.maximum(result, 0.0).astype(np.float32)
    result[invalid] = np.nan
    return result


def read_window(item, band, bbox_wgs84, provider, out_shape=None,
                resampling=rasterio.enums.Resampling.bilinear):
    """Read only the pixels covering bbox_wgs84 for one band.

    If out_shape is given, the window is resampled to that shape (used to
    bring 20 m bands onto the 10 m grid). Returns (array, transform, crs).
    Categorical bands (e.g. SCL) should pass resampling=nearest.
    """
    href = _asset_href(item, config.S2_BANDS.get(band, band), provider)
    with rasterio.open(href) as src:
        left, bottom, right, top = transform_bounds(
            config.WGS84, src.crs, *bbox_wgs84
        )
        window = from_bounds(left, bottom, right, top, src.transform)
        if out_shape is None:
            arr = src.read(1, window=window)
            win_transform = src.window_transform(window)
        else:
            arr = src.read(1, window=window, out_shape=out_shape,
                           resampling=resampling)
            # transform for the resampled grid over the same window bounds
            win_transform = rasterio.windows.transform(
                window, src.transform
            ) * rasterio.Affine.scale(
                window.width / out_shape[1], window.height / out_shape[0]
            )
        return harmonize_band(arr, item, band, provider), win_transform, src.crs


def read_stack(item, bands, bbox_wgs84, provider):
    """Read several bands onto a common 10 m grid.

    Returns (stack HxWxB float32, transform, crs, band_order). The first
    band defines the grid; others are resampled to match.
    """
    first, transform, crs = read_window(item, bands[0], bbox_wgs84, provider)
    shape = first.shape
    layers = [first.astype("float32")]
    for b in bands[1:]:
        arr, _, _ = read_window(item, b, bbox_wgs84, provider, out_shape=shape)
        layers.append(arr.astype("float32"))
    import numpy as np
    stack = np.stack(layers, axis=-1)
    return stack, transform, crs, list(bands)
