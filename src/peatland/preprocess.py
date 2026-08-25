"""Preprocessing: scene quality control before any detection runs.

Tile-level cloud metadata (``eo:cloud_cover``) describes a whole 100 km
Sentinel-2 tile — a scene can be "20% cloudy" overall yet completely
clear (or completely clouded) over our small bog. So we judge each scene
by the Sentinel-2 Scene Classification Layer (SCL) *inside the site's
window*: how many pixels are actually clear ground, not cloud or shadow.

A scene is accepted only if its clear fraction over the area of interest
clears a threshold; scene selection then picks the best clear scene in a
date range.
"""

import numpy as np
import rasterio

from . import imagery

# Sentinel-2 SCL classes.
#   0 no-data          1 saturated/defective   2 dark area
#   3 cloud shadow     4 vegetation            5 bare soil
#   6 water            7 unclassified          8 cloud medium prob
#   9 cloud high prob 10 thin cirrus          11 snow/ice
# Pixels we cannot trust for land-surface analysis:
INVALID_SCL = {0, 1, 3, 8, 9, 10, 11}


def read_scl(item, bbox_wgs84, provider, out_shape):
    """Read the SCL band over the site window at out_shape (nearest)."""
    scl, _, _ = imagery.read_window(
        item, "scl", bbox_wgs84, provider, out_shape=out_shape,
        resampling=rasterio.enums.Resampling.nearest,
    )
    return scl


def valid_mask(scl):
    """Boolean mask of usable pixels (not cloud / shadow / no-data)."""
    return ~np.isin(scl, list(INVALID_SCL))


def clear_fraction(scl, aoi_mask=None):
    """Fraction of clear pixels, optionally restricted to an AOI mask."""
    valid = valid_mask(scl)
    if aoi_mask is not None:
        denom = int(aoi_mask.sum())
        if denom == 0:
            return 0.0
        return float((valid & aoi_mask).sum()) / denom
    return float(valid.mean())


def assess_scene(item, bbox_wgs84, provider, out_shape, aoi_mask=None):
    """Return a small quality report for one candidate scene."""
    scl = read_scl(item, bbox_wgs84, provider, out_shape)
    return {
        "id": item.id,
        "datetime": item.properties.get("datetime", ""),
        "tile_cloud": float(item.properties.get("eo:cloud_cover", 100.0)),
        "aoi_clear": clear_fraction(scl, aoi_mask),
        "scl": scl,
    }


def pick_clear_scene(provider, bbox_wgs84, date_range, out_shape,
                     aoi_mask=None, min_clear=0.90, max_cloud=60, limit=12):
    """Search a date range and return the scene clearest over the AOI.

    Unlike ``imagery.search_scene`` (which ranks by tile-level cloud %),
    this reads the SCL for each candidate and ranks by the clear fraction
    *inside the bog window*. Returns (item, report) or (None, None).
    """
    client = imagery.open_catalog(provider)
    search = client.search(
        collections=[imagery.PROVIDERS[provider]["collection"]],
        bbox=bbox_wgs84,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
        max_items=limit,
    )
    items = list(search.items())
    if not items:
        return None, None

    # cheapest first: try low tile-cloud candidates before costly reads
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))

    best = None
    for it in items:
        try:
            rep = assess_scene(it, bbox_wgs84, provider, out_shape, aoi_mask)
        except Exception:
            continue
        if best is None or rep["aoi_clear"] > best["aoi_clear"]:
            best = {**rep, "_item": it}
        if rep["aoi_clear"] >= min_clear:
            break  # good enough, stop paying for reads

    if best is None:
        return None, None
    item = best.pop("_item")
    return item, best


def clear_scenes(provider, bbox_wgs84, date_range, out_shape,
                 aoi_mask=None, min_clear=0.85, max_cloud=60, limit=14,
                 max_scenes=4):
    """All scenes in the range that are clear over the AOI, best first.

    Returns a list of (item, report) with aoi_clear >= min_clear, at most
    `max_scenes`, ordered by acquisition date. Used for season-max
    detection: bare cut peat is most visible right after cutting, so the
    per-year figure is the maximum over every clear scene in the season,
    which removes the acquisition-date lottery a single scene suffers from.
    """
    client = imagery.open_catalog(provider)
    search = client.search(
        collections=[imagery.PROVIDERS[provider]["collection"]],
        bbox=bbox_wgs84,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
        max_items=limit,
    )
    items = list(search.items())
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))

    out = []
    for it in items:
        if len(out) >= max_scenes:
            break
        try:
            rep = assess_scene(it, bbox_wgs84, provider, out_shape, aoi_mask)
        except Exception:
            continue
        if rep["aoi_clear"] >= min_clear:
            out.append((it, rep))
    out.sort(key=lambda pair: pair[1]["datetime"])
    return out
