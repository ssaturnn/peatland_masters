"""Read exact released scenes without modifying release caches or detectors."""

import json
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

from . import boundaries, config, geo, imagery, pipeline, preprocess

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def site_boundaries():
    return boundaries.build_sites()


def release_grid(code):
    """Return the original boundary, bbox and frozen release grid."""
    record = json.loads((ROOT / f"outputs/cache-release-v3/{code}.json").read_text())
    if record["detector_version"] != pipeline.DETECTOR_VERSION:
        raise ValueError("Release detector version differs")
    sites = site_boundaries()
    index = sites.index[sites.SITECODE == code][0]
    grid = record["grid"]
    shape = tuple(grid["shape"])
    transform = rasterio.Affine(*grid["transform"][:6])
    geom = gpd.GeoSeries([sites.loc[index].geometry], crs=config.ITM).to_crs(grid["crs"]).iloc[0]
    inside = geo.polygon_mask(geom, shape, transform)
    tidal = pipeline.mask_from_b64(record["tidal_mask_b64"], shape) if record.get("tidal_mask_b64") else np.zeros(shape, bool)
    return record, inside, tidal, boundaries.site_bbox_wgs84(sites, index, buffer_m=200)


def exact_scene(key, destination):
    """Rerun v3 on the card's exact scene and retain the masks privately."""
    from .planet import private_path, write_json
    destination = private_path(destination)
    code, year = key.split("_")
    record, inside, tidal, bbox = release_grid(code)
    grid = record["grid"]
    card = json.loads((ROOT / f"web/data/tiles/{key}.json").read_text())
    i = record["years"].index(int(year))
    if card["scene_id"] != record["scene_ids"][i] or card["scene_date"] != record["dates"][i]:
        raise ValueError(f"{key}: cache/card scene identity differs")
    meta = dict(scene_id=card["scene_id"], scene_date=card["scene_date"],
                detector_version=pipeline.DETECTOR_VERSION, crs=grid["crs"],
                transform=grid["transform"], shape=grid["shape"], bbox=list(bbox))
    cache = destination / "sentinel2.npz"
    metadata = destination / "sentinel2.json"
    if cache.exists() and metadata.exists():
        old = json.loads(metadata.read_text())
        if all(old.get(k) == v for k, v in meta.items()):
            with np.load(cache) as z:
                return z["inside"], z["prediction_gng"], meta
    catalog = imagery.open_catalog("planetary")
    item = catalog.get_collection("sentinel-2-l2a").get_item(card["scene_id"])
    if item is None:
        raise ValueError(f"{key}: exact release scene unavailable")
    rep = preprocess.assess_scene(item, bbox, "planetary", inside.shape, aoi_mask=inside)
    detected = pipeline.detect_year(item, bbox, "planetary", inside.shape, inside & ~tidal, rep["scl"])
    if not pipeline.scene_passes(detected):
        raise ValueError(f"{key}: release scene fails quality gates")
    transform = rasterio.Affine(*grid["transform"][:6])
    area = geo.mask_area_ha(detected["gng"], transform)
    if abs(round(area, 2) - record["gng_series"][i]) > .011:
        raise ValueError(f"{key}: rerun area differs from published area")
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, inside=inside, prediction_gng=detected["gng"],
                        valid=detected["valid"], stack=detected["stack"],
                        transform=np.array(grid["transform"]))
    write_json(metadata, meta | {"gng_ha": area})
    return inside, detected["gng"], meta
