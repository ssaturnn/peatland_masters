"""Step 12 — per-bog, per-year image tiles for the web card.

For every bog and every survey year, renders a small PNG: natural-colour
Sentinel-2 clipped to the bog, with the two detectors overlaid — GNG as a
translucent blue fill, NDVI as a red outline — so a viewer can compare
them and drag a year slider to watch bare peat change.

Same clear-scene selection as the dataset build, so the pictures match
the numbers. Resumable: existing tiles are skipped.
"""

import sys
import argparse
import json
import rasterio
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from peatland import (boundaries, imagery, detect, geo, config, preprocess,
                      pipeline)

PROVIDER = "planetary"
YEARS = pipeline.YEARS
BANDS = pipeline.BANDS
OUT_DIR = Path(__file__).resolve().parents[1] / "web" / "data" / "tiles"
CACHE_DIR = config.OUT_DIR / "cache"
MAXPX = 360  # cap long side of the saved image


def natural_rgb(stack, order):
    r = stack[:, :, order.index("red")] / 10000.0
    g = stack[:, :, order.index("green")] / 10000.0
    b = stack[:, :, order.index("blue")] / 10000.0
    return np.clip(np.dstack([r, g, b]) * 3.2, 0, 1) ** (1.0 / 1.4)


def render(rgb, gng_mask, ndvi_mask, inside, out_path, clean_path=None):
    """Save the overlay tile and, optionally, a clean-RGB twin (no
    markings) so the card can toggle overlays off to inspect the raw
    surface beneath the detections."""
    h, w = rgb.shape[:2]
    scale = MAXPX / max(h, w)
    fig_w, fig_h = w * scale / 100, h * scale / 100

    if clean_path is not None:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
        ax.imshow(rgb)
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(clean_path, dpi=100, pad_inches=0,
                    pil_kwargs={"quality": 82, "optimize": True})
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.imshow(rgb)
    # GNG: translucent blue fill
    blue = np.zeros((h, w, 4))
    blue[gng_mask] = [0.15, 0.5, 1.0, 0.6]
    ax.imshow(blue)
    # NDVI: red outline
    if ndvi_mask.any():
        ax.contour(ndvi_mask.astype(float), levels=[0.5],
                   colors=["#ff2d2d"], linewidths=0.8)
    # bog boundary
    ax.contour(inside.astype(float), levels=[0.5],
               colors=["#ffffff"], linewidths=0.6, alpha=0.7)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=100, pad_inches=0,
                pil_kwargs={"quality": 82, "optimize": True})
    plt.close(fig)


def process_site(row, bbox):
    cache_file = CACHE_DIR / f"{row['SITECODE']}.json"
    record = json.loads(cache_file.read_text())
    if record.get("detector_version") != pipeline.DETECTOR_VERSION:
        raise ValueError("Stale detector cache: rebuild into a new cache directory first")
    grid = record["grid"]
    shape = tuple(grid["shape"])
    transform = rasterio.Affine(*grid["transform"][:6])
    crs = grid["crs"]
    geom = gpd.GeoSeries([row["geometry"]], crs=config.ITM).to_crs(crs).iloc[0]
    inside = geo.polygon_mask(geom, shape, transform)
    # the dataset build screens intertidal pixels out of detection; the
    # rendered masks must use the exact same exclusion or images and
    # numbers diverge (the site outline still shows the full boundary)
    det_inside = inside
    if record.get("tidal_mask_b64"):
        det_inside = inside & ~pipeline.mask_from_b64(
            record["tidal_mask_b64"], shape)
    collection = None
    made = 0
    for yr, day, scene_id in zip(record["years"], record["dates"], record["scene_ids"]):
        out = OUT_DIR / f"{row['SITECODE']}_{yr}.jpg"
        clean = OUT_DIR / f"{row['SITECODE']}_{yr}c.jpg"
        manifest = out.with_suffix(".json")
        identity = {"scene_id": scene_id, "scene_date": day,
                    "detector_version": pipeline.DETECTOR_VERSION}
        if (out.exists() and clean.exists() and manifest.exists()
                and json.loads(manifest.read_text()) == identity):
            continue
        if collection is None:
            collection = imagery.open_catalog(PROVIDER).get_collection(
                imagery.PROVIDERS[PROVIDER]["collection"])
        item = collection.get_item(scene_id)
        if item is None:
            raise ValueError(f"Chosen scene is unavailable: {scene_id}; no substitute scene is rendered")
        rep = preprocess.assess_scene(item, bbox, PROVIDER, shape, aoi_mask=inside)
        d = pipeline.detect_year(item, bbox, PROVIDER, shape, det_inside,
                                 rep["scl"])
        if not pipeline.scene_passes(d):
            raise ValueError(f"Chosen scene no longer passes the shared quality gates: {scene_id}")
        render(natural_rgb(d["stack"], BANDS), d["gng"], d["ndvi"], inside, out, clean_path=clean)
        manifest.write_text(json.dumps(identity, indent=2))
        made += 1
    return made


def main():
    global CACHE_DIR, OUT_DIR
    parser = argparse.ArgumentParser(description="Render exact versioned dataset scenes")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--site", help="Optional literal site-name fragment")
    args = parser.parse_args()
    CACHE_DIR, OUT_DIR = args.cache_dir, args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sites = boundaries.build_sites()
    if args.site:
        sites = sites[sites.SITE_NAME.str.contains(args.site, regex=False)]
        if sites.empty:
            parser.error(f"No site matches {args.site!r}")
    failures = 0
    print(f"{len(sites)} sites\n")
    for pos, (idx, row) in enumerate(sites.iterrows(), 1):
        bbox = boundaries.site_bbox_wgs84(sites, idx, buffer_m=200)
        try:
            t0 = time.time()
            n = process_site(row, bbox)
            print(f"[{pos}/{len(sites)}] {row['SITE_NAME'][:34]:34} "
                  f"+{n} tiles [{time.time()-t0:.0f}s]")
        except Exception as e:
            failures += 1
            print(f"[{pos}/{len(sites)}] FAIL {row['SITE_NAME']}: "
                  f"{type(e).__name__}: {e}")
    total = len(list(OUT_DIR.glob("*.jpg")))
    print(f"\nDone. {total} tiles in {OUT_DIR}; {failures} failed sites")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
