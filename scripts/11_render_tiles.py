"""Step 12 — per-bog, per-year image tiles for the web card.

For every bog and every survey year, renders a small PNG: natural-colour
Sentinel-2 clipped to the bog, with the two detectors overlaid — GNG as a
translucent blue fill, NDVI as a red outline — so a viewer can compare
them and drag a year slider to watch bare peat change.

Same clear-scene selection as the dataset build, so the pictures match
the numbers. Resumable: existing tiles are skipped.
"""

import sys
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
MAXPX = 360  # cap long side of the saved image


def natural_rgb(stack, order):
    r = stack[:, :, order.index("red")] / 10000.0
    g = stack[:, :, order.index("green")] / 10000.0
    b = stack[:, :, order.index("blue")] / 10000.0
    return np.clip(np.dstack([r, g, b]) * 3.2, 0, 1) ** (1.0 / 1.4)


def render(rgb, gng_mask, ndvi_mask, inside, out_path):
    h, w = rgb.shape[:2]
    scale = MAXPX / max(h, w)
    fig_w, fig_h = w * scale / 100, h * scale / 100
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


def _cache_dates(code):
    """year -> chosen max-activity scene date, from the dataset cache, so
    the rendered picture matches the numbers on the card."""
    f = config.OUT_DIR / "cache" / f"{code}.json"
    if not f.exists():
        return {}
    import json
    r = json.loads(f.read_text())
    return {int(y): d for y, d in zip(r.get("years", []), r.get("dates", []))}


def process_site(row, bbox):
    ref = imagery.search_scene(PROVIDER, bbox, "2021-05-01/2021-09-15",
                               max_cloud=40)
    if ref is None:
        return 0
    red0, transform, crs = imagery.read_window(ref, "red", bbox, PROVIDER)
    shape = red0.shape
    geom = gpd.GeoSeries([row["geometry"]], crs=config.ITM).to_crs(crs).iloc[0]
    inside = geo.polygon_mask(geom, shape, transform)
    dates = _cache_dates(row["SITECODE"])

    made = 0
    for yr in YEARS:
        out = OUT_DIR / f"{row['SITECODE']}_{yr}.jpg"
        if out.exists():
            continue
        # render the exact scene the dataset chose for this year, so the
        # image matches the reported number; fall back to clearest-in-window
        rng = (f"{dates[yr]}/{dates[yr]}" if yr in dates
               else pipeline._year_range(yr))
        item, rep = preprocess.pick_clear_scene(
            PROVIDER, bbox, rng, shape,
            aoi_mask=inside, min_clear=0.80, limit=12)
        if item is None and yr in dates:
            item, rep = preprocess.pick_clear_scene(
                PROVIDER, bbox, pipeline._year_range(yr), shape,
                aoi_mask=inside, min_clear=0.85, limit=12)
        if item is None:
            continue
        # don't render a visibly cloudy tile — the card shows "no clear
        # image" for that year instead of a misleading cloudy picture
        if rep["aoi_clear"] < 0.82:
            continue
        order = list(BANDS)
        layers = [imagery.read_window(item, b, bbox, PROVIDER, out_shape=shape)[0]
                  .astype("float32") for b in order]
        stack = np.stack(layers, axis=-1)
        red = stack[:, :, order.index("red")]
        nir = stack[:, :, order.index("nir")]
        ndvi = detect.ndvi(red, nir)
        valid = preprocess.valid_mask(rep["scl"])
        ndvi_mask = pipeline.ndvi_bare(stack, order, ndvi, valid, inside)
        gng_mask = pipeline.gng_bare(stack, order, valid, inside)
        rgb = natural_rgb(stack, order)
        # haze guard: SCL sometimes passes thin cloud/haze as clear. If too
        # much of the bog is near-white, skip — the card shows "no clear
        # image" rather than a washed-out picture.
        gray = rgb.mean(axis=2)
        if (gray[inside] > 0.75).mean() > 0.18:
            continue
        render(rgb, gng_mask, ndvi_mask, inside, out)
        made += 1
    return made


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sites = boundaries.build_sites()
    print(f"{len(sites)} sites\n")
    for pos, (idx, row) in enumerate(sites.iterrows(), 1):
        bbox = boundaries.site_bbox_wgs84(sites, idx, buffer_m=200)
        try:
            t0 = time.time()
            n = process_site(row, bbox)
            print(f"[{pos}/{len(sites)}] {row['SITE_NAME'][:34]:34} "
                  f"+{n} tiles [{time.time()-t0:.0f}s]")
        except Exception as e:
            print(f"[{pos}/{len(sites)}] FAIL {row['SITE_NAME']}: "
                  f"{type(e).__name__}: {e}")
    total = len(list(OUT_DIR.glob("*.jpg")))
    print(f"\nDone. {total} tiles in {OUT_DIR}")


if __name__ == "__main__":
    main()
