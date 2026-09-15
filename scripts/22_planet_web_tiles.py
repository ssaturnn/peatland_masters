"""PlanetScope 3 m tiles for the password-protected web map.

For every evaluation site-year with a same-day PlanetScope mosaic
(scripts/20_planet_reference.py), renders the mosaic on the exact extent of
the published Sentinel-2 card tile, so the card can swipe between 10 m and
3 m. Two images per site-year, like the Sentinel-2 tiles: one with the
frozen Sentinel-2 GNG outline (plus the calibrated 3 m candidates where the
case study ran, scripts/21_planet_case_study.py) and a clean twin.

Planet imagery is licensed for non-commercial research only, so the output
goes to web/private/ (gitignored, served behind a password), never to the
public site.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.warp import Resampling, reproject
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from peatland.planet import first_frame_clear, private_path, write_json
from peatland.release_scene import exact_scene
RUN = ROOT / "outputs" / "evaluation" / "2026-09-13-v3-sample"
CASE = ROOT / "outputs" / "planet_case_study"
WEB_TILES = ROOT / "web" / "data" / "tiles"
OUT = ROOT / "web" / "private" / "planet"
UPSAMPLE = 3                # output pixels per Sentinel-2 pixel side (3.3 m)
MAXPX = 1400                # cap long side of the saved image
GNG_EDGE = (40, 128, 255)   # the card's GNG blue
PS_FILL = (255, 197, 61)    # calibrated PlanetScope candidates
BOG_EDGE = (255, 255, 255)
NODATA_RGB = (10, 28, 21)   # the card's figure background
MIN_CLEAR_PCT = 50          # withhold tiles where UDM2 marks less of the bog clear
LICENCE = ("PlanetScope imagery, Planet Education & Research programme: "
           "non-commercial research only")
CITATION = ("Planet Team (2026). Planet Application Program Interface: "
            "In Space for Life on Earth. San Francisco, CA. https://api.planet.com")


def natural_rgb(bands):
    """Same stretch as the Sentinel-2 tiles (scripts/11_render_tiles.py)."""
    rgb = np.dstack([bands[2], bands[1], bands[0]]) / 10000.0  # B, G, R, NIR order
    return np.clip(rgb * 3.2, 0, 1) ** (1.0 / 1.4)


def read_on_grid(path, transform, shape, crs):
    out = np.zeros((4, *shape), dtype=np.float32)
    with rasterio.open(path) as ds:
        if ds.count not in (4, 8):
            raise ValueError("Expected a four- or eight-band PlanetScope mosaic")
        indices = (1, 2, 3, 4) if ds.count == 4 else (2, 4, 6, 8)
        for b in range(4):
            reproject(rasterio.band(ds, indices[b]), out[b], dst_transform=transform,
                      dst_crs=crs, dst_nodata=0, resampling=Resampling.bilinear)
    return out


def case_candidates(transform, shape, crs, s2_transform):
    m = np.load(CASE / "masks.npz")
    if not np.allclose(m["sentinel2_transform"], s2_transform):
        raise ValueError("case-study masks were made on a different Sentinel-2 grid")
    out = np.zeros(shape, np.uint8)
    reproject(m["planet_rules"].astype(np.uint8), out,
              src_transform=rasterio.Affine(*m["planet_transform"][:6]),
              src_crs=str(m["crs"]), dst_transform=transform, dst_crs=crs,
              dst_nodata=0, resampling=Resampling.nearest)
    return out.astype(bool)


def check_matches_card(key, scene_id, shape):
    """The swipe compares against the published card tile: same scene, same extent."""
    meta = WEB_TILES / f"{key}.json"
    if not meta.exists() or json.loads(meta.read_text()).get("scene_id") != scene_id:
        raise ValueError(f"{key}: the card tile shows a different Sentinel-2 scene")
    with Image.open(WEB_TILES / f"{key}.jpg") as im:
        tw, th = im.size
    h, w = shape
    if abs(tw / th - w / h) > 0.01 * w / h:
        raise ValueError(f"{key}: the card tile extent differs from the frozen grid")


def upsample(mask):
    return mask.repeat(UPSAMPLE, 0).repeat(UPSAMPLE, 1)


def fit(shape):
    h, w = shape
    s = min(1.0, MAXPX / max(h, w))
    return max(1, round(w * s)), max(1, round(h * s))


def resize_mask(mask, size):
    if mask.shape[::-1] == size:
        return mask
    im = Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)
    return np.asarray(im) > 127


def edge(mask, width):
    return mask & ~ndimage.binary_erosion(mask, iterations=width)


def to_image(rgb, valid, size):
    arr = np.where(valid[..., None], rgb, np.array(NODATA_RGB) / 255.0)
    im = Image.fromarray((arr * 255).round().astype(np.uint8))
    if im.size != size:
        im = im.resize(size, Image.LANCZOS)
    return np.asarray(im).astype(np.float32)


def paint(img, mask, rgb, alpha):
    img[mask] = img[mask] * (1 - alpha) + np.array(rgb, np.float32) * alpha


def save(img, path):
    Image.fromarray(img.round().clip(0, 255).astype(np.uint8)).save(
        path, quality=85, optimize=True, progressive=True)


def udm2_clear(folder, item_ids, transform, shape, crs):
    """Planet's UDM2 clear flag on the tile grid, frame by frame in mosaic order."""
    frames = []
    for item in item_ids:
        sr = sorted(folder.glob(f"{item}*AnalyticMS_SR*.tif"))
        udm = sorted(folder.glob(f"{item}*udm2*.tif"))
        if not sr or not udm:
            return None
        footprint = np.zeros(shape, np.uint16)
        clear = np.zeros(shape, np.uint8)
        with rasterio.open(sr[0]) as ds:
            reproject(rasterio.band(ds, 1), footprint, dst_transform=transform, dst_crs=crs,
                      dst_nodata=0, resampling=Resampling.nearest)
        with rasterio.open(udm[0]) as ds:
            reproject(rasterio.band(ds, 1), clear, dst_transform=transform, dst_crs=crs,
                      dst_nodata=0, resampling=Resampling.nearest)
        frames.append((clear == 1, footprint > 0))
    return first_frame_clear(frames) if frames else None


def render(key, rec, mosaic, run, out_dir, case_key, release=False):
    if release:
        inside, gng, meta = exact_scene(key, mosaic.parent)
        transform = meta["transform"]
    else:
        with np.load(run / f"{key}.npz") as z:
            inside, gng, transform = z["inside"], z["prediction_gng"], z["transform"]
        meta = json.loads((run / f"{key}.json").read_text())
    check_matches_card(key, meta["scene_id"], inside.shape)
    t10 = rasterio.Affine(*transform[:6])
    t3 = t10 * rasterio.Affine.scale(1 / UPSAMPLE)
    shape3 = (inside.shape[0] * UPSAMPLE, inside.shape[1] * UPSAMPLE)
    bands = read_on_grid(mosaic, t3, shape3, meta["crs"])
    valid = (bands > 0).all(axis=0)
    cand = None
    if key == case_key and (CASE / "masks.npz").exists():
        cand = case_candidates(t3, shape3, meta["crs"], transform)
    clear_pct = None
    frames = udm2_clear(mosaic.parent, [i["id"] for i in rec["items"]], t3, shape3, meta["crs"])
    if frames is not None:
        clear, covered = frames
        bog = upsample(inside) & valid & covered
        if bog.any():
            clear_pct = round(100 * float((clear & bog).sum() / bog.sum()), 1)

    size = fit(shape3)
    clean = to_image(natural_rgb(bands), valid, size)
    over = clean.copy()
    if cand is not None:
        paint(over, resize_mask(cand & valid, size), PS_FILL, 0.45)
    paint(over, edge(resize_mask(upsample(inside), size), 1), BOG_EDGE, 0.7)
    paint(over, edge(resize_mask(upsample(gng), size), 2), GNG_EDGE, 1.0)
    save(clean, out_dir / f"{key}_psc.jpg")
    save(over, out_dir / f"{key}_ps.jpg")
    return {
        "site": rec["site"], "date": rec["date"], "s2_date": rec["s2_date"],
        "delta_days": rec["delta_days"],
        "instruments": sorted({i["instrument"] for i in rec["items"]}),
        "items": [i["id"] for i in rec["items"]],
        "sentinel2_scene": meta["scene_id"],
        "candidates": cand is not None,
        "size": list(size),
        "harmonized": rec.get("harmonized", False),
        "aoi_area_km2": rec.get("aoi_area_km2"),
        "clear_pct": clear_pct,
        "quality_note": ("Clear share of the bog from Planet's UDM2 mask" if clear_pct is not None
                         else "No UDM2 mask; source-frame clear percentages only"),
    }


def main():
    ap = argparse.ArgumentParser(description="Render PlanetScope tiles for the private web map")
    ap.add_argument("--run", type=Path, default=RUN, help="frozen evaluation run with planet/manifest.json")
    ap.add_argument("--out", type=Path, default=OUT, help="output folder (must stay private)")
    ap.add_argument("--release", type=Path, help="release folder containing manifest.json")
    ap.add_argument("--key", help="render one site-year")
    ap.add_argument("--force", action="store_true", help="re-render site-years already rendered")
    args = ap.parse_args()
    args.out = private_path(args.out)
    source = args.release or args.run / "planet"
    manifest = json.loads((source / "manifest.json").read_text())
    case_key = None
    if (CASE / "summary.json").exists():
        s = json.loads((CASE / "summary.json").read_text())
        case_key = f"{s['site_code']}_{s['scene_date'][:4]}"
    args.out.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    tiles, withheld = previous.get("tiles", {}), previous.get("withheld", {})

    def save_manifest():
        write_json(manifest_path, {"citation": CITATION, "licence": LICENCE,
                                   "tiles": tiles, "withheld": withheld})

    failures = []
    for key, rec in sorted(manifest.items()):
        if args.key and key != args.key:
            continue
        if (key in tiles or key in withheld) and not args.force:
            continue
        # the manifest records repo-relative paths; the mosaic sits in the run
        mosaic = source / key / "ps_mosaic.tif"
        if rec.get("order_state") != "success" or not mosaic.exists():
            print(f"{key}: skipped ({rec.get('status', 'no mosaic')})")
            continue
        try:
            tile = render(key, rec, mosaic, args.run, args.out, case_key, release=bool(args.release))
        except Exception as exc:
            failures.append(key)
            print(f"{key}: render deferred ({type(exc).__name__}: {str(exc)[:180]})", flush=True)
            continue
        tile["source"] = "release" if args.release else "evaluation"
        tiles.pop(key, None)
        withheld.pop(key, None)
        clear = tile["clear_pct"]
        if clear is not None and clear < MIN_CLEAR_PCT:
            # a mostly cloudy frame would mislead in the swipe
            withheld[key] = tile
            print(f"{key}: withheld, {clear}% of the bog clear", flush=True)
        else:
            tiles[key] = tile
            print(f"{key}: {tile['size'][0]}x{tile['size'][1]} px, PlanetScope {tile['date']}"
                  + (f", {clear}% clear" if clear is not None else "")
                  + (" + 3 m candidates" if tile["candidates"] else ""), flush=True)
        save_manifest()
    save_manifest()
    print(f"\n{len(tiles)} site-years -> {args.out} ({len(withheld)} withheld as cloudy)")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
