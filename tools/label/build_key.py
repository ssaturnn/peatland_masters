"""Build the interpretation key shown in the labelling tool's guide.

For each reference class it finds one clear-cut example pixel in the frozen
evaluation scenes and renders it exactly as the tool does (Sentinel-2 true
and false colour close-ups), next to very-high-resolution imagery of the
same spot. Examples are chosen by simple spectral rules as the most central
pixel of a homogeneous patch, at least MIN_GAP_PX away from every sampled
point, so the key never shows a point the annotator will label.

The key is written into the annotation directory (not the repository),
because it embeds third-party imagery.

Usage:
    python3 tools/label/build_key.py outputs/evaluation/<run>
"""

import argparse
import csv
import importlib.util
import io
import json
import math
import ssl
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer
from scipy import ndimage
import certifi

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("label_tool", ROOT / "scripts" / "18_label_tool.py")
lt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lt)

MIN_GAP_PX = 25          # 250 m from any sampled point
CORE_PX = 1              # example must sit in a homogeneous 3 x 3 patch
HALF, SCALE = lt.CHIP_SCOPES["near"]
SIZE = (2 * HALF + 1) * SCALE
VHR_ZOOM = 17
# python.org builds ship without system CA roots; verify against certifi's bundle
SSL_CTX = ssl.create_default_context(cafile=certifi.where())
VHR_URL = ("https://services.arcgisonline.com/ArcGIS/rest/services/"
           "World_Imagery/MapServer/tile/{z}/{y}/{x}")

# (title, scene stem, cue, rule on reflectance indices)
EXAMPLES = [
    # bare-peat examples come from cutting sites outside the evaluation
    # sample (--extra-run), so no sampled point can appear in the key
    ("Bare peat · cut banks", "extra:0",
     "Brown to pale tan strips along the bog margin; beige-brown in false colour, no red.",
     lambda I: (I["ndvi"] < 0.25) & (I["mndwi"] < 0) & (I["blue"] < 0.14) & I["inside"]),
    ("Bare peat · cut banks, another bog", "extra:1",
     "Worked faces and turf strips; brown-grey in false colour, no red.",
     lambda I: (I["ndvi"] < 0.25) & (I["mndwi"] < 0) & (I["blue"] < 0.14) & I["inside"]),
    ("Vegetated · green", "000221_2025",
     "Fields and fresh growth: green in true colour, bright red in false colour.",
     lambda I: I["ndvi"] > 0.75),
    ("Vegetated · brown bog (the tricky one)", "002381_2023",
     "Straw-brown heath and grass: brown in true colour, still pinkish in false colour.",
     lambda I: (I["ndvi"] > 0.42) & (I["ndvi"] < 0.58) & (I["mndwi"] < -0.2) & I["inside"]),
    ("Water", "002364_2018",
     "Dark blue-green to black in true colour, very dark blue in false colour; smooth.",
     lambda I: (I["mndwi"] > 0.2) & (I["nir"] < 0.05)),
    ("Other · sand and tidal mud", "001567_2026",
     "Bright beige sand or grey mud beside the channel; no red.",
     lambda I: (I["blue"] > 0.10) & (I["ndvi"] < 0.15) & (I["mndwi"] < 0.2)),
]


def indices(stack, order, inside):
    band = lambda n: np.nan_to_num(stack[..., order.index(n)] / 10000.0)
    blue, green, red, nir = band("blue"), band("green"), band("red"), band("nir")
    swir1, swir2 = band("swir1"), band("swir2")
    eps = 1e-9
    return {"blue": blue, "nir": nir, "inside": inside.astype(bool),
            "ndvi": (nir - red) / (nir + red + eps),
            "mndwi": (green - swir1) / (green + swir1 + eps),
            "nbr": (nir - swir2) / (nir + swir2 + eps)}


def pick(mask, avoid, shape, core=CORE_PX):
    """Most central pixel of a homogeneous patch, clear of sampled points."""
    pure = (ndimage.binary_erosion(mask, structure=np.ones((2 * core + 1,) * 2, bool))
            if core else mask.copy())
    for r, c in avoid:
        pure[max(r - MIN_GAP_PX, 0):r + MIN_GAP_PX + 1,
             max(c - MIN_GAP_PX, 0):c + MIN_GAP_PX + 1] = False
    pure[:HALF, :] = pure[-HALF:, :] = False
    pure[:, :HALF] = pure[:, -HALF:] = False
    if not pure.any():
        return None
    score = np.where(pure, ndimage.distance_transform_edt(mask), -1.0)
    return tuple(int(v) for v in np.unravel_index(np.argmax(score), shape))


def vhr_chip(lat, lon):
    """High-resolution crop of the same 330 m window, target drawn alike."""
    n = 2 ** VHR_ZOOM
    x = (lon + 180.0) / 360.0 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    tx, ty = int(x), int(y)
    canvas = Image.new("RGB", (768, 768))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            req = urllib.request.Request(VHR_URL.format(z=VHR_ZOOM, x=tx + dx, y=ty + dy),
                                         headers={"User-Agent": "peatland-label-key"})
            with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
                tile = Image.open(io.BytesIO(resp.read())).convert("RGB")
            canvas.paste(tile, ((dx + 1) * 256, (dy + 1) * 256))
    m_per_px = 156543.03392 * math.cos(math.radians(lat)) / n
    half = (SIZE / 2) * 10 / SCALE / m_per_px   # same ground extent as the S2 chip
    px, py = (x - tx + 1) * 256, (y - ty + 1) * 256
    img = canvas.crop((px - half, py - half, px + half, py + half)).resize((SIZE, SIZE), Image.LANCZOS)
    lt._draw_target(img, HALF, SCALE, box=True)
    return img


def s2_chip(stack, order, row, col, kind):
    rgb = lt._composite(lt._window(stack, row, col, HALF), order, kind)
    img = Image.fromarray(rgb).resize((SIZE, SIZE), Image.NEAREST)
    lt._draw_target(img, HALF, SCALE, box=True)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="frozen evaluation run with annotation/")
    ap.add_argument("--extra-run", type=Path,
                    help="frozen scenes of sites outside the sample, for the bare-peat examples")
    args = ap.parse_args()
    extra = sorted(args.extra_run.glob("*.npz")) if args.extra_run else []
    ann = args.run / "annotation"
    sampled = {}
    with (ann / "accuracy_points.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            stem = r["point_id"].rsplit("_r", 1)[0]
            sampled.setdefault(stem, []).append((int(r["pixel_row"]), int(r["pixel_col"])))

    font_t = ImageFont.load_default(size=19)
    font_s = ImageFont.load_default(size=14)
    gap, head, foot = 12, 54, 24
    width = 3 * SIZE + 4 * gap
    rows = []
    for title, stem, cue, rule in EXAMPLES:
        # cut banks are strips one or two pixels wide: no homogeneity core
        core = 0 if stem.startswith("extra:") else CORE_PX
        if stem.startswith("extra:"):
            k = int(stem.split(":")[1])
            if k >= len(extra):
                print(f"  no extra scene #{k} for {title!r}; pass --extra-run")
                continue
            bundle, stem = extra[k], extra[k].stem
        else:
            bundle = args.run / f"{stem}.npz"
        meta = json.loads(bundle.with_suffix(".json").read_text())
        order = meta["bands"]
        with np.load(bundle, allow_pickle=False) as data:
            stack, inside, valid = data["stack"], data["inside"], data["valid"]
        idx = indices(stack, order, inside)
        # only pixels the detector itself accepts as clear, never cloud/shadow
        raw = rule(idx)
        mask = raw & valid.astype(bool)
        if title.startswith("Bare peat"):
            # keep cut-bank examples off lake shores and flooded cuttings:
            # 30 m from water, or 20 m where a bog has nothing further out
            water = (idx["mndwi"] > 0) | (idx["nir"] < 0.05)
            for buf in (7, 5):
                off = mask & ~ndimage.binary_dilation(water, structure=np.ones((buf, buf), bool))
                if off.any():
                    break
            print(f"    {title}: {int(raw.sum())} rule px, {int(mask.sum())} clear, "
                  f"{int(off.sum())} off water")
            mask = off
        where = pick(mask, sampled.get(stem, []), inside.shape, core)
        if where is None:
            print(f"  no clean example for {title!r} in {stem}; skipped")
            continue
        r, c = where
        x, y = rasterio.transform.xy(rasterio.Affine(*meta["transform"][:6]), r, c)
        lon, lat = Transformer.from_crs(meta["crs"], "EPSG:4326", always_xy=True).transform(x, y)
        panel = Image.new("RGB", (width, head + SIZE + foot), (21, 28, 25))
        d = ImageDraw.Draw(panel)
        d.text((gap, 8), title, font=font_t, fill=(229, 236, 232))
        d.text((gap, 32), f"{cue}   ({meta['site']}, {meta['scene_date']})", font=font_s, fill=(160, 178, 170))
        imgs = [(s2_chip(stack, order, r, c, "tc"), "Sentinel-2 true colour"),
                (s2_chip(stack, order, r, c, "fc"), "Sentinel-2 false colour"),
                (vhr_chip(lat, lon), "High-resolution imagery (other date)")]
        for k, (img, cap) in enumerate(imgs):
            x0 = gap + k * (SIZE + gap)
            panel.paste(img, (x0, head))
            d.text((x0, head + SIZE + 4), cap, font=font_s, fill=(143, 163, 154))
        rows.append(panel)
        print(f"  {title:42} {stem} r{r} c{c}  {lat:.5f}, {lon:.5f}")

    note = ("Examples lie at least 250 m from every sampled point. "
            "High-resolution imagery: Esri, Maxar, Earthstar Geographics.")
    key = Image.new("RGB", (width, sum(p.height for p in rows) + 34), (21, 28, 25))
    y0 = 0
    for p in rows:
        key.paste(p, (0, y0)); y0 += p.height
    ImageDraw.Draw(key).text((gap, y0 + 8), note, font=font_s, fill=(143, 163, 154))
    out = ann / "key.png"
    key.save(out, optimize=True)
    print(f"Saved {out} ({len(rows)} examples)")


if __name__ == "__main__":
    main()
