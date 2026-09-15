"""Step 20 — date-matched PlanetScope reference imagery for the labelled sample.

For every frozen site-year of an evaluation run it searches PlanetScope
(3 m) acquisitions within +-3 days of the Sentinel-2 date and picks the
nearest-in-time clear frame, or a same-day mosaic when one frame does not
cover the site. It orders 4-band surface reflectance clipped to the site
window through the Planet Orders API, downloads it, and renders a 3 m
true- and false-colour close-up for every sampled point, drawn like the
Sentinel-2 close-ups so the labelling tool can show them side by side.

Unlike the very-high-resolution basemaps, these frames are from the same
week as the Sentinel-2 acquisition, so they show the state on the date.
A manifest records item ids, acquisition times and coverage.

Planet imagery is licensed for non-commercial research only: it stays in
outputs/ (not in the repository or on the public web map). Cite as
Planet Team (2026). Planet Application Program Interface: In Space for
Life on Earth. San Francisco, CA. https://api.planet.com

Usage (PL_API_KEY must be set in the environment):
    python3 scripts/20_planet_reference.py outputs/evaluation/<run>
"""

import argparse
import csv
import datetime as dt
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import rasterio
import requests
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio.windows import Window
from shapely.geometry import box, shape

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from peatland.planet import (session, search, choose, place_order, wait,
                             download, mosaic, CITATION, MAX_DAYS)

HALF, SCALE = 55, 3      # 111 x 111 px at 3 m (~330 m) drawn at 3x, like the S2 close-up
TARGET = (253, 224, 71)
def site_window(npz):
    """Metadata and WGS84 bounding box of one frozen Sentinel-2 site window."""
    meta = json.loads(Path(npz).with_suffix(".json").read_text())
    with np.load(npz) as data:
        h, w = data["inside"].shape
    west, south, east, north = array_bounds(h, w, rasterio.Affine(*meta["transform"][:6]))
    xs, ys = Transformer.from_crs(meta["crs"], "EPSG:4326", always_xy=True).transform(
        [west, east, west, east], [south, south, north, north])
    return meta, box(min(xs), min(ys), max(xs), max(ys))


def _target(img):
    """10 m square and ticks at the centre, matching the Sentinel-2 close-ups."""
    d = ImageDraw.Draw(img)
    ctr = (HALF + 0.5) * SCALE
    half = 10 / 3 * SCALE / 2
    d.rectangle([ctr - half - 3, ctr - half - 3, ctr + half + 2, ctr + half + 2], outline=(0, 0, 0), width=1)
    d.rectangle([ctr - half - 2, ctr - half - 2, ctr + half + 1, ctr + half + 1], outline=TARGET, width=2)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        a = (ctr + dx * 16, ctr + dy * 16)
        b = (ctr + dx * 42, ctr + dy * 42)
        d.line([a, b], fill=(0, 0, 0), width=4)
        d.line([a, b], fill=TARGET, width=2)


def chips(mosaic_path, points, chips_dir, label):
    font = ImageFont.load_default(size=13)
    made = 0
    with rasterio.open(mosaic_path) as ds:
        order = {"blue": 1, "green": 2, "red": 3, "nir": 4} if ds.count == 4 else \
                {"blue": 2, "green": 4, "red": 6, "nir": 8}
        to_ds = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        for p in points:
            x, y = to_ds.transform(float(p["lon"]), float(p["lat"]))
            row, col = ds.index(x, y)
            win = Window(col - HALF, row - HALF, 2 * HALF + 1, 2 * HALF + 1)
            arr = ds.read([order[b] for b in ("blue", "green", "red", "nir")], window=win,
                          boundless=True, fill_value=0).astype("float32") / 10000.0
            empty = (arr == 0).all(axis=0)
            if empty.mean() > 0.5:
                continue                      # point outside the PlanetScope footprint
            blue, green, red, nir = arr
            for kind, rgb, gamma in (("tc", np.dstack([red, green, blue]) * 3.2, 1.4),
                                     ("fc", np.dstack([nir * 2.2, red * 3.2, green * 3.2]), 1.2)):
                img = np.clip(rgb, 0, 1) ** (1.0 / gamma)
                img[empty] = 0.13
                size = (2 * HALF + 1) * SCALE
                im = Image.fromarray((img * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
                _target(im)
                d = ImageDraw.Draw(im)
                d.rectangle([0, 0, size, 18], fill=(0, 0, 0))
                d.text((4, 2), label, font=font, fill=(240, 240, 240))
                im.save(chips_dir / f"{p['point_id']}_ps_{kind}.png", optimize=True)
            made += 1
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="frozen evaluation run with annotation/")
    ap.add_argument("--out", type=Path, help="download directory (default: <run>/planet)")
    args = ap.parse_args()
    s = session()
    out = args.out or args.run / "planet"
    out.mkdir(parents=True, exist_ok=True)
    ann = args.run / "annotation"
    rows = list(csv.DictReader(open(ann / "labels.csv", newline="", encoding="utf-8-sig")))
    manifest_f = out / "manifest.json"
    manifest = json.loads(manifest_f.read_text()) if manifest_f.exists() else {}

    pending = {}
    for npz in sorted(glob.glob(str(args.run / "*.npz"))):
        stem = Path(npz).stem
        if manifest.get(stem, {}).get("mosaic") and Path(manifest[stem]["mosaic"]).exists():
            continue
        meta, aoi = site_window(npz)
        d0 = dt.date.fromisoformat(meta["scene_date"])
        # the evaluation reference accepted 50% site cover (6 of 8 site-years)
        choice = choose(search(s, aoi, d0), aoi, min_cover=0.5)
        if choice is None:
            manifest[stem] = {"site": meta["site"], "s2_date": str(d0),
                              "status": f"no clear downloadable PlanetScope within ±{MAX_DAYS} days"}
            print(f"  {stem} {meta['site'][:28]:28} no usable PlanetScope")
            continue
        date, frames, cover = choice
        oid, bundle = place_order(s, f"peatland-reference-{stem}", [c["id"] for c in frames], aoi)
        pending[stem] = (oid, bundle, date, frames, cover, meta, d0)
        print(f"  {stem} {meta['site'][:28]:28} ordered {len(frames)} frame(s) of {date} "
              f"(cover {cover:.0%}) · order {oid}", flush=True)

    for stem, (oid, bundle, date, frames, cover, meta, d0) in pending.items():
        od = wait(s, oid)
        files = download(s, od, out / stem) if od.get("state") in ("success", "partial") else []
        ids = [c["id"] for c in frames]
        files.sort(key=lambda f: next((k for k, i in enumerate(ids) if f.name.startswith(i)), 99))
        entry = {"site": meta["site"], "s2_date": str(d0), "date": date,
                 "delta_days": (dt.date.fromisoformat(date) - d0).days, "order_id": oid,
                 "order_state": od.get("state"), "bundle": bundle, "cover": round(cover, 3),
                 "items": [{"id": c["id"], "acquired": c["acquired"], "clear_percent": c["clear"],
                            "instrument": c["instrument"]} for c in frames],
                 "citation": CITATION}
        if files:
            mos = out / stem / "ps_mosaic.tif"
            mosaic(files, mos)
            entry["mosaic"] = str(mos)
        manifest[stem] = entry
        manifest_f.write_text(json.dumps(manifest, indent=2))
        print(f"  {stem} order {od.get('state')}: {len(files)} file(s)", flush=True)

    manifest_f.write_text(json.dumps(manifest, indent=2))
    total = 0
    for stem, m in manifest.items():
        if not m.get("mosaic") or not Path(m["mosaic"]).exists():
            continue
        delta = m["delta_days"]
        when = "same day" if delta == 0 else f"{delta:+d} day" + ("s" if abs(delta) > 1 else "")
        pts = [r for r in rows if r["point_id"].startswith(stem + "_")]
        n = chips(m["mosaic"], pts, ann / "chips", f"PlanetScope 3 m · {m['date']} · {when}")
        total += n
        print(f"  {stem} {m['site'][:28]:28} {n}/{len(pts)} points with a PlanetScope close-up")
    print(f"Done: {total} points have date-matched 3 m reference imagery. Manifest: {manifest_f}")


if __name__ == "__main__":
    main()
