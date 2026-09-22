"""Shared PlanetScope search, clipped orders and local mosaics."""

import datetime as dt
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import rasterio
import requests
from pyproj import Geod, Transformer
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject
from shapely.geometry import Polygon, MultiPolygon, shape
from shapely.ops import transform as transform_geom

DATA = "https://api.planet.com/data/v1"
ORDERS = "https://api.planet.com/compute/ops/orders/v2"
MAX_DAYS = 3
MIN_CLEAR = 50
TARGET_COVER = 0.95
BUNDLES = ("analytic_sr_udm2", "analytic_8b_sr_udm2")
CITATION = ("Planet Team (2026). Planet Application Program Interface: In Space for Life "
            "on Earth. San Francisco, CA. https://api.planet.com")
LICENCE = "Planet Education & Research programme: non-commercial research only"
VERSION = "2026-09-15-planet-release-v1"
ROOT = Path(__file__).resolve().parents[2]

def session():
    key = os.environ.get("PL_API_KEY")
    if not key:
        raise SystemExit("PL_API_KEY is not set")
    s = requests.Session()
    s.auth = (key, "")
    return s


def private_path(path):
    """Planet imagery may only be written under the gitignored outputs/ or web/private/.

    The path is resolved first, so a symlink cannot lead out of those folders.
    """
    path = Path(path).resolve()
    roots = (ROOT / "outputs", ROOT / "web/private")
    if not any(path.is_relative_to(r.resolve()) for r in roots):
        raise ValueError("Planet data must stay inside outputs/ or web/private/")
    return path


def write_json(path, value):
    """Replace a private JSON file atomically."""
    path = private_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def vertex_count(geom):
    """Count all ring coordinates, including closing vertices."""
    polygons = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    return sum(len(p.exterior.coords) + sum(len(r.coords) for r in p.interiors)
               for p in polygons)


def area_km2(geom):
    """Geodesic WGS84 polygon area, with holes subtracted."""
    from shapely.geometry.polygon import orient
    parts = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    geod = Geod(ellps="WGS84")
    return sum(abs(geod.geometry_area_perimeter(orient(p, sign=1))[0])
               for p in parts) / 1e6


def buffered_aoi(geom, crs="EPSG:2157", buffer_m=250):
    """Buffer an Irish site in metres and simplify to fewer than 500 vertices."""
    projected = transform_geom(Transformer.from_crs(crs, "EPSG:2157",
                               always_xy=True).transform, geom)
    buffered = projected.buffer(buffer_m)
    # Fill small interior gaps: they save negligible quota and complicate clips.
    parts = [buffered] if buffered.geom_type == "Polygon" else list(buffered.geoms)
    buffered = MultiPolygon([Polygon(p.exterior) for p in parts]) if len(parts) > 1 else Polygon(parts[0].exterior)
    tolerance = 5.0
    for _ in range(50):
        simplified = buffered.simplify(tolerance, preserve_topology=True)
        if vertex_count(simplified) < 500:
            break
        tolerance *= 1.4
    else:
        raise ValueError("Cannot simplify the clip below 500 vertices")
    aoi = transform_geom(Transformer.from_crs("EPSG:2157", "EPSG:4326",
                         always_xy=True).transform, simplified)
    if aoi.is_empty or not aoi.is_valid:
        raise ValueError("Invalid clip geometry")
    return aoi, tolerance


MONTHLY_KM2 = 3000          # the Education & Research allowance


def reserve_area(ledger, key, area, cap=1300):
    """Reserve full AOI area per item before a potentially ambiguous POST."""
    import math
    if not math.isfinite(area) or area <= 0 or not 0 < cap <= MONTHLY_KM2:
        raise ValueError("Invalid quota reservation")
    if key in ledger["orders"]:
        raise ValueError("Order already reserved; reconcile before retrying")
    used = sum(r["reserved_km2"] for r in ledger["orders"].values())
    if used + area > cap:
        raise ValueError("Clipped-area cap would be exceeded")
    ledger["orders"][key] = {"reserved_km2": area, "status": "submitting"}
    ledger["reserved_km2"] = used + area
    return ledger["orders"][key]


def search(s, aoi, d0, max_days=MAX_DAYS):
    filt = {"type": "AndFilter", "config": [
        {"type": "GeometryFilter", "field_name": "geometry", "config": aoi.__geo_interface__},
        {"type": "DateRangeFilter", "field_name": "acquired", "config": {
            "gte": f"{d0 - dt.timedelta(days=max_days)}T00:00:00Z",
            "lte": f"{d0 + dt.timedelta(days=max_days)}T23:59:59Z"}},
        {"type": "RangeFilter", "field_name": "cloud_cover", "config": {"lte": 0.5}}]}
    r = s.post(f"{DATA}/quick-search", json={"item_types": ["PSScene"], "filter": filt}, timeout=60)
    r.raise_for_status()
    out = []
    features = r.json().get("features", [])
    next_url = r.json().get("_links", {}).get("_next")
    while next_url:
        if urlparse(next_url).hostname != "api.planet.com":
            raise ValueError("Unexpected search pagination host")
        r = s.get(next_url, timeout=60)
        r.raise_for_status()
        features.extend(r.json().get("features", []))
        next_url = r.json().get("_links", {}).get("_next")
    for f in features:
        p = f["properties"]
        perms = " ".join(f.get("_permissions", []))
        day = dt.date.fromisoformat(p["acquired"][:10])
        out.append({"id": f["id"], "acquired": p["acquired"], "date": str(day),
                    "delta": (day - d0).days, "clear": p.get("clear_percent") or 0,
                    "instrument": p.get("instrument"), "geom": shape(f["geometry"]),
                    "download": ("ortho_analytic_4b_sr:download" in perms
                                 or "ortho_analytic_8b_sr:download" in perms)})
    return out


def choose(cands, aoi, min_cover=0.90):
    """Nearest date first; on that date the clearest frames until the site is covered."""
    if aoi.is_empty or aoi.area <= 0 or not 0 < min_cover <= 1:
        raise ValueError("Invalid AOI or coverage threshold")
    for gap in sorted({abs(c["delta"]) for c in cands}):
        usable = [c for c in cands if abs(c["delta"]) == gap and c["download"]
                  and c["clear"] >= MIN_CLEAR]
        best = None
        for date in sorted({c["date"] for c in usable}):
            frames = sorted((c for c in usable if c["date"] == date),
                            key=lambda c: (-c["clear"], -c["geom"].intersection(aoi).area))
            chosen, union = [], None
            for c in frames:
                if c["geom"].intersection(aoi).area == 0:
                    continue
                if union is not None and c["geom"].difference(union).intersection(aoi).area == 0:
                    continue
                chosen.append(c)
                union = c["geom"] if union is None else union.union(c["geom"])
                if union.intersection(aoi).area / aoi.area >= TARGET_COVER:
                    break
            if union is None:
                continue
            cover = union.intersection(aoi).area / aoi.area
            score = (cover >= TARGET_COVER, cover, -len(chosen))
            if best is None or score > best[0]:
                best = (score, date, chosen, cover)
        if best and best[3] >= min_cover:
            return best[1], best[2], best[3]
    return None


def place_order(s, name, ids, aoi):
    if not ids or not aoi.is_valid or aoi.is_empty or vertex_count(aoi) >= 500:
        raise ValueError("Orders require items and a valid clip below 500 vertices")
    last = None
    for bundle in BUNDLES:
        body = {"name": name,
                "products": [{"item_ids": ids, "item_type": "PSScene", "product_bundle": bundle}],
                "tools": [{"clip": {"aoi": json.loads(json.dumps(aoi.__geo_interface__))}}]}
        r = s.post(ORDERS, json=body, timeout=60)
        if r.status_code in (200, 202):
            return r.json()["id"], bundle
        last = r.status_code
        # Only a definite validation rejection permits trying another bundle.
        if r.status_code not in (400, 422):
            break
    raise RuntimeError(f"order rejected for {name}: HTTP {last}")


def wait(s, oid, timeout_s=3600):
    start = time.time()
    while True:
        od = s.get(f"{ORDERS}/{oid}", timeout=60).json()
        if od.get("state") in ("success", "partial", "failed", "cancelled"):
            return od
        if time.time() - start > timeout_s:
            raise TimeoutError(f"order {oid} still {od.get('state')}")
        time.sleep(30)


def _wanted(name, product):
    """Surface-reflectance frames, or their UDM2 quality masks."""
    if not name.endswith(".tif"):
        return False
    if product == "udm2":
        return "_udm2" in name
    return "AnalyticMS_SR" in name and "udm" not in name.lower()


def first_frame_clear(frames):
    """UDM2 clear flags in mosaic order: each pixel comes from the first frame covering it.

    `frames` holds (clear, footprint) boolean arrays on one grid, in the same
    clearest-first order as the reflectance mosaic. Returns (clear, covered).
    """
    clear = covered = None
    for c, f in frames:
        c, f = np.asarray(c, bool), np.asarray(f, bool)
        if clear is None:
            clear, covered = np.zeros(c.shape, bool), np.zeros(c.shape, bool)
        take = f & ~covered
        clear[take] = c[take]
        covered |= f
    if clear is None:
        raise ValueError("No UDM2 frames")
    return clear, covered


def udm2_clear(folder, item_ids, transform, shape, crs):
    """Planet's UDM2 clear flag on a target grid, frame by frame in mosaic order.

    Each pixel is taken from the first frame that covers it, matching how the
    reflectance mosaic was merged. Returns (clear, covered), or None when a
    frame has no UDM2 file next to it.
    """
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


def download(s, od, dest, product="sr"):
    dest = private_path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    files = []
    for res in od.get("_links", {}).get("results", []):
        name = res["name"]
        if _wanted(name, product):
            f = dest / Path(name).name
            if not f.exists():
                # Signed delivery URLs need no account credential.
                with requests.get(res["location"], stream=True, timeout=120) as r:
                    r.raise_for_status()
                    temp = f.with_suffix(".tif.part")
                    with open(temp, "wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                    with rasterio.open(temp) as ds:
                        if ds.count not in (4, 8):
                            raise ValueError("Unexpected Planet band count")
                    temp.replace(f)
            files.append(f)
    return files


def mosaic(files, out):
    """Same-day frames merged clearest-first into one GeoTIFF."""
    out = private_path(out)
    srcs = [rasterio.open(f) for f in files]
    try:
        arr, transform = merge(srcs, method="first", nodata=0)
        profile = srcs[0].profile | {"height": arr.shape[1], "width": arr.shape[2],
                                     "transform": transform, "nodata": 0, "count": arr.shape[0]}
    finally:
        for src in srcs:
            src.close()
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(arr)

