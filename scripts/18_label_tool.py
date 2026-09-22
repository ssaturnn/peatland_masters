"""Step 18 — local labelling tool for the blind accuracy sample.

Serves a keyboard-driven page for labelling the points of a blind
labels.csv (produced by 13_sample_points.py). For every point it shows the
Sentinel-2 acquisition the detectors saw — true colour and NIR false
colour, close-up and context — next to very-high-resolution imagery of the
same spot, and writes each label straight back into labels.csv with an
atomic rewrite, so an interrupted session never corrupts the file.
Detector predictions and strata are never loaded: labelling stays blind.

Usage:
    python3 scripts/18_label_tool.py outputs/evaluation/<run>/annotation
    # the page opens at http://localhost:8765 — Ctrl+C to stop, resume any time
"""

import argparse
import csv
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from peatland.evaluation import TRUTH_CLASSES

UI_FILE = ROOT / "tools" / "label" / "index.html"
BLIND_FIELDS = ["point_id", "site", "year", "scene_date", "lat", "lon",
                "truth", "reference_source", "reference_date", "notes"]
# (half-window in pixels, upscale): close-up 33 x 33 px = 330 m at 10x,
# context 121 x 121 px = 1.2 km at 3x. Nearest-neighbour upscaling keeps
# the real 10 m pixels visible, which is what is being labelled.
CHIP_SCOPES = {"near": (16, 10), "far": (60, 3)}
COMPOSITES = {  # bands, gains, gamma
    "tc": (("red", "green", "blue"), (3.2, 3.2, 3.2), 1.4),   # as the map tiles
    "fc": (("nir", "red", "green"), (2.2, 3.2, 3.2), 1.2),    # vegetation shows red
}
TARGET = (253, 224, 71)
POINT_ID = re.compile(r"^(?P<stem>.+_\d{4})_r(?P<row>\d+)_c(?P<col>\d+)$")


def _window(arr, row, col, half):
    """Square window centred on (row, col); cells beyond the array are NaN/False."""
    size = 2 * half + 1
    fill = np.nan if arr.dtype.kind == "f" else False
    out = np.full((size, size) + arr.shape[2:], fill, dtype=arr.dtype)
    r0, c0 = row - half, col - half
    sr0, sr1 = max(r0, 0), min(r0 + size, arr.shape[0])
    sc0, sc1 = max(c0, 0), min(c0 + size, arr.shape[1])
    if sr0 < sr1 and sc0 < sc1:
        out[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = arr[sr0:sr1, sc0:sc1]
    return out


def _composite(win, order, kind):
    bands, gains, gamma = COMPOSITES[kind]
    rgb = np.dstack([win[..., order.index(b)] / 10000.0 * g
                     for b, g in zip(bands, gains)])
    rgb = np.clip(np.nan_to_num(rgb, nan=0.0), 0, 1) ** (1.0 / gamma)
    rgb[~np.isfinite(win).all(axis=2)] = 0.13
    return (rgb * 255).astype(np.uint8)


def _outline(inside_win):
    """Edge pixels of the assessed area (bog minus excluded water)."""
    m = inside_win.astype(bool)
    edge = np.zeros_like(m)
    edge[1:, :] |= m[1:, :] != m[:-1, :]
    edge[:-1, :] |= m[1:, :] != m[:-1, :]
    edge[:, 1:] |= m[:, 1:] != m[:, :-1]
    edge[:, :-1] |= m[:, 1:] != m[:, :-1]
    return edge & m


def _draw_target(img, half, scale, box):
    d = ImageDraw.Draw(img)
    c0, c1 = half * scale, (half + 1) * scale
    mid = (c0 + c1 - 1) / 2
    if box:  # frame the exact 10 m pixel, leaving the pixel itself visible
        d.rectangle([c0 - 3, c0 - 3, c1 + 2, c1 + 2], outline=(0, 0, 0), width=1)
        d.rectangle([c0 - 2, c0 - 2, c1 + 1, c1 + 1], outline=TARGET, width=2)
    gap, length = (16, 26) if box else (9, 20)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        a = (mid + dx * gap, mid + dy * gap)
        b = (mid + dx * (gap + length), mid + dy * (gap + length))
        d.line([a, b], fill=(0, 0, 0), width=4)
        d.line([a, b], fill=TARGET, width=2)


def chip_name(point_id, kind, scope):
    return f"{point_id}_{kind}_{scope}.png"


def render_chips(rows, run_dir, chips_dir):
    """Pre-render the four chips per point once; later starts reuse them."""
    chips_dir.mkdir(exist_ok=True)
    todo = {}
    for r in rows:
        m = POINT_ID.match(r["point_id"])
        if not m:
            raise ValueError(f"Unexpected point_id format: {r['point_id']!r}")
        names = [chip_name(r["point_id"], k, s) for k in COMPOSITES for s in CHIP_SCOPES]
        if not all((chips_dir / n).exists() for n in names):
            todo.setdefault(m["stem"], []).append(
                (r["point_id"], int(m["row"]), int(m["col"])))
    made = 0
    for stem, points in todo.items():
        bundle = run_dir / f"{stem}.npz"
        if not bundle.exists():
            raise FileNotFoundError(f"Frozen scene {bundle} not found; pass --run")
        order = json.loads(bundle.with_suffix(".json").read_text())["bands"]
        with np.load(bundle, allow_pickle=False) as data:
            stack, inside = data["stack"], data["inside"]
        for pid, row, col in points:
            for scope, (half, scale) in CHIP_SCOPES.items():
                win = _window(stack, row, col, half)
                edge = _outline(_window(inside, row, col, half)) if scope == "far" else None
                for kind in COMPOSITES:
                    rgb = _composite(win, order, kind)
                    if edge is not None:
                        rgb[edge] = (rgb[edge] * 0.35 + 255 * 0.65).astype(np.uint8)
                    size = (2 * half + 1) * scale
                    img = Image.fromarray(rgb).resize((size, size), Image.NEAREST)
                    _draw_target(img, half, scale, box=(scope == "near"))
                    img.save(chips_dir / chip_name(pid, kind, scope), optimize=True)
                    made += 1
        print(f"  chips for {stem}: {len(points)} points", flush=True)
    return made


def planet_dates(run_dir):
    """PlanetScope acquisition per site-year, for the 3 m chip caption."""
    manifest = run_dir / "planet" / "manifest.json"
    if not manifest.exists():
        return {}
    return {key: {"date": rec["date"], "delta_days": rec.get("delta_days")}
            for key, rec in json.loads(manifest.read_text()).items() if rec.get("date")}


class LabelStore:
    """labels.csv as the single source of truth, rewritten atomically."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            self.fields = list(reader.fieldnames or [])
            self.rows = list(reader)
        leaks = [k for k in self.fields if k.startswith("prediction_")
                 or k in ("stratum", "stratum_population", "stratum_sample_size")]
        if leaks:
            raise ValueError(f"{path} exposes {leaks}: open the blind labels.csv instead")
        missing = [k for k in BLIND_FIELDS if k not in self.fields]
        if missing:
            raise ValueError(f"{path} lacks columns {missing}")
        self.index = {r["point_id"]: r for r in self.rows}

    def labelled(self):
        return sum(1 for r in self.rows if r.get("truth", "").strip())

    def snapshot(self):
        return [{k: r.get(k, "") for k in BLIND_FIELDS} for r in self.rows]

    def update(self, data):
        row = self.index.get(str(data.get("point_id", "")))
        if row is None:
            raise ValueError("Unknown point_id")
        truth = str(data.get("truth", "")).strip().lower()
        if truth and truth not in TRUTH_CLASSES:
            raise ValueError(f"Unknown class {truth!r}")
        with self.lock:
            row["truth"] = truth
            for key in ("reference_source", "reference_date", "notes"):
                row[key] = " ".join(str(data.get(key, "")).split())[:500] if truth else ""
            if not truth:
                row["notes"] = " ".join(str(data.get("notes", "")).split())[:500]
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()
                writer.writerows(self.rows)
            os.replace(tmp, self.path)
            return {"ok": True, "labelled": self.labelled(), "total": len(self.rows)}


def make_handler(store, chips_dir, info):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype, cache="no-store"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            path = unquote(self.path.split("?", 1)[0])
            if path in ("/", "/index.html"):
                return self._send(200, UI_FILE.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/points":
                return self._json(200, {"points": store.snapshot(), **info})
            if path.startswith("/chips/"):
                f = chips_dir / Path(path).name
                if f.suffix == ".png" and f.is_file():
                    return self._send(200, f.read_bytes(), "image/png", "max-age=86400")
            if path == "/key.png":
                f = chips_dir.parent / "key.png"
                if f.is_file():
                    return self._send(200, f.read_bytes(), "image/png")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/label":
                return self._send(404, b"not found", "text/plain")
            try:
                length = int(self.headers.get("Content-Length", 0))
                result = store.update(json.loads(self.rfile.read(length) or b"{}"))
            except (ValueError, json.JSONDecodeError) as exc:
                return self._json(400, {"error": str(exc)})
            self._json(200, result)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("annotation", type=Path,
                    help="annotation directory holding the blind labels.csv")
    ap.add_argument("--run", type=Path,
                    help="frozen scene run with the .npz bundles (default: from manifest.json)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    labels = args.annotation / "labels.csv"
    if not labels.exists():
        ap.error(f"{labels} not found")
    run_dir = args.run
    if run_dir is None:
        manifest = args.annotation / "manifest.json"
        if not manifest.exists():
            ap.error("no manifest.json; pass --run")
        run_dir = Path(json.loads(manifest.read_text())["source_run"])
    store = LabelStore(labels)
    backup = labels.with_name(f"labels.backup-{datetime.now():%Y%m%d-%H%M%S}.csv")
    shutil.copy2(labels, backup)

    chips_dir = args.annotation / "chips"
    print("Preparing image chips...", flush=True)
    made = render_chips(store.rows, run_dir, chips_dir)
    print(f"  {made} new chips" if made else "  chips up to date")

    try:
        shown = args.annotation.resolve().relative_to(ROOT)
    except ValueError:
        shown = args.annotation.resolve()
    info = {"score_cmd": f"python3 scripts/14_score_points.py {shown}/accuracy_points.csv "
                         f"--labels {shown}/labels.csv --output {shown}/accuracy.json",
            "planet": planet_dates(run_dir)}
    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                 make_handler(store, chips_dir, info))
    url = f"http://localhost:{args.port}"
    print(f"\n{store.labelled()}/{len(store.rows)} labelled · open {url}  (Ctrl+C to stop)")
    print(f"Backup of labels.csv at start: {backup.name}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nStopped. {store.labelled()}/{len(store.rows)} labelled; "
              f"everything is saved in {labels}")


if __name__ == "__main__":
    main()
