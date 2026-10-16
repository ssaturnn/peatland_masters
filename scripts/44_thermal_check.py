"""Step 44 — are the exposed fields warm because they smoulder, or because they are bare?

A smouldering (underground) peat fire heats the surface above it. Bare dry peat
is also warmer than vegetation on a sunny day, simply because it is dark and
does not transpire. This script compares Landsat 8/9 surface temperature
(Collection 2 Level-2 ST_B10, `lwir11`) on clear summer days 2017-2025 for:

- the worked fields of Castle Ffrench East (001244): pixels flagged v5 exposed
  peat on at least 40% of the clear April-September scenes of 2017-2023
  (about 47 ha, from scripts/38_time_series.py);
- the intact bog of the same site: pixels never flagged in those years;
- Bord na Móna Mountdillon production bogs (BNMS, BNMW from
  scripts/32_positive_control.py): v5 exposed peat on the clearest summer
  scene of 2018, against vegetation (NDVI >= 0.6) in the same rectangles.
  These fields are known to be worked or abandoned, not burning.

Only Landsat scenes clear (QA_PIXEL) over at least 80% of both the Castle
site and a Mountdillon area are kept, so the two sites are compared on the
same overpass. A smouldering front would also show as a hot outlier inside the
fields, so the 99th percentile minus the median of the fields is recorded.

Usage:
    python3 scripts/44_thermal_check.py              # download and compute
    python3 scripts/44_thermal_check.py --plot-only  # redraw the figure from the json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import planetary_computer as pc
import rasterio
from pystac_client import Client
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from peatland import config, v5

TS = config.OUT_DIR / "time-series"
PC_DIR = config.OUT_DIR / "positive-control"
OUT = config.OUT_DIR / "evaluation/thermal-check.json"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
CLOUD_BITS = (1 << 1) | (1 << 3) | (1 << 4)   # dilated cloud, cloud, cloud shadow
ST_SCALE, ST_OFFSET = 0.00341802, 149.0
FIELD_FREQ = 0.4
MIN_CLEAR = 0.8
YEARS = range(2017, 2026)


def read(item, asset, grid, nearest=False):
    shape, transform, crs, bbox = grid
    with rasterio.open(item.assets[asset].href) as src:
        win = from_bounds(*transform_bounds("EPSG:4326", src.crs, *bbox), src.transform)
        arr = src.read(1, window=win, boundless=True, fill_value=0)
        out = np.zeros(shape, np.float32)
        reproject(arr.astype(np.float32), out, src_transform=src.window_transform(win), src_crs=src.crs,
                  dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.nearest if nearest else Resampling.bilinear)
    return out


def bbox_of(shape, transform, crs):
    h, w = shape
    x0, y0 = transform * (0, 0)
    x1, y1 = transform * (w, h)
    return transform_bounds(crs, "EPSG:4326", min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def castle_grid():
    z = np.load(TS / "001244_2017_2026.npz")
    meta = json.loads((TS / "001244_2017_2026.json").read_text())
    dates = z["dates"]
    keep = np.array([4 <= int(d[5:7]) <= 9 and int(d[:4]) <= 2023 for d in dates])
    valid, peat = z["valid"][keep], z["peat"][keep]
    freq = (peat & valid).sum(0) / np.maximum(valid.sum(0), 1)
    inside = z["inside"]
    fields = inside & (freq >= FIELD_FREQ)
    intact = inside & (freq == 0) & (valid.sum(0) >= 20)
    transform = rasterio.Affine(*z["transform"][:6])
    shape = inside.shape
    return {"grid": (shape, transform, meta["crs"], bbox_of(shape, transform, meta["crs"])),
            "classes": {"fields": fields, "intact_bog": intact}, "inside": inside}


def bnm_grid(area):
    meta = json.loads((PC_DIR / "areas.json").read_text())["runs"][f"{area}-su_2018"]
    z = np.load(PC_DIR / "scenes" / f"{area}-su_2018.npz")
    stack, valid = z["stack"], z["valid"]
    inside = np.ones(valid.shape, bool)
    peat = v5.peat_mask(stack, valid, inside, tuple(meta["bands"]))
    red, nir = stack[..., meta["bands"].index("red")], stack[..., meta["bands"].index("nir")]
    ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
    veg = valid & (ndvi >= 0.6) & ~peat
    transform = rasterio.Affine(*meta["transform"][:6])
    return {"grid": (valid.shape, transform, meta["crs"], bbox_of(valid.shape, transform, meta["crs"])),
            "classes": {"bare_peat": peat, "vegetation": veg}, "inside": inside}


def stats(lst, clear, mask):
    v = lst[clear & mask]
    if v.size < 20:
        return None
    return {"median_c": round(float(np.median(v)), 2), "p99_c": round(float(np.percentile(v, 99)), 2),
            "max_c": round(float(v.max()), 2), "n": int(v.size)}


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pairs = out["pairs"]
    x = np.arange(len(pairs))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), dpi=130, sharex=True)
    w = 0.38
    axes[0].bar(x - w / 2, [p["castle_fields_minus_intact"] for p in pairs], w, color="#b5651d",
                label="Castle Ffrench East: worked fields minus intact bog")
    axes[0].bar(x + w / 2, [p["bnm_peat_minus_veg"] for p in pairs], w, color="#7a7a7a",
                label="Bord na Móna: exposed peat minus vegetation")
    axes[0].set_ylabel("surface temperature\ndifference, K"); axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(axis="y", alpha=.3)
    axes[1].bar(x - w / 2, [p["castle_fields_p99_minus_median"] for p in pairs], w, color="#b5651d",
                label="Castle Ffrench East fields")
    axes[1].bar(x + w / 2, [p["bnm_peat_p99_minus_median"] for p in pairs], w, color="#7a7a7a",
                label="Bord na Móna exposed peat")
    axes[1].set_ylabel("hottest 1% minus\nmedian, K"); axes[1].legend(fontsize=8, loc="upper left")
    axes[1].grid(axis="y", alpha=.3)
    axes[1].set_xticks(x, [p["date"] for p in pairs], rotation=60, fontsize=8)
    axes[0].set_title("Landsat 8/9 surface temperature on the same overpass (clear summer days)")
    fig.tight_layout(); fig.savefig(OUT.with_suffix(".png")); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if args.plot_only:
        plot(json.loads(OUT.read_text()))
        return
    sites = {"castle": castle_grid(), "BNMS": bnm_grid("BNMS"), "BNMW": bnm_grid("BNMW")}
    for k, s in sites.items():
        print(k, {c: round(m.sum() / 100, 1) for c, m in s["classes"].items()}, "ha")
    catalog = Client.open(STAC, modifier=pc.sign_inplace)
    lon = [sites["castle"]["grid"][3][0], sites["BNMW"]["grid"][3][2]]
    lat = [sites["castle"]["grid"][3][1], sites["BNMW"]["grid"][3][3]]
    bbox = [min(lon), min(lat), max(lon), max(lat)]
    rows = []
    for year in YEARS:
        items = list(catalog.search(collections=["landsat-c2-l2"], bbox=bbox,
                                    datetime=f"{year}-05-01/{year}-09-30",
                                    query={"eo:cloud_cover": {"lt": 50},
                                           "platform": {"in": ["landsat-8", "landsat-9"]}}).items())
        for it in sorted(items, key=lambda i: i.properties["datetime"]):
            row = {"date": it.properties["datetime"][:10], "platform": it.properties["platform"],
                   "scene": it.id, "sites": {}}
            for name, s in sites.items():
                try:
                    qa = read(it, "qa_pixel", s["grid"], nearest=True).astype(np.uint32)
                except Exception:
                    continue
                clear = (qa > 1) & ((qa & CLOUD_BITS) == 0)
                frac = float((clear & s["inside"]).sum() / s["inside"].sum())
                if frac < MIN_CLEAR:
                    continue
                st = read(it, "lwir11", s["grid"])
                lst = np.where(st > 0, st * ST_SCALE + ST_OFFSET - 273.15, np.nan)
                clear &= np.isfinite(lst)
                row["sites"][name] = {"clear": round(frac, 3),
                                      **{c: stats(lst, clear, m) for c, m in s["classes"].items()}}
            if "castle" in row["sites"] and ({"BNMS", "BNMW"} & set(row["sites"])):
                rows.append(row)
                c = row["sites"]["castle"]
                b = row["sites"].get("BNMS") or row["sites"]["BNMW"]
                if c["fields"] and c["intact_bog"] and b["bare_peat"] and b["vegetation"]:
                    print(row["date"], row["platform"],
                          f"castle fields {c['fields']['median_c']:.1f} intact {c['intact_bog']['median_c']:.1f} | "
                          f"BnM peat {b['bare_peat']['median_c']:.1f} veg {b['vegetation']['median_c']:.1f}", flush=True)

    pairs = []
    for r in rows:
        c = r["sites"]["castle"]
        for a in ("BNMS", "BNMW"):
            b = r["sites"].get(a)
            if b and c["fields"] and c["intact_bog"] and b["bare_peat"] and b["vegetation"]:
                pairs.append({"date": r["date"], "area": a,
                              "castle_fields_minus_intact": round(c["fields"]["median_c"] - c["intact_bog"]["median_c"], 2),
                              "bnm_peat_minus_veg": round(b["bare_peat"]["median_c"] - b["vegetation"]["median_c"], 2),
                              "castle_fields_minus_bnm_peat": round(c["fields"]["median_c"] - b["bare_peat"]["median_c"], 2),
                              "castle_fields_p99_minus_median": round(c["fields"]["p99_c"] - c["fields"]["median_c"], 2),
                              "bnm_peat_p99_minus_median": round(b["bare_peat"]["p99_c"] - b["bare_peat"]["median_c"], 2),
                              "castle_fields_max_c": c["fields"]["max_c"]})
                break
    summary = {}
    for k in ("castle_fields_minus_intact", "bnm_peat_minus_veg", "castle_fields_minus_bnm_peat",
              "castle_fields_p99_minus_median", "bnm_peat_p99_minus_median", "castle_fields_max_c"):
        v = np.array([p[k] for p in pairs])
        if v.size:
            summary[k] = {"median": round(float(np.median(v)), 2), "min": round(float(v.min()), 2),
                          "max": round(float(v.max()), 2)}
    out = {"field_freq": FIELD_FREQ, "min_clear": MIN_CLEAR, "st_scale": ST_SCALE, "st_offset": ST_OFFSET,
           "n_paired_dates": len(pairs), "summary": summary, "pairs": pairs, "scenes": rows}
    OUT.write_text(json.dumps(out, indent=2))
    plot(out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("scenes",)}, indent=2))


if __name__ == "__main__":
    main()
