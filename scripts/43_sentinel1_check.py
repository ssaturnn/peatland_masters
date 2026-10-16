"""Step 43 — does Sentinel-1 radar add anything to the SWIR rule?

Sentinel-1 C-band SAR sees through cloud and responds to surface roughness and
moisture, not to reflected sunlight. The script reads the radiometrically
terrain-corrected backscatter (Planetary Computer collection sentinel-1-rtc,
gamma0, VV and VH) on the bog's 10 m release grid and stores dB values per
date. To keep the incidence angle fixed, the time series uses one relative
orbit (the most frequent over the bog unless --orbit is given), thinned to at
most one acquisition per --step days; the acquisition count over all orbits is
kept for the cloud-gap comparison.

Two analyses:

- fields (a bog with outputs/time-series/{code}_fields.json and a dense
  Sentinel-2 bundle): backscatter of the worked fields against intact bog,
  per-year separability (AUC of per-pixel seasonal medians), relation to v5
  field exposure on near-coincident Sentinel-2 dates and to ERA5 rainfall, and
  usable acquisitions per April-September season against clear Sentinel-2 days;
- fire (a dense Sentinel-2 bundle and a FIRMS date): the burn mapped by dNBR as
  in scripts/40_fire_signal.py, backscatter of burnt against unburnt bog
  through time.

Usage:
    python3 scripts/43_sentinel1_check.py fields outputs/time-series/001244_2017_2026.npz
    python3 scripts/43_sentinel1_check.py fire outputs/time-series/002381_2018_2021.npz 2019-04-22
"""

import argparse
import datetime as dt
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer as pc
import rasterio
import requests
from pystac_client import Client
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds
from scipy.ndimage import binary_dilation
from scipy.stats import mannwhitneyu, spearmanr

# a stalled range request must fail and be retried, not hang the run
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
RAIN_API = "https://archive-api.open-meteo.com/v1/archive"
STEP_DAYS = 6
MIN_COVER = 0.8      # share of the bog a date's slices must cover
FIELD_SHARE = 0.3       # as for {code}_fields.json: flagged on >= 30% of clear May-August scenes
INTACT_SHARE = 0.02     # intact bog: flagged on < 2% of them
INTACT_GAP_PX = 3       # and at least 30 m from the fields
DRY_MM, WET_MM = 2.0, 5.0
PRE_DAYS, POST_DAYS, DNBR = 270, 45, 0.15   # as scripts/40_fire_signal.py


def grid_of(bundle):
    z = np.load(bundle)
    meta = json.loads(bundle.with_suffix(".json").read_text())
    transform = rasterio.Affine(*z["transform"][:6])
    return z, meta, transform, meta["crs"]


def search(shape, transform, crs, start, end):
    h, w = shape
    left, top = transform.c, transform.f
    right, bottom = left + w * transform.a, top + h * transform.e
    bbox = transform_bounds(crs, "EPSG:4326", left, bottom, right, top)
    catalog = Client.open(STAC)
    items = list(catalog.search(collections=["sentinel-1-rtc"], bbox=bbox,
                                datetime=f"{start}/{end}").items())
    return items, (left, bottom, right, top)


def thin(items, orbit, step):
    """Acquisition dates of one orbit, at most one per `step` days; each date
    keeps all its slices, since a bog can sit across a slice boundary."""
    by_date = {}
    for it in items:
        if it.properties["sat:relative_orbit"] == orbit:
            by_date.setdefault(it.properties["datetime"][:10], []).append(it)
    out, last = [], None
    for day in sorted(by_date):
        d = dt.date.fromisoformat(day)
        if last is None or (d - last).days >= step:
            out.append((day, by_date[day]))
            last = d
    return out


def read_db(item, bounds, shape, transform, crs):
    out = {}
    for pol in ("vv", "vh"):
        href = pc.sign(item.assets[pol].href)
        with rasterio.open(href) as src:
            win = from_bounds(*transform_bounds(crs, src.crs, *bounds), src.transform).round_offsets().round_lengths()
            arr = src.read(1, window=win, boundless=True, fill_value=np.nan).astype(np.float32)
            arr[~np.isfinite(arr) | (arr <= 1e-4)] = np.nan   # nodata and swath-edge border noise
            dst = np.full(shape, np.nan, np.float32)
            reproject(arr, dst, src_transform=src.window_transform(win), src_crs=src.crs, src_nodata=np.nan,
                      dst_transform=transform, dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
        with np.errstate(all="ignore"):
            out[pol] = (10 * np.log10(dst)).astype(np.float16)
    return out


def fetch(bundle, start, end, orbit, step):
    """Backscatter stack on the bundle's grid, cached next to the bundle."""
    z, meta, transform, crs = grid_of(bundle)
    shape = z["inside"].shape
    cache = bundle.with_name(f"{meta['code']}_s1_{start[:4]}_{end[:4]}.npz")
    if cache.exists():
        c = np.load(cache, allow_pickle=True)
        return {k: c[k] for k in c.files}, cache
    items, bounds = search(shape, transform, crs, start, end)
    orbits = Counter(it.properties["sat:relative_orbit"] for it in items)
    orbit = orbit or orbits.most_common(1)[0][0]
    chosen = thin(items, orbit, step)
    print(f"{len(items)} acquisitions over {dict(orbits)}; orbit {orbit}, {len(chosen)} after thinning", flush=True)

    done = [0]

    def one(group):
        day, slices = group
        merged = None
        for it in slices:
            r = None
            for attempt in range(4):   # transient network and DNS failures
                try:
                    r = read_db(it, bounds, shape, transform, crs)
                    break
                except Exception as e:
                    err = e
                    time.sleep(10 * (attempt + 1))
            if r is None:
                print("skip", it.id, repr(err)[:120], flush=True)
                continue
            if merged is None:
                merged = r
            else:
                for pol in ("vv", "vh"):
                    merged[pol] = np.where(np.isfinite(merged[pol]), merged[pol], r[pol])
        done[0] += 1
        if done[0] % 50 == 0:
            print(f"{done[0]}/{len(chosen)} dates read", flush=True)
        inside = z["inside"]
        if merged is None or np.isfinite(merged["vv"][inside].astype(np.float32)).mean() < MIN_COVER:
            return day, None
        return day, merged

    with ThreadPoolExecutor(8) as ex:
        res = [r for r in ex.map(one, chosen) if r[1] is not None]
    print(f"{len(res)} dates cover >= {MIN_COVER:.0%} of the bog", flush=True)
    all_dates = sorted({it.properties["datetime"][:10] for it in items})
    out = {"dates": np.array([day for day, _ in res]),
           "vv": np.stack([r["vv"] for _, r in res]), "vh": np.stack([r["vh"] for _, r in res]),
           "orbit": np.array(orbit), "pass": np.array(chosen[0][1][0].properties["sat:orbit_state"]),
           "all_dates": np.array(all_dates)}
    np.savez_compressed(cache, **out)
    return out, cache


def med(a, m):
    v = a[m].astype(np.float32)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else None


def rain_sums(lat, lon, first, last):
    w = requests.get(RAIN_API, params={"latitude": round(lat, 4), "longitude": round(lon, 4),
                                       "start_date": first, "end_date": last,
                                       "daily": "precipitation_sum", "timezone": "Europe/Dublin"},
                     timeout=120).json()["daily"]
    return dict(zip(w["time"], w["precipitation_sum"]))


def before(series, d, days):
    return sum(series.get(str(d - dt.timedelta(days=i))) or 0.0 for i in range(1, days + 1))


def auc(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not a.size or not b.size:
        return None
    return float(mannwhitneyu(a, b).statistic / (a.size * b.size))


def fields(args):
    z, meta, transform, crs = grid_of(args.bundle)
    s2_dates = [dt.date.fromisoformat(d) for d in z["dates"]]
    valid, peat, inside = z["valid"], z["peat"], z["inside"]
    summer = [i for i, d in enumerate(s2_dates) if 5 <= d.month <= 8]
    cnt = valid[summer].sum(0)
    share = np.where(cnt > 0, peat[summer].sum(0) / np.maximum(cnt, 1), 0)
    field = (share >= FIELD_SHARE) & inside
    intact = inside & (share < INTACT_SHARE) & ~binary_dilation(field, iterations=INTACT_GAP_PX)
    s1, cache = fetch(args.bundle, args.start or "2017-01-01", args.end or "2026-09-30", args.orbit, args.step)
    dates = [dt.date.fromisoformat(d) for d in s1["dates"]]
    # rainfall at the bog centre (ERA5)
    rows, cols = np.nonzero(inside)
    x, y = transform * (cols.mean(), rows.mean())
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(x, y)
    rain = rain_sums(lat, lon, str(dates[0] - dt.timedelta(days=8)), str(dates[-1]))
    fx = {d: s for d, s, _ in json.loads(args.bundle.with_name(f"{meta['code']}_fields.json").read_text())["rows"]}
    fx = {dt.date.fromisoformat(k): v for k, v in fx.items()}

    series = []
    for i, d in enumerate(dates):
        r = {"date": str(d), "rain_3d": round(before(rain, d, 3), 1)}
        for pol in ("vv", "vh"):
            r[f"field_{pol}"] = med(s1[pol][i], field)
            r[f"intact_{pol}"] = med(s1[pol][i], intact)
        if None in (r["field_vv"], r["intact_vv"]):
            continue
        r["contrast_vv"] = r["field_vv"] - r["intact_vv"]
        r["contrast_vh"] = r["field_vh"] - r["intact_vh"]
        r["field_cross"] = r["field_vh"] - r["field_vv"]
        r["intact_cross"] = r["intact_vh"] - r["intact_vv"]
        near = [(abs((e - d).days), v) for e, v in fx.items() if abs((e - d).days) <= 2]
        r["s2_field_share"] = min(near)[1] if near else None
        series.append(r)

    def corr(xk, yk, cond=lambda r: True):
        pts = [(r[xk], r[yk]) for r in series if r.get(xk) is not None and r.get(yk) is not None and cond(r)]
        if len(pts) < 8:
            return None
        s = spearmanr(*zip(*pts))
        return {"rho": round(float(s.statistic), 3), "p": float(s.pvalue), "n": len(pts)}

    season = lambda r: 4 <= int(r["date"][5:7]) <= 9
    out = {"site": meta["site"], "code": meta["code"], "collection": "sentinel-1-rtc (gamma0, dB)",
           "orbit": int(s1["orbit"]), "pass": str(s1["pass"]), "dates_used": len(series),
           "field_ha": round(field.sum() / 100, 2), "intact_ha": round(intact.sum() / 100, 2),
           "spearman": {
               "field_vv~rain_3d (Apr-Sep)": corr("field_vv", "rain_3d", season),
               "intact_vv~rain_3d (Apr-Sep)": corr("intact_vv", "rain_3d", season),
               "contrast_vv~rain_3d (Apr-Sep)": corr("contrast_vv", "rain_3d", season),
               "field_vv~s2_field_share (within 2 d)": corr("field_vv", "s2_field_share"),
               "contrast_vv~s2_field_share (within 2 d)": corr("contrast_vv", "s2_field_share"),
               "contrast_vh~s2_field_share (within 2 d)": corr("contrast_vh", "s2_field_share"),
               "field_cross~s2_field_share (within 2 d)": corr("field_cross", "s2_field_share")}}

    # per year, April-September: contrasts on dry and wet days and per-pixel separability
    years = {}
    for y in sorted({d.year for d in dates}):
        idx = [i for i, d in enumerate(dates) if d.year == y and 4 <= d.month <= 9]
        rs = [r for r in series if r["date"][:4] == str(y) and season(r)]
        if len(idx) < 4:
            continue
        rec = {"n": len(idx)}
        for pol in ("vv", "vh"):
            with np.errstate(all="ignore"):
                pm = np.nanmedian(s1[pol][idx].astype(np.float32), axis=0)
            rec[f"auc_{pol}"] = None if (a := auc(pm[field], pm[intact])) is None else round(a, 3)
        for key in ("contrast_vv", "contrast_vh", "field_vv", "intact_vv"):
            rec[f"{key}_median"] = round(float(np.median([r[key] for r in rs])), 2)
        dry = [r["contrast_vv"] for r in rs if r["rain_3d"] < DRY_MM]
        wet = [r["contrast_vv"] for r in rs if r["rain_3d"] >= WET_MM]
        rec["contrast_vv_dry"] = round(float(np.median(dry)), 2) if dry else None
        rec["contrast_vv_wet"] = round(float(np.median(wet)), 2) if wet else None
        years[str(y)] = rec
    out["by_year_apr_sep"] = years
    early = [r["contrast_vv"] for r in series if season(r) and int(r["date"][:4]) <= 2023]
    late = [r["contrast_vv"] for r in series if season(r) and int(r["date"][:4]) >= 2024]
    if early and late:
        t = mannwhitneyu(early, late)
        out["contrast_vv_2017_2023_vs_2024_2026"] = {"median_early": round(float(np.median(early)), 2),
                                                     "median_late": round(float(np.median(late)), 2), "p": float(t.pvalue)}
    # cloud gaps: usable radar acquisitions (all orbits) against clear S2 days, per April-September season
    s1_all = Counter(int(d[:4]) for d in s1["all_dates"] if 4 <= int(d[5:7]) <= 9)
    s2_clear = Counter(d.year for d in fx if 4 <= d.month <= 9)
    out["apr_sep_acquisitions"] = {str(y): {"s1_all_orbits": s1_all.get(y, 0), "s2_clear_fields": s2_clear.get(y, 0)}
                                   for y in sorted(set(s1_all) | set(s2_clear))}
    out["series"] = series
    stem = args.bundle.with_name(f"{meta['code']}_s1")
    Path(f"{stem}_fields.json").write_text(json.dumps(out, indent=2))

    t = [dt.date.fromisoformat(r["date"]) for r in series]
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), dpi=130, sharex=True)
    ax[0].plot(t, [r["field_vv"] for r in series], "-", lw=.8, color="#8b5a2b", label="worked fields VV")
    ax[0].plot(t, [r["intact_vv"] for r in series], "-", lw=.8, color="#2e7d5b", label="intact bog VV")
    ax[0].set_ylabel("gamma0 VV, dB"); ax[0].legend(fontsize=8, loc="lower left")
    ax[1].plot(t, [r["contrast_vv"] for r in series], "-", lw=.8, color="#444", label="VV fields - intact")
    ax[1].plot(t, [r["contrast_vh"] for r in series], "-", lw=.8, color="#b0506e", alpha=.8, label="VH fields - intact")
    ax[1].axhline(0, color="#999", lw=.6); ax[1].set_ylabel("contrast, dB"); ax[1].legend(fontsize=8, loc="lower left")
    ts = [(dt.date.fromisoformat(r["date"]), r["s2_field_share"]) for r in series if r["s2_field_share"] is not None]
    ax[2].bar(t, [r["rain_3d"] for r in series], width=4, color="#6fa8dc", label="rain, 3 days before (mm)")
    ax2 = ax[2].twinx()
    ax2.plot(sorted(fx), [fx[d] for d in sorted(fx)], ".", ms=3, color="#e07b00", label="S2 v5 field share")
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("S2 v5 share")
    ax[2].set_ylabel("rain, mm"); ax[2].legend(fontsize=8, loc="upper left"); ax2.legend(fontsize=8, loc="upper right")
    for a in ax:
        a.grid(alpha=.3)
    ax[0].set_title(f"{meta['site']}: Sentinel-1 backscatter, orbit {int(s1['orbit'])} ({s1['pass']})")
    fig.tight_layout(); fig.savefig(f"{stem}_fields.png"); plt.close(fig)
    print(json.dumps({k: v for k, v in out.items() if k != "series"}, indent=2))


def fire(args):
    z, meta, transform, crs = grid_of(args.bundle)
    s2_dates = [dt.date.fromisoformat(d) for d in z["dates"]]
    valid, inside, nbr = z["valid"], z["inside"], z["nbr"]
    f = dt.date.fromisoformat(args.fire_date)

    def window(lo, hi):
        sel = [i for i, d in enumerate(s2_dates) if lo <= d < hi]
        v = np.where(valid[sel], nbr[sel].astype(np.float32), np.nan)
        with np.errstate(all="ignore"):
            return np.nanmedian(v, axis=0)

    pre = window(f - dt.timedelta(days=PRE_DAYS), f)
    post = window(f + dt.timedelta(days=1), f + dt.timedelta(days=POST_DAYS))
    ok = inside & np.isfinite(pre) & np.isfinite(post)
    burn, control = ok & (pre - post >= DNBR), ok & (pre - post < 0.05)
    start = str(f - dt.timedelta(days=args.pre))
    end = str(f + dt.timedelta(days=args.post))
    s1, cache = fetch(args.bundle, start, end, args.orbit, args.step)
    rows = []
    for i, d in enumerate(s1["dates"]):
        r = {"date": str(d), "days_from_fire": (dt.date.fromisoformat(str(d)) - f).days}
        for pol in ("vv", "vh"):
            r[f"burn_{pol}"], r[f"control_{pol}"] = med(s1[pol][i], burn), med(s1[pol][i], control)
        if None in (r["burn_vv"], r["control_vv"]):
            continue
        r["diff_vv"], r["diff_vh"] = r["burn_vv"] - r["control_vv"], r["burn_vh"] - r["control_vh"]
        rows.append(r)
    base = {pol: [r[f"diff_{pol}"] for r in rows if r["days_from_fire"] < 0] for pol in ("vv", "vh")}
    summary = {}
    for pol in ("vv", "vh"):
        m, s = float(np.mean(base[pol])), float(np.std(base[pol]))
        after = [r for r in rows if r["days_from_fire"] >= 0]
        out_of_range = [r["days_from_fire"] for r in after if abs(r[f"diff_{pol}"] - m) > 2 * s]
        first = after[:3]
        summary[pol] = {"pre_mean_diff": round(m, 2), "pre_sd": round(s, 2), "n_pre": len(base[pol]),
                        "first_after": [(r["days_from_fire"], round(r[f"diff_{pol}"], 2)) for r in first],
                        "days_outside_2sd": out_of_range}
    out = {"site": meta["site"], "code": meta["code"], "fire_date": str(f), "orbit": int(s1["orbit"]),
           "pass": str(s1["pass"]), "burn_ha": round(burn.sum() / 100, 2), "control_ha": round(control.sum() / 100, 2),
           "summary": summary, "scenes": rows}
    stem = args.bundle.with_name(f"{meta['code']}_s1")
    Path(f"{stem}_fire.json").write_text(json.dumps(out, indent=2))
    t = [dt.date.fromisoformat(r["date"]) for r in rows]
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), dpi=130, sharex=True)
    for a, pol in zip(ax, ("vv", "vh")):
        a.plot(t, [r[f"burn_{pol}"] for r in rows], "-o", ms=2.5, color="#c0392b", label="burnt area")
        a.plot(t, [r[f"control_{pol}"] for r in rows], "-o", ms=2.5, color="#2e7d5b", label="unburnt bog")
        a.axvline(f, color="#ff7a00", lw=2, label="FIRMS fire")
        a.set_ylabel(f"gamma0 {pol.upper()}, dB"); a.grid(alpha=.3)
    ax[0].legend(fontsize=8, loc="lower left")
    ax[0].set_title(f"{meta['site']}: Sentinel-1 around the fire of {f} ({out['burn_ha']} ha burnt by dNBR)")
    fig.tight_layout(); fig.savefig(f"{stem}_fire.png"); plt.close(fig)
    print(json.dumps({k: v for k, v in out.items() if k != "scenes"}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("fields")
    a.add_argument("bundle", type=Path)
    a.add_argument("--start"); a.add_argument("--end")
    b = sub.add_parser("fire")
    b.add_argument("bundle", type=Path)
    b.add_argument("fire_date")
    b.add_argument("--pre", type=int, default=270)
    b.add_argument("--post", type=int, default=450)
    for p in (a, b):
        p.add_argument("--orbit", type=int)
        p.add_argument("--step", type=int, default=STEP_DAYS)
    args = ap.parse_args()
    fields(args) if args.mode == "fields" else fire(args)


if __name__ == "__main__":
    main()
