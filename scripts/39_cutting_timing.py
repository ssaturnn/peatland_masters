"""Step 39 — when was the peat exposed? Per-pixel timing on one bog.

From a dense time series (scripts/38_time_series.py) it finds, for every
pixel, the transitions from vegetation to exposed peat:

- before: the last two clear observations have NDVI >= 0.55 (a vegetation
  canopy) and are not v5 peat;
- after: the next two clear observations are both v5 exposed peat.

A pixel's exposure is dated between the last vegetated and the first peat
observation. Wet milled peat can drop out of v5 for a while, but it never has
a vegetation NDVI, so such gaps are not counted as new cutting.

Outputs a JSON summary, an area-through-time chart and an exposure-date map.

Usage:
    python3 scripts/39_cutting_timing.py outputs/time-series/001244_2017_2026.npz
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VEG_NDVI = 0.55
RUN = 2
CLEAR_FOR_AREA = 0.8


def transitions(dates, valid, peat, ndvi):
    """Per pixel: list of (last vegetated date index, first peat date index)."""
    T, H, W = peat.shape
    events = {}
    for r in range(H):
        for c in range(W):
            obs = [t for t in range(T) if valid[t, r, c]]
            if len(obs) < 2 * RUN:
                continue
            for k in range(RUN, len(obs) - RUN + 1):
                before, after = obs[k - RUN:k], obs[k:k + RUN]
                if (all(not peat[t, r, c] and ndvi[t, r, c] >= VEG_NDVI for t in before)
                        and all(peat[t, r, c] for t in after)):
                    events.setdefault((r, c), []).append((before[-1], after[0]))
    return events


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path)
    args = ap.parse_args()
    z = np.load(args.bundle)
    meta = json.loads(args.bundle.with_suffix(".json").read_text())
    dates = [dt.date.fromisoformat(d) for d in z["dates"]]
    valid, peat = z["valid"], z["peat"]
    ndvi = z["ndvi"].astype(np.float32)
    inside = z["inside"]
    px_ha = abs(z["transform"][0] * z["transform"][4]) / 1e4

    # area of exposed peat through time, on scenes that see most of the bog
    clear = valid.reshape(len(dates), -1).sum(1) / inside.sum()
    area = [(d, float(p.sum() * px_ha / max(c, 1e-9)), float(c))
            for d, p, c in zip(dates, peat, clear) if c >= CLEAR_FOR_AREA]

    events = transitions(dates, valid, peat, ndvi)
    first = {}
    for (r, c), ev in events.items():
        a, b = ev[0]
        first[(r, c)] = (a, b)
    gaps = [(dates[b] - dates[a]).days for a, b in first.values()]
    by_year = {}
    for a, b in first.values():
        y = dates[b].year
        by_year[y] = by_year.get(y, 0) + px_ha
    # contiguous dated patches: pixels exposed in the same bracket
    brackets = {}
    for (r, c), (a, b) in first.items():
        brackets.setdefault((a, b), []).append((r, c))
    top = sorted(brackets.items(), key=lambda kv: -len(kv[1]))[:12]

    out = {"site": meta["site"], "code": meta["code"], "scenes": len(dates),
           "scenes_for_area": len(area), "veg_ndvi": VEG_NDVI, "run": RUN,
           "exposed_pixels_dated": len(first), "exposed_ha_dated": round(len(first) * px_ha, 2),
           "exposed_ha_by_year": {str(k): round(v, 2) for k, v in sorted(by_year.items())},
           "bracket_days": {"median": float(np.median(gaps)) if gaps else None,
                            "p25": float(np.percentile(gaps, 25)) if gaps else None,
                            "p75": float(np.percentile(gaps, 75)) if gaps else None},
           "largest_dated_events": [{"after": str(dates[a]), "before": str(dates[b]),
                                     "days": (dates[b] - dates[a]).days,
                                     "ha": round(len(px) * px_ha, 2)} for (a, b), px in top],
           "area_series": [{"date": str(d), "ha": round(h, 2), "clear": round(c, 2)} for d, h, c in area]}
    stem = args.bundle.with_suffix("")
    Path(f"{stem}_timing.json").write_text(json.dumps(out, indent=2))

    # charts
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=130)
    ax.plot([a[0] for a in area], [a[1] for a in area], "-o", ms=3, color="#e07b00", lw=1.2)
    ax.set_ylabel("exposed peat, ha"); ax.set_title(f"{meta['site']}: v5 exposed peat on every clear Sentinel-2 scene")
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(f"{stem}_area.png"); plt.close(fig)

    years = sorted(by_year)
    if years:
        H, W = inside.shape
        img = np.full((H, W), np.nan)
        for (r, c), (a, b) in first.items():
            img[r, c] = dates[b].year + (dates[b].timetuple().tm_yday - 1) / 365
        fig, ax = plt.subplots(figsize=(6.5, 6.5 * H / W + 0.6), dpi=130)
        ax.imshow(np.where(inside, 0.85, 0.6), cmap="gray", vmin=0, vmax=1)
        im = ax.imshow(img, cmap="viridis", vmin=min(years), vmax=max(years) + 1)
        ax.set_axis_off(); ax.set_title(f"{meta['site']}: when vegetation gave way to exposed peat")
        fig.colorbar(im, ax=ax, fraction=0.035, label="date of first exposed-peat observation")
        fig.tight_layout(); fig.savefig(f"{stem}_onset.png"); plt.close(fig)
    print(json.dumps({k: v for k, v in out.items() if k != "area_series"}, indent=2))


if __name__ == "__main__":
    main()
