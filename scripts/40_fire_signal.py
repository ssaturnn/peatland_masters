"""Step 40 — how a peatland fire looks through time in the infrared.

From a dense time series (scripts/38_time_series.py) and a fire date recorded
by NASA FIRMS, it maps the burn independently of v5, from the drop in NBR
(median of clear observations in the 270 days before the fire minus the median
in the 45 days after, from the day after the fire; dNBR >= 0.15), and follows on every clear scene the
median NBR, SWIR1/SWIR2 ratio and NDVI of the burnt pixels, the share flagged
as v5 exposed peat, and the same for unburnt bog as a control.

Usage:
    python3 scripts/40_fire_signal.py outputs/time-series/002381_2018_2021.npz 2019-04-22
"""

import argparse
import datetime as dt
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DNBR = 0.15
PRE_DAYS, POST_DAYS = 270, 45   # Irish winters can leave months without a clear scene
MIN_SHARE = 0.6   # share of the burn that must be clear for a scene to count


def window_median(values, valid, dates, lo, hi):
    sel = [i for i, d in enumerate(dates) if lo <= d < hi]
    if not sel:
        return None
    v = np.where(valid[sel], values[sel].astype(np.float32), np.nan)
    with np.errstate(all="ignore"):
        return np.nanmedian(v, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("fire_date")
    args = ap.parse_args()
    z = np.load(args.bundle)
    meta = json.loads(args.bundle.with_suffix(".json").read_text())
    fire = dt.date.fromisoformat(args.fire_date)
    dates = [dt.date.fromisoformat(d) for d in z["dates"]]
    valid, peat, inside = z["valid"], z["peat"], z["inside"]
    nbr, ratio, ndvi = z["nbr"], z["ratio"], z["ndvi"]
    px_ha = abs(z["transform"][0] * z["transform"][4]) / 1e4

    pre = window_median(nbr, valid, dates, fire - dt.timedelta(days=PRE_DAYS), fire)
    # the fire day itself is excluded: the morning acquisition can precede the fire
    post = window_median(nbr, valid, dates, fire + dt.timedelta(days=1), fire + dt.timedelta(days=POST_DAYS))
    if pre is None or post is None:
        raise SystemExit("No clear scene before or after the fire inside the windows")
    burn = inside & np.isfinite(pre) & np.isfinite(post) & (pre - post >= DNBR)
    control = inside & np.isfinite(pre) & np.isfinite(post) & (pre - post < 0.05)
    rows = []
    for i, d in enumerate(dates):
        ok_b = valid[i] & burn
        ok_c = valid[i] & control
        if burn.sum() == 0 or ok_b.sum() < MIN_SHARE * burn.sum():
            continue
        f = lambda a, m: float(np.nanmedian(a[i][m].astype(np.float32))) if m.any() else None
        rows.append({"date": str(d), "days_from_fire": (d - fire).days,
                     "burn_nbr": f(nbr, ok_b), "burn_ratio": f(ratio, ok_b), "burn_ndvi": f(ndvi, ok_b),
                     "burn_peat_share": float(peat[i][ok_b].mean()),
                     "control_nbr": f(nbr, ok_c), "control_ratio": f(ratio, ok_c),
                     "control_peat_share": float(peat[i][ok_c].mean()) if ok_c.any() else None})
    out = {"site": meta["site"], "code": meta["code"], "fire_date": str(fire), "dnbr_threshold": DNBR,
           "burn_ha": round(float(burn.sum() * px_ha), 2), "scenes": rows}
    stem = args.bundle.with_suffix("")
    Path(f"{stem}_fire.json").write_text(json.dumps(out, indent=2))

    t = [dt.date.fromisoformat(r["date"]) for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), dpi=130, sharex=True)
    for ax, key, ctrl, lab, ref in [
            (axes[0], "burn_nbr", "control_nbr", "NBR", 0.0),
            (axes[1], "burn_ratio", "control_ratio", "SWIR1 / SWIR2", 1.25),
            (axes[2], "burn_peat_share", "control_peat_share", "share flagged v5 exposed peat", None)]:
        ax.plot(t, [r[key] for r in rows], "-o", ms=3, color="#c0392b", label="burnt area")
        ax.plot(t, [r[ctrl] for r in rows], "-o", ms=2.5, color="#2e7d5b", alpha=.8, label="unburnt bog")
        if ref is not None:
            ax.axhline(ref, color="#555", ls=":", lw=1)
        ax.axvline(fire, color="#ff7a00", lw=2, label="FIRMS fire")
        ax.set_ylabel(lab); ax.grid(alpha=.3)
    axes[0].legend(loc="lower left", fontsize=8)
    axes[0].set_title(f"{meta['site']}: fire of {fire} ({out['burn_ha']} ha burnt by dNBR)")
    fig.tight_layout(); fig.savefig(f"{stem}_fire.png"); plt.close(fig)
    print(json.dumps({k: v for k, v in out.items() if k != "scenes"}, indent=2))
    for r in rows:
        if -120 <= r["days_from_fire"] <= 900:
            print(f"{r['date']} ({r['days_from_fire']:+4d} d): NBR {r['burn_nbr']:+.2f} ratio {r['burn_ratio']:.2f} "
                  f"NDVI {r['burn_ndvi']:.2f} v5 {100 * r['burn_peat_share']:4.0f}%")


if __name__ == "__main__":
    main()
