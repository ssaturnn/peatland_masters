"""Step 33 — how wide must a strip of exposed peat be to be detected?

Linear mixing experiment. The exposed-peat spectrum is the median of the v5
candidates on the south Mountdillon production fields (summer 2018). The
background is every clear pixel of a protected bog on its frozen evaluation
scene. A straight strip of exposed peat of width w metres crossing the centre
of a pixel covers w/10 of a 10 m band footprint and w/20 of a 20 m one
(Sentinel-2 SWIR1 and SWIR2 are acquired at 20 m), capped at 1. For each width
the script mixes the two spectra and reports the share of background pixels
that each detector rule then flags:

- v5: the SWIR signature (src/peatland/v5.py);
- the v1-v4 spectral rule: NDVI below the bog's adaptive threshold and NBR > 0.

PlanetScope has no SWIR band, so at 3 m only an NDVI contrast is available;
the script reports the NDVI of the mixed pixel for reference.

Usage:
    python3 scripts/33_detection_limit.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from peatland import pipeline, v5

PEAT = ROOT / "outputs/positive-control/scenes/BNMS-su_2018"
BACKGROUNDS = [ROOT / "outputs/evaluation/2026-10-07-v4-test" / s for s in
               ("000595_2021", "000600_2021", "000247_2026")] + \
              [ROOT / "outputs/evaluation/2026-09-13-v3-sample" / s for s in
               ("002352_2022", "002110_2021", "000600_2022")]
WIDTHS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20]
TWENTY_M = {"swir1", "swir2"}
OUT = ROOT / "outputs/evaluation/detection-limit.json"


def load(stem):
    z = np.load(f"{stem}.npz")
    meta = json.loads(Path(f"{stem}.json").read_text())
    return z, meta, tuple(meta["bands"])


def fractions(width, order):
    return np.array([min(width / (20.0 if b in TWENTY_M else 10.0), 1.0) for b in order])


def main():
    z, meta, order = load(PEAT)
    peat_px = v5.peat_mask(z["stack"], z["valid"], z["inside"], order)
    peat = np.median(z["stack"][peat_px], axis=0)
    result = {"peat_spectrum_dn": dict(zip(order, peat.round(1).tolist())), "widths_m": WIDTHS, "sites": {}}
    pooled = {w: [0, 0, 0] for w in WIDTHS}
    for stem in BACKGROUNDS:
        zb, mb, ob = load(stem)
        assert ob == order
        sel = zb["valid"] & zb["inside"] & ~v5.peat_mask(zb["stack"], zb["valid"], zb["inside"], ob)
        bg = zb["stack"][sel].astype(float)
        red, nir = bg[:, order.index("red")], bg[:, order.index("nir")]
        ndvi_bg = (nir - red) / (nir + red)
        # the scene's own adaptive NDVI threshold, as the v1-v4 rules compute it
        eff = min(pipeline.GNG_BARE_NDVI, max(np.quantile(ndvi_bg, 0.75) - pipeline.CONTRAST_DROP,
                                              pipeline.GNG_NDVI_FLOOR))
        rows = {}
        for w in WIDTHS:
            f = fractions(w, order)
            mix = bg * (1 - f) + peat * f
            img = mix[None, :, :]
            ones = np.ones(img.shape[:2], bool)
            hit5 = v5.peat_mask(img, ones, ones, order)[0]
            r, n, s2 = (mix[:, order.index(b)] for b in ("red", "nir", "swir2"))
            ndvi = (n - r) / (n + r)
            nbr = (n - s2) / (n + s2)
            hit_old = (ndvi < eff) & (nbr > pipeline.BURN_NBR_FLOOR)
            rows[w] = {"v5": round(float(hit5.mean()), 4), "v1_v4_rule": round(float(hit_old.mean()), 4),
                       "ndvi_median": round(float(np.median(ndvi)), 3)}
            pooled[w][0] += int(hit5.sum()); pooled[w][1] += int(hit_old.sum()); pooled[w][2] += len(bg)
        result["sites"][mb["site"] + f" {mb['year']}"] = {"scene_date": mb["scene_date"], "pixels": int(len(bg)),
                                                           "adaptive_ndvi_threshold": round(float(eff), 3),
                                                           "background_ndvi_median": round(float(np.median(ndvi_bg)), 3),
                                                           "by_width": rows}
    result["pooled"] = {w: {"v5": round(a / n, 4), "v1_v4_rule": round(b / n, 4)} for w, (a, b, n) in pooled.items()}
    OUT.write_text(json.dumps(result, indent=2))
    print("width m   v5 detected   v1-v4 rule detected")
    for w in WIDTHS:
        p = result["pooled"][w]
        print(f"{w:7}   {p['v5']:11.1%}   {p['v1_v4_rule']:11.1%}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
