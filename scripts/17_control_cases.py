"""Step 17 — control-case check for detector changes.

Re-runs the detector on site-years where the ground truth is known from
earlier visual QA and public records, and prints per-scene diagnostics:
the bog's vegetated-matrix NDVI (p75), the effective adaptive threshold,
cloud-shadow fraction, and both detectors' areas. Use it before any full
rebuild: the blanket-bog false positives should collapse while the
documented raised-bog cutting and the burn discrimination hold.

Control cases:
  * Doogort East Bog 2023   — blanket; winter-brown Molinia false positive
                              (old detector: 223 ha, visually no cutting)
  * Tullaghan Bay 2023      — blanket; largest reported figure, suspect
  * Monivea Bog SAC 2021/22 — raised; most-cut SAC in NPWS records (51/49
                              plots) — detection must NOT drop to zero
  * Callow Bog SAC 2022     — raised; 31 documented plots
  * Moorfield Bog 2025      — burn scar; must stay ~4 ha, not 60
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import geopandas as gpd

from peatland import boundaries, imagery, geo, config, preprocess, pipeline

CASES = [
    ("Doogort East", 2023, "blanket FP: expect big drop from 223 ha"),
    ("Tullaghan Bay", 2023, "blanket: expect drop"),
    ("Monivea", 2021, "raised TP (51 plots): must hold"),
    ("Monivea", 2022, "raised TP (49 plots): must hold"),
    ("Callow", 2022, "raised TP (31 plots): must hold"),
    ("Moorfield", 2025, "burn scar: must stay ~4 ha"),
]


def run_case(sites, name_frag, year, note):
    row = sites[sites.SITE_NAME.str.contains(name_frag, case=False)].iloc[0]
    bog_type = row["bog_type"]
    print(f"\n=== {row['SITE_NAME']} ({row['SITECODE']}) — {year} "
          f"[{bog_type}] — {note}")

    bbox = boundaries.site_bbox_wgs84(sites, row.name, buffer_m=200)
    ref = imagery.search_scene("planetary", bbox, "2021-05-01/2021-09-15",
                               max_cloud=40)
    red0, transform, crs = imagery.read_window(ref, "red", bbox, "planetary")
    shape = red0.shape
    geom = gpd.GeoSeries([row.geometry], crs=config.ITM).to_crs(crs).iloc[0]
    inside = geo.polygon_mask(geom, shape, transform)

    rng = pipeline._year_range(year, bog_type)
    scenes = preprocess.clear_scenes("planetary", bbox, rng, shape,
                                     aoi_mask=inside, min_clear=0.85)
    if not scenes:
        print("  no clear scenes")
        return None

    season_max = None
    for item, rep in scenes:
        d = pipeline.detect_year(item, bbox, "planetary", shape, inside,
                                 rep["scl"])
        stack = d["stack"]
        order = list(pipeline.BANDS)
        refl = stack / 10000.0
        red = refl[:, :, order.index("red")]
        nir = refl[:, :, order.index("nir")]
        ndvi = (nir - red) / (nir + red + 1e-9)
        sel = d["valid"] & inside
        matrix = float(np.quantile(ndvi[sel], 0.75)) if sel.sum() else float("nan")
        eff = max(min(pipeline.GNG_BARE_NDVI, matrix - pipeline.CONTRAST_DROP),
                  pipeline.GNG_NDVI_FLOOR)
        g_ha = geo.mask_area_ha(d["gng"], transform)
        n_ha = geo.mask_area_ha(d["ndvi"], transform)
        skip = "" if pipeline.scene_passes(d) else "  [gated]"
        print(f"  {rep['datetime'][:10]}  matrix_p75={matrix:5.2f} "
              f"eff_thresh={eff:5.2f}  gng={g_ha:7.1f} ha  ndvi={n_ha:7.1f} ha"
              f"  shadow={100*d['shadow_frac']:4.1f}% cloud={100*d['cloud_frac']:4.1f}%"
              f" hot={d['hot_frac']:.2f}{skip}")
        if not skip:
            season_max = g_ha if season_max is None else max(season_max, g_ha)
    print(f"  season-max (gng): "
          f"{'n/a' if season_max is None else f'{season_max:.1f} ha'}")
    return season_max


def main():
    sites = boundaries.build_sites()
    results = {}
    for frag, year, note in CASES:
        try:
            results[f"{frag} {year}"] = run_case(sites, frag, year, note)
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")
    out = Path("outputs/control_cases.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
