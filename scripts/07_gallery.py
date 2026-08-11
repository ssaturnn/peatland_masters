"""Step 8 — a gallery of several bog sites for the demo.

Runs the full pipeline (imagery -> NDVI -> GNG -> clip -> area) over a
handful of West-of-Ireland bog sites and stacks them into one figure:
one row per site, columns = natural-colour RGB / NDVI / GNG bare cluster.

The RGB here uses a shared, gamma-corrected stretch (not per-band) so the
peat surface reads as its true brown, not the magenta artefact of an
independent per-band stretch.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from peatland import boundaries, imagery, detect, geo, gng, config

DATE_RANGE = "2023-05-01/2023-09-30"
PROVIDER = "planetary"
BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]
GNG_STEPS = 15000
GNG_MAX_NODES = 40
N_SITES = 4
# a cluster counts as bare/cut peat only if its mean NDVI is physically low,
# not merely the lowest present — otherwise a whole low-vigour bog matrix
# gets mislabelled as bare on large sites.
BARE_NDVI_MAX = 0.25
OUT_PNG = Path(__file__).resolve().parents[1] / "outputs" / "gallery.png"


def natural_rgb(stack, order, scale=3.2, gamma=1.4):
    """True-colour RGB with a shared stretch, so peat looks brown."""
    r = stack[:, :, order.index("red")] / 10000.0
    g = stack[:, :, order.index("green")] / 10000.0
    b = stack[:, :, order.index("blue")] / 10000.0
    rgb = np.dstack([r, g, b]) * scale
    rgb = np.clip(rgb, 0, 1) ** (1.0 / gamma)
    return rgb


def process_site(west, label):
    site_row = west.loc[label]
    name = site_row["SITE_NAME"]
    bbox = boundaries.site_bbox_wgs84(west, label, buffer_m=200)
    item = imagery.search_scene(PROVIDER, bbox, DATE_RANGE, max_cloud=15)
    if item is None:
        return None
    stack, transform, crs, order = imagery.read_stack(item, BANDS, bbox, PROVIDER)
    h, w, b = stack.shape

    red = stack[:, :, order.index("red")]
    nir = stack[:, :, order.index("nir")]
    ndvi = detect.ndvi(red, nir)
    rgb = natural_rgb(stack, order)

    feats = stack.reshape(-1, b) / 10000.0
    rng = np.random.default_rng(42)
    train_idx = rng.choice(len(feats), size=min(8000, len(feats)), replace=False)
    net = gng.GrowingNeuralGas(max_nodes=GNG_MAX_NODES, rng=rng)
    net.fit(feats[train_idx], n_steps=GNG_STEPS)
    net.prune_long_edges(factor=1.0)
    cluster = np.array([net.components()[n] for n in net.predict(feats)])

    # GNG regularises, physics decides: keep the spectrally low-vigour
    # clusters (mean NDVI below the scene mean), then require each pixel to
    # be genuinely bare (NDVI < BARE_NDVI_MAX). The cluster step removes
    # scattered noise; the per-pixel step stops a whole low-NDVI bog matrix
    # being swallowed as "cut peat".
    ndvi_flat = ndvi.reshape(-1)
    scene_mean = float(ndvi_flat.mean())
    low_clusters = [c for c in set(cluster)
                    if ndvi_flat[cluster == c].mean() < scene_mean]
    gng_bare = (np.isin(cluster, low_clusters)
                & (ndvi_flat < BARE_NDVI_MAX)).reshape(h, w)

    site_geom = west.loc[[label]].to_crs(crs).geometry.iloc[0]
    inside = geo.polygon_mask(site_geom, (h, w), transform)
    gng_in = gng_bare & inside
    site_ha = geo.mask_area_ha(inside, transform)
    gng_ha = geo.mask_area_ha(gng_in, transform)

    return {
        "name": name, "county": config.TARGET_COUNTIES[site_row["COUNTY"]],
        "date": item.properties.get("datetime", "")[:10],
        "cloud": item.properties.get("eo:cloud_cover", 0),
        "rgb": rgb, "ndvi": ndvi, "gng_in": gng_in,
        "site_ha": site_ha, "gng_ha": gng_ha, "shape": (h, w),
    }


def main():
    west = boundaries.west_bog_sites(boundaries.load_nha())
    # a spread of sizes: quartile positions, skip the tiny ones
    ranked = west.sort_values("HA")
    n = len(ranked)
    picks = [ranked.iloc[int(n * f)].name for f in (0.35, 0.55, 0.75, 0.92)]

    results = []
    for label in picks:
        try:
            t0 = time.time()
            r = process_site(west, label)
            if r:
                results.append(r)
                print(f"OK  {r['name']} ({r['county']}) "
                      f"site {r['site_ha']:.0f} ha, GNG {r['gng_ha']:.1f} ha "
                      f"[{time.time()-t0:.0f}s]")
            else:
                print(f"skip {label}: no scene")
        except Exception as e:
            print(f"FAIL {label}: {type(e).__name__}: {e}")

    if not results:
        print("nothing to plot"); sys.exit(1)

    rows = len(results)
    fig, ax = plt.subplots(rows, 3, figsize=(13, 4.2 * rows))
    if rows == 1:
        ax = ax[None, :]

    for i, r in enumerate(results):
        h, w = r["shape"]
        ax[i, 0].imshow(r["rgb"])
        ax[i, 0].set_ylabel(f"{r['name']}\n{r['county']} · {r['date']}",
                            fontsize=10)
        ax[i, 0].set_title("True-colour RGB" if i == 0 else "")

        im = ax[i, 1].imshow(r["ndvi"], cmap="RdYlGn", vmin=-0.1, vmax=0.9)
        ax[i, 1].set_title("NDVI" if i == 0 else "")

        overlay = np.zeros((h, w, 4))
        overlay[r["gng_in"]] = [0, 0.4, 1, 1]
        ax[i, 2].imshow(r["rgb"])
        ax[i, 2].imshow(overlay)
        ax[i, 2].set_title("GNG bare peat" if i == 0 else "")
        ax[i, 2].text(0.5, -0.08,
                      f"{r['gng_ha']:.1f} ha / {r['site_ha']:.0f} ha "
                      f"({100*r['gng_ha']/r['site_ha']:.1f}%)",
                      transform=ax[i, 2].transAxes, ha="center", fontsize=9)

        for j in range(3):
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])

    fig.suptitle("Bare-peat detection across West-of-Ireland bog sites "
                 "(Sentinel-2, GNG unsupervised clustering)", fontsize=13)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT_PNG, dpi=130)
    print(f"\nSaved gallery -> {OUT_PNG}  ({len(results)} sites)")


if __name__ == "__main__":
    main()
