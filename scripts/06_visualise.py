"""Step 7 — a picture for the demo.

Runs the same pipeline as scripts 04/05 but instead of only printing
numbers, it renders a four-panel figure over one bog site and saves it
as a PNG:

  1. True-colour RGB (what the eye sees)
  2. NDVI map (vegetation vigour; dark = bare peat)
  3. NDVI-threshold baseline mask, clipped to the site polygon
  4. GNG bare-peat cluster mask, clipped to the site polygon

Nothing here is new science — it just makes the existing result visible.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")                 # headless: write a file, no window
import matplotlib.pyplot as plt

from peatland import boundaries, imagery, detect, geo, gng, config

DATE_RANGE = "2023-05-01/2023-09-30"
PROVIDER = "planetary"
BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]
GNG_STEPS = 15000
GNG_MAX_NODES = 40
OUT_PNG = Path(__file__).resolve().parents[1] / "outputs" / "demo_site.png"


def stretch(band, lo=2, hi=98):
    """Percentile stretch to [0,1] for display."""
    a, b = np.percentile(band, [lo, hi])
    return np.clip((band - a) / (b - a + 1e-9), 0, 1)


def main():
    west = boundaries.west_bog_sites(boundaries.load_nha())
    site_row = west.sort_values("HA").iloc[len(west) // 2]
    label = site_row.name
    name = site_row["SITE_NAME"]
    print(f"Site: {name} "
          f"({config.TARGET_COUNTIES[site_row['COUNTY']]}, "
          f"{site_row['HA']:,.0f} ha)")

    bbox = boundaries.site_bbox_wgs84(west, label, buffer_m=200)
    item = imagery.search_scene(PROVIDER, bbox, DATE_RANGE, max_cloud=15)
    if item is None:
        print("No scene."); sys.exit(1)
    print(f"Scene: {item.id}  cloud={item.properties.get('eo:cloud_cover'):.2f}%")

    stack, transform, crs, order = imagery.read_stack(item, BANDS, bbox, PROVIDER)
    h, w, b = stack.shape

    red = stack[:, :, order.index("red")]
    nir = stack[:, :, order.index("nir")]
    ndvi = detect.ndvi(red, nir)

    # true-colour RGB, each band stretched independently
    rgb = np.dstack([
        stretch(stack[:, :, order.index("red")]),
        stretch(stack[:, :, order.index("green")]),
        stretch(stack[:, :, order.index("blue")]),
    ])

    # --- GNG on per-pixel spectra ---
    feats = stack.reshape(-1, b) / 10000.0
    rng = np.random.default_rng(42)
    train_idx = rng.choice(len(feats), size=min(8000, len(feats)), replace=False)
    print("Training GNG...")
    t0 = time.time()
    net = gng.GrowingNeuralGas(max_nodes=GNG_MAX_NODES, rng=rng)
    net.fit(feats[train_idx], n_steps=GNG_STEPS)
    net.prune_long_edges(factor=1.0)
    node_of_pixel = net.predict(feats)
    comp_of_node = net.components()
    cluster = np.array([comp_of_node[n] for n in node_of_pixel])
    print(f"  GNG done in {time.time()-t0:.1f}s, "
          f"{len(set(comp_of_node.values()))} clusters")

    ndvi_flat = ndvi.reshape(-1)
    bare_cluster = min(
        set(cluster), key=lambda c: ndvi_flat[cluster == c].mean()
    )
    gng_bare = (cluster == bare_cluster).reshape(h, w)

    # clip both detectors to the site polygon
    site_geom = west.loc[[label]].to_crs(crs).geometry.iloc[0]
    inside = geo.polygon_mask(site_geom, (h, w), transform)
    ndvi_bare = detect.exposed_peat_mask(ndvi, 0.25) & inside
    gng_in = gng_bare & inside

    site_ha = geo.mask_area_ha(inside, transform)
    ndvi_ha = geo.mask_area_ha(ndvi_bare, transform)
    gng_ha = geo.mask_area_ha(gng_in, transform)

    # --- render ---
    fig, ax = plt.subplots(1, 4, figsize=(18, 5.2))
    fig.suptitle(
        f"{name} — Sentinel-2 {item.properties.get('datetime', '')[:10]} "
        f"(cloud {item.properties.get('eo:cloud_cover'):.2f}%)",
        fontsize=13,
    )

    ax[0].imshow(rgb)
    ax[0].set_title("True-colour RGB")

    im = ax[1].imshow(ndvi, cmap="RdYlGn", vmin=-0.1, vmax=0.9)
    ax[1].set_title("NDVI (dark = bare peat)")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

    base = np.zeros((h, w, 4))
    base[ndvi_bare] = [1, 0, 0, 1]
    ax[2].imshow(rgb)
    ax[2].imshow(base)
    ax[2].set_title(f"NDVI baseline\n{ndvi_ha:.1f} ha "
                    f"({100*ndvi_ha/site_ha:.1f}% of site)")

    gm = np.zeros((h, w, 4))
    gm[gng_in] = [0, 0.4, 1, 1]
    ax[3].imshow(rgb)
    ax[3].imshow(gm)
    ax[3].set_title(f"GNG bare cluster\n{gng_ha:.1f} ha "
                    f"({100*gng_ha/site_ha:.1f}% of site)")

    for a in ax:
        a.set_xticks([]); a.set_yticks([])

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=130)
    print(f"\nSaved figure -> {OUT_PNG}")
    print(f"Site {site_ha:.1f} ha | NDVI {ndvi_ha:.1f} ha | GNG {gng_ha:.1f} ha")


if __name__ == "__main__":
    main()
