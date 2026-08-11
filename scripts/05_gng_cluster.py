"""Step 6 — Growing Neural Gas on multi-spectral pixels.

Reads a 6-band Sentinel-2 stack over a bog site, trains a GNG on the
per-pixel spectral vectors, partitions the pixels by the learned graph's
connected components, identifies the component that best matches bare /
cut peat (lowest mean NDVI), clips it to the NPWS polygon, and reports
its area. The result is compared against the plain NDVI-threshold
baseline on the same scene.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from peatland import boundaries, imagery, detect, geo, gng, config

DATE_RANGE = "2023-05-01/2023-09-30"
PROVIDER = "planetary"
BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]
GNG_STEPS = 15000
GNG_MAX_NODES = 40


def main():
    west = boundaries.west_bog_sites(boundaries.load_nha())
    site_row = west.sort_values("HA").iloc[len(west) // 2]
    label = site_row.name
    print(f"Site: {site_row['SITE_NAME']} "
          f"({config.TARGET_COUNTIES[site_row['COUNTY']]}, "
          f"{site_row['HA']:,.0f} ha)")

    bbox = boundaries.site_bbox_wgs84(west, label, buffer_m=200)
    item = imagery.search_scene(PROVIDER, bbox, DATE_RANGE, max_cloud=15)
    if item is None:
        print("No scene."); sys.exit(1)
    print(f"Scene: {item.id}  cloud={item.properties.get('eo:cloud_cover'):.2f}%")

    stack, transform, crs, order = imagery.read_stack(item, BANDS, bbox, PROVIDER)
    h, w, b = stack.shape
    print(f"Stack: {h}x{w}x{b} bands {order}")

    # per-pixel feature vectors, scaled to reflectance ~[0,1]
    feats = stack.reshape(-1, b) / 10000.0
    red = stack[:, :, order.index("red")]
    nir = stack[:, :, order.index("nir")]
    ndvi = detect.ndvi(red, nir).reshape(-1)

    # subsample for training (GNG doesn't need every pixel)
    rng = np.random.default_rng(42)
    train_idx = rng.choice(len(feats), size=min(8000, len(feats)), replace=False)

    print(f"\nTraining GNG on {len(train_idx)} sampled pixels...")
    t0 = time.time()
    net = gng.GrowingNeuralGas(max_nodes=GNG_MAX_NODES, rng=rng)
    net.fit(feats[train_idx], n_steps=GNG_STEPS, verbose=True)
    print(f"  done in {time.time()-t0:.1f}s: "
          f"{len(net.weights)} nodes, {len(net.edges)} edges")

    # cut long edges so components correspond to spectral clusters
    net.prune_long_edges(factor=1.0)
    print(f"  after edge pruning: {len(net.edges)} edges")

    # assign every pixel to a node, then to a graph component
    node_of_pixel = net.predict(feats)
    comp_of_node = net.components()
    cluster = np.array([comp_of_node[n] for n in node_of_pixel])
    n_clusters = len(set(comp_of_node.values()))
    print(f"\nGraph components (clusters): {n_clusters}")

    # characterise each cluster by mean NDVI; lowest = bare/cut peat
    print(f"{'cluster':>8} {'pixels':>8} {'mean_NDVI':>10}")
    stats = []
    for c in sorted(set(cluster)):
        m = cluster == c
        mean_ndvi = float(ndvi[m].mean())
        stats.append((c, int(m.sum()), mean_ndvi))
        print(f"{c:>8} {int(m.sum()):>8} {mean_ndvi:>10.3f}")
    bare_cluster = min(stats, key=lambda s: s[2])[0]
    print(f"-> bare/cut-peat cluster = {bare_cluster}")

    gng_bare = (cluster == bare_cluster).reshape(h, w)

    # clip both detectors to the actual site polygon
    site_geom = west.loc[[label]].to_crs(crs).geometry.iloc[0]
    inside = geo.polygon_mask(site_geom, (h, w), transform)

    ndvi_bare = detect.exposed_peat_mask(ndvi.reshape(h, w), 0.25) & inside
    gng_in = gng_bare & inside

    site_ha = geo.mask_area_ha(inside, transform)
    ndvi_ha = geo.mask_area_ha(ndvi_bare, transform)
    gng_ha = geo.mask_area_ha(gng_in, transform)

    print(f"\nSite area in window:      {site_ha:8.1f} ha")
    print(f"NDVI-threshold bare area: {ndvi_ha:8.1f} ha "
          f"({100*ndvi_ha/site_ha:.1f}%)")
    print(f"GNG bare-cluster area:    {gng_ha:8.1f} ha "
          f"({100*gng_ha/site_ha:.1f}%)")

    inside_flat = inside.reshape(-1)
    agree = np.mean(
        (ndvi_bare.reshape(-1)[inside_flat]) == (gng_in.reshape(-1)[inside_flat])
    )
    print(f"Pixel agreement inside site: {100*agree:.1f}%")
    print("\nGNG detector runs end to end on real imagery.")


if __name__ == "__main__":
    main()
