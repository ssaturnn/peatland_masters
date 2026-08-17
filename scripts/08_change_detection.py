"""Step 9 — change detection between two epochs, with cloud-aware preprocessing.

Pipeline:
  1. establish the 10 m grid + site polygon mask for one bog
  2. PREPROCESS: for each epoch, search a summer date range and pick the
     scene that is clearest *over the bog* (SCL-based, not tile metadata)
  3. detect bare peat on each epoch (NDVI threshold, cloud-masked)
  4. classify change: newly-bare (new cutting) / re-vegetated / stable
  5. report areas in hectares, save a figure and a JSON summary
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from peatland import boundaries, imagery, detect, geo, config, preprocess, change

PROVIDER = "planetary"
EPOCHS = {"A": ("2018", "2018-05-01/2018-09-15"),
          "B": ("2024", "2024-05-01/2024-09-15")}
NDVI_BARE = 0.25
OUT_PNG = config.OUT_DIR / "change_detection.png"
OUT_JSON = config.OUT_DIR / "change_detection.json"


def natural_rgb(r, g, b):
    rgb = np.dstack([r, g, b]) / 10000.0 * 3.2
    return np.clip(rgb, 0, 1) ** (1.0 / 1.4)


def read_epoch(item, bbox, shape):
    """Read R,G,B,NIR onto the fixed grid and return rgb + ndvi."""
    r, _, _ = imagery.read_window(item, "red", bbox, PROVIDER, out_shape=shape)
    g, _, _ = imagery.read_window(item, "green", bbox, PROVIDER, out_shape=shape)
    b, _, _ = imagery.read_window(item, "blue", bbox, PROVIDER, out_shape=shape)
    nir, _, _ = imagery.read_window(item, "nir", bbox, PROVIDER, out_shape=shape)
    return natural_rgb(r, g, b), detect.ndvi(r, nir)


def main():
    west = boundaries.west_bog_sites(boundaries.load_nha())
    site_row = west.sort_values("HA").iloc[len(west) // 2]
    label = site_row.name
    name = site_row["SITE_NAME"]
    county = config.TARGET_COUNTIES[site_row["COUNTY"]]
    print(f"Site: {name} ({county}, {site_row['HA']:,.0f} ha)")

    bbox = boundaries.site_bbox_wgs84(west, label, buffer_m=200)

    # 1. establish the grid from any reference scene
    ref = imagery.search_scene(PROVIDER, bbox, "2022-05-01/2022-09-15",
                               max_cloud=30)
    if ref is None:
        print("No reference scene."); sys.exit(1)
    red0, transform, crs = imagery.read_window(ref, "red", bbox, PROVIDER)
    shape = red0.shape
    site_geom = west.loc[[label]].to_crs(crs).geometry.iloc[0]
    inside = geo.polygon_mask(site_geom, shape, transform)
    print(f"Grid: {shape[0]}x{shape[1]} px, site = "
          f"{geo.mask_area_ha(inside, transform):.1f} ha in window")

    # 2. preprocessing: pick the clearest scene over the bog per epoch
    epochs = {}
    for key, (yr, rng) in EPOCHS.items():
        t0 = time.time()
        item, rep = preprocess.pick_clear_scene(
            PROVIDER, bbox, rng, shape, aoi_mask=inside, min_clear=0.90)
        if item is None:
            print(f"  {yr}: no scene at all"); sys.exit(1)
        print(f"  {yr}: {rep['id'][:28]}…  {rep['datetime'][:10]}  "
              f"tile_cloud={rep['tile_cloud']:.1f}%  "
              f"AOI_clear={100*rep['aoi_clear']:.1f}%  [{time.time()-t0:.0f}s]")
        rgb, ndvi = read_epoch(item, bbox, shape)
        valid = preprocess.valid_mask(rep["scl"])
        bare = detect.exposed_peat_mask(ndvi, NDVI_BARE) & valid & inside
        epochs[key] = dict(yr=yr, rgb=rgb, ndvi=ndvi, valid=valid, bare=bare,
                           date=rep["datetime"][:10], clear=rep["aoi_clear"])

    A, B = epochs["A"], epochs["B"]

    # 3-4. classify change on pixels clear in BOTH epochs
    both_valid = A["valid"] & B["valid"] & inside
    masks = change.compare(A["bare"], B["bare"], both_valid)

    ha = lambda m: geo.mask_area_ha(m, transform)
    bare_a_ha, bare_b_ha = ha(A["bare"]), ha(B["bare"])
    newly, reveg, stable = (ha(masks["newly_bare"]),
                            ha(masks["revegetated"]),
                            ha(masks["stable_bare"]))
    site_ha = ha(inside)
    coverage = 100 * both_valid.sum() / max(inside.sum(), 1)

    print(f"\n{'':14}{A['yr']:>10}{B['yr']:>10}")
    print(f"{'bare peat ha':14}{bare_a_ha:>10.1f}{bare_b_ha:>10.1f}")
    print(f"newly bare (new cutting): {newly:6.1f} ha")
    print(f"re-vegetated:             {reveg:6.1f} ha")
    print(f"stable bare:              {stable:6.1f} ha")
    print(f"both-date valid coverage: {coverage:5.1f}% of site")

    # 5a. JSON summary
    summary = {
        "site": name, "county": county,
        "site_ha": round(site_ha, 1),
        "epoch_a": {"year": A["yr"], "date": A["date"],
                    "bare_ha": round(bare_a_ha, 1),
                    "aoi_clear": round(A["clear"], 3)},
        "epoch_b": {"year": B["yr"], "date": B["date"],
                    "bare_ha": round(bare_b_ha, 1),
                    "aoi_clear": round(B["clear"], 3)},
        "newly_bare_ha": round(newly, 1),
        "revegetated_ha": round(reveg, 1),
        "stable_bare_ha": round(stable, 1),
        "net_bare_change_ha": round(bare_b_ha - bare_a_ha, 1),
        "valid_coverage_pct": round(coverage, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved JSON  -> {OUT_JSON}")

    # 5b. figure: RGB A | RGB B | change map + text
    change_raster = change.change_class_raster(masks, shape)
    cmap = ListedColormap(["#00000000", "#cfe8cf", "#8a6d3b",
                           "#2c7fb8", "#e31a1c"])

    fig = plt.figure(figsize=(16, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[3.0, 1.15], hspace=0.12)
    ax = [fig.add_subplot(gs[0, i]) for i in range(3)]
    fig.suptitle(f"{name} — turf-cutting change {A['yr']} → {B['yr']}",
                 fontsize=14, y=0.98)

    ax[0].imshow(A["rgb"]); ax[0].set_title(f"{A['yr']}  ({A['date']})")
    ax[1].imshow(B["rgb"]); ax[1].set_title(f"{B['yr']}  ({B['date']})")
    ax[2].imshow(B["rgb"])
    ax[2].imshow(np.ma.masked_equal(change_raster, 0), cmap=cmap, vmin=0, vmax=4,
                 interpolation="nearest", alpha=0.85)
    ax[2].set_title("Change map")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])

    # legend
    from matplotlib.patches import Patch
    ax[2].legend(handles=[
        Patch(facecolor="#e31a1c", label=f"newly bare (new cutting) {newly:.1f} ha"),
        Patch(facecolor="#2c7fb8", label=f"re-vegetated {reveg:.1f} ha"),
        Patch(facecolor="#8a6d3b", label=f"stable bare {stable:.1f} ha"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.32), fontsize=8, frameon=False)

    axt = fig.add_subplot(gs[1, :]); axt.axis("off")
    lines = [
        ("Preprocessing + change detection", "head"),
        (f"Each epoch's scene was chosen by clear-pixel fraction over the "
         f"bog (SCL cloud mask), not tile cloud %: {A['yr']} "
         f"{100*A['clear']:.0f}% clear, {B['yr']} {100*B['clear']:.0f}% clear.",
         "body"),
        (f"Bare peat: {bare_a_ha:.1f} ha in {A['yr']} → {bare_b_ha:.1f} ha in "
         f"{B['yr']} (net {bare_b_ha-bare_a_ha:+.1f} ha).", "body"),
        (f"Newly bare (vegetated then bare) = {newly:.1f} ha — the candidate "
         f"NEW-cutting signal. Re-vegetated = {reveg:.1f} ha.", "body"),
        (f"Only pixels cloud-clear in BOTH dates are compared "
         f"({coverage:.0f}% of the site), so cloud cannot masquerade as "
         f"change.", "note"),
    ]
    y = 0.98
    for text, kind in lines:
        if kind == "head":
            axt.text(0, y, text, fontsize=12, fontweight="bold", va="top",
                     transform=axt.transAxes); y -= 0.19
        elif kind == "note":
            axt.text(0, y, text, fontsize=10, style="italic", color="#555",
                     va="top", transform=axt.transAxes); y -= 0.17
        else:
            axt.text(0, y, "•  " + text, fontsize=10, va="top",
                     transform=axt.transAxes); y -= 0.17

    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"Saved figure -> {OUT_PNG}")


if __name__ == "__main__":
    main()
