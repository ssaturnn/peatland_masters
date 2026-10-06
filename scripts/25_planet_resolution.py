"""Step 25 — what 10 m misses: PlanetScope 3 m against Sentinel-2 across every bog.

The Monivea case study (scripts/21) answers the resolution question on one
site-year. This repeats the same calculation wherever a cloud-screened
PlanetScope mosaic exists: fit the cross-sensor NDVI calibration on paired
10 m cells, detect calibrated 3 m candidates, aggregate them back to the
10 m grid and compare with the frozen v3 GNG mask.

Every number is a candidate bare surface, never confirmed turf cutting. No
reference label is read. PlanetScope has no SWIR, so the Sentinel-2 burn and
water/deep-shadow guards cannot be reproduced; cloudy pixels are removed with
Planet's own UDM2 mask instead.

Usage from the repository root:
    python3 scripts/25_planet_resolution.py
    python3 scripts/25_planet_resolution.py --key 002352_2022 --gng
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True

from peatland.cross_sensor import (
    MIN_COVERAGE, adaptive_threshold, aggregate_labels, aggregate_to_grid,
    area_mean_reflectance, candidate_mask, compare_masks, fit_ndvi_calibration,
    gng_candidates, indices, north_up, overlap_statistics, project_mask,
    summarise, true_colour, zoom_bounds)
from peatland.planet import CITATION, LICENCE, private_path, udm2_clear

DEFAULT_MANIFEST = Path("web/private/planet/manifest.json")
DEFAULT_RELEASE = Path("outputs/planet_release")
DEFAULT_EVALUATION = Path("outputs/evaluation/2026-09-13-v3-sample")
DEFAULT_OUT = Path("outputs/planet_resolution")
VERSION = "2026-09-16-planet-resolution-v1"
LIMITATIONS = [
    "Candidate bare surface only: no reference label enters this study, so none of "
    "these numbers is precision, recall or accuracy.",
    "One date per site-year. A bog is compared on the single release scene, so a "
    "difference mixes resolution with anything that changed between the two "
    "acquisitions; frames are within three days of Sentinel-2, not simultaneous.",
    "PlanetScope has no SWIR: the Sentinel-2 NBR burn guard and the SWIR water and "
    "deep-shadow guards are unavailable, so the 3 m mask is less protected against "
    "burn scars, wet shadow and dark water than the 10 m one.",
    "Co-registration is not estimated or corrected. A 1-2 pixel shift is 3-6 m and "
    "changes edge agreement and narrow-feature counts.",
    "The NDVI calibration is fitted per scene on paired 10 m cells and applied to "
    "native 3 m pixels, which assumes the relationship transfers across scale; "
    "its residuals are in-sample.",
    "The narrow share is a morphological opening proxy. It includes small objects and "
    "boundary protrusions and is not a measured cutting-strip width.",
    "Cloudy PlanetScope pixels are removed with UDM2, which counts light haze as not "
    "clear; tiles below 50% clear were already withheld upstream.",
]


def site_paths(key, record, release, evaluation):
    """Mosaic folder plus the frozen Sentinel-2 arrays for one site-year."""
    if record.get("source") == "release":
        folder = release / key
        return folder, folder / "sentinel2.npz", folder / "sentinel2.json"
    return (evaluation / "planet" / key, evaluation / f"{key}.npz",
            evaluation / f"{key}.json")


def read_site(key, record, release, evaluation):
    """Load both sensors onto their native grids, with the assessed masks."""
    folder, npz_path, json_path = site_paths(key, record, release, evaluation)
    meta = json.loads(json_path.read_text())
    with np.load(npz_path, allow_pickle=False) as data:
        s2 = {"stack": data["stack"].copy(), "valid": data["valid"].astype(bool),
              "inside": data["inside"].astype(bool),
              "frozen": data["prediction_gng"].astype(bool)}
    s2["transform"] = rasterio.Affine(*meta["transform"][:6])
    s2["crs"] = rasterio.crs.CRS.from_user_input(meta["crs"])
    with rasterio.open(folder / "ps_mosaic.tif") as ds:
        if ds.count not in (4, 8):
            raise ValueError("Expected a four-band or eight-band PlanetScope mosaic")
        bands = [1, 2, 3, 4] if ds.count == 4 else [2, 4, 6, 8]
        stack = np.moveaxis(ds.read(bands), 0, -1).astype(float)
        valid = ((ds.read_masks(bands) > 0).all(axis=0)
                 & np.isfinite(stack).all(axis=2) & (stack != 0).all(axis=2))
        ps = {"stack": stack, "valid": valid, "transform": ds.transform,
              "crs": ds.crs, "bands": bands}
    if ps["crs"] != s2["crs"] or not s2["crs"].is_projected or s2["crs"].linear_units != "metre":
        raise ValueError("Exact area aggregation needs both grids in one projected metre CRS")
    north_up(ps["transform"])
    north_up(s2["transform"])
    clear = udm2_clear(folder, record.get("items", []), ps["transform"],
                       ps["valid"].shape, ps["crs"])
    ps["clear"], ps["covered"] = (None, None) if clear is None else clear
    return meta, s2, ps


def run_site(key, record, release, evaluation, bog_type=None, with_gng=False):
    """Calibrate, detect at 3 m and compare with the frozen 10 m mask."""
    meta, s2, ps = read_site(key, record, release, evaluation)
    if meta.get("detector_version") != "2026-09-13-cloud-screen-v3":
        raise ValueError(f"{key}: expected the frozen Sentinel-2 v3 detector")
    ps_inside = project_mask(s2["inside"], s2["transform"], s2["crs"],
                             ps["valid"].shape, ps["transform"], ps["crs"])
    cloudy = 0
    analysis = ps_inside & ps["valid"]
    if ps["clear"] is not None:
        cloudy = int((analysis & ps["covered"] & ~ps["clear"]).sum())
        analysis = analysis & ps["clear"]
    s2_analysis = s2["inside"] & s2["valid"]
    s2_mask = s2["frozen"] & s2_analysis
    if not analysis.any():
        raise ValueError(f"{key}: no assessed PlanetScope pixels remain")

    ndvi, ndwi = indices(ps["stack"] / 10000)
    s2_ndvi, s2_ndwi = indices(s2["stack"][:, :, :4] / 10000)
    ps_mean, coverage = area_mean_reflectance(ps["stack"] / 10000, analysis,
                                              ps["transform"], s2_mask.shape, s2["transform"])
    ps_10_ndvi, ps_10_ndwi = indices(ps_mean)
    paired = s2_analysis & (coverage > 0)
    fit, _ = fit_ndvi_calibration(ps_10_ndvi, s2_ndvi, ps_10_ndwi, s2_ndwi, paired)
    common = s2_analysis & (coverage >= MIN_COVERAGE) & (coverage > 0)
    calibrated = fit["a"] * ndvi + fit["b"]
    threshold = adaptive_threshold(calibrated, analysis)
    rules = candidate_mask(calibrated, ndwi, analysis, threshold)
    metrics, aggregated = compare_masks(rules, analysis, ps["transform"],
                                        s2_mask, s2["transform"], common)
    ps_area = abs(ps["transform"].a * ps["transform"].e)
    s2_area = abs(s2["transform"].a * s2["transform"].e)
    summary = {
        "key": key, "site": record["site"], "source": record.get("source"),
        "bog_type": bog_type,
        "sentinel2_date": record["s2_date"], "planet_date": record["date"],
        "delta_days": record["delta_days"], "clear_pct": record.get("clear_pct"),
        "sentinel2_scene": meta["scene_id"],
        "assessed": {
            "sentinel2_valid_inside_ha": float(s2_analysis.sum() * s2_area / 10000),
            "planet_assessed_ha": float(analysis.sum() * ps_area / 10000),
            "planet_cloudy_pixels_removed": cloudy,
            "common_10m_cells": int(common.sum()),
            "common_10m_ha": float(common.sum() * s2_area / 10000)},
        "calibration": {**fit, "threshold": threshold,
                        "equation": "NDVI_S2 = a * NDVI_PS10 + b",
                        # a weak correlation means the 3 m threshold rests on a fit
                        # that does not describe this scene: flag, never silently use
                        "reliable": fit["r"] is not None and fit["r"] >= RELIABLE_R},
        **metrics,
        "sentinel2_area_ha": float(s2_mask.sum() * s2_area / 10000),
    }
    if with_gng:
        gng, info = gng_candidates(ps["stack"] / 10000, analysis, threshold, ndvi=calibrated)
        gng_fraction, _ = aggregate_to_grid(gng, analysis, ps["transform"],
                                            s2_mask.shape, s2["transform"])
        gng_labels, _ = aggregate_labels(gng_fraction, coverage)
        summary["gng_variant"] = {
            "area_ha": float(gng.sum() * ps_area / 10000),
            "changed_pixels_vs_rules": int((gng ^ rules).sum()),
            "iou_vs_sentinel2": overlap_statistics(gng_labels, s2_mask, common)["iou"],
            "nodes": info.get("nodes"), "water_nodes": info.get("water_nodes")}
    arrays = {"planet_rules": rules, "planet_analysis": analysis,
              "planet_aggregated_10m": aggregated & common,
              "sentinel2_gng": s2_mask, "comparison_support": common,
              # the 3 m masks sit on the mosaic grid: keep it, so the map tiles
              # can draw them without guessing the geometry
              "planet_transform": np.array(tuple(ps["transform"]), dtype=float),
              "crs": np.array(ps["crs"].to_string())}
    if with_gng:
        arrays["planet_gng"] = gng
    return summary, arrays, (s2, ps, rules, s2_mask)


SUBSTANTIAL_HA = 0.5   # below this, a per-site ratio is a few pixels of noise
RELIABLE_R = 0.8       # weaker cross-sensor correlation makes the transfer unsafe


def aggregate(sites):
    """Medians and ranges over the site-years, split by what each sensor found.

    Per-site IoU and area ratios are reported where Sentinel-2 actually has
    candidates: where it has none, the ratio is undefined and the IoU is zero
    by construction, which would otherwise dominate the medians. Pooled cell
    counts give the same comparison weighted by area instead of by site.
    """
    defined = [s for s in sites if s["iou"] is not None]
    paired = [s for s in sites if s["sentinel2_area_common_support_ha"] > 0]
    planet_only = [s for s in sites if s["sentinel2_area_common_support_ha"] == 0
                   and s["planet_area_common_support_ha"] > 0]
    silent = [s for s in sites if s["sentinel2_area_common_support_ha"] == 0
              and s["planet_area_common_support_ha"] == 0]
    with_candidates = [s for s in sites if s["planet_area_ha"] > 0]
    substantial = [s for s in paired
                   if s["sentinel2_area_common_support_ha"] >= SUBSTANTIAL_HA]
    shared = sum(s["shared_cells"] for s in sites)
    union = shared + sum(s["only_3m_cells"] + s["only_10m_cells"] for s in sites)
    planet_total = float(sum(s["planet_area_common_support_ha"] for s in sites))
    sentinel2_total = float(sum(s["sentinel2_area_common_support_ha"] for s in sites))
    return {
        "site_years": len(sites),
        "site_years_with_sentinel2_candidates": len(paired),
        "site_years_with_planet_only": len(planet_only),
        "site_years_both_empty": len(silent),
        "site_years_above_half_hectare": len(substantial),
        "iou_defined_on": len(defined),
        "iou": summarise([s["iou"] for s in paired]),
        "iou_including_planet_only": summarise([s["iou"] for s in defined]),
        "iou_above_half_hectare": summarise([s["iou"] for s in substantial]),
        "iou_pooled_cells": shared / union if union else None,
        "area_ratio_3m_over_10m": summarise(
            [s["area_ratio_common_support"] for s in paired]),
        "area_ratio_above_half_hectare": summarise(
            [s["area_ratio_common_support"] for s in substantial]),
        "area_ratio_pooled": planet_total / sentinel2_total if sentinel2_total else None,
        "narrow_share": summarise([s["narrow_share"] for s in with_candidates]),
        "narrow_share_above_half_hectare": summarise(
            [s["narrow_share"] for s in substantial]),
        "unreliable_calibrations": [s["key"] for s in sites if not s["calibration"]["reliable"]],
        "by_bog_type": {
            kind: {"site_years": len(group),
                   "with_sentinel2_candidates": len([s for s in group
                                                     if s["sentinel2_area_common_support_ha"] > 0]),
                   "iou": summarise([s["iou"] for s in group
                                     if s["sentinel2_area_common_support_ha"] > 0]),
                   "area_ratio": summarise([s["area_ratio_common_support"] for s in group
                                            if s["sentinel2_area_common_support_ha"] > 0]),
                   "calibration_r": summarise([s["calibration"]["r"] for s in group])}
            for kind, group in [(k, [s for s in sites if s["bog_type"] == k])
                                for k in sorted({s["bog_type"] for s in sites if s["bog_type"]})]},
        "planet_area_ha": summarise([s["planet_area_ha"] for s in sites]),
        "sentinel2_area_ha": summarise([s["sentinel2_area_ha"] for s in sites]),
        "calibration_slope": summarise([s["calibration"]["a"] for s in sites]),
        "calibration_intercept": summarise([s["calibration"]["b"] for s in sites]),
        "calibration_r": summarise([s["calibration"]["r"] for s in sites]),
        "calibration_rmse": summarise([s["calibration"]["rmse"] for s in sites]),
        "cells_only_3m_total": int(sum(s["only_3m_cells"] for s in sites)),
        "cells_only_10m_total": int(sum(s["only_10m_cells"] for s in sites)),
        "cells_shared_total": int(sum(s["shared_cells"] for s in sites)),
        "planet_area_total_ha": planet_total,
        "sentinel2_area_total_ha": sentinel2_total,
    }


def _figure_setup():
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".mplcache")
    os.environ["XDG_CACHE_HOME"] = str(ROOT / ".mplcache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def write_distributions(out, sites, totals):
    plt = _figure_setup()
    panels = [
        ("Agreement where Sentinel-2 fires (IoU)",
         [s["iou"] for s in sites if s["sentinel2_area_common_support_ha"] > 0],
         "IoU of aggregated 3 m candidates with the frozen 10 m mask", "#3f7a5e"),
        ("Candidate area, 3 m ÷ 10 m", [s["area_ratio_common_support"] for s in sites
                                        if s["sentinel2_area_common_support_ha"] > 0],
         "Ratio on common support (1 = same area)", "#c2452a"),
        ("Narrow-feature share at 3 m", [s["narrow_share"] for s in sites
                                         if s["planet_area_ha"] > 0 and s["narrow_share"] is not None],
         "Share of 3 m candidate area removed by a 10 m opening", "#2f6690"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax, (title, values, xlabel, colour) in zip(axes, panels):
        values = np.asarray(values, dtype=float)
        # a handful of tiny-area ratios reach 20 and would flatten the rest
        limit = float(np.quantile(values, 0.95)) * 1.3 if values.size else 1
        shown = values[values <= limit] if values.size else values
        ax.hist(shown, bins=12, color=colour, alpha=0.85, edgecolor="white")
        if values.size:
            median = float(np.median(values))
            ax.axvline(median, color="black", ls="--", lw=1.2)
            beyond = int((values > limit).sum())
            ax.text(0.97, 0.94, f"n = {values.size}\nmedian {median:.2f}"
                    + (f"\n{beyond} beyond axis" if beyond else ""),
                    transform=ax.transAxes, ha="right", va="top", fontsize=9)
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("site-years", fontsize=9)
    fig.suptitle("PlanetScope 3 m against Sentinel-2 10 m, "
                 f"{totals['site_years']} bog site-years", fontsize=13, x=0.02, ha="left")
    fig.text(0.02, 0.02, "Candidate bare surface, not confirmed cutting. " + CITATION, fontsize=7.5)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.2, top=0.82, wspace=0.28)
    fig.savefig(out / "distributions.png", dpi=150, facecolor="white")
    plt.close(fig)


def write_example(out, name, key, summary, data):
    """Two panels on the densest 400 m window: 10 m mask beside 3 m candidates."""
    from rasterio.transform import array_bounds
    plt = _figure_setup()
    s2, ps, rules, s2_mask = data
    reference = s2_mask if s2_mask.any() else None
    window, _ = zoom_bounds(reference if reference is not None else
                            project_mask(rules, ps["transform"], ps["crs"], s2_mask.shape,
                                         s2["transform"], s2["crs"]), s2["transform"])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.6))
    panels = [(s2["stack"], s2["valid"], s2_mask, s2["transform"],
               f"Sentinel-2 · 10 m · frozen v3 GNG\n{summary['sentinel2_area_ha']:.2f} ha on the site"),
              (ps["stack"], ps["valid"], rules, ps["transform"],
               f"PlanetScope · 3 m · calibrated NDVI + NDWI\n{summary['planet_area_ha']:.2f} ha on the site")]
    for ax, (stack, valid, mask, transform, title) in zip(axes, panels):
        west, south, east, north = array_bounds(*mask.shape, transform)
        ax.imshow(true_colour(stack, valid), extent=(west, east, south, north),
                  interpolation="nearest", origin="upper")
        xs = transform.c + (np.arange(mask.shape[1]) + 0.5) * transform.a
        ys = transform.f + (np.arange(mask.shape[0]) + 0.5) * transform.e
        if mask.any() and not mask.all():
            ax.contour(xs, ys, mask.astype(float), levels=[0.5], colors=["#ffd447"], linewidths=1.1)
        ax.set(xlim=(window[0], window[2]), ylim=(window[1], window[3]), xticks=[], yticks=[])
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11, loc="left", pad=8)
        x = window[0] + (window[2] - window[0]) * 0.06
        y = window[1] + (window[3] - window[1]) * 0.06
        ax.plot([x, x + 100], [y, y], color="black", lw=5)
        ax.plot([x, x + 100], [y, y], color="white", lw=2)
        ax.text(x + 50, y + (window[3] - window[1]) * 0.02, "100 m", ha="center",
                color="white", fontsize=9,
                bbox={"facecolor": "black", "alpha": 0.65, "pad": 2, "edgecolor": "none"})
    iou = summary["iou"]
    fig.suptitle(f"{summary['site']} · {summary['sentinel2_date']} "
                 f"(PlanetScope {summary['planet_date']})", fontsize=14, x=0.045, ha="left")
    fig.text(0.045, 0.925, f"Densest 400 m window · IoU {iou:.2f}" if iou is not None
             else "Densest 400 m window · Sentinel-2 found nothing here", fontsize=10)
    fig.text(0.045, 0.03, "Candidate bare surface, not confirmed cutting. " + CITATION, fontsize=7.5)
    fig.subplots_adjust(left=0.04, right=0.97, bottom=0.08, top=0.8, wspace=0.06)
    fig.savefig(out / name, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    ap.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--key", help="run one site-year")
    ap.add_argument("--gng", action="store_true",
                    help="also fit the four-band GNG variant on every site (slower)")
    args = ap.parse_args()
    if any(p.is_absolute() for p in (args.manifest, args.release, args.evaluation, args.out)):
        ap.error("paths must be repository-relative")
    out = private_path(ROOT / args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / args.manifest).read_text())
    bog_types = {f["properties"]["code"]: f["properties"].get("bog_type")
                 for f in json.loads((ROOT / "web/data/sites.geojson").read_text())["features"]}
    tiles = {k: v for k, v in sorted(manifest["tiles"].items())
             if not args.key or k == args.key}
    if not tiles:
        ap.error("no site-years selected")

    def example_data(key):
        """Reload one site for a figure: holding every site's rasters would not fit in memory."""
        return run_site(key, tiles[key], ROOT / args.release, ROOT / args.evaluation)[2]

    sites, failures = [], {}
    for key, record in tiles.items():
        try:
            summary, arrays, _ = run_site(key, record, ROOT / args.release,
                                          ROOT / args.evaluation,
                                          bog_type=bog_types.get(key.split("_")[0]),
                                          with_gng=args.gng)
        except Exception as exc:
            failures[key] = f"{type(exc).__name__}: {exc}"
            print(f"{key}: skipped ({failures[key]})", flush=True)
            continue
        sites.append(summary)
        np.savez_compressed(out / f"{key}_masks.npz", **arrays)
        iou = summary["iou"]
        print(f"{key}: 3 m {summary['planet_area_ha']:6.2f} ha · 10 m "
              f"{summary['sentinel2_area_ha']:6.2f} ha · IoU "
              f"{'n/a' if iou is None else f'{iou:.2f}'} · narrow "
              f"{summary['narrow_share'] if summary['narrow_share'] is None else round(summary['narrow_share'], 2)}",
              flush=True)

    totals = aggregate(sites)
    report = {"version": VERSION, "citation": CITATION, "licence": LICENCE,
              "method": {
                  "calibration": "per scene OLS of Sentinel-2 NDVI on PlanetScope NDVI "
                                 "of area-mean bands, all finite nonwater paired 10 m cells",
                  "planet_candidates": "calibrated NDVI < max(0.10, min(0.25, site p75 - 0.18)) "
                                       "with NDWI > 0 water removed",
                  "cloud_screen": "Planet UDM2 clear flag per pixel, frame by frame",
                  "aggregation": "exact area-weighted 3 m to 10 m regridding; a cell is bare at "
                                 f"fraction >= 0.5 with assessed coverage >= {MIN_COVERAGE}",
                  "reference": "frozen v3 GNG mask on the same release scene",
                  "labels_used": False},
              "totals": totals, "failures": failures,
              "limitations": LIMITATIONS, "sites": sites}
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    if sites:
        write_distributions(out, sites, totals)
        ranked = [s for s in sites if s["iou"] is not None]
        if ranked:
            ranked.sort(key=lambda s: s["iou"])
            for name, pick in [("example_agreement.png", ranked[-1]),
                               ("example_disagreement.png", ranked[0])]:
                write_example(out, name, pick["key"], pick, example_data(pick["key"]))
        biggest = max(sites, key=lambda s: s["planet_area_ha"])
        write_example(out, "example_largest.png", biggest["key"], biggest,
                      example_data(biggest["key"]))
    print(json.dumps(totals, indent=2))
    print(f"\n{len(sites)} site-years -> {out.relative_to(ROOT)}"
          + (f"; {len(failures)} skipped" if failures else ""))


if __name__ == "__main__":
    main()
