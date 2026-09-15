"""Freeze guards, unavailable pairs and scoring columns for the seasonal experiment."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from peatland import multitemporal as mt
from peatland import temporal_experiment as te

ROOT = Path(__file__).resolve().parents[1]


def test_add_predictions_leaves_unobserved_points_blank(tmp_path):
    support = np.zeros((3, 3), bool); support[0] = True
    gng = np.zeros((3, 3), bool); gng[0, 1] = True
    np.savez(tmp_path / "000001_2022.npz", support=support,
             prediction_trajectory_gng=gng, prediction_trajectory_rules=gng)
    rows = [{"point_id": "000001_2022_r0_c1", "pixel_row": "0", "pixel_col": "1"},
            {"point_id": "000001_2022_r0_c0", "pixel_row": "0", "pixel_col": "0"},
            {"point_id": "000001_2022_r2_c2", "pixel_row": "2", "pixel_col": "2"},
            {"point_id": "000002_2018_r0_c0", "pixel_row": "0", "pixel_col": "0"}]
    te.add_predictions(rows, tmp_path)
    assert [r["prediction_trajectory_gng"] for r in rows] == ["1", "0", "", ""]


def test_missing_pair_is_unavailable_not_zero(tmp_path):
    run = tmp_path / "run"; run.mkdir()
    v3 = np.zeros((4, 4), bool); v3[0, 0] = True
    np.savez(run / "000001_2022.npz", inside=np.ones((4, 4), bool), prediction_gng=v3)
    meta = {"site": "Test Bog", "year": 2022, "role": "held-out", "scene_date": "2022-04-23"}
    row, result = te.evaluate("000001_2022", meta, run, tmp_path / "out", mt.Parameters())
    assert result is None and row["status"] == "unavailable"
    assert "persistent_gng_ha" not in row and row["v3_gng_ha"] == 0.01


def test_no_early_candidates_is_empty_by_construction(tmp_path):
    run, pairs = tmp_path / "run", tmp_path / "out" / "pairs" / "000001_2022"
    run.mkdir(); pairs.mkdir(parents=True)
    np.savez(run / "000001_2022.npz", inside=np.ones((4, 4), bool), prediction_gng=np.zeros((4, 4), bool))
    green = np.tile(np.array([.04, .07, .03, .35, .18, .09]) * 10000, (4, 4, 1))
    np.savez(pairs / "early.npz", stack=green, valid=np.ones((4, 4), bool))
    (pairs / "early.json").write_text(json.dumps({"status": "available"}))
    (pairs / "late.json").write_text(json.dumps({"status": "no_usable_scene"}))
    meta = {"site": "Green Bog", "year": 2022, "role": "calibration", "scene_date": "2022-04-23"}
    row, result = te.evaluate("000001_2022", meta, run, tmp_path / "out", mt.Parameters())
    assert result is None and row["status"] == "no_early_candidates"
    assert row["persistent_gng_ha"] == 0.0 and row["early_candidates_ha"] == 0.0


def test_freeze_is_blind_and_apply_refuses_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(te, "write_json", lambda p, v: Path(p).write_text(json.dumps(v)))
    metas = {"a_2022": {"role": "calibration"}, "b_2021": {"role": "held-out"}}
    rules = {"fetch_version": "x"}
    out = tmp_path / "out"; out.mkdir()
    with pytest.raises(ValueError, match="freeze first"):
        te.check_frozen(out, rules)
    (out / "pairs" / "b_2021").mkdir(parents=True)
    with pytest.raises(ValueError, match="blind"):
        te.freeze(metas, tmp_path, out, None, rules)
    (out / "pairs" / "b_2021").rmdir()
    te.freeze(metas, tmp_path, out, None, rules)
    assert te.check_frozen(out, rules)[0] == mt.Parameters()
    with pytest.raises(ValueError, match="Scene rules"):
        te.check_frozen(out, {"fetch_version": "y"})
    record = json.loads((out / te.FROZEN).read_text())
    record["parameters"]["max_greenup"] = .5
    (out / te.FROZEN).write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Detector"):
        te.check_frozen(out, rules)


def test_scorer_scores_extra_methods_on_common_support(tmp_path):
    fields = ["point_id", "site", "year", "stratum", "stratum_population",
              "stratum_sample_size", "prediction_gng", "prediction_trajectory_gng", "truth"]
    points, labels, report = tmp_path / "points.csv", tmp_path / "labels.csv", tmp_path / "r.json"
    with points.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for i, value in enumerate(["1", "0", "", "1"]):
            w.writerow(dict(point_id=f"p{i}", site="S", year="2022", stratum="detected",
                            stratum_population="10", stratum_sample_size="4",
                            prediction_gng="1", prediction_trajectory_gng=value, truth=""))
    with labels.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["point_id", "truth"]); w.writeheader()
        for i, truth in enumerate(["bare_peat", "vegetated", "bare_peat", "bare_peat"]):
            w.writerow({"point_id": f"p{i}", "truth": truth})
    subprocess.run([sys.executable, str(ROOT / "scripts/14_score_points.py"), str(points),
                    "--labels", str(labels), "--methods", "gng", "trajectory_gng",
                    "--common-support", "--output", str(report)],
                   check=True, capture_output=True)
    result = json.loads(report.read_text())
    assert result["common_support"]["dropped_points"] == 1
    assert result["methods"]["trajectory_gng"]["sample"]["confusion"] == {"tp": 2, "fp": 0, "fn": 0, "tn": 1}
    assert result["methods"]["gng"]["sample"]["confusion"] == {"tp": 2, "fp": 1, "fn": 0, "tn": 0}
    assert result["weighting_available"] is False
