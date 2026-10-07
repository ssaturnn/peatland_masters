"""Summer-checked reading of the reference labels."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("acc", ROOT / "scripts/28_accuracy_report.py")
acc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acc)


def test_only_bare_labels_green_in_summer_change(tmp_path):
    check = tmp_path / "summer.json"
    check.write_text(json.dumps({"green_ndvi": 0.5, "points": [
        {"point_id": "a", "summer_ndvi": 0.8}, {"point_id": "b", "summer_ndvi": 0.2},
        {"point_id": "c", "summer_ndvi": 0.9}, {"point_id": "d", "summer_ndvi": None}]}))
    rows = [{"point_id": "a", "truth": "bare_peat"}, {"point_id": "b", "truth": "bare_peat"},
            {"point_id": "c", "truth": "water"}, {"point_id": "d", "truth": "bare_peat"}]
    out, changed = acc.summer_checked(rows, check)
    assert changed == 1
    assert [r["truth"] for r in out] == ["vegetated", "bare_peat", "water", "bare_peat"]


def test_no_summer_check_file_means_no_reading(tmp_path):
    assert acc.summer_checked([], tmp_path / "missing.json") == (None, 0)
