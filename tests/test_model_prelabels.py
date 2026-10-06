"""Merging two blind model passes into review proposals."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "27_model_prelabels.py"
_spec = importlib.util.spec_from_file_location("model_prelabels", SCRIPT)
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)


def rec(label, confidence="high"):
    return {"label": label, "confidence": confidence, "reason": "r"}


def test_agreement_keeps_the_label_at_the_lower_confidence():
    assert mp.consensus(rec("water", "high"), rec("water", "medium")) == ("water", "medium", "agree")


def test_unsure_defers_and_confidence_breaks_disagreement():
    assert mp.consensus(rec("unsure"), rec("bare_peat", "medium")) == ("bare_peat", "low", "disagree")
    assert mp.consensus(rec("vegetated", "high"), rec("bare_peat", "low"))[0] == "vegetated"


def test_tie_uses_the_earlier_annotator_only_when_it_sides_with_a_pass():
    a, b = rec("vegetated", "medium"), rec("bare_peat", "medium")
    assert mp.consensus(a, b, rec("bare_peat"))[0] == "bare_peat"
    assert mp.consensus(a, b, rec("water"))[0] == "unsure"
    assert mp.consensus(a, b)[0] == "unsure"


def test_missing_passes():
    assert mp.consensus(None, None) == (None, None, "missing")
    assert mp.consensus(rec("burn"), None) == ("burn", "low", "single pass")


def test_kappa_matches_hand_calculation():
    pairs = [("a", "a")] * 3 + [("a", "b")] + [("b", "b")] * 2
    # observed 5/6; expected (4*3 + 2*3)/36 = 0.5
    assert mp.cohen_kappa(pairs) == pytest.approx((5 / 6 - 0.5) / 0.5)
    assert mp.cohen_kappa([]) is None
    assert mp.cohen_kappa([("a", "a")]) is None


def test_load_jsonl_tolerates_joined_objects(tmp_path):
    f = tmp_path / "p.jsonl"
    f.write_text('{"n": 1, "label": "Water"}{"n": 2, "label": "nonsense"}\n{"n": 3, "label": "burn"}\n')
    out = mp.load_jsonl([f])
    assert sorted(out) == [1, 3] and out[1]["label"] == "water"
