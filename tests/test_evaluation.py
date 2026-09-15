import pytest

from peatland.evaluation import score_points


def point(pid, stratum, truth, population=10):
    return {"point_id": pid, "site": "Bog", "year": "2022", "stratum": stratum,
            "truth": truth, "stratum_population": str(population),
            "stratum_sample_size": "2"}


def test_empty_truth_is_pending_not_false_negative():
    report = score_points([point("a", "detected", ""), point("b", "undetected", "unsure")])
    assert report["status"] == "awaiting_labels"
    assert report["methods"] == {}
    assert report["labelled_points"] == 0


def test_stratified_weights_change_recall():
    rows = [point("a", "detected", "bare_peat"), point("b", "detected", "water"),
            point("c", "undetected", "bare_peat", 90), point("d", "undetected", "vegetated", 90)]
    report = score_points(rows)
    assert report["methods"]["gng"]["sample"]["recall"] == 0.5
    assert report["methods"]["gng"]["weighted"]["recall"] == pytest.approx(0.1)


def test_legacy_rows_have_only_sample_metrics():
    rows = [point("a", "detected", "bare_peat")]
    rows[0].pop("stratum_population")
    report = score_points(rows)
    assert report["methods"]["gng"]["sample"]["precision"] == 1
    assert report["methods"]["gng"]["weighted"] is None
    assert "kmeans" not in report["methods"]


def test_undefined_precision_is_null():
    report = score_points([point("a", "undetected", "vegetated")])
    assert report["methods"]["gng"]["sample"]["precision"] is None


@pytest.mark.parametrize("bad", ["typo", "0"])
def test_invalid_labels_fail(bad):
    with pytest.raises(ValueError, match="unknown truth"):
        score_points([point("a", "detected", bad)])


def test_duplicate_points_fail():
    with pytest.raises(ValueError, match="duplicate"):
        score_points([point("a", "detected", "water")] * 2)


def test_all_methods_use_same_reference_labels():
    rows = [point("a", "detected", "bare_peat"), point("b", "detected", "water")]
    for row in rows:
        row.update(prediction_ndvi="1", prediction_kmeans="0", prediction_rules="0")
    result = score_points(rows)["methods"]
    assert set(result) == {"ndvi", "rules", "kmeans", "gng"}
    assert result["ndvi"]["sample"]["confusion"]["fp"] == 1
    assert result["kmeans"]["sample"]["confusion"]["fn"] == 1
