"""Synthetic quota, clip and date-selection checks."""

import importlib.util
from pathlib import Path

import pytest
from shapely.geometry import box
from peatland import planet


def frame(day, delta, geom, clear=100, download=True):
    return dict(date=day, delta=delta, geom=geom, clear=clear, download=download)


def test_choose_prefers_same_day_and_requires_coverage():
    aoi = box(0, 0, 1, 1)
    today = frame("2022-04-23", 0, box(0, 0, .8, 1))
    tomorrow = frame("2022-04-24", 1, aoi)
    assert planet.choose([today, tomorrow], aoi)[0] == "2022-04-24"
    today["geom"] = aoi
    assert planet.choose([today, tomorrow], aoi)[0] == "2022-04-23"
    assert planet.choose([frame("2022-04-23", 0, aoi, download=False)], aoi) is None


def test_choose_combines_only_same_date_and_nonredundant_frames():
    aoi = box(0, 0, 1, 1)
    left, right = box(0, 0, .6, 1), box(.5, 0, 1, 1)
    assert planet.choose([frame("a", 0, left), frame("b", 1, right)], aoi) is None
    result = planet.choose([frame("a", 0, left), frame("a", 0, left), frame("a", 0, right)], aoi)
    assert len(result[1]) == 2
    assert result[2] == pytest.approx(1)


def test_buffer_is_metric_valid_and_under_vertex_limit():
    aoi, _ = planet.buffered_aoi(box(500000, 700000, 501000, 701000))
    assert aoi.is_valid
    assert planet.vertex_count(aoi) < 500
    assert 2 < planet.area_km2(aoi) < 2.3


def test_reservation_accounts_for_pending_and_blocks_cap_or_duplicates():
    ledger = {"orders": {}}
    planet.reserve_area(ledger, "a", 1299)
    planet.reserve_area(ledger, "b", 1)
    assert ledger["reserved_km2"] == 1300
    with pytest.raises(ValueError, match="cap"):
        planet.reserve_area(ledger, "c", .001)
    with pytest.raises(ValueError, match="already"):
        planet.reserve_area(ledger, "a", 1)
    assert len(ledger["orders"]) == 2


def test_planet_data_stays_in_private_folders():
    for path in (planet.ROOT / "web/data/image.tif", planet.ROOT / "docs/image.tif",
                 Path("/tmp/planet-output")):
        with pytest.raises(ValueError):
            planet.private_path(path)
    assert planet.private_path(planet.ROOT / "outputs/planet_release/x.tif").name == "x.tif"


def test_orders_always_clip_and_fallback_only_after_definite_rejection():
    class Response:
        def __init__(self, status): self.status_code = status
        def json(self): return {"id": "test-order"}
    class Session:
        def __init__(self): self.bodies = []
        def post(self, url, json, timeout):
            self.bodies.append(json)
            return Response(400 if len(self.bodies) == 1 else 202)
    s = Session()
    assert planet.place_order(s, "test", ["item"], box(-9, 53, -8.99, 53.01)) == ("test-order", "analytic_8b_sr_udm2")
    assert len(s.bodies) == 2
    assert all(b["tools"][0]["clip"]["aoi"]["type"] == "Polygon" for b in s.bodies)


def test_download_selects_reflectance_or_udm2_files():
    sr, udm, meta = ("o/PSScene/a_3B_AnalyticMS_SR_clip.tif", "o/PSScene/a_3B_udm2_clip.tif",
                     "o/PSScene/a_metadata.json")
    assert planet._wanted(sr, "sr") and not planet._wanted(udm, "sr")
    assert planet._wanted(udm, "udm2") and not planet._wanted(sr, "udm2")
    assert not planet._wanted(meta, "sr") and not planet._wanted(meta, "udm2")


def test_udm2_clear_takes_each_pixel_from_the_first_covering_frame():
    import numpy as np
    first = (np.array([[1, 0, 0]], bool), np.array([[1, 1, 0]], bool))
    second = (np.array([[0, 1, 1]], bool), np.array([[1, 1, 1]], bool))
    clear, covered = planet.first_frame_clear([first, second])
    assert clear.tolist() == [[True, False, True]]
    assert covered.all()
    with pytest.raises(ValueError):
        planet.first_frame_clear([])
