"""Synthetic checks for the FIRMS burn layer (scripts/26_firms_burns.py)."""

import datetime as dt
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/26_firms_burns.py"
SPEC = importlib.util.spec_from_file_location("firms_burns", SCRIPT)
firms = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(firms)

D = dt.date


def viirs_row(**kw):
    row = {"latitude": "53.5", "longitude": "-8.5", "bright_ti4": "320", "scan": "0.4",
           "track": "0.4", "acq_date": "2020-06-03", "acq_time": "124", "satellite": "N",
           "instrument": "VIIRS", "confidence": "n", "version": "2", "bright_ti5": "290",
           "frp": "12.5", "daynight": "D", "type": "0"}
    row.update(kw)
    return row


def test_api_calls_cover_the_range_ten_days_at_a_time():
    urls = firms.api_urls("KEY", "VIIRS_SNPP_NRT", (-10, 53, -8, 54), D(2025, 5, 1), D(2025, 5, 23))
    assert [u.rsplit("/", 2)[-2:] for u in urls] == [
        ["10", "2025-05-01"], ["10", "2025-05-11"], ["3", "2025-05-21"]]
    assert "/KEY/VIIRS_SNPP_NRT/-10.0000,53.0000,-8.0000,54.0000/" in urls[0]
    assert firms.api_urls("KEY", "MODIS_NRT", (0, 0, 1, 1), D(2025, 5, 2), D(2025, 5, 1)) == []


def test_rows_are_normalised_with_local_summer_time_and_footprint():
    r = firms.normalise(viirs_row(), "archive:viirs-snpp")
    assert (r["time_utc"], r["time_local"]) == ("01:24", "2020-06-03 02:24")
    assert r["footprint_m"] == 375 and r["frp"] == 12.5
    modis = {k: v for k, v in viirs_row(instrument="MODIS", acq_date="2020-01-10",
                                        acq_time="1302").items() if k != "bright_ti4"}
    m = firms.normalise({**modis, "brightness": "310"}, "archive:modis")
    assert m["footprint_m"] == 1000 and m["time_local"] == "2020-01-10 13:02"   # winter: UTC


def test_duplicate_passes_are_dropped():
    a = firms.normalise(viirs_row(), "archive:viirs-snpp")
    b = firms.normalise(viirs_row(latitude="53.50004"), "api:VIIRS_SNPP_SP")
    c = firms.normalise(viirs_row(acq_time="1302"), "archive:viirs-snpp")
    assert len(firms.dedupe([a, b, c])) == 2


def test_events_split_on_gaps_and_keep_the_nearest_distance():
    def det(day, dist, conf="n"):
        return {"code": "1", "name": "Bog", "date": day, "time_utc": "13:00",
                "instrument": "VIIRS", "frp": 2.0, "distance_m": dist, "confidence": conf}
    events = firms.group_events([det("2020-06-03", 120.0), det("2020-05-31", 0.0, "h"),
                                 det("2020-06-10", 50.0)], gap_days=3)
    assert [(e["start"], e["end"], e["detections"]) for e in events] == [
        ("2020-05-31", "2020-06-03", 2), ("2020-06-10", "2020-06-10", 1)]
    assert events[0]["inside_bog"] and not events[1]["inside_bog"]
    assert events[0]["confidence"] == {"h": 1, "n": 1}


@pytest.mark.parametrize("day, closed", [(D(2020, 2, 29), False), (D(2020, 3, 1), True),
                                         (D(2020, 8, 31), True), (D(2020, 9, 1), False)])
def test_closed_season_is_1_march_to_31_august(day, closed):
    assert firms.in_closed_season(day) is closed


def daily(start, values):
    days = [(start + dt.timedelta(days=k)).isoformat() for k in range(len(values))]
    return days, values, [10.0] * len(values)


def test_dry_spell_and_totals_over_the_window_before_the_event():
    rain = [5.0] * 20 + [0.0] * 10          # 30 days, the last ten dry
    days, precip, temp = daily(D(2020, 5, 1), rain)
    s = firms.weather_summary(days, precip, temp, D(2020, 5, 31))
    assert s == {"precip_mm": 100.0, "tmean_c": 10.0, "dry_days_before": 10,
                 "days_with_data": 30}
    wet = firms.weather_summary(*daily(D(2020, 5, 1), [0.0] * 29 + [3.0]), D(2020, 5, 31))
    assert wet["dry_days_before"] == 0
    parched = firms.weather_summary(*daily(D(2020, 5, 1), [0.0] * 30), D(2020, 5, 31))
    assert parched["dry_days_before"] == ">=30"


def test_missing_days_never_count_as_dry():
    days, precip, temp = daily(D(2020, 5, 1), [2.0] * 25 + [None] * 5)
    s = firms.weather_summary(days, precip, temp, D(2020, 5, 31))
    assert s["precip_mm"] is None and s["dry_days_before"] is None and s["days_with_data"] == 25


def test_normal_is_the_median_of_the_same_window_in_other_years():
    days, precip = [], []
    for year, mm in [(2018, 1.0), (2019, 2.0), (2020, 9.0), (2021, 4.0)]:
        d, p, _ = daily(D(year, 5, 1), [mm] * 30)
        days += d
        precip += p
    assert firms.window_normal(days, precip, D(2020, 5, 31)) == 60.0    # median of 30, 60, 120
    assert firms.window_normal(days, precip, D(2016, 5, 31)) == 90.0    # median of all four


def test_dry_flag_needs_a_long_spell_or_half_the_usual_rain():
    base = {"precip_mm": 60.0, "dry_days_before": 3}
    assert not firms.dry_flag(base, 100.0)
    assert firms.dry_flag({**base, "dry_days_before": 12}, 100.0)
    assert firms.dry_flag({**base, "precip_mm": 30.0}, 100.0)
    assert firms.dry_flag({**base, "dry_days_before": ">=30"}, None)


def test_events_are_placed_among_the_published_years():
    props = {"years": [2020, 2021], "dates": ["2020-04-15", "2021-04-25"],
             "gng_series": [0.31, 5.72]}
    after = firms.published_context({"start": "2020-06-01"}, props)
    assert after == {"year_published": True, "scene_date": "2020-04-15", "gng_ha_that_year": 0.31,
                     "fire_before_scene": False, "next_year_scene": "2021-04-25",
                     "gng_ha_next_year": 5.72}
    assert firms.published_context({"start": "2019-05-01"}, props)["year_published"] is False
