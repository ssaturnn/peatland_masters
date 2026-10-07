"""Bog-type tagging: Atlantic longitude line plus the upland rule."""

from peatland import boundaries


def test_west_of_line_is_blanket_at_any_height():
    assert boundaries.bog_type(-9.8, 10) == "blanket"
    assert boundaries.bog_type(-9.8) == "blanket"


def test_upland_east_of_line_is_blanket():
    assert boundaries.bog_type(-8.47, 285) == "blanket"     # Slieve Aughty upland
    assert boundaries.bog_type(-8.44, 123) == "raised"      # highest lowland raised bog (p90)
    assert boundaries.bog_type(-8.5) == "raised"            # no elevation known


def test_elevation_table_covers_every_site_and_tags_the_uplands():
    elev = boundaries._site_elevations()
    assert len(elev) >= 80
    uplands = {c for c, e in elev.items() if e >= boundaries.UPLAND_M}
    assert {"000308", "001913", "001229", "002379"} <= uplands   # Slieve Aughty bogs
    assert "000592" not in uplands                            # Bellanagare, a raised bog
