"""Project-wide constants: region, coordinate systems, band names, paths."""

from pathlib import Path

# --- Coordinate reference systems ---------------------------------------
ITM = "EPSG:2157"      # Irish Transverse Mercator — NPWS polygons use this
WGS84 = "EPSG:4326"    # lat/lon for GPS output
WEBMERC = "EPSG:3857"  # web-map display

# --- Study region: West of Ireland (Galway / Mayo / Roscommon) ----------
# Bounding box in ITM, taken from the extent of NPWS NHA bog sites in the
# three counties (measured in the early POC): ~144 km x 145 km.
REGION_BBOX_ITM = (460_000, 691_000, 605_000, 836_000)  # (minx, miny, maxx, maxy)

# Same box in WGS84 (approximate), for STAC imagery search.
REGION_BBOX_WGS84 = (-10.30, 53.15, -8.00, 54.55)

TARGET_COUNTIES = {"ga": "Galway", "ma": "Mayo", "ro": "Roscommon"}

# --- Sentinel-2 bands we care about -------------------------------------
# Names as exposed by the Planetary Computer / element84 STAC assets.
S2_BANDS = {
    "blue": "B02",   # 10 m
    "green": "B03",  # 10 m
    "red": "B04",    # 10 m
    "nir": "B08",    # 10 m
    "swir1": "B11",  # 20 m
    "swir2": "B12",  # 20 m
    "scl": "SCL",    # scene classification (cloud mask)
}

# --- Paths --------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "outputs"

# NPWS boundary shapefile (downloaded in the early POC; copied under data/).
NPWS_NHA_SHP = DATA_DIR / "NHA_ITM_2019_06.shp"
