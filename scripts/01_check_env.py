"""Step 1 — environment check.

Verifies the geospatial stack imports and prints versions. This is the
first place a blocker shows up (GDAL / rasterio wheels on macOS).
"""

import sys


def check(name, attr="__version__"):
    try:
        mod = __import__(name)
        ver = getattr(mod, attr, "?")
        print(f"  ok    {name:<20} {ver}")
        return True
    except Exception as e:
        print(f"  FAIL  {name:<20} {type(e).__name__}: {e}")
        return False


def main():
    print(f"Python {sys.version.split()[0]}")
    print("Geospatial stack:")
    results = [
        check("numpy"),
        check("rasterio"),
        check("geopandas"),
        check("shapely"),
        check("pyproj"),
        check("sklearn"),
    ]
    print("STAC / imagery access:")
    results += [
        check("pystac_client"),
        check("planetary_computer"),
        check("rioxarray"),
    ]

    ok = sum(bool(r) for r in results)
    print(f"\n{ok}/{len(results)} imports succeeded")
    sys.exit(0 if ok == len(results) else 1)


if __name__ == "__main__":
    main()
