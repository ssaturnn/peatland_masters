"""Step 26 — NASA FIRMS active fires on the monitored bogs.

Lists every FIRMS active-fire detection within a sensor footprint of the
published bogs, groups them into fire events per bog, and adds context to
each event: the weather of the 30 days before it (Open-Meteo archive,
ERA5-based, at the bog centroid, compared with the same calendar window in
other years) and whether it fell between 1 March and 31 August, when
section 40 of the Wildlife Act prohibits burning vegetation on uncultivated
land. Each event is also matched to the survey years the release publishes
for that bog.

A detection shows that something hot was seen in a footprint of 375 m
(VIIRS) or 1 km (MODIS); it does not say why. Weather and the calendar are
context, never a cause.

Complete years come from the keyless FIRMS country archives (VIIRS S-NPP,
VIIRS NOAA-20 and MODIS, 2018 onwards; a year is published there some months
after it ends). More recent months need the FIRMS area API and a free
MAP_KEY (https://firms.modaps.eosdis.nasa.gov/api/map_key/):

    python3 scripts/26_firms_burns.py
    FIRMS_MAP_KEY=... python3 scripts/26_firms_burns.py --api-start 2025-01-01
"""

import argparse
import csv
import datetime as dt
import io
import json
import os
import statistics
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[1]
GEOJSON = ROOT / "web" / "data" / "sites.geojson"
OUT = ROOT / "outputs" / "firms"
VERSION = "2026-09-28-firms-burns-v1"
ARCHIVE = ("https://firms.modaps.eosdis.nasa.gov/data/country/"
           "{sensor}/{year}/{sensor}_{year}_Ireland.csv")
ARCHIVE_SENSORS = ("viirs-snpp", "viirs-jpss1", "modis")   # jpss1 = NOAA-20
API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{bbox}/{days}/{date}"
API_SOURCES = ("VIIRS_SNPP_SP", "VIIRS_NOAA20_SP", "MODIS_SP",
               "VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT")
API_MAX_DAYS = 10              # the area API returns at most ten days per call
WEATHER = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat:.4f}"
           "&longitude={lon:.4f}&start_date={start}&end_date={end}"
           "&daily=precipitation_sum,temperature_2m_mean&timezone=Europe%2FDublin")
GAP_DAYS = 3                   # detections on one bog closer than this are one event
WINDOW_DAYS = 30               # weather window before an event
WET_MM = 1.0                   # a day with at least this much rain ends a dry spell
DRY_SPELL_DAYS = 10            # heuristic flags, descriptive only
DRY_RATIO = 0.5
DUBLIN = ZoneInfo("Europe/Dublin")
NON_VEGETATION_TYPES = {"1", "2", "3"}   # volcano, static land source, offshore
LIMITATIONS = [
    "Footprints are large: a VIIRS detection stands for a pixel of about 375 m and a "
    "MODIS one for about 1 km, far coarser than a cutting strip. A detection within the "
    "buffer may be a fire just outside the bog.",
    "Omission is common: small or smouldering fires, fires under cloud and fires that "
    "start and end between satellite overpasses are often not detected. The absence of "
    "a detection does not mean that a bog did not burn.",
    "Commission errors are possible (hot surfaces, sun glint); FIRMS confidence is "
    "recorded with every detection.",
    "Satellites do not show why a fire started. Weather and the closed season are "
    "context only.",
    "The weather is modelled (ERA5-based reanalysis at the bog centroid), not measured "
    "on the bog.",
]


# ---- pure functions -------------------------------------------------------

def archive_url(sensor, year):
    return ARCHIVE.format(sensor=sensor, year=year)


def api_urls(key, source, bbox, start, end):
    """Area-API calls covering start..end inclusive, ten days at a time."""
    if end < start:
        return []
    area = ",".join(f"{v:.4f}" for v in bbox)
    urls, day = [], start
    while day <= end:
        days = min(API_MAX_DAYS, (end - day).days + 1)
        urls.append(API.format(key=key, source=source, bbox=area, days=days, date=day.isoformat()))
        day += dt.timedelta(days=days)
    return urls


def footprint_m(instrument):
    """Nominal pixel size, used as the buffer around each bog."""
    return 1000 if "MODIS" in instrument.upper() else 375


def normalise(row, source):
    """One FIRMS CSV row (VIIRS or MODIS layout) as a uniform record."""
    instrument = row.get("instrument") or ("MODIS" if "brightness" in row else "VIIRS")
    when = dt.datetime.strptime(row["acq_date"] + row["acq_time"].zfill(4), "%Y-%m-%d%H%M")
    utc = when.replace(tzinfo=dt.timezone.utc)
    frp = row.get("frp")
    return {
        "latitude": float(row["latitude"]), "longitude": float(row["longitude"]),
        "date": row["acq_date"], "time_utc": utc.strftime("%H:%M"),
        "time_local": utc.astimezone(DUBLIN).strftime("%Y-%m-%d %H:%M"),
        "satellite": row.get("satellite", ""), "instrument": instrument,
        "confidence": row.get("confidence", ""), "frp": float(frp) if frp else None,
        "daynight": row.get("daynight", ""), "type": row.get("type", ""),
        "source": source, "footprint_m": footprint_m(instrument),
    }


def dedupe(records):
    """Drop repeats of one pass delivered by two products (archive and NRT)."""
    seen, out = set(), []
    for r in records:
        key = (round(r["latitude"], 3), round(r["longitude"], 3), r["date"],
               r["time_utc"], r["satellite"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def group_events(detections, gap_days=GAP_DAYS):
    """Detections on one bog within gap_days of each other form one event."""
    by_bog = {}
    for d in sorted(detections, key=lambda d: (d["code"], d["date"], d["time_utc"])):
        by_bog.setdefault(d["code"], []).append(d)
    events = []
    for code, rows in by_bog.items():
        current = None
        for d in rows:
            day = dt.date.fromisoformat(d["date"])
            if current is None or (day - current["_last"]).days > gap_days:
                current = {"code": code, "name": d["name"], "start": d["date"], "end": d["date"],
                           "detections": 0, "instruments": set(), "max_frp": None,
                           "min_distance_m": None, "confidence": {}, "_last": day}
                events.append(current)
            current["end"], current["_last"] = d["date"], day
            current["detections"] += 1
            current["instruments"].add(d["instrument"])
            if d.get("frp") is not None:
                current["max_frp"] = max(current["max_frp"] or 0.0, d["frp"])
            dist = d.get("distance_m")
            if dist is not None:
                current["min_distance_m"] = dist if current["min_distance_m"] is None \
                    else min(current["min_distance_m"], dist)
            conf = str(d.get("confidence", ""))
            current["confidence"][conf] = current["confidence"].get(conf, 0) + 1
    for e in events:
        e.pop("_last")
        e["instruments"] = sorted(e["instruments"])
        e["inside_bog"] = e["min_distance_m"] == 0
    return events


def in_closed_season(day):
    """1 March to 31 August: burning vegetation on uncultivated land is prohibited."""
    return dt.date(day.year, 3, 1) <= day <= dt.date(day.year, 8, 31)


def weather_summary(days, precip, temp, event_day, window=WINDOW_DAYS, wet_mm=WET_MM):
    """Rain and temperature over the `window` days before event_day, and the
    length of the dry spell that preceded it. Days with missing values count as
    unknown, never as dry."""
    series = {dt.date.fromisoformat(d): (p, t) for d, p, t in zip(days, precip, temp)}
    wanted = [event_day - dt.timedelta(days=k) for k in range(window, 0, -1)]
    rain = [series[d][0] for d in wanted if d in series and series[d][0] is not None]
    warm = [series[d][1] for d in wanted if d in series and series[d][1] is not None]
    since = None
    for k in range(1, window + 1):
        p = series.get(event_day - dt.timedelta(days=k), (None, None))[0]
        if p is None:
            break
        if p >= wet_mm:
            since = k - 1
            break
    complete = len(rain) == window
    return {"precip_mm": round(sum(rain), 1) if complete else None,
            "tmean_c": round(statistics.fmean(warm), 1) if warm else None,
            "dry_days_before": since if since is not None else (f">={window}" if complete else None),
            "days_with_data": len(rain)}


def window_normal(days, precip, event_day, window=WINDOW_DAYS):
    """Median rain of the same calendar window in the other years of the record."""
    series = {dt.date.fromisoformat(d): p for d, p in zip(days, precip)}
    years = sorted({dt.date.fromisoformat(d).year for d in days})
    totals = []
    for y in years:
        if y == event_day.year:
            continue
        try:
            end = event_day.replace(year=y)
        except ValueError:       # 29 February
            end = event_day.replace(year=y, day=28)
        wanted = [end - dt.timedelta(days=k) for k in range(window, 0, -1)]
        values = [series.get(d) for d in wanted]
        if all(v is not None for v in values):
            totals.append(sum(values))
    return round(statistics.median(totals), 1) if totals else None


def dry_flag(summary, normal):
    """Descriptive only: a long dry spell or under half the usual rain."""
    spell = summary["dry_days_before"]
    long_spell = spell is not None and (isinstance(spell, str) or spell >= DRY_SPELL_DAYS)
    low_rain = (summary["precip_mm"] is not None and normal
                and summary["precip_mm"] < DRY_RATIO * normal)
    return bool(long_spell or low_rain)


def published_context(event, props):
    """Where the event sits among the survey years the release publishes."""
    start = dt.date.fromisoformat(event["start"])
    years, dates, gng = props["years"], props["dates"], props["gng_series"]
    ctx = {"year_published": start.year in years}
    if start.year in years:
        i = years.index(start.year)
        scene = dt.date.fromisoformat(dates[i])
        ctx.update(scene_date=dates[i], gng_ha_that_year=gng[i],
                   fire_before_scene=start <= scene)
    if start.year + 1 in years:
        j = years.index(start.year + 1)
        ctx.update(next_year_scene=dates[j], gng_ha_next_year=gng[j])
    return ctx


# ---- input / output -------------------------------------------------------

class NotPublished(Exception):
    """The archive has no file for this sensor and year yet."""


def fetch(url, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=120)
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 404:
            raise NotPublished(url)
        if r.status_code == 429 and attempt < retries - 1:   # rate limit: back off
            time.sleep(65 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.text


def read_csv(text, source):
    return [normalise(row, source) for row in csv.DictReader(io.StringIO(text))
            if row.get("latitude")]


def archive_detections(years, cache):
    records, used = [], []
    for sensor in ARCHIVE_SENSORS:
        for year in years:
            path = cache / f"{sensor}_{year}.csv"
            if not path.exists():
                try:
                    path.write_text(fetch(archive_url(sensor, year)))
                except NotPublished:
                    print(f"  {sensor} {year}: not in the archive yet", flush=True)
                    continue
            rows = read_csv(path.read_text(), f"archive:{sensor}")
            records += rows
            used.append({"source": f"archive:{sensor}", "year": year, "rows": len(rows),
                         "url": archive_url(sensor, year)})
    return records, used


def api_detections(key, bbox, start, end, cache):
    records, used = [], []
    for source in API_SOURCES:
        rows_for_source = 0
        for url in api_urls(key, source, bbox, start, end):
            text = fetch(url)
            if text.lstrip().startswith(("Invalid", "Error")):
                raise SystemExit(f"FIRMS API refused {source}: check FIRMS_MAP_KEY")
            rows = read_csv(text, f"api:{source}")
            rows_for_source += len(rows)
            records += rows
        used.append({"source": f"api:{source}", "start": start.isoformat(),
                     "end": end.isoformat(), "rows": rows_for_source})
        (cache / f"api_{source}_{start}_{end}.json").write_text(json.dumps(used[-1]))
    return records, used


def match_bogs(records, bogs):
    """Every detection inside its sensor footprint around a bog, with the distance."""
    import geopandas as gpd
    from shapely.geometry import Point
    if not records:
        return []
    points = gpd.GeoDataFrame(records, geometry=[Point(r["longitude"], r["latitude"])
                                                 for r in records], crs=4326).to_crs(2157)
    out = []
    for size in sorted({r["footprint_m"] for r in records}):
        group = points[points["footprint_m"] == size]
        zones = bogs.copy()
        zones["geometry"] = bogs.geometry.buffer(size)
        hits = gpd.sjoin(group, zones[["code", "name", "geometry"]], predicate="within")
        for idx, hit in hits.iterrows():
            shape = bogs.loc[bogs["code"] == hit["code"], "geometry"].iloc[0]
            row = {k: hit[k] for k in records[0].keys()}
            row.update(code=hit["code"], name=hit["name"],
                       distance_m=round(float(group.loc[idx, "geometry"].distance(shape)), 1))
            out.append(row)
    return out


def weather_window(lat, lon, start, end, cache):
    """Daily rain and temperature for one short window, cached on disk: the free
    archive weighs calls by their length, so only the windows needed are asked for."""
    path = cache / f"{lat:.3f}_{lon:.3f}_{start}_{end}.json"
    if path.exists():
        return json.loads(path.read_text())
    data = json.loads(fetch(WEATHER.format(lat=lat, lon=lon, start=start, end=end),
                            retries=5))["daily"]
    path.write_text(json.dumps(data))
    return data


def weather_for(bog_centroid, events, first_year, last_year, cache):
    lat, lon = bog_centroid
    newest = dt.date.today() - dt.timedelta(days=7)
    for e in events:
        day = dt.date.fromisoformat(e["start"])
        data = {"time": [], "precipitation_sum": [], "temperature_2m_mean": []}
        for year in range(first_year - 2, last_year + 1):
            try:
                end = day.replace(year=year) - dt.timedelta(days=1)
            except ValueError:   # 29 February
                end = day.replace(year=year, day=28) - dt.timedelta(days=1)
            if end > newest:
                continue
            window = weather_window(lat, lon, end - dt.timedelta(days=WINDOW_DAYS - 1), end, cache)
            for k in data:
                data[k] += window[k]
        summary = weather_summary(data["time"], data["precipitation_sum"],
                                  data["temperature_2m_mean"], day)
        normal = window_normal(data["time"], data["precipitation_sum"], day)
        e["weather_30d"] = {**summary, "precip_normal_mm": normal,
                            "dry": dry_flag(summary, normal)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first-year", type=int, default=2018)
    ap.add_argument("--last-year", type=int, default=dt.date.today().year)
    ap.add_argument("--api-start", type=dt.date.fromisoformat,
                    help="also query the FIRMS area API from this date (needs FIRMS_MAP_KEY)")
    ap.add_argument("--gap-days", type=int, default=GAP_DAYS)
    ap.add_argument("--no-weather", action="store_true")
    args = ap.parse_args()

    import geopandas as gpd
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "raw"
    cache.mkdir(exist_ok=True)
    weather_cache = OUT / "weather"
    weather_cache.mkdir(exist_ok=True)
    bogs = gpd.read_file(GEOJSON)[["code", "name", "geometry"]].to_crs(2157)
    props = {f["properties"]["code"]: f["properties"]
             for f in json.loads(GEOJSON.read_text())["features"]}
    centroids = {c: (g.y, g.x) for c, g in
                 zip(bogs["code"], bogs.geometry.centroid.to_crs(4326))}

    years = range(args.first_year, args.last_year + 1)
    records, used = archive_detections(years, cache)
    if args.api_start:
        key = os.environ.get("FIRMS_MAP_KEY")
        if not key:
            raise SystemExit("--api-start needs FIRMS_MAP_KEY (free: "
                             "https://firms.modaps.eosdis.nasa.gov/api/map_key/)")
        west, south, east, north = bogs.to_crs(4326).total_bounds
        bbox = (west - 0.05, south - 0.05, east + 0.05, north + 0.05)
        api_rows, api_used = api_detections(key, bbox, args.api_start,
                                            dt.date.today() - dt.timedelta(days=1), cache)
        records += api_rows
        used += api_used
    records = dedupe(records)
    covered = sorted(r["date"] for r in records)
    print(f"{len(records)} detections over Ireland, "
          f"{covered[0] if covered else '-'} to {covered[-1] if covered else '-'}", flush=True)

    hits = match_bogs(records, bogs)
    vegetation = [h for h in hits if str(h.get("type", "")) not in NON_VEGETATION_TYPES]
    events = group_events(vegetation, args.gap_days)
    for e in events:
        e["closed_season"] = in_closed_season(dt.date.fromisoformat(e["start"]))
        e["published"] = published_context(e, props[e["code"]])
    if not args.no_weather:
        for code in sorted({e["code"] for e in events}):
            mine = [e for e in events if e["code"] == code]
            weather_for(centroids[code], mine, args.first_year, args.last_year, weather_cache)
            print(f"  weather for {mine[0]['name']}: {len(mine)} event(s)", flush=True)

    fields = ["code", "name", "date", "time_local", "time_utc", "instrument", "satellite",
              "confidence", "frp", "daynight", "type", "distance_m", "footprint_m",
              "latitude", "longitude", "source"]
    with (OUT / "detections.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(hits, key=lambda h: (h["date"], h["code"])))

    per_year = {}
    for e in events:
        y = e["start"][:4]
        per_year.setdefault(y, {"events": 0, "bogs": set(), "inside_bog": 0})
        per_year[y]["events"] += 1
        per_year[y]["bogs"].add(e["code"])
        per_year[y]["inside_bog"] += e["inside_bog"]
    per_year = {y: {**v, "bogs": len(v["bogs"])} for y, v in sorted(per_year.items())}
    report = {
        "version": VERSION, "sources": used,
        "coverage": {"first_detection": covered[0] if covered else None,
                     "last_detection": covered[-1] if covered else None,
                     "archive_years": [u["year"] for u in used if "year" in u]},
        "method": {"footprint_buffer_m": {"VIIRS": 375, "MODIS": 1000},
                   "event_gap_days": args.gap_days, "weather_window_days": WINDOW_DAYS,
                   "wet_day_mm": WET_MM, "dry_flag": f"dry spell >= {DRY_SPELL_DAYS} days or "
                   f"rain < {DRY_RATIO} x the same window's median in other years",
                   "closed_season": "1 March - 31 August (Wildlife Act s.40)",
                   "excluded_types": "FIRMS type 1-3 (volcano, static land source, offshore)"},
        "counts_by_year": per_year, "detections_on_bogs": len(hits),
        "events": events, "limitations": LIMITATIONS,
    }
    (OUT / "events.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(per_year, indent=2))
    print(f"\n{len(hits)} detections on {len({h['code'] for h in hits})} bogs, "
          f"{len(events)} events -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
