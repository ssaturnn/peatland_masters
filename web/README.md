# Web map — peatland turf-cutting monitor

Interactive MapLibre map of protected bogs in Galway / Mayo / Roscommon,
showing detected bare-peat area and how much has been newly cut between
two survey years. Reads `data/sites.geojson`, produced by
`scripts/09_export_web.py`.

## Run locally

```bash
cd web
pip install -r requirements.txt
uvicorn server:app --reload --port 8000
# open http://localhost:8000
```

Or, with no dependencies at all (static only):

```bash
cd web
python3 -m http.server 8000
```

## Deploy on Railway

1. Push the repo to GitHub.
2. In Railway: **New Project → Deploy from GitHub repo**, pick this repo.
3. In the service **Settings → Root Directory**, set `web`.
   (So Railway uses `web/requirements.txt`, `web/Procfile`, `web/railway.json`.)
4. Railway builds with Nixpacks and starts
   `uvicorn server:app --host 0.0.0.0 --port $PORT`.
5. Open the generated URL. Health check is at `/healthz`.

## Private area (licensed imagery)

`web/private/` holds material that must not be public, such as figures
with PlanetScope imagery (Planet's Education & Research licence covers
non-commercial research only). The folder is gitignored and served only
at `/private/`, behind HTTP Basic auth. Credentials live in the Railway
environment, never in the repo:

```bash
railway variables --set PRIVATE_USER=... --set PRIVATE_PASSWORD=... --skip-deploys
cd web && railway up --no-gitignore
```

`/private/` itself is the same map: in the bog card, evaluation site-years
with a same-day PlanetScope scene get a swipe between Sentinel-2 10 m and
PlanetScope 3 m. `/private/briefing/` is the supervisor briefing.

The evaluation site-years come from `python3 scripts/22_planet_web_tiles.py`.
The other bogs (latest year, plus the NPWS-documented years) are ordered
under a clipped-area cap and then rendered:

```bash
python3 scripts/23_planet_release.py plan      # search + quota plan, no orders
python3 scripts/23_planet_release.py order     # clipped orders within the cap
python3 scripts/23_planet_release.py collect   # rerun until nothing is pending
python3 scripts/23_planet_release.py udm2      # Planet's cloud masks for the clear share
python3 scripts/22_planet_web_tiles.py --release outputs/planet_release
```

Every tile carries the clear share of the bog from Planet's UDM2 mask. A tile
under 50% clear is withheld, because a cloudy frame would mislead in the
swipe. For the evaluation tiles, fetch their masks first with
`python3 scripts/23_planet_release.py udm2 --out outputs/evaluation/2026-09-13-v3-sample/planet`.
Re-render existing tiles with `--force`.

GitHub autodeploys and a plain `railway up` skip the folder, so
`/private/` then returns 404 after login. Without the two variables the
area stays closed.

## Refresh the data

Re-run the export and redeploy (or commit the new file):

```bash
python3 scripts/09_export_web.py   # rewrites web/data/sites.geojson
```

## Files

| File | Purpose |
|---|---|
| `index.html` | Page shell, sidebar, legend |
| `app.js` | MapLibre map, loads GeoJSON, colours sites by newly-cut area |
| `style.css` | Styling |
| `server.py` | FastAPI static server (Railway) |
| `data/sites.geojson` | Per-site results (generated) |
| `private/` | Licensed material behind a password (gitignored) |
