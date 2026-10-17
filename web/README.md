# Web map — peatland turf-cutting monitor

Interactive MapLibre map of protected bogs in Galway / Mayo / Roscommon,
showing exposed peat mapped by detector v5 (Sentinel-2 SWIR, one clear summer
scene per year, 2018–2026), with PlanetScope 3 m imagery for comparison. Reads
`data/sites.geojson` and the card tiles in `data/tiles-v5/`, produced by
`scripts/36_v5_release.py` and `scripts/46_compare_tiles.py`.

The whole site is behind HTTP Basic auth (see below); only `/healthz` is open.

## Run locally

```bash
cd web
pip install -r requirements.txt
PRIVATE_USER=me PRIVATE_PASSWORD=local uvicorn server:app --reload --port 8000
# open http://localhost:8000 and log in with those values
```

## Deploy on Railway

1. Push the repo to GitHub.
2. In Railway: **New Project → Deploy from GitHub repo**, pick this repo.
3. In the service **Settings → Root Directory**, set `web`.
   (So Railway uses `web/requirements.txt`, `web/Procfile`, `web/railway.json`.)
4. Railway builds with Nixpacks and starts
   `uvicorn server:app --host 0.0.0.0 --port $PORT --proxy-headers ...`.
5. Open the generated URL. Health check is at `/healthz`.

## Password and licensed imagery

Every path except `/healthz` needs the Basic-auth credentials; `/` and
`/index.html` redirect to `/private/`. From `web/` itself only `index.html`,
`app.js`, `style.css` and `data/` are served. `web/private/` holds material
that must not be public, such as figures with PlanetScope imagery (Planet's
Education & Research licence covers non-commercial research only); it is
gitignored. Credentials live in the Railway environment, never in the repo:

```bash
railway variables --set PRIVATE_USER=... --set PRIVATE_PASSWORD=... --skip-deploys
cd web && railway up --no-gitignore
```

`/private/` is the map with PlanetScope layers: the bog card has a swipe
between Sentinel-2 10 m and PlanetScope 3 m when a frame lies within 10 days of
the summer scene, and the "Compare this summer" panel. `/private/findings/` and
`/private/briefing/` are the results pages.

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

```bash
python3 scripts/36_v5_release.py          # build v5 card tiles (resumable)
python3 scripts/36_v5_release.py --write  # merge into sites.geojson
python3 scripts/46_compare_tiles.py swir  # SWIR false-colour tiles
```

## Files

| File | Purpose |
|---|---|
| `index.html` | Page shell, sidebar, legend |
| `app.js` | MapLibre map, bog cards, compare panel, PlanetScope swipe |
| `style.css` | Styling |
| `server.py` | FastAPI static server (Railway) |
| `data/sites.geojson` | Per-site results (generated) |
| `data/tiles-v5/` | Card tiles: v5 mask, natural colour (`c`), SWIR (`s`) |
| `private/` | Licensed material behind a password (gitignored) |
