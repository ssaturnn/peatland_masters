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
