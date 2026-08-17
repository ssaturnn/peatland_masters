"""Tiny static server for the peatland web map (Railway-friendly).

Serves index.html, app.js, style.css and data/sites.geojson. Railway
sets $PORT; locally it defaults to 8000. Point Railway's service "root
directory" at web/ so this folder's requirements.txt / Procfile are used.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

WEB = Path(__file__).parent

app = FastAPI(title="Peatland turf-cutting monitor")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# everything else is static; html=True serves index.html at /
app.mount("/", StaticFiles(directory=WEB, html=True), name="static")
