"""Tiny static server for the peatland web map (Railway-friendly).

Serves index.html, app.js, style.css and data/sites.geojson. Railway
sets $PORT; locally it defaults to 8000. Point Railway's service "root
directory" at web/ so this folder's requirements.txt / Procfile are used.

The whole site sits behind HTTP Basic auth (PRIVATE_USER /
PRIVATE_PASSWORD in the Railway environment); only /healthz is open, for
the platform health check. "/" redirects to /private/, the map with
same-day PlanetScope 3 m imagery. web/private/ holds that licensed
material (PlanetScope is for non-commercial research only); the folder is
gitignored and never served by the root mount. Deploy with
`railway up --no-gitignore` so the folder is uploaded.
"""

import base64
import os
import secrets
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

WEB = Path(__file__).parent
PRIVATE_PREFIX = "/private"


def _inside(path: str, directory: str) -> bool:
    # lower() also covers /PRIVATE/... on case-insensitive local disks
    path, directory = os.path.realpath(path).lower(), directory.lower()
    return os.path.commonpath([path, directory]) == directory


# the only files served from web/ itself: the map page, its script and styles,
# and the data folder (never server code, config or the private folder)
SITE_FILES = {"index.html", "app.js", "style.css"}
SITE_DIRS = {"data"}


class PublicFiles(StaticFiles):
    """Map assets in web/: the page, script, styles and data/, nothing else."""

    def __init__(self, *, private_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.private_dir = private_dir

    def lookup_path(self, path):
        full_path, stat_result = super().lookup_path(path)
        if not full_path or _inside(full_path, self.private_dir):
            return "", None
        rel = os.path.relpath(os.path.realpath(full_path), os.path.realpath(self.directory))
        parts = rel.split(os.sep)
        if not (rel in SITE_FILES or (len(parts) > 1 and parts[0] in SITE_DIRS)):
            return "", None
        return full_path, stat_result


def authorised(header: str | None) -> bool:
    """Check an HTTP Basic Authorization header against the env credentials."""
    user = os.environ.get("PRIVATE_USER", "")
    password = os.environ.get("PRIVATE_PASSWORD", "")
    scheme, _, token = (header or "").partition(" ")
    if not (user and password) or scheme.lower() != "basic":
        return False
    try:
        name, _, given = base64.b64decode(token, validate=True).decode().partition(":")
    except ValueError:
        return False
    return secrets.compare_digest(name.encode(), user.encode()) & secrets.compare_digest(
        given.encode(), password.encode())


def create_app(web: Path = WEB) -> FastAPI:
    private_dir = os.path.realpath(web / "private")
    app = FastAPI(title="Peatland turf-cutting monitor",
                  docs_url=None, redoc_url=None, openapi_url=None)
    get = lambda path: app.api_route(path, methods=["GET", "HEAD"], include_in_schema=False)

    @get("/healthz")
    def healthz():
        return {"status": "ok"}

    # the map lives at /private/; the old public addresses lead there
    @get("/")
    def root():
        return RedirectResponse(PRIVATE_PREFIX + "/")

    @get("/index.html")
    def index_page():
        return RedirectResponse(PRIVATE_PREFIX + "/")

    @get(PRIVATE_PREFIX)
    def private_root():
        return RedirectResponse(PRIVATE_PREFIX + "/")

    @get(PRIVATE_PREFIX + "/")
    def private_map():
        # the public page, re-based so its relative asset URLs resolve to /;
        # app.js adds the PlanetScope layer when served under /private/
        page = (web / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(page.replace("<head>", '<head>\n  <base href="/" />', 1))

    @app.middleware("http")
    async def password_wall(request: Request, call_next):
        path = request.scope["path"]
        if path == "/healthz":
            return await call_next(request)
        if not authorised(request.headers.get("authorization")):
            return Response(status_code=401, headers={
                "WWW-Authenticate": 'Basic realm="Peatland private", charset="UTF-8"'})
        response = await call_next(request)
        if path.startswith(PRIVATE_PREFIX):
            response.headers["Cache-Control"] = "private, no-store"
        elif path.endswith((".html", ".js", ".css", ".geojson")):
            # the page, script and styles change together on every release:
            # make browsers revalidate them so a cached page never meets a new script
            response.headers["Cache-Control"] = "private, no-cache"
        response.headers["X-Robots-Tag"] = "noindex, nofollow"
        return response

    app.mount(PRIVATE_PREFIX, StaticFiles(directory=private_dir, html=True, check_dir=False),
              name="private")
    # map assets (script, styles, Sentinel-2 tiles), also behind the password
    app.mount("/", PublicFiles(directory=web, html=True, private_dir=private_dir), name="static")
    return app


app = create_app()
