"""Tiny static server for the peatland web map (Railway-friendly).

Serves index.html, app.js, style.css and data/sites.geojson. Railway
sets $PORT; locally it defaults to 8000. Point Railway's service "root
directory" at web/ so this folder's requirements.txt / Procfile are used.

/private/ is the same map with same-day PlanetScope 3 m imagery, and
web/private/ holds that licensed material (PlanetScope is for
non-commercial research only). Everything under /private/ sits behind
HTTP Basic auth (PRIVATE_USER / PRIVATE_PASSWORD in the Railway
environment); the folder is gitignored and never served by the public
mount. Deploy with `railway up --no-gitignore` so the folder is uploaded.
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


class PublicFiles(StaticFiles):
    """The public site: every file in web/ except the private folder."""

    def __init__(self, *, private_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.private_dir = private_dir

    def lookup_path(self, path):
        full_path, stat_result = super().lookup_path(path)
        if full_path and _inside(full_path, self.private_dir):
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
    app = FastAPI(title="Peatland turf-cutting monitor")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get(PRIVATE_PREFIX, include_in_schema=False)
    def private_root():
        return RedirectResponse(PRIVATE_PREFIX + "/")

    @app.get(PRIVATE_PREFIX + "/", include_in_schema=False)
    def private_map():
        # the public page, re-based so its relative asset URLs resolve to /;
        # app.js adds the PlanetScope layer when served under /private/
        page = (web / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(page.replace("<head>", '<head>\n  <base href="/" />', 1))

    @app.middleware("http")
    async def private_area(request: Request, call_next):
        path = request.scope["path"]
        if not path.startswith(PRIVATE_PREFIX):
            response = await call_next(request)
            # the page, script and styles change together on every release:
            # make browsers revalidate them so a cached page never meets a new script
            if path == "/" or path.endswith((".html", ".js", ".css", ".geojson")):
                response.headers["Cache-Control"] = "no-cache"
            return response
        if not authorised(request.headers.get("authorization")):
            return Response(status_code=401, headers={
                "WWW-Authenticate": 'Basic realm="Peatland private", charset="UTF-8"'})
        response = await call_next(request)
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["X-Robots-Tag"] = "noindex, nofollow"
        return response

    app.mount(PRIVATE_PREFIX, StaticFiles(directory=private_dir, html=True, check_dir=False),
              name="private")
    # everything else is static; html=True serves index.html at /
    app.mount("/", PublicFiles(directory=web, html=True, private_dir=private_dir), name="static")
    return app


app = create_app()
