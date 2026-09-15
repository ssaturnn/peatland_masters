"""The web server keeps licensed files (web/private/) behind Basic auth."""

import base64
import importlib.util
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SERVER = Path(__file__).resolve().parents[1] / "web" / "server.py"
_spec = importlib.util.spec_from_file_location("web_server", SERVER)
server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server)


def basic(user, password):
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


AUTH = basic("reader", "secret")


@pytest.fixture
def site(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIVATE_USER", "reader")
    monkeypatch.setenv("PRIVATE_PASSWORD", "secret")
    (tmp_path / "index.html").write_text("<html><head></head><body>public map</body></html>")
    (tmp_path / "private" / "briefing").mkdir(parents=True)
    (tmp_path / "private" / "briefing" / "index.html").write_text("licensed")
    return tmp_path


@pytest.fixture
def client(site):
    return TestClient(server.create_app(site))


def test_public_site_needs_no_password(client):
    r = client.get("/")
    assert r.status_code == 200 and "public map" in r.text
    assert client.get("/healthz").json() == {"status": "ok"}


def test_private_area_requires_the_configured_credentials(client):
    assert client.get("/private/briefing/").status_code == 401
    assert client.get("/private/briefing/", headers=basic("reader", "wrong")).status_code == 401
    assert client.get("/private/briefing/", headers=basic("other", "secret")).status_code == 401
    r = client.get("/private/briefing/", headers=AUTH)
    assert r.status_code == 200 and r.text == "licensed"
    assert r.headers["cache-control"] == "private, no-store"


def test_private_map_is_the_public_page_rebased_to_root(client):
    assert client.get("/private/").status_code == 401
    r = client.get("/private/", headers=AUTH)
    assert r.status_code == 200 and "public map" in r.text
    assert '<head>\n  <base href="/" />' in r.text
    assert r.headers["cache-control"] == "private, no-store"


def test_private_root_redirects_only_after_login(client):
    assert client.get("/private", follow_redirects=False).status_code == 401
    r = client.get("/private", headers=AUTH, follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/private/"


def test_private_area_stays_closed_without_configured_credentials(client, monkeypatch):
    monkeypatch.delenv("PRIVATE_PASSWORD")
    assert client.get("/private/", headers=basic("reader", "")).status_code == 401


@pytest.mark.parametrize("url", [
    "http://testserver//private/briefing/index.html",
    "http://testserver/PRIVATE/briefing/index.html",
])
def test_private_files_are_not_served_by_the_public_mount(client, url):
    r = client.get(url)
    assert r.status_code in (401, 404) and "licensed" not in r.text


def test_public_lookup_refuses_every_spelling_of_the_private_folder(site):
    files = server.PublicFiles(directory=site, html=True,
                               private_dir=os.path.realpath(site / "private"))
    assert files.lookup_path("index.html")[1] is not None
    for path in ("private/briefing/index.html", "x/../private/briefing/index.html",
                 "private", "./private/briefing/index.html"):
        assert files.lookup_path(path) == ("", None), path
