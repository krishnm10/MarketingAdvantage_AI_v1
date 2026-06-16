"""Auth API contract tests — login, token, and role paths."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v2.auth_api import router as auth_router


@pytest.fixture()
def auth_app() -> FastAPI:
    app = FastAPI()
    app.include_router(auth_router)
    return app


def test_login_success_returns_admin_role(auth_app: FastAPI) -> None:
    client = TestClient(auth_app, raise_server_exceptions=True)
    r = client.post(
        "/api/v2/auth/login",
        json={"username": "admin", "password": "admin"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["access_token"]
    assert body["role"] == "admin"
    assert body["token_type"] == "bearer"
    assert r.cookies.get("mai_access_token")
    assert r.cookies.get("mai_auth_hint") == "1"


def test_me_accepts_http_only_cookie(auth_app: FastAPI) -> None:
    client = TestClient(auth_app, raise_server_exceptions=True)
    login = client.post(
        "/api/v2/auth/login",
        json={"username": "admin", "password": "admin"},
    )
    assert login.status_code == 200
    client.cookies.set("mai_access_token", login.cookies["mai_access_token"])
    client.cookies.set("mai_auth_hint", "1")
    me = client.get("/api/v2/auth/me")
    assert me.status_code == 200, me.text
    assert me.json()["role"] == "admin"
    assert me.json()["username"] == "admin"


def test_logout_clears_session_cookies(auth_app: FastAPI) -> None:
    client = TestClient(auth_app, raise_server_exceptions=True)
    login = client.post(
        "/api/v2/auth/login",
        json={"username": "admin", "password": "admin"},
    )
    client.cookies.set("mai_access_token", login.cookies["mai_access_token"])
    out = client.post("/api/v2/auth/logout")
    assert out.status_code == 200
    assert out.cookies.get("mai_access_token") in ("", None)


def test_login_invalid_credentials_rejected(auth_app: FastAPI) -> None:
    client = TestClient(auth_app, raise_server_exceptions=True)
    r = client.post(
        "/api/v2/auth/login",
        json={"username": "admin", "password": "wrong-password"},
    )
    assert r.status_code == 401


def test_oauth2_token_endpoint_accepts_form_data(auth_app: FastAPI) -> None:
    client = TestClient(auth_app, raise_server_exceptions=True)
    r = client.post(
        "/api/v2/auth/token",
        data={"username": "viewer", "password": "viewer"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["access_token"]
    assert body["role"] == "viewer"


def test_viewer_login_returns_viewer_role(auth_app: FastAPI) -> None:
    client = TestClient(auth_app, raise_server_exceptions=True)
    r = client.post(
        "/api/v2/auth/login",
        json={"username": "viewer", "password": "viewer"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["role"] == "viewer"
