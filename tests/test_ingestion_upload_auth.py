"""Ingestion upload endpoints require admin role."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v2.ingestion_api_v2 import router as ingestion_router
from app.auth.deps import get_current_user
from app.db.session_v2 import get_db


@pytest.fixture()
def ingestion_app() -> FastAPI:
    app = FastAPI()
    app.include_router(ingestion_router)
    return app


async def _fake_db():
    yield MagicMock()


def test_upload_requires_auth(ingestion_app: FastAPI) -> None:
    client = TestClient(ingestion_app, raise_server_exceptions=True)
    r = client.post(
        "/api/v2/ingestion/upload",
        files={"file": ("test.txt", b"hello", "text/plain")},
        data={"business_id": "default"},
    )
    assert r.status_code == 401


def test_upload_forbidden_for_viewer(ingestion_app: FastAPI) -> None:
    def _viewer():
        return {"role": "viewer", "sub": "viewer-1"}

    ingestion_app.dependency_overrides[get_current_user] = _viewer
    ingestion_app.dependency_overrides[get_db] = _fake_db
    try:
        client = TestClient(ingestion_app, raise_server_exceptions=True)
        r = client.post(
            "/api/v2/ingestion/upload",
            files={"file": ("test.txt", b"hello", "text/plain")},
            data={"business_id": "default"},
        )
        assert r.status_code == 403
    finally:
        ingestion_app.dependency_overrides.clear()


def test_upload_ok_for_admin(ingestion_app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _mock_route(*_args, **_kwargs):
        return {"status": "success", "file_id": "test-file-id"}

    monkeypatch.setattr(
        "app.api.v2.ingestion_api_v2.route_file_ingestion",
        _mock_route,
    )

    def _admin():
        return {"role": "admin", "sub": "admin-1"}

    ingestion_app.dependency_overrides[get_current_user] = _admin
    ingestion_app.dependency_overrides[get_db] = _fake_db
    try:
        client = TestClient(ingestion_app, raise_server_exceptions=True)
        r = client.post(
            "/api/v2/ingestion/upload",
            files={"file": ("test.txt", b"hello", "text/plain")},
            data={"business_id": "default"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "success"
    finally:
        ingestion_app.dependency_overrides.clear()


def test_media_upload_requires_auth(ingestion_app: FastAPI) -> None:
    client = TestClient(ingestion_app, raise_server_exceptions=True)
    r = client.post(
        "/api/v2/ingestion/media/upload",
        files={"file": ("clip.mp3", b"audio-bytes", "audio/mpeg")},
        data={"business_id": "default", "media_kind": "audio"},
    )
    assert r.status_code == 401
