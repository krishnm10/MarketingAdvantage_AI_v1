"""RAG API auth gate — endpoints require admin role."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v2.rag_api import router as rag_router
from app.auth.deps import get_current_user


@pytest.fixture()
def rag_app() -> FastAPI:
    app = FastAPI()
    app.include_router(rag_router, prefix="/api/v2/rag")
    return app


def test_rag_query_requires_auth(rag_app: FastAPI) -> None:
    client = TestClient(rag_app, raise_server_exceptions=True)
    r = client.post(
        "/api/v2/rag/query",
        json={"client_id": "default", "query": "hello"},
    )
    assert r.status_code == 401


def test_rag_pipeline_list_forbidden_for_viewer(rag_app: FastAPI) -> None:
    def _viewer():
        return {"role": "viewer", "sub": "viewer-1"}

    rag_app.dependency_overrides[get_current_user] = _viewer
    try:
        client = TestClient(rag_app, raise_server_exceptions=True)
        r = client.get("/api/v2/rag/pipeline/list")
        assert r.status_code == 403
    finally:
        rag_app.dependency_overrides.clear()


def test_rag_pipeline_list_ok_for_admin(rag_app: FastAPI) -> None:
    def _admin():
        return {"role": "admin", "sub": "admin-1"}

    rag_app.dependency_overrides[get_current_user] = _admin
    try:
        client = TestClient(rag_app, raise_server_exceptions=True)
        r = client.get("/api/v2/rag/pipeline/list")
        assert r.status_code == 200, r.text
        assert "cached_pipelines" in r.json()
    finally:
        rag_app.dependency_overrides.clear()
