"""Tenant corpus erasure — batched delete_many, partial failure reporting."""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v2.ingestion_admin_api import router as ingestion_admin_router
from app.auth.deps import get_current_user


@pytest.fixture()
def admin_app() -> FastAPI:
    app = FastAPI()
    app.include_router(ingestion_admin_router)

    def _admin():
        return {"role": "admin", "sub": "admin-1"}

    app.dependency_overrides[get_current_user] = _admin
    yield app
    app.dependency_overrides.clear()


def test_erase_tenant_corpus_reports_partial_batch_failure(admin_app: FastAPI) -> None:
    tenant_id = "tenant-a"
    storage_uid = uuid.uuid5(uuid.NAMESPACE_DNS, f"mai:tenant:{tenant_id}")

    mock_pipeline = MagicMock()
    mock_pipeline.config.vectordb.collection = "docs"
    mock_vectordb = MagicMock()
    mock_vectordb.delete_many.side_effect = [2, RuntimeError("vdb down")]
    mock_pipeline.vectordb = mock_vectordb

    chunk_ids_a = [uuid.uuid4(), uuid.uuid4()]
    chunk_ids_b = [uuid.uuid4()]

    class _Scalars:
        def __init__(self, values):
            self._values = values

        def all(self):
            return self._values

    class _Result:
        def __init__(self, values):
            self._values = values

        def scalars(self):
            return _Scalars(self._values)

    execute_results = [
        _Result(chunk_ids_a),
        _Result(chunk_ids_b),
        _Result([]),
        _Result([]),  # files query
        _Result([]),  # delete content
        _Result([]),  # delete gci
        _Result([]),  # delete files
    ]

    mock_db = MagicMock()
    mock_db.execute = AsyncMock(side_effect=execute_results)
    mock_db.commit = AsyncMock()

    async def _get_db():
        yield mock_db

    from app.db.session_v2 import get_db

    admin_app.dependency_overrides[get_db] = _get_db

    with patch("app.api.v2.ingestion_admin_api._get_pipeline", return_value=mock_pipeline), patch(
        "app.api.v2.ingestion_admin_api.get_storage_uuid",
        return_value=storage_uid,
    ), patch(
        "app.api.v2.ingestion_admin_api.validate_tenant_id_strict",
        return_value=MagicMock(tenant_id=tenant_id),
    ):
        client = TestClient(admin_app, raise_server_exceptions=True)
        r = client.delete(f"/api/v2/ingestion-admin/tenant/{tenant_id}/corpus")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["deleted_chunks"] == 2
    assert body["failed_chunks"] == 1
    assert mock_vectordb.delete_many.call_count == 2


def test_erase_tenant_a_does_not_invoke_tenant_b_vectordb(admin_app: FastAPI) -> None:
    """Erasing tenant A must not call delete_many on tenant B's vector store."""
    tenant_a = "tenant-a"
    tenant_b = "tenant-b"
    uid_a = uuid.uuid5(uuid.NAMESPACE_DNS, f"mai:tenant:{tenant_a}")
    uid_b = uuid.uuid5(uuid.NAMESPACE_DNS, f"mai:tenant:{tenant_b}")

    pipeline_a = MagicMock()
    pipeline_a.config.vectordb.collection = "shared-collection"
    pipeline_a.vectordb.delete_many.return_value = 1

    pipeline_b = MagicMock()
    pipeline_b.config.vectordb.collection = "shared-collection"
    pipeline_b.vectordb.delete_many.return_value = 0

    chunk_a = uuid.uuid4()

    class _Scalars:
        def __init__(self, values):
            self._values = values

        def all(self):
            return self._values

    class _Result:
        def __init__(self, values):
            self._values = values

        def scalars(self):
            return _Scalars(self._values)

    execute_results = [
        _Result([chunk_a]),
        _Result([]),
        _Result([]),
        _Result([]),
        _Result([]),
        _Result([]),
    ]

    mock_db = MagicMock()
    mock_db.execute = AsyncMock(side_effect=execute_results)
    mock_db.commit = AsyncMock()

    async def _get_db():
        yield mock_db

    from app.db.session_v2 import get_db

    admin_app.dependency_overrides[get_db] = _get_db

    def _pipeline_for(tenant_id: str):
        if tenant_id == tenant_a:
            return pipeline_a
        if tenant_id == tenant_b:
            return pipeline_b
        raise AssertionError(f"unexpected tenant {tenant_id}")

    def _storage_uuid(ctx):
        if ctx.tenant_id == tenant_a:
            return uid_a
        if ctx.tenant_id == tenant_b:
            return uid_b
        raise AssertionError(f"unexpected tenant {ctx.tenant_id}")

    with patch(
        "app.api.v2.ingestion_admin_api._get_pipeline",
        side_effect=_pipeline_for,
    ), patch(
        "app.api.v2.ingestion_admin_api.get_storage_uuid",
        side_effect=_storage_uuid,
    ), patch(
        "app.api.v2.ingestion_admin_api.validate_tenant_id_strict",
        side_effect=lambda tid, **_: MagicMock(tenant_id=tid),
    ):
        client = TestClient(admin_app, raise_server_exceptions=True)
        r = client.delete(f"/api/v2/ingestion-admin/tenant/{tenant_a}/corpus")

    assert r.status_code == 200, r.text
    pipeline_a.vectordb.delete_many.assert_called_once()
    pipeline_b.vectordb.delete_many.assert_not_called()
    called_ids = pipeline_a.vectordb.delete_many.call_args.kwargs.get("doc_ids") or []
    assert called_ids == [str(chunk_a)]
