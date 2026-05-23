"""Tests for config client registry and isolated admin dashboard router."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.v2.rag_config_api as rag_config_api
from app.api.v2.admin_customers_rag_dashboard_api import router as admin_dashboard_router
from app.auth.deps import get_current_user
from app.services.admin.config_client_registry import list_config_client_stems


@pytest.fixture()
def admin_app():
    app = FastAPI()
    app.include_router(admin_dashboard_router)

    def _admin():
        return {"role": "admin", "sub": "test-admin"}

    app.dependency_overrides[get_current_user] = _admin
    yield app
    app.dependency_overrides.clear()


def test_list_config_client_stems_returns_sorted_strings():
    stems = list_config_client_stems()
    assert isinstance(stems, list)
    assert stems == sorted(stems)


def test_customers_rag_dashboard_ok_for_admin(admin_app: FastAPI):
    with TestClient(admin_app, raise_server_exceptions=True) as client:
        r = client.get("/api/v2/admin/customers-rag-dashboard")
        assert r.status_code == 200, r.text
        data = r.json()
        assert "customers" in data
        assert "generated_at" in data
        assert "evaluation_templates_count" in data
        assert "customer_ids_from_config_files" in data
        assert "status" in data
        assert data["status"] in ("ok", "partial")


def test_customers_rag_dashboard_forbidden_for_viewer(admin_app: FastAPI):
    def _viewer():
        return {"role": "viewer", "sub": "viewer-1"}

    admin_app.dependency_overrides[get_current_user] = _viewer
    try:
        with TestClient(admin_app, raise_server_exceptions=True) as client:
            r = client.get("/api/v2/admin/customers-rag-dashboard")
            assert r.status_code == 403
    finally:
        del admin_app.dependency_overrides[get_current_user]


def test_archive_client_config_moves_to_archived(tmp_path, monkeypatch, admin_app: FastAPI):
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "acme_archive_test.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(rag_config_api, "_CONFIG_DIRS", [cfg])

    with TestClient(admin_app, raise_server_exceptions=True) as client:
        r = client.delete("/api/v2/admin/tenants/acme_archive_test/client-config")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "archived"
    assert body["client_id"] == "acme_archive_test"
    assert "_archived" in body["archived_to"]
    assert not (cfg / "acme_archive_test.json").exists()
    archived = list((cfg / "_archived").glob("*.json"))
    assert len(archived) == 1
    assert "acme_archive_test.json" in archived[0].name


def test_archive_client_config_forbidden_for_default(admin_app: FastAPI):
    with TestClient(admin_app, raise_server_exceptions=True) as client:
        r = client.delete("/api/v2/admin/tenants/default/client-config")
    assert r.status_code == 403
    assert "default" in r.json()["detail"].lower()


def test_archive_client_config_404_when_no_overlay(tmp_path, monkeypatch, admin_app: FastAPI):
    cfg = tmp_path / "empty_cfg"
    cfg.mkdir()
    monkeypatch.setattr(rag_config_api, "_CONFIG_DIRS", [cfg])

    with TestClient(admin_app, raise_server_exceptions=True) as client:
        r = client.delete("/api/v2/admin/tenants/ghost_client_no_file/client-config")
    assert r.status_code == 404


def test_archive_client_config_forbidden_for_viewer(admin_app: FastAPI):
    def _viewer():
        return {"role": "viewer", "sub": "viewer-1"}

    admin_app.dependency_overrides[get_current_user] = _viewer
    try:
        with TestClient(admin_app, raise_server_exceptions=True) as client:
            r = client.delete("/api/v2/admin/tenants/some_client/client-config")
        assert r.status_code == 403
    finally:
        del admin_app.dependency_overrides[get_current_user]
