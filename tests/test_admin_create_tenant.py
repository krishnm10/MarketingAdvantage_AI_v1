"""Tests for POST /api/v2/admin/tenants and tenant copy allowlist helpers."""
from __future__ import annotations

import json
from typing import Any, Dict, List

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


def _collect_keys(obj: Any, key: str, found: List[Any] | None = None) -> List[Any]:
    found = found if found is not None else []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                found.append(v)
            _collect_keys(v, key, found)
    elif isinstance(obj, list):
        for item in obj:
            _collect_keys(item, key, found)
    return found


def test_build_tenant_config_from_copy_scrubs_secrets_and_infra(monkeypatch, tmp_path):
    cfg_root = tmp_path / "cfg"
    cfg_root.mkdir()
    source_cfg = {
        "client_id": "default",
        "client_name": "Source Tenant",
        "secrets_backend": {"provider": "hashicorp_vault", "vault_addr": "http://vault:8200"},
        "vectordb": {
            "type": "chroma",
            "collection": "source_collection",
            "chroma": {
                "persist_directory": "./source_db",
                "tenant": "source_tenant",
                "secret_ref": {"uri": "vault://default/keys#CHROMA"},
            },
        },
        "embedder": {
            "type": "gemini",
            "gemini": {
                "model": "gemini-embedding-2",
                "secret_ref": {"uri": "vault://default/keys#EMBED"},
            },
        },
        "retrieval": {"search_mode": "hybrid", "hybrid_alpha": 0.55},
        "ingestion": {"batch_size": 128, "chunking": {"strategy": "semantic"}},
        "parsers": {"pdf_enabled": True},
        "prompt": {"enabled": True, "prompt_type": "rag_context"},
    }
    (cfg_root / "default.json").write_text(json.dumps(source_cfg), encoding="utf-8")
    monkeypatch.setattr(rag_config_api, "_CONFIG_DIRS", [cfg_root])
    monkeypatch.setattr(rag_config_api, "_CONFIGS_DIR", cfg_root)

    out = rag_config_api.build_tenant_config_from_copy("acme_new", "default")

    assert out["client_id"] == "acme_new"
    assert out.get("secrets_backend") is None
    serialized = json.dumps(out)
    assert "vault://default/keys" not in serialized
    assert "source_collection" not in serialized
    assert "./source_db" not in serialized
    assert out["retrieval"]["search_mode"] == "hybrid"
    assert out["retrieval"]["hybrid_alpha"] == 0.55
    assert out["ingestion"]["batch_size"] == 128
    assert out["parsers"]["pdf_enabled"] is True
    assert out["embedder"]["type"] == "gemini"
    assert out["embedder"]["gemini"]["model"] == "gemini-embedding-2"
    assert out["vectordb"]["type"] == "chroma"
    assert out["vectordb"]["collection"] != "source_collection"
    chroma = out["vectordb"].get("chroma") or {}
    assert chroma.get("persist_directory") != "./source_db"
    assert chroma.get("tenant") != "source_tenant"


def test_create_tenant_writes_exclusive_file(tmp_path, monkeypatch, admin_app: FastAPI):
    cfg_root = tmp_path / "cfg"
    cfg_root.mkdir()
    default_cfg = rag_config_api._build_default_config_dict("default")
    (cfg_root / "default.json").write_text(json.dumps(default_cfg), encoding="utf-8")
    monkeypatch.setattr(rag_config_api, "_CONFIG_DIRS", [cfg_root])
    monkeypatch.setattr(rag_config_api, "_CONFIGS_DIR", cfg_root)

    with TestClient(admin_app, raise_server_exceptions=True) as client:
        r = client.post(
            "/api/v2/admin/tenants",
            json={"client_id": "brand_new_tenant", "client_name": "Brand New"},
        )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["client_id"] == "brand_new_tenant"
    assert body["client_name"] == "Brand New"
    assert (cfg_root / "brand_new_tenant.json").is_file()


def test_create_tenant_conflict_when_file_exists(tmp_path, monkeypatch, admin_app: FastAPI):
    cfg_root = tmp_path / "cfg"
    cfg_root.mkdir()
    (cfg_root / "default.json").write_text("{}", encoding="utf-8")
    (cfg_root / "dup_tenant.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(rag_config_api, "_CONFIG_DIRS", [cfg_root])
    monkeypatch.setattr(rag_config_api, "_CONFIGS_DIR", cfg_root)

    with TestClient(admin_app, raise_server_exceptions=True) as client:
        r = client.post("/api/v2/admin/tenants", json={"client_id": "dup_tenant"})
    assert r.status_code == 409


def test_create_tenant_rejects_default(admin_app: FastAPI):
    with TestClient(admin_app, raise_server_exceptions=True) as client:
        r = client.post("/api/v2/admin/tenants", json={"client_id": "default"})
    assert r.status_code == 422


def test_create_tenant_rejects_invalid_id(admin_app: FastAPI):
    with TestClient(admin_app, raise_server_exceptions=True) as client:
        r = client.post("/api/v2/admin/tenants", json={"client_id": "../etc"})
    assert r.status_code == 422


def test_create_tenant_copy_from_default(tmp_path, monkeypatch, admin_app: FastAPI):
    cfg_root = tmp_path / "cfg"
    cfg_root.mkdir()
    source = {
        "client_id": "default",
        "secrets_backend": {"provider": "env"},
        "retrieval": {"search_mode": "semantic", "top_k_final": 7},
        "ingestion": {"batch_size": 64},
        "embedder": {"type": "gemini", "gemini": {"model": "gemini-embedding-2", "secret_ref": {"uri": "vault://x"}}},
        "vectordb": {"type": "chroma", "collection": "tenant_coll", "chroma": {"persist_directory": "./x"}},
    }
    (cfg_root / "default.json").write_text(json.dumps(source), encoding="utf-8")
    monkeypatch.setattr(rag_config_api, "_CONFIG_DIRS", [cfg_root])
    monkeypatch.setattr(rag_config_api, "_CONFIGS_DIR", cfg_root)

    with TestClient(admin_app, raise_server_exceptions=True) as client:
        r = client.post(
            "/api/v2/admin/tenants",
            json={"client_id": "cloned_tenant", "copy_from": "default"},
        )
    assert r.status_code == 201, r.text
    stored = json.loads((cfg_root / "cloned_tenant.json").read_text(encoding="utf-8"))
    assert stored["retrieval"]["top_k_final"] == 7
    assert stored["ingestion"]["batch_size"] == 64
    assert stored.get("secrets_backend") is None
    assert "vault://x" not in json.dumps(stored)
    assert stored["vectordb"]["collection"] != "tenant_coll"
