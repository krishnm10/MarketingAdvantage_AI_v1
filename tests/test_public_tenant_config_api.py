from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v2.public_tenant_config_api import router as public_config_router
from app.core.config.config_store import FileSystemConfigStore, set_config_store
from app.core.config.public_tenant_config import PublicTenantConfig

_REPO_DEFAULT = (
    Path(__file__).resolve().parents[1] / "app" / "core" / "configs" / "default.json"
)

_FORBIDDEN_SUBSTRINGS = (
    "secret_ref",
    "secrets_backend",
    "api_key_env",
    "password_env",
    "token_env",
    "vault://",
    "aws-sm://",
    "azure-kv://",
    "gcp-sm://",
    "role_id",
    "vault_addr",
    "persist_directory",
    "sk-proj-",
)


@pytest.fixture()
def public_config_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    shutil.copy(_REPO_DEFAULT, cfg_dir / "default.json")

    tenant_id = "acme_public"
    overlay = {
        "client_id": tenant_id,
        "client_name": "Acme Public Corp",
        "description": "Phase 6 public read test tenant",
        "secrets_backend": {
            "provider": "hashicorp_vault",
            "vault_addr": "https://vault.example.com:8200",
            "namespace": "acme",
            "role_id": "role-123",
            "secret_id_ref": {"uri": "vault://acme/worker/approle-secret-id"},
        },
        "vectordb": {
            "type": "qdrant",
            "collection": "docs",
            "qdrant": {
                "url": "https://xyz.qdrant.io",
                "secret_ref": {"uri": "aws-sm://acme/qdrant-api-key"},
            },
        },
        "embedder": {
            "type": "openai",
            "openai": {
                "model": "text-embedding-3-small",
                "secret_ref": {"uri": "vault://acme/openai/embed-key"},
            },
        },
        "llm": {
            "single": {
                "type": "openai",
                "model": "gpt-4o-mini",
                "base_url": "https://api.openai.com/v1",
                "secret_ref": {"uri": "vault://acme/openai/llm-key"},
            }
        },
        "reranker": {
            "type": "cohere",
            "model": "rerank-english-v3.0",
            "secret_ref": {"uri": "vault://acme/cohere/key"},
        },
        "ingestion": {
            "vision": {
                "ai_profile": "api",
                "vision_api_secret_ref": {"uri": "vault://acme/openai/vision-key"},
            }
        },
        "features": {
            "enable_hybrid_search": True,
            "enable_pii_middleware": True,
        },
    }
    (cfg_dir / f"{tenant_id}.json").write_text(
        json.dumps(overlay, indent=2),
        encoding="utf-8",
    )

    store = FileSystemConfigStore([cfg_dir])
    set_config_store(store)
    monkeypatch.setenv("GOOGLE_API_KEY", "unit-test-key-not-in-response")
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-openai-not-in-response")
    monkeypatch.setenv("COHERE_API_KEY", "unit-test-cohere-not-in-response")

    app = FastAPI()
    app.include_router(public_config_router)
    yield app, tenant_id
    set_config_store(None)


def test_get_public_config_matches_schema_and_leaks_no_secrets(
    public_config_app: tuple[FastAPI, str],
) -> None:
    app, tenant_id = public_config_app
    client = TestClient(app, raise_server_exceptions=True)

    response = client.get(f"/api/v2/config/{tenant_id}")
    assert response.status_code == 200, response.text

    payload = response.json()
    serialized = json.dumps(payload).lower()

    for needle in _FORBIDDEN_SUBSTRINGS:
        assert needle not in serialized, f"leaked forbidden substring {needle!r}"

    public = PublicTenantConfig.model_validate(payload)
    assert public.client_id == tenant_id
    assert public.client_name == "Acme Public Corp"
    assert public.providers.embedder_type == "openai"
    assert public.providers.embedder_configured is True
    assert public.providers.llm_provider == "openai"
    assert public.providers.llm_model == "gpt-4o-mini"
    assert public.providers.vectordb_type == "qdrant"
    assert public.providers.secret_store_provider == "hashicorp_vault"
    assert public.features.enable_hybrid_search is True
    assert public.features.enable_pii_middleware is True
    assert public.ingestion.vision.vision_api_configured is True

    public.model_dump_public()


def test_get_public_config_default_tenant(
    public_config_app: tuple[FastAPI, str],
) -> None:
    app, _ = public_config_app
    client = TestClient(app, raise_server_exceptions=True)

    response = client.get("/api/v2/config/default")
    assert response.status_code == 200, response.text
    public = PublicTenantConfig.model_validate(response.json())
    assert public.client_id == "default"


def test_get_public_config_unknown_tenant_uses_default_merge(
    public_config_app: tuple[FastAPI, str],
) -> None:
    app, _ = public_config_app
    client = TestClient(app, raise_server_exceptions=True)

    response = client.get("/api/v2/config/brand_new_tenant")
    assert response.status_code == 200, response.text
    public = PublicTenantConfig.model_validate(response.json())
    assert public.client_id == "brand_new_tenant"
