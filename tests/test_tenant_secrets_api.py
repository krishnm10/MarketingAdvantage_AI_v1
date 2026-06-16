from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.v2.tenant_secrets_api as tenant_secrets_api
from app.api.v2.tenant_secrets_api import router as tenant_secrets_router
from app.auth.deps import get_current_user
from app.core.config.config_store import FileSystemConfigStore, set_config_store
from app.core.config.secret_ref import SecretsBackendConfig, SecretsBackendProvider

_REPO_DEFAULT = (
    Path(__file__).resolve().parents[1] / "app" / "core" / "configs" / "default.json"
)


@pytest.fixture()
def secrets_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    shutil.copy(_REPO_DEFAULT, cfg_dir / "default.json")

    store = FileSystemConfigStore([cfg_dir])
    set_config_store(store)
    monkeypatch.setenv("GOOGLE_API_KEY", "unit-test-google-key")

    app = FastAPI()
    app.include_router(tenant_secrets_router)

    def _admin(client_id: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"sub": "admin", "role": "admin"}
        if client_id is not None:
            payload["client_id"] = client_id
        return payload

    app.state._admin_user_factory = _admin
    yield app, cfg_dir, store
    set_config_store(None)


def _set_user(app: FastAPI, client_id: Optional[str] = None) -> None:
    factory = app.state._admin_user_factory

    def _override():
        return factory(client_id)

    app.dependency_overrides[get_current_user] = _override


def test_put_secrets_backend_forbidden_when_jwt_client_id_mismatch(
    secrets_app: tuple[FastAPI, Path, FileSystemConfigStore],
) -> None:
    app, _, _ = secrets_app
    _set_user(app, client_id="tenant_a")
    client = TestClient(app, raise_server_exceptions=True)

    r = client.put(
        "/api/v2/tenants/tenant_b/secrets-backend",
        json={"provider": "env"},
    )
    assert r.status_code == 403
    assert "does not match" in r.json()["detail"]


def test_put_secrets_backend_forbidden_when_jwt_missing_client_id(
    secrets_app: tuple[FastAPI, Path, FileSystemConfigStore],
) -> None:
    app, _, _ = secrets_app
    _set_user(app, client_id=None)
    client = TestClient(app, raise_server_exceptions=True)

    r = client.put(
        "/api/v2/tenants/acme/secrets-backend",
        json={"provider": "env"},
    )
    assert r.status_code == 403
    assert "Missing client_id" in r.json()["detail"]


def test_put_secrets_backend_persists_overlay(
    secrets_app: tuple[FastAPI, Path, FileSystemConfigStore],
) -> None:
    app, cfg_dir, _ = secrets_app
    tenant_id = "acme_secrets"
    _set_user(app, client_id=tenant_id)
    client = TestClient(app, raise_server_exceptions=True)

    payload = {
        "provider": "hashicorp_vault",
        "vault_addr": "https://vault.example.com:8200",
        "namespace": "acme",
        "mount_path": "secret",
    }
    r = client.put(
        f"/api/v2/tenants/{tenant_id}/secrets-backend",
        json=payload,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["client_id"] == tenant_id
    assert body["secrets_backend"]["provider"] == "hashicorp_vault"
    assert body["version"]

    stored_path = cfg_dir / f"{tenant_id}.json"
    assert stored_path.is_file()
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    assert stored["secrets_backend"]["vault_addr"] == payload["vault_addr"]


def test_put_secret_refs_updates_integration_uris(
    secrets_app: tuple[FastAPI, Path, FileSystemConfigStore],
) -> None:
    app, cfg_dir, _ = secrets_app
    tenant_id = "acme_refs"
    _set_user(app, client_id=tenant_id)
    client = TestClient(app, raise_server_exceptions=True)

    refs = {
        "embedder": "env://GOOGLE_API_KEY",
        "llm": "env://GOOGLE_API_KEY",
        "vectordb": "env://QDRANT_API_KEY",
    }
    r = client.put(
        f"/api/v2/tenants/{tenant_id}/secret-refs",
        json={"refs": refs},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["refs"]["embedder"] == refs["embedder"]
    assert body["refs"]["llm"] == refs["llm"]

    stored = json.loads((cfg_dir / f"{tenant_id}.json").read_text(encoding="utf-8"))
    embedder = stored.get("embedder") or {}
    ref_blocks = {
        k: v
        for k, v in embedder.items()
        if isinstance(v, dict) and (v.get("secret_ref") or {}).get("uri")
    }
    assert len(ref_blocks) == 1
    assert next(iter(ref_blocks.values()))["secret_ref"]["uri"] == refs["embedder"]
    assert stored["llm"]["single"]["secret_ref"]["uri"] == refs["llm"]


def test_put_secret_refs_accepts_vault_uris_without_env_keys(
    secrets_app: tuple[FastAPI, Path, FileSystemConfigStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vault-backed secret_ref URIs must not require process env vars at save time."""
    app, _, _ = secrets_app
    tenant_id = "acme_vault_refs"
    _set_user(app, client_id=tenant_id)
    client = TestClient(app, raise_server_exceptions=True)

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    client.put(
        f"/api/v2/tenants/{tenant_id}/secrets-backend",
        json={
            "provider": "hashicorp_vault",
            "vault_addr": "http://127.0.0.1:8200",
            "mount_path": "secret",
        },
    )

    refs = {
        "embedder": "vault://acme_vault_refs/mai_secrets#GOOGLE_API_KEY",
        "llm": "vault://acme_vault_refs/mai_secrets#GOOGLE_LLM_KEY",
    }
    r = client.put(
        f"/api/v2/tenants/{tenant_id}/secret-refs",
        json={"refs": refs},
    )
    assert r.status_code == 200, r.text
    assert r.json()["refs"]["embedder"] == refs["embedder"]


def test_put_secrets_backend_redacts_token_on_get(
    secrets_app: tuple[FastAPI, Path, FileSystemConfigStore],
) -> None:
    app, cfg_dir, _ = secrets_app
    tenant_id = "acme_vault_token"
    _set_user(app, client_id=tenant_id)
    client = TestClient(app, raise_server_exceptions=True)

    payload = {
        "provider": "hashicorp_vault",
        "vault_addr": "https://vault.example.com:8200",
        "mount_path": "secret",
        "vault_token": "hvs.super-secret-token",
    }
    put = client.put(
        f"/api/v2/tenants/{tenant_id}/secrets-backend",
        json=payload,
    )
    assert put.status_code == 200, put.text
    put_body = put.json()
    assert put_body["secrets_backend"]["vault_token_configured"] is True
    assert "vault_token" not in put_body["secrets_backend"]
    assert "hvs.super-secret-token" not in json.dumps(put_body)

    get = client.get(f"/api/v2/tenants/{tenant_id}/secrets-backend")
    assert get.status_code == 200, get.text
    get_body = get.json()
    assert get_body["secrets_backend"]["vault_token_configured"] is True
    assert "vault_token" not in get_body["secrets_backend"]

    stored = json.loads((cfg_dir / f"{tenant_id}.json").read_text(encoding="utf-8"))
    assert stored["secrets_backend"]["vault_token"] == "hvs.super-secret-token"


def test_put_secrets_backend_preserves_token_when_blank(
    secrets_app: tuple[FastAPI, Path, FileSystemConfigStore],
) -> None:
    app, cfg_dir, _ = secrets_app
    tenant_id = "acme_keep_token"
    _set_user(app, client_id=tenant_id)
    client = TestClient(app, raise_server_exceptions=True)

    client.put(
        f"/api/v2/tenants/{tenant_id}/secrets-backend",
        json={
            "provider": "hashicorp_vault",
            "vault_addr": "https://vault.example.com:8200",
            "vault_token": "hvs.keep-me",
        },
    )
    client.put(
        f"/api/v2/tenants/{tenant_id}/secrets-backend",
        json={
            "provider": "hashicorp_vault",
            "vault_addr": "https://vault.example.com:8200",
            "mount_path": "kv",
        },
    )
    stored = json.loads((cfg_dir / f"{tenant_id}.json").read_text(encoding="utf-8"))
    assert stored["secrets_backend"]["vault_token"] == "hvs.keep-me"
    assert stored["secrets_backend"]["mount_path"] == "kv"


def test_probe_secret_refs_env_backend(
    secrets_app: tuple[FastAPI, Path, FileSystemConfigStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _ = secrets_app
    tenant_id = "acme_probe_env"
    _set_user(app, client_id=tenant_id)
    client = TestClient(app, raise_server_exceptions=True)

    monkeypatch.setenv("GOOGLE_API_KEY", "probe-test-key")

    r = client.post(
        f"/api/v2/tenants/{tenant_id}/secret-refs/test",
        json={
            "secrets_backend": {"provider": "env"},
            "refs": {
                "embedder": "env://GOOGLE_API_KEY",
                "llm": "env://MISSING_PROBE_VAR_XYZ",
            },
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["results"]["embedder"]["exists"] is True
    assert body["results"]["embedder"]["field"] == "GOOGLE_API_KEY"
    assert body["results"]["llm"]["exists"] is False
    assert "probe-test-key" not in json.dumps(body)


def test_probe_secret_refs_vault_never_returns_secret_value(
    secrets_app: tuple[FastAPI, Path, FileSystemConfigStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _ = secrets_app
    tenant_id = "acme_probe_vault"
    _set_user(app, client_id=tenant_id)
    client = TestClient(app, raise_server_exceptions=True)

    async def _fake_probe(client_id, backend, refs, *, vault_token=None):
        return {
            "embedder": tenant_secrets_api.SecretRefProbeResult(
                integration="embedder",
                uri=refs["embedder"],
                exists=True,
                secret_path=f"{client_id}/mai_secrets",
                field="GOOGLE_API_KEY",
            )
        }

    monkeypatch.setattr(tenant_secrets_api, "probe_secret_refs", _fake_probe)

    r = client.post(
        f"/api/v2/tenants/{tenant_id}/secret-refs/test",
        json={
            "secrets_backend": {
                "provider": "hashicorp_vault",
                "vault_addr": "http://127.0.0.1:8200",
            },
            "refs": {"embedder": f"vault://{tenant_id}/mai_secrets#GOOGLE_API_KEY"},
        },
    )
    assert r.status_code == 200, r.text
    payload = json.dumps(r.json())
    assert "super-secret" not in payload
    assert r.json()["results"]["embedder"]["secret_path"] == f"{tenant_id}/mai_secrets"


def test_secrets_backend_test_never_returns_secret_payload(
    secrets_app: tuple[FastAPI, Path, FileSystemConfigStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _ = secrets_app
    tenant_id = "acme_test"
    _set_user(app, client_id=tenant_id)
    client = TestClient(app, raise_server_exceptions=True)

    class _FakeResponse:
        status_code = 200

        def json(self) -> dict:
            return {
                "data": {
                    "secret": "super-secret-value",
                    "token": "vault-token-abc",
                }
            }

    class _FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url: str) -> _FakeResponse:
            return _FakeResponse()

    monkeypatch.setattr(tenant_secrets_api.httpx, "Client", _FakeClient)

    backend = SecretsBackendConfig(
        provider=SecretsBackendProvider.HASHICORP_VAULT,
        vault_addr="https://vault.example.com:8200",
    )
    success, error = tenant_secrets_api.test_secrets_backend_connectivity(backend)
    assert success is True
    assert error is None

    r = client.post(
        f"/api/v2/tenants/{tenant_id}/secrets-backend/test",
        json={"secrets_backend": backend.model_dump(mode="json")},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert set(data.keys()) <= {"success", "error"}
    assert data["success"] is True
    assert "super-secret" not in json.dumps(data)
    assert "vault-token" not in json.dumps(data)


def test_viewer_role_forbidden(
    secrets_app: tuple[FastAPI, Path, FileSystemConfigStore],
) -> None:
    app, _, _ = secrets_app

    def _viewer():
        return {"sub": "viewer", "role": "viewer", "client_id": "acme"}

    app.dependency_overrides[get_current_user] = _viewer
    client = TestClient(app, raise_server_exceptions=True)

    r = client.put(
        "/api/v2/tenants/acme/secrets-backend",
        json={"provider": "env"},
    )
    assert r.status_code == 403
