"""
Tenant secrets backend and SecretRef configuration API (Phase 5).

Endpoints (admin + JWT ``client_id`` must match path ``{client_id}``):
  GET  /api/v2/tenants/{client_id}/secrets-backend
  PUT  /api/v2/tenants/{client_id}/secrets-backend
  GET  /api/v2/tenants/{client_id}/secret-refs
  PUT  /api/v2/tenants/{client_id}/secret-refs
  POST /api/v2/tenants/{client_id}/secrets-backend/test
  POST /api/v2/tenants/{client_id}/secret-refs/test
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.v2.rag_config_api import _validate_config_dict_raises
from app.auth.deps import get_current_user
from app.auth.guards import require_role
from app.core.config.config_store import FileSystemConfigStore, get_config_store
from app.core.config.secret_ref import (
    SecretRef,
    SecretsBackendConfig,
    SecretsBackendProvider,
    merge_secrets_backend_write_only,
    redact_secrets_backend_for_api,
    resolve_vault_token,
)
from app.core.secrets.connectors.base import (
    SecretNamespaceViolationError,
    SecretResolutionError,
)
from app.core.secrets.connectors.vault_connector import parse_vault_uri
from app.core.secrets.resolver import SecretResolver
from app.utils.path_sanitizer import sanitize_client_id
from app.utils.tenant_validator import TenantValidationError, validate_tenant_id

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v2/tenants",
    tags=["Tenant Secrets"],
)

_INTEGRATION_KEYS = frozenset({"embedder", "llm", "vectordb", "reranker"})

_AWS_REGION_RE = re.compile(r"^[a-z]{2}-[a-z]+-\d$")


# ---------------------------------------------------------------------------
# Auth / path validation
# ---------------------------------------------------------------------------


def _validated_tenant_path(client_id: str, endpoint: str) -> str:
    try:
        ctx = validate_tenant_id(
            client_id,
            source="path",
            endpoint=endpoint,
            allow_default=False,
        )
        return ctx.tenant_id
    except TenantValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _enforce_jwt_client_id_matches_path(
    user: Dict[str, Any],
    path_client_id: str,
    *,
    endpoint: str,
) -> str:
    """Require JWT ``client_id`` claim to match the URL tenant id."""
    safe_path = _validated_tenant_path(path_client_id, endpoint)
    jwt_client_id = user.get("client_id")
    if not jwt_client_id:
        raise HTTPException(
            status_code=403,
            detail="Missing client_id claim in JWT; bind tenant scope before configuring secrets.",
        )
    jwt_safe = sanitize_client_id(str(jwt_client_id))
    if jwt_safe != safe_path:
        raise HTTPException(
            status_code=403,
            detail="JWT client_id does not match URL tenant.",
        )
    return safe_path


async def require_secrets_admin(
    client_id: str,
    user: Dict[str, Any] = Depends(require_role("admin")),
) -> Dict[str, Any]:
    _enforce_jwt_client_id_matches_path(user, client_id, endpoint="tenant_secrets")
    return user


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _load_client_overlay(client_id: str) -> Dict[str, Any]:
    store = get_config_store()
    if isinstance(store, FileSystemConfigStore):
        path = store.find_config_path(client_id)
        if path is not None and client_id != "default":
            return store._read_file(path)
    return {"client_id": client_id}


def _invalidate_runtime_caches(client_id: str) -> None:
    try:
        from app.core.config.effective_tenant_runtime import (
            invalidate_effective_tenant_runtime_cache,
        )
        from app.core.pipeline_factory import pipeline_factory

        pipeline_factory.invalidate(client_id)
        invalidate_effective_tenant_runtime_cache(client_id)
    except Exception as exc:
        logger.warning(
            "[tenant_secrets_api] Cache invalidation failed for %s: %s",
            client_id,
            exc,
        )


def _persist_overlay_merge(client_id: str, updates: Dict[str, Any]) -> str:
    store = get_config_store()
    overlay = _load_client_overlay(client_id)
    if isinstance(store, FileSystemConfigStore):
        merged_overlay = store._deep_merge(overlay, updates)
    else:
        merged_overlay = {**overlay, **updates}
    merged_overlay["client_id"] = client_id

    preview = store.load_raw(client_id)
    if isinstance(store, FileSystemConfigStore):
        preview = store._deep_merge(preview, updates)
    else:
        preview = {**preview, **updates}
    preview["client_id"] = client_id
    _validate_config_dict_raises(preview)

    if client_id == "default" and isinstance(store, FileSystemConfigStore):
        default_path = store.find_config_path("default")
        if default_path is None:
            raise HTTPException(status_code=500, detail="Default config file not found.")
        tmp_path = default_path.with_suffix(default_path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(preview, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(default_path)
        version = store.get_version(client_id)
        from app.core.config.config_change_bus import get_config_change_bus

        get_config_change_bus().publish(client_id, version)
        _invalidate_runtime_caches(client_id)
        return version

    version = store.save_raw(client_id, merged_overlay)
    _invalidate_runtime_caches(client_id)
    return version


def _embedder_type(cfg: Dict[str, Any]) -> str:
    return str((cfg.get("embedder") or {}).get("type") or "gemini")


def _vectordb_type(cfg: Dict[str, Any]) -> str:
    return str((cfg.get("vectordb") or {}).get("type") or "chroma")


def _read_secret_ref_uri(cfg: Dict[str, Any], integration: str) -> Optional[str]:
    if integration == "embedder":
        block = (cfg.get("embedder") or {}).get(_embedder_type(cfg)) or {}
    elif integration == "llm":
        block = (cfg.get("llm") or {}).get("single") or {}
    elif integration == "vectordb":
        block = (cfg.get("vectordb") or {}).get(_vectordb_type(cfg)) or {}
    elif integration == "reranker":
        block = cfg.get("reranker") or {}
    else:
        return None
    ref = block.get("secret_ref")
    if isinstance(ref, dict):
        uri = ref.get("uri")
        return str(uri).strip() if uri else None
    if isinstance(ref, str) and ref.strip():
        return ref.strip()
    return None


def _apply_secret_ref(cfg: Dict[str, Any], integration: str, uri: str) -> None:
    SecretRef(uri=uri)
    ref_obj = {"uri": uri.strip()}
    if integration == "embedder":
        et = _embedder_type(cfg)
        embedder = cfg.setdefault("embedder", {})
        block = embedder.setdefault(et, {})
        if not isinstance(block, dict):
            block = {}
            embedder[et] = block
        block["secret_ref"] = ref_obj
    elif integration == "llm":
        single = cfg.setdefault("llm", {}).setdefault("single", {})
        if not isinstance(single, dict):
            single = {}
            cfg["llm"]["single"] = single
        single["secret_ref"] = ref_obj
    elif integration == "vectordb":
        vt = _vectordb_type(cfg)
        vectordb = cfg.setdefault("vectordb", {})
        block = vectordb.setdefault(vt, {})
        if not isinstance(block, dict):
            block = {}
            vectordb[vt] = block
        block["secret_ref"] = ref_obj
    elif integration == "reranker":
        reranker = cfg.setdefault("reranker", {})
        if not isinstance(reranker, dict):
            reranker = {}
            cfg["reranker"] = reranker
        reranker["secret_ref"] = ref_obj
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown integration key '{integration}'. Allowed: {sorted(_INTEGRATION_KEYS)}",
        )


# ---------------------------------------------------------------------------
# Connectivity test (never return secret values)
# ---------------------------------------------------------------------------


def _safe_error_message(exc: Exception) -> str:
    return type(exc).__name__


def test_secrets_backend_connectivity(
    backend: SecretsBackendConfig,
) -> tuple[bool, Optional[str]]:
    """
    Probe the configured secrets backend. Returns (success, error_message).
    Must never surface secret payloads to callers.
    """
    provider = backend.provider

    if provider == SecretsBackendProvider.ENV:
        return True, None

    if provider == SecretsBackendProvider.HASHICORP_VAULT:
        addr = (backend.vault_addr or "").rstrip("/")
        if not addr:
            return False, "vault_addr is required"
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{addr}/v1/sys/health")
            if resp.status_code == 200:
                return True, None
            return False, f"Vault health check failed (HTTP {resp.status_code})"
        except Exception as exc:
            return False, _safe_error_message(exc)

    if provider == SecretsBackendProvider.AWS_SECRETS_MANAGER:
        region = (backend.aws_region or "").strip()
        if not region or not _AWS_REGION_RE.match(region):
            return False, "Invalid or missing aws_region"
        return True, None

    if provider == SecretsBackendProvider.AZURE_KEY_VAULT:
        url = (backend.azure_vault_url or "").strip()
        if not url.startswith("https://"):
            return False, "azure_vault_url must be an https URL"
        return True, None

    if provider == SecretsBackendProvider.GCP_SECRET_MANAGER:
        project = (backend.gcp_project_id or "").strip()
        if not project:
            return False, "gcp_project_id is required"
        return True, None

    return False, f"Unsupported provider: {provider}"


def _probe_error_message(exc: Exception) -> str:
    if isinstance(exc, (SecretResolutionError, SecretNamespaceViolationError, ValueError)):
        return str(exc)
    return type(exc).__name__


def _probe_ref_metadata(uri: str) -> tuple[Optional[str], Optional[str]]:
    trimmed = uri.strip()
    if trimmed.startswith("vault://"):
        path, field = parse_vault_uri(trimmed)
        return path, field
    if trimmed.startswith("env://"):
        ref = SecretRef(uri=trimmed)
        return None, ref.env_var_name()
    return None, None


async def probe_secret_refs(
    client_id: str,
    backend: SecretsBackendConfig,
    refs: Dict[str, str],
) -> Dict[str, SecretRefProbeResult]:
    """
    Check that each SecretRef URI resolves. Never returns secret values.
    """
    resolver = SecretResolver(
        vault_token=resolve_vault_token(backend),
    )
    results: Dict[str, SecretRefProbeResult] = {}

    for integration, raw_uri in refs.items():
        if integration not in _INTEGRATION_KEYS:
            continue
        uri = raw_uri.strip()
        if not uri:
            continue

        secret_path, field = _probe_ref_metadata(uri)
        try:
            ref = SecretRef(uri=uri)
            await resolver.probe(
                ref,
                client_id=client_id,
                backend=backend,
                purpose=f"probe.{integration}",
            )
            results[integration] = SecretRefProbeResult(
                integration=integration,
                uri=uri,
                exists=True,
                secret_path=secret_path,
                field=field,
            )
        except Exception as exc:
            results[integration] = SecretRefProbeResult(
                integration=integration,
                uri=uri,
                exists=False,
                error=_probe_error_message(exc),
                secret_path=secret_path,
                field=field,
            )

    return results


def _resolve_secrets_backend_for_request(
    client_id: str,
    draft: Optional[SecretsBackendConfig],
) -> SecretsBackendConfig:
    store = get_config_store()
    merged = store.load_raw(client_id)
    existing_raw = merged.get("secrets_backend")
    if draft is None:
        if not existing_raw:
            raise HTTPException(
                status_code=400,
                detail="Configure a secrets backend before verifying secret references.",
            )
        return SecretsBackendConfig.model_validate(existing_raw)

    incoming = draft.model_dump(mode="json", exclude_none=True)
    merged_backend = merge_secrets_backend_write_only(existing_raw, incoming)
    return SecretsBackendConfig.model_validate(merged_backend)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SecretsBackendResponse(BaseModel):
    client_id: str
    version: str
    secrets_backend: Optional[Dict[str, Any]] = None


class SecretRefsResponse(BaseModel):
    client_id: str
    version: str
    refs: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="Integration key → SecretRef URI (embedder, llm, vectordb, reranker).",
    )


class SecretRefsUpdate(BaseModel):
    refs: Dict[str, str] = Field(
        ...,
        description="Integration key → SecretRef URI.",
        examples=[
            {
                "embedder": "vault://acme/embedder-key",
                "llm": "env://GOOGLE_API_KEY",
                "vectordb": "aws-sm://acme/vectordb-key",
            }
        ],
    )


class SecretsBackendTestRequest(BaseModel):
    secrets_backend: SecretsBackendConfig


class SecretsBackendTestResponse(BaseModel):
    success: bool
    error: Optional[str] = None


class SecretRefProbeResult(BaseModel):
    integration: str
    uri: str
    exists: bool
    error: Optional[str] = None
    secret_path: Optional[str] = Field(
        None,
        description="Vault KV path segment (no secret values).",
    )
    field: Optional[str] = Field(
        None,
        description="Field or env var name within the secret (no values).",
    )


class SecretRefsTestRequest(BaseModel):
    secrets_backend: Optional[SecretsBackendConfig] = Field(
        None,
        description="Optional draft backend from the UI; falls back to saved tenant config.",
    )
    refs: Dict[str, str] = Field(
        default_factory=dict,
        description="Integration key → SecretRef URI to probe.",
    )


class SecretRefsTestResponse(BaseModel):
    results: Dict[str, SecretRefProbeResult] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/{client_id}/secrets-backend",
    response_model=SecretsBackendResponse,
    dependencies=[Depends(require_role("admin"))],
)
async def get_secrets_backend(
    client_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
):
    cid = _enforce_jwt_client_id_matches_path(
        user, client_id, endpoint="get_secrets_backend"
    )
    store = get_config_store()
    merged = store.load_raw(cid)
    raw_backend = merged.get("secrets_backend")
    backend_view = None
    if raw_backend:
        validated = SecretsBackendConfig.model_validate(raw_backend)
        backend_view = redact_secrets_backend_for_api(
            validated.model_dump(mode="json", exclude_none=True)
        )
    return SecretsBackendResponse(
        client_id=cid,
        version=store.get_version(cid),
        secrets_backend=backend_view,
    )


@router.put(
    "/{client_id}/secrets-backend",
    response_model=SecretsBackendResponse,
)
async def put_secrets_backend(
    client_id: str,
    payload: SecretsBackendConfig,
    _user: Dict[str, Any] = Depends(require_secrets_admin),
):
    cid = _validated_tenant_path(client_id, "put_secrets_backend")
    store = get_config_store()
    existing_raw = store.load_raw(cid).get("secrets_backend")
    incoming = payload.model_dump(mode="json", exclude_none=True)
    merged_backend = merge_secrets_backend_write_only(existing_raw, incoming)
    SecretsBackendConfig.model_validate(merged_backend)

    version = _persist_overlay_merge(
        cid,
        {"secrets_backend": merged_backend},
    )
    backend_view = redact_secrets_backend_for_api(merged_backend)
    return SecretsBackendResponse(
        client_id=cid,
        version=version,
        secrets_backend=backend_view,
    )


@router.get(
    "/{client_id}/secret-refs",
    response_model=SecretRefsResponse,
    dependencies=[Depends(require_role("admin"))],
)
async def get_secret_refs(
    client_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
):
    cid = _enforce_jwt_client_id_matches_path(
        user, client_id, endpoint="get_secret_refs"
    )
    store = get_config_store()
    merged = store.load_raw(cid)
    refs = {
        key: _read_secret_ref_uri(merged, key)
        for key in sorted(_INTEGRATION_KEYS)
    }
    return SecretRefsResponse(
        client_id=cid,
        version=store.get_version(cid),
        refs=refs,
    )


@router.put(
    "/{client_id}/secret-refs",
    response_model=SecretRefsResponse,
)
async def put_secret_refs(
    client_id: str,
    payload: SecretRefsUpdate,
    _user: Dict[str, Any] = Depends(require_secrets_admin),
):
    cid = _validated_tenant_path(client_id, "put_secret_refs")
    unknown = set(payload.refs) - _INTEGRATION_KEYS
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown integration keys: {sorted(unknown)}",
        )

    store = get_config_store()
    preview = store.load_raw(cid)
    for integration, uri in payload.refs.items():
        _apply_secret_ref(preview, integration, uri)
    preview["client_id"] = cid
    _validate_config_dict_raises(preview)

    # Apply refs onto overlay only (delta persistence)
    delta: Dict[str, Any] = {"client_id": cid}
    for integration in payload.refs:
        if integration == "embedder":
            et = _embedder_type(preview)
            delta.setdefault("embedder", {})[et] = (preview.get("embedder") or {}).get(et)
        elif integration == "llm":
            delta.setdefault("llm", {})["single"] = (preview.get("llm") or {}).get("single")
        elif integration == "vectordb":
            vt = _vectordb_type(preview)
            delta.setdefault("vectordb", {})[vt] = (preview.get("vectordb") or {}).get(vt)
        elif integration == "reranker":
            delta["reranker"] = preview.get("reranker")

    version = _persist_overlay_merge(cid, delta)
    refs = {key: _read_secret_ref_uri(preview, key) for key in sorted(_INTEGRATION_KEYS)}
    return SecretRefsResponse(client_id=cid, version=version, refs=refs)


@router.post(
    "/{client_id}/secrets-backend/test",
    response_model=SecretsBackendTestResponse,
)
async def test_secrets_backend(
    client_id: str,
    payload: SecretsBackendTestRequest,
    _user: Dict[str, Any] = Depends(require_secrets_admin),
):
    _validated_tenant_path(client_id, "test_secrets_backend")
    success, error = test_secrets_backend_connectivity(payload.secrets_backend)
    return SecretsBackendTestResponse(success=success, error=error)


@router.post(
    "/{client_id}/secret-refs/test",
    response_model=SecretRefsTestResponse,
)
async def test_secret_refs(
    client_id: str,
    payload: SecretRefsTestRequest,
    _user: Dict[str, Any] = Depends(require_secrets_admin),
):
    cid = _validated_tenant_path(client_id, "test_secret_refs")

    backend = _resolve_secrets_backend_for_request(cid, payload.secrets_backend)

    unknown = set(payload.refs) - _INTEGRATION_KEYS
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown integration keys: {sorted(unknown)}",
        )

    if not any(v.strip() for v in payload.refs.values()):
        raise HTTPException(
            status_code=400,
            detail="Provide at least one non-empty SecretRef URI to verify.",
        )

    results = await probe_secret_refs(cid, backend, payload.refs)
    return SecretRefsTestResponse(results=results)
