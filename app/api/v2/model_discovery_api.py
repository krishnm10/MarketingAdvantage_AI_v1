"""
================================================================================
Marketing Advantage AI — Model Discovery API
File: app/api/v2/model_discovery_api.py

Exposes available LLM providers/models and reranker plugins for the admin UI.
Read-only endpoints used to populate dropdowns in the chat console.

Endpoints:
  GET /api/v2/models/llm       → Available LLM providers with API key status
  GET /api/v2/models/reranker  → Registered reranker plugins
  GET /api/v2/models/defaults  → Server-side retrieval defaults
  GET /api/v2/models/runtime   → Effective tenant runtime (SSOT bridge)
================================================================================
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.auth.guards import require_role
from app.core.config.client_config_schema import EffectiveTenantRuntime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/models")


# ─────────────────────────────────────────────────────────────────────────────
# Response Models
# ─────────────────────────────────────────────────────────────────────────────

class LLMProviderInfo(BaseModel):
    provider: str
    display_name: str
    default_model: str
    api_key_set: bool
    api_key_env: str
    recommended: bool = False


class RerankerInfo(BaseModel):
    name: str
    description: str
    requires_gpu: bool = False
    requires_api_key: bool = False


class RetrievalDefaults(BaseModel):
    top_k: int
    similarity_threshold: float
    rag_answer_min_score: float
    default_llm_provider: str
    default_reranker: str
    embedder: str


# ─────────────────────────────────────────────────────────────────────────────
# LLM Discovery
# ─────────────────────────────────────────────────────────────────────────────

_LLM_PROVIDER_DEFS = [
    {
        "provider": "openai",
        "display_name": "OpenAI",
        "default_model": "gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
        "recommended": True,
    },
    {
        "provider": "gemini",
        "display_name": "Google Gemini",
        "default_model": "gemini-1.5-flash",
        "key_env": "GOOGLE_API_KEY",
        "recommended": True,
    },
    {
        "provider": "anthropic",
        "display_name": "Anthropic Claude",
        "default_model": "claude-3-5-sonnet-20241022",
        "key_env": "ANTHROPIC_API_KEY",
        "recommended": False,
    },
    {
        "provider": "groq",
        "display_name": "Groq",
        "default_model": "llama-3.1-70b-versatile",
        "key_env": "GROQ_API_KEY",
        "recommended": False,
    },
    {
        "provider": "ollama",
        "display_name": "Ollama (Local)",
        "default_model": "llama3.2",
        "key_env": "",
        "recommended": False,
    },
]


@router.get("/llm", response_model=List[LLMProviderInfo])
async def list_llm_providers(_user=Depends(require_role("admin"))):
    """List available LLM providers with API key readiness."""
    from app.core.config.client_config_resolver import get_client_config
    from app.middleware.security_middleware import validate_business_id

    try:
        cfg = get_client_config(validate_business_id(os.getenv("MAI_DEFAULT_BUSINESS_ID")))
    except Exception:
        cfg = None

    result = []
    for d in _LLM_PROVIDER_DEFS:
        key_env = d["key_env"]
        default_model = d["default_model"]
        if cfg and cfg.llm and cfg.llm.single:
            ll = cfg.llm.single
            prov = ll.type.value.lower()
            if prov == "google":
                prov = "gemini"
            if prov == d["provider"]:
                default_model = ll.model or default_model
                if ll.api_key_env:
                    key_env = ll.api_key_env
        if d["provider"] == "gemini":
            api_key_set = bool(os.getenv(key_env or "GOOGLE_API_KEY", "").strip()) or bool(
                os.getenv("GEMINI_API_KEY", "").strip()
            )
        else:
            api_key_set = True if not key_env else bool(os.getenv(key_env, "").strip())
        result.append(LLMProviderInfo(
            provider=d["provider"],
            display_name=d["display_name"],
            default_model=default_model,
            api_key_set=api_key_set,
            api_key_env=key_env or "(none — local)",
            recommended=d["recommended"],
        ))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Reranker Discovery
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/reranker", response_model=List[RerankerInfo])
async def list_rerankers(_user=Depends(require_role("admin"))):
    """List registered reranker plugins with descriptions."""
    items: List[RerankerInfo] = [
        RerankerInfo(name="none", description="No reranking — use raw vector scores"),
    ]
    try:
        from app.core.plugin_registry import reranker_registry
        import app.core.rerankers.register  # noqa: F401 — side-effect
        for name, desc in reranker_registry.list().items():
            items.append(RerankerInfo(
                name=name,
                description=desc,
                requires_gpu=name in ("crossencoder", "bge_reranker", "colbert"),
                requires_api_key=name == "cohere",
            ))
    except Exception as e:
        logger.warning("[ModelDiscovery] Failed to list rerankers: %s", e)

    return items


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Defaults
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/defaults", response_model=RetrievalDefaults)
async def retrieval_defaults(
    client_id: Optional[str] = Query(None, description="Tenant id; uses MAI_DEFAULT_BUSINESS_ID when omitted."),
    _user=Depends(require_role("admin")),
):
    """Return retrieval defaults; prefers effective runtime when client_id is set."""
    if client_id and str(client_id).strip():
        try:
            rt = build_effective_tenant_runtime(str(client_id).strip())
            return RetrievalDefaults(
                top_k=rt.retrieval.top_k_final,
                similarity_threshold=0.0,
                rag_answer_min_score=0.25,
                default_llm_provider=rt.llm.effective_provider,
                default_reranker=rt.reranker.effective_plugin,
                embedder=rt.embedder.type,
            )
        except Exception as exc:
            logger.warning("[ModelDiscovery] defaults for %r failed: %s", client_id, exc)

    try:
        from app.core.config.pipeline_runtime import get_pipeline_identity
        from app.middleware.security_middleware import validate_business_id
        from app.core.config.client_config_resolver import get_client_config

        _cid = validate_business_id(os.getenv("MAI_DEFAULT_BUSINESS_ID"))
        pi = get_pipeline_identity(_cid)
        d_llm = str(pi["llm"]).lower()
        d_rr = str(pi["reranker"]).lower()
        emb = str(pi["embedder"])
        _cfg = get_client_config(_cid)
        rag_min = float(_cfg.retrieval.answer_min_score)
    except Exception:
        d_llm = "openai"
        d_rr = "none"
        emb = "unknown"
        rag_min = 0.25
    return RetrievalDefaults(
        top_k=5,
        similarity_threshold=0.0,
        rag_answer_min_score=rag_min,
        default_llm_provider=d_llm,
        default_reranker=d_rr,
        embedder=emb,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Effective tenant runtime (SSOT bridge)
# ─────────────────────────────────────────────────────────────────────────────

def build_effective_tenant_runtime(client_id: str) -> EffectiveTenantRuntime:
    """Delegate to cached SSOT builder."""
    from app.core.config.effective_tenant_runtime import build_effective_tenant_runtime as _build

    return _build(client_id)


@router.get("/runtime", response_model=EffectiveTenantRuntime)
async def get_effective_tenant_runtime(
    client_id: str = Query(..., min_length=1, description="Tenant / client id"),
    _user=Depends(require_role("admin")),
):
    """
    Server-computed effective pipeline state for a tenant.

    Used by admin UI to hydrate session controls without configuration drift.
    """
    from app.utils.tenant_validator import TenantValidationError, validate_tenant_id_strict

    try:
        tenant_ctx = validate_tenant_id_strict(
            client_id,
            source="query",
            endpoint="models_runtime",
        )
    except TenantValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        return build_effective_tenant_runtime(tenant_ctx.tenant_id)
    except Exception as exc:
        logger.exception("[ModelDiscovery] runtime build failed for %r", client_id)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build effective runtime: {exc}",
        ) from exc
