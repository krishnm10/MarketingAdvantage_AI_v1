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
================================================================================
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.auth.guards import require_role

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
        "model_env": "OPENAI_LLM_MODEL",
        "default_model": "gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
        "recommended": True,
    },
    {
        "provider": "gemini",
        "display_name": "Google Gemini",
        "model_env": "GEMINI_LLM_MODEL",
        "default_model": "gemini-1.5-flash",
        "key_env": "GEMINI_API_KEY",
        "recommended": True,
    },
    {
        "provider": "anthropic",
        "display_name": "Anthropic Claude",
        "model_env": "ANTHROPIC_LLM_MODEL",
        "default_model": "claude-3-5-sonnet-20241022",
        "key_env": "ANTHROPIC_API_KEY",
        "recommended": False,
    },
    {
        "provider": "groq",
        "display_name": "Groq",
        "model_env": "GROQ_LLM_MODEL",
        "default_model": "llama-3.1-70b-versatile",
        "key_env": "GROQ_API_KEY",
        "recommended": False,
    },
    {
        "provider": "ollama",
        "display_name": "Ollama (Local)",
        "model_env": "OLLAMA_LLM_MODEL",
        "default_model": "llama3.1:8b",
        "key_env": "",
        "recommended": False,
    },
]


@router.get("/llm", response_model=List[LLMProviderInfo])
async def list_llm_providers(_user=Depends(require_role("admin"))):
    """List available LLM providers with API key readiness."""
    result = []
    for d in _LLM_PROVIDER_DEFS:
        key_env = d["key_env"]
        api_key_set = True if not key_env else bool(os.getenv(key_env, "").strip())
        result.append(LLMProviderInfo(
            provider=d["provider"],
            display_name=d["display_name"],
            default_model=os.getenv(d["model_env"], d["default_model"]),
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
async def retrieval_defaults(_user=Depends(require_role("admin"))):
    """Return server-side default retrieval configuration."""
    return RetrievalDefaults(
        top_k=5,
        similarity_threshold=0.0,
        rag_answer_min_score=float(os.getenv("RAG_ANSWER_MIN_SCORE", "0.25")),
        default_llm_provider=os.getenv("MAI_LLM", "openai").lower(),
        default_reranker=os.getenv("MAI_RERANKER", "none").lower(),
        embedder=os.getenv("MAI_EMBEDDER", "unknown"),
    )
