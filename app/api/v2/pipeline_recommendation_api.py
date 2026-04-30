# =============================================================================
# app/api/v2/pipeline_recommendation_api.py
#
# Pipeline Recommendation API — read-only, catalog-driven.
#
# Endpoints:
#   GET  /api/v2/pipeline-recommendations/models
#       → List all available embedding models from embedder_catalog.yaml.
#
#   POST /api/v2/pipeline-recommendations/preview
#       → Generate a full pipeline recommendation for a chosen model,
#         including VectorDB, Reranker, LLM, chunking strategy, safe
#         token limits, and ready-to-apply env delta sets.
#
# Security:
#   - Auth-protected (admin/superadmin only).
#   - Read-only — never modifies .env or tenant configs.
#   - Structured logging with correlation fields; no secrets or PII logged.
#
# Observability:
#   - Request/response shape logged at INFO level with stable fields.
#   - Errors logged at WARNING/ERROR with full context.
# =============================================================================

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.utils.tenant_validator import validate_tenant_id, TenantValidationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/pipeline-recommendations", tags=["Pipeline Recommendations"])


# ---------------------------------------------------------------------------
# Auth dependency (reuses the pattern from embedding_alignment_api.py)
# ---------------------------------------------------------------------------

def _get_auth_dep():
    """Lazy import to avoid circular imports at module load time."""
    try:
        from app.api.v2.auth_api import require_role
        return Depends(require_role("admin"))
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PreviewRequest(BaseModel):
    embedder_model_id: str = Field(
        ...,
        description="Canonical catalog model ID (e.g. 'openai/text-embedding-3-large').",
        examples=["openai/text-embedding-3-large"],
    )
    client_id: str = Field(
        default="default",
        description="Tenant identifier. 'default' returns global-only env deltas.",
    )
    prefer_onprem: bool = Field(
        default=False,
        description=(
            "When true, prefer self-hosted / local VectorDB and LLM options "
            "over cloud-managed ones."
        ),
    )


class ModelListResponse(BaseModel):
    models: List[Dict[str, Any]]
    total: int
    catalog_hash: str


class PreviewResponse(BaseModel):
    success: bool
    catalog_entry: Optional[Dict[str, Any]] = None
    recommendation: Optional[Dict[str, Any]] = None
    env_deltas: Optional[Dict[str, Any]] = None
    alignment_notes: List[str] = []
    error: Optional[str] = None
    catalog_hash: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper — lazy auth
# ---------------------------------------------------------------------------

def _require_admin():
    """
    Returns the auth dependency, or a no-op dependency if auth is unavailable
    (development mode fallback — never expose unauthenticated in production).
    """
    try:
        from app.api.v2.auth_api import require_role
        return Depends(require_role("admin"))
    except ImportError:
        logger.warning(
            "[PipelineRecommendationAPI] auth_api unavailable — "
            "endpoints running without auth guard (dev mode only)"
        )
        # Return a dummy dependency that always passes
        async def _no_auth():
            return {"role": "admin", "sub": "dev"}
        return Depends(_no_auth)


# ---------------------------------------------------------------------------
# GET /models — list all catalog models
# ---------------------------------------------------------------------------

@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="List available embedding models",
    description=(
        "Returns all embedding models from embedder_catalog.yaml with display-ready fields. "
        "Optionally filter by provider, verification status, or language."
    ),
)
async def list_models(
    provider: Optional[str] = Query(None, description="Filter by provider (openai, huggingface, ollama, cohere)."),
    verified_only: bool = Query(False, description="Return only verified models."),
    lang: Optional[str] = Query(None, description="Filter by language support (en, multilingual)."),
    user=_require_admin(),
):
    """
    List all available embedding models from the catalog.

    This is used by the Pipeline Builder UI to populate the model picker.
    """
    from app.ai.recommendations.pipeline_recommender import list_catalog_models
    from app.ai.catalog.catalog_loader import get_catalog_hash

    try:
        models = list_catalog_models(
            provider_filter=provider,
            verified_only=verified_only,
            lang_filter=lang,
        )
        catalog_hash = get_catalog_hash()

        logger.info(
            "[PipelineRecommendationAPI] list_models count=%d provider=%r verified_only=%s lang=%r",
            len(models), provider, verified_only, lang,
        )
        return ModelListResponse(
            models=models,
            total=len(models),
            catalog_hash=catalog_hash,
        )

    except Exception as exc:
        logger.error("[PipelineRecommendationAPI] list_models error: %s", exc)
        return ModelListResponse(models=[], total=0, catalog_hash="")


# ---------------------------------------------------------------------------
# POST /preview — generate full pipeline recommendation
# ---------------------------------------------------------------------------

@router.post(
    "/preview",
    response_model=PreviewResponse,
    summary="Preview pipeline recommendation",
    description=(
        "Given an embedding model, returns a complete pipeline recommendation: "
        "compatible VectorDB, Reranker, LLM, chunking strategy, safe token limits, "
        "and ready-to-apply env deltas for global and per-tenant scopes. "
        "This endpoint is read-only — it does not modify any configuration."
    ),
)
async def preview_recommendation(
    body: PreviewRequest,
    user=_require_admin(),
):
    """
    Generate a full pipeline recommendation for the given embedding model.

    Returns a structured recommendation including:
    - Catalog entry metadata (tokenizer family, dimension, distance metric, etc.)
    - Recommended VectorDB (with alternatives), Reranker, LLM
    - Safe chunk size and overlap computed from the model's context window
    - Ready-to-apply env variable deltas (global and per-tenant)
    - Alignment notes (warnings for unverified models, small context windows, etc.)

    This endpoint is completely read-only and safe to call at any time.
    """
    from app.ai.recommendations.pipeline_recommender import generate_recommendation

    try:
        _tctx = validate_tenant_id(
            body.client_id, source="body", endpoint="preview_recommendation",
            allow_default=True,
        )
        _tenant = _tctx.tenant_id
    except TenantValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    logger.info(
        "[PipelineRecommendationAPI] preview model_id=%r client_id=%r prefer_onprem=%s",
        body.embedder_model_id,
        _tenant,
        body.prefer_onprem,
    )

    try:
        result = generate_recommendation(
            embedder_model_id=body.embedder_model_id,
            client_id=_tenant,
            prefer_onprem=body.prefer_onprem,
        )

        if result["success"]:
            logger.info(
                "[PipelineRecommendationAPI] recommendation generated "
                "model_id=%r vectordb=%r llm=%r chunk_size=%s",
                body.embedder_model_id,
                result.get("recommendation", {}).get("vectordb", {}).get("selected"),
                result.get("recommendation", {}).get("llm", {}).get("provider"),
                result.get("recommendation", {}).get("safe_chunk_size"),
            )
        else:
            logger.warning(
                "[PipelineRecommendationAPI] recommendation failed model_id=%r error=%r",
                body.embedder_model_id,
                result.get("error"),
            )

        return PreviewResponse(**result)

    except Exception as exc:
        logger.error(
            "[PipelineRecommendationAPI] unexpected error model_id=%r: %s",
            body.embedder_model_id, exc,
        )
        return PreviewResponse(
            success=False,
            error=f"Internal error: {exc}",
            catalog_entry=None,
            recommendation=None,
            env_deltas=None,
            alignment_notes=[],
            catalog_hash=None,
        )
