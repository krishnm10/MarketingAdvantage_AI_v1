"""
================================================================================
Marketing Advantage AI — Pluggable RAG API Router
File: app/api/v2/rag_api.py

Endpoints:
  POST /api/v2/rag/query              → Run RAG query for a client
  POST /api/v2/rag/pipeline/build     → Pre-warm a client pipeline
  DELETE /api/v2/rag/pipeline/cache   → Invalidate cached pipeline
  GET  /api/v2/rag/pipeline/health    → Per-client pipeline health
  GET  /api/v2/rag/pipeline/list      → List cached pipelines

CONFIG RESOLUTION:
  All runtime config loading goes through get_client_config() — the
  authoritative resolver. No direct filesystem config parsing in any
  request handler. The build endpoint accepts explicit admin overrides
  but routes them through validation.
================================================================================
"""

from __future__ import annotations

import json as _json_module
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.pipeline_factory import pipeline_factory
from app.core.config.client_config_schema import ClientConfig
from app.core.config.client_config_resolver import (
    get_client_config,
    get_config_fingerprint,
    validate_config_compatibility,
    ConfigValidationError,
    IssueSeverity,
)
from app.utils.pipeline_logger import PipelineLogger
from app.utils.tenant_validator import (
    validate_tenant_id,
    TenantValidationError,
)
from app.core.runtime.runtime_constants import AUTHORITATIVE_RUNTIME
from app.core.runtime.runtime_context import RAGRuntimeContext

logger = logging.getLogger(__name__)

router = APIRouter()


def _validated_tenant(
    raw_id: Optional[str],
    *,
    source: str = "body",
    endpoint: str = "rag_api",
    allow_default: bool = False,
) -> str:
    """Validate tenant ID or raise HTTPException(422)."""
    try:
        ctx = validate_tenant_id(
            raw_id, source=source, endpoint=endpoint,
            allow_default=allow_default,
        )
        return ctx.tenant_id
    except TenantValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROMPTS_DIR = _REPO_ROOT / "app" / "core" / "configs" / "prompts"


def _resolve_prompt_template(template_id: str) -> Optional[str]:
    """
    Load a prompt template by ID from the prompts store and return
    the system_instructions string for use as the LLM system prompt.
    Returns None if the template is not found or cannot be loaded.
    """
    template_path = _PROMPTS_DIR / f"{template_id}.json"
    if not template_path.exists():
        logger.warning(
            "[rag_api] Prompt template '%s' not found at %s",
            template_id, template_path,
        )
        return None
    try:
        with template_path.open() as f:
            data = _json_module.load(f)
        return data.get("system_instructions") or data.get("content") or None
    except Exception as e:
        logger.warning(
            "[rag_api] Failed to load prompt template '%s': %s", template_id, e
        )
        return None




# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class RAGQueryRequest(BaseModel):
    """Request body for POST /api/v2/rag/query"""
    client_id:        str  = Field(..., description="Client ID (must have a built pipeline)")
    query:            str  = Field(..., description="Natural language question")
    metadata_filters: Optional[Dict[str, Any]] = None
    top_k_retrieval:  Optional[int] = None
    top_k_final:      Optional[int] = None
    system_prompt:    Optional[str] = None
    temperature:      Optional[float] = None
    max_tokens:       Optional[int] = None


class BuildPipelineRequest(BaseModel):
    """
    Request body for POST /api/v2/rag/pipeline/build.

    Resolution order:
        1. client_id → get_client_config() authoritative resolution (preferred).
        2. config_dict → inline override (admin use only, validated).
    """
    client_id: Optional[str] = Field(
        None,
        description="Client ID — resolved via authoritative config resolver.",
    )
    config_dict: Optional[Dict[str, Any]] = Field(
        None,
        description="Inline ClientConfig as JSON object (admin override).",
    )


class RAGQueryResponse(BaseModel):
    """Response from POST /api/v2/rag/query"""
    client_id:     str
    query:         str
    final_answer:  Optional[str]
    reranked:      bool
    trust_score:   Optional[float]
    chunk_count:   int
    latency_ms:    Dict[str, Any]
    metadata:      Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(req: RAGQueryRequest):
    """
    Run a full RAG query using a pre-built client pipeline.

    If no cached pipeline exists, one is built on-the-fly from the
    authoritative config resolver (get_client_config).
    """
    import uuid as _uuid

    tenant = _validated_tenant(req.client_id, endpoint="rag_query")
    _request_id = str(_uuid.uuid4())
    _trace_id = str(_uuid.uuid4())

    _rtx = RAGRuntimeContext(
        request_id=_request_id,
        trace_id=_trace_id,
        tenant_id=tenant,
        client_id=tenant,
        pipeline_id=tenant,
        user_query=req.query,
        stack_used="rag_pipeline",
        runtime_authority=AUTHORITATIVE_RUNTIME,
    )

    plog = PipelineLogger(request_path="rag_api", client_id=tenant)
    plog.info(
        "RAG query received",
        query_length=len(req.query),
        stack_used="rag_pipeline",
        runtime_authority=AUTHORITATIVE_RUNTIME,
    )

    # ── Security gate: PII redaction + prompt injection detection ─────────
    from app.middleware.security_middleware import scan_text as _security_scan_text

    _scan_result = _security_scan_text(req.query, context="rag_query")
    if _scan_result.injection_detected:
        raise HTTPException(
            status_code=400,
            detail="Query rejected: potential prompt injection detected.",
        )
    sanitized_query = _scan_result.redacted_text
    plog.info(
        "Query passed through security middleware",
        sanitized=sanitized_query != req.query,
        pii_detected=_scan_result.has_pii,
    )

    # ── Pipeline resolution: cache hit or authoritative build ─────────────
    pipeline = pipeline_factory.get_cached(tenant)
    if pipeline is None:
        try:
            config = get_client_config(tenant)
            fingerprint = get_config_fingerprint(config)
            plog.info(
                "Auto-building pipeline from authoritative config",
                authoritative_config_used=True,
                config_fingerprint=fingerprint,
            )
            pipeline_factory.build(config)
            pipeline = pipeline_factory.get_cached(tenant)
        except ConfigValidationError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Config validation failed for '{tenant}': {e}",
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No config found for client_id='{tenant}'. "
                    f"Use POST /api/v2/rag/pipeline/build first."
                ),
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to auto-build pipeline: {e}",
            )

    try:
        if not pipeline:
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline for '{tenant}' not in cache after build.",
            )

        plog = plog.bind(
            pipeline_id=tenant,
            embedder_model=getattr(getattr(pipeline, "embedder", None), "info", None) and pipeline.embedder.info.model,
            vectordb_backend=getattr(pipeline.vectordb, "kind", None),
        )

        effective_system_prompt = req.system_prompt
        if not effective_system_prompt:
            retrieval_cfg = getattr(
                getattr(pipeline, "config", None), "retrieval", None
            )
            template_id = getattr(retrieval_cfg, "prompt_template_id", None)
            if template_id:
                effective_system_prompt = _resolve_prompt_template(template_id)
                if effective_system_prompt:
                    logger.info(
                        "[rag_api] Using prompt template '%s' for client '%s'.",
                        template_id, tenant,
                    )

        result = pipeline.query(
            sanitized_query,
            metadata_filters=req.metadata_filters,
            top_k_retrieval=req.top_k_retrieval,
            top_k_final=req.top_k_final,
            system_prompt=effective_system_prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            runtime_context=_rtx,
        )

        plog.query_complete(
            chunks_used=len(result.context_chunks),
            reranked=result.reranked,
            trust_score=result.trust_score,
            total_ms=result.latency.get("total_ms", 0),
            token_usage=result.metadata.get("token_usage"),
        )

        _shadow_dual_execute(
            client_id=tenant,
            sanitized_query=sanitized_query,
            old_result=result,
            query_kwargs=dict(
                metadata_filters=req.metadata_filters,
                top_k_retrieval=req.top_k_retrieval,
                top_k_final=req.top_k_final,
                system_prompt=effective_system_prompt,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            ),
            plog=plog,
        )

        _metadata = {
            **result.metadata,
            "security_scan": {
                "pii_detected": _scan_result.has_pii,
                "injection_detected": _scan_result.injection_detected,
            },
        }

        return RAGQueryResponse(
            client_id=tenant,
            query=req.query,
            final_answer=result.final_answer,
            reranked=result.reranked,
            trust_score=result.trust_score,
            chunk_count=len(result.context_chunks),
            latency_ms=result.latency,
            metadata=_metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        plog.error("RAG query failed", error=str(e)[:300])
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipeline/build")
async def build_pipeline(req: BuildPipelineRequest):
    """
    Build (or rebuild) a client pipeline.

    Resolution order:
        1. client_id → authoritative get_client_config() (preferred).
        2. config_dict → inline admin override (validated before build).

    At least one of client_id or config_dict must be provided.
    """
    plog = PipelineLogger(request_path="rag_api")
    try:
        if req.client_id and not req.config_dict:
            tenant = _validated_tenant(
                req.client_id, endpoint="build_pipeline",
            )
            config = get_client_config(tenant)
            plog.info(
                "Pipeline build from authoritative config",
                authoritative_config_used=True,
                config_fingerprint=get_config_fingerprint(config),
            )
        elif req.config_dict:
            config = ClientConfig.from_dict(req.config_dict)
            issues = validate_config_compatibility(config)
            errors = [i for i in issues if i.severity == IssueSeverity.ERROR]
            if errors:
                raise ConfigValidationError(client_id=config.client_id, issues=errors)
            plog.info(
                "Pipeline build from inline admin override",
                authoritative_config_used=False,
                config_fingerprint=get_config_fingerprint(config),
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either 'client_id' or 'config_dict'.",
            )

        plog = plog.bind(client_id=config.client_id)
        plog.info("Pipeline build requested")

        pipeline_factory.invalidate(config.client_id)
        pipeline = pipeline_factory.build(config)

        plog.info(
            "Pipeline built successfully",
            pipeline_id=config.client_id,
            embedder_model=pipeline.embedder.info.model,
            vectordb_backend=pipeline.vectordb.kind,
        )

        return {
            "status":     "built",
            "client_id":  config.client_id,
            "vectordb":   pipeline.vectordb.kind,
            "embedder":   pipeline.embedder.info.model,
            "llm":        repr(pipeline.llm) if pipeline.llm else "none",
            "reranker":   pipeline.reranker.info.model if pipeline.reranker else "none",
            "cached":     True,
        }

    except ConfigValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        plog.error("Pipeline build failed", error=str(e)[:300])
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/pipeline/cache")
async def invalidate_pipeline(
    client_id: str = Query(..., description="Client ID to remove from cache"),
):
    """
    Invalidate a cached pipeline.
    Next query for this client will trigger a fresh pipeline build.
    """
    tenant = _validated_tenant(
        client_id, source="query", endpoint="invalidate_pipeline",
    )
    pipeline_factory.invalidate(tenant)
    return {"status": "invalidated", "client_id": tenant}


@router.get("/pipeline/health")
async def pipeline_health(
    client_id: str = Query(..., description="Client ID to check"),
):
    """Per-client pipeline health check (VectorDB reachability, etc.)"""
    tenant = _validated_tenant(
        client_id, source="query", endpoint="pipeline_health",
    )
    pipeline = pipeline_factory.get_cached(tenant)
    if pipeline is None:
        raise HTTPException(
            status_code=404,
            detail=f"No cached pipeline for client_id='{tenant}'.",
        )
    return pipeline.health_check()


@router.get("/pipeline/list")
async def list_pipelines():
    """List all currently cached client pipeline IDs."""
    return {
        "cached_pipelines": pipeline_factory.list_cached(),
        "count": len(pipeline_factory.list_cached()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shadow dual execution (migration observability)
# ─────────────────────────────────────────────────────────────────────────────

def _shadow_dual_execute(
    *,
    client_id: str,
    sanitized_query: str,
    old_result: Any,
    query_kwargs: Dict[str, Any],
    plog: PipelineLogger,
) -> None:
    """
    If the migration controller indicates shadow dual execution for this
    client, build the new ClientConfig pipeline, run the same query, compare
    results, and log metrics. ALWAYS non-blocking — failures are swallowed.

    The old_result is NEVER replaced. This function has no return value.
    """
    import asyncio
    import time

    try:
        from app.core.migration.migration_controller import migration_controller

        if not migration_controller.should_run_shadow_dual_execution(client_id):
            return

        try:
            new_config = get_client_config(client_id)
        except Exception:
            return
        new_pipeline = pipeline_factory.build(new_config)

        plog.info("Shadow dual execution: running new pipeline", client_id=client_id)

        t0 = time.perf_counter()
        new_result = new_pipeline.query(sanitized_query, **query_kwargs)
        new_latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        metrics = _evaluate_shadow_results(old_result, new_result, new_latency_ms)

        try:
            logger.info(
                "%s",
                _json_module.dumps({
                    "event": "SHADOW_DUAL_EXECUTION",
                    "client_id": client_id,
                    **metrics,
                }, default=str),
            )
        except Exception:
            pass

        plog.info(
            "Shadow dual execution complete",
            client_id=client_id,
            answer_match=metrics.get("answer_match"),
            chunk_overlap_pct=metrics.get("chunk_overlap_pct"),
            new_latency_ms=new_latency_ms,
        )

    except Exception as exc:
        logger.debug(
            "[rag_api] Shadow dual execution failed for '%s': %s — non-fatal",
            client_id, exc,
        )


def _evaluate_shadow_results(old_result: Any, new_result: Any, new_latency_ms: float) -> Dict[str, Any]:
    """
    Compare old and new pipeline results for shadow observability.
    Returns a metrics dict — never raises.
    """
    try:
        old_chunks = set(
            c.get("chunk_id") or c.get("id", "")
            for c in getattr(old_result, "context_chunks", [])
            if isinstance(c, dict)
        )
        new_chunks = set(
            c.get("chunk_id") or c.get("id", "")
            for c in getattr(new_result, "context_chunks", [])
            if isinstance(c, dict)
        )

        overlap = old_chunks & new_chunks if old_chunks and new_chunks else set()
        union = old_chunks | new_chunks if old_chunks or new_chunks else {None}
        overlap_pct = round(len(overlap) / max(len(union), 1) * 100, 1)

        old_answer = (getattr(old_result, "final_answer", None) or "").strip()
        new_answer = (getattr(new_result, "final_answer", None) or "").strip()
        answer_match = old_answer == new_answer if old_answer and new_answer else None

        old_trust = getattr(old_result, "trust_score", None)
        new_trust = getattr(new_result, "trust_score", None)

        old_latency = getattr(old_result, "latency", {}).get("total_ms", 0)

        return {
            "old_chunk_count": len(old_chunks),
            "new_chunk_count": len(new_chunks),
            "chunk_overlap_pct": overlap_pct,
            "answer_match": answer_match,
            "old_trust_score": old_trust,
            "new_trust_score": new_trust,
            "old_latency_ms": old_latency,
            "new_latency_ms": new_latency_ms,
            "latency_delta_ms": round(new_latency_ms - old_latency, 2),
        }
    except Exception:
        return {"evaluation_error": True, "new_latency_ms": new_latency_ms}
