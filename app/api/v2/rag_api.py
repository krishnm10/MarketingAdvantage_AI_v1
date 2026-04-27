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
================================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.pipeline_factory import pipeline_factory
from app.core.config.client_config_schema import ClientConfig
from app.utils.path_sanitizer import sanitize_client_id
from app.utils.pipeline_logger import PipelineLogger

logger = logging.getLogger(__name__)

router = APIRouter()

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIRS = [
    _REPO_ROOT / "app" / "core" / "configs",
    _REPO_ROOT / "configs",  # backward-compatible fallback
]
_PROMPTS_DIR = _REPO_ROOT / "app" / "core" / "configs" / "prompts"


def _find_client_config(client_id: str) -> Optional[Path]:
    safe_id = sanitize_client_id(client_id)
    for base in _CONFIG_DIRS:
        for ext in ("json", "yaml"):
            candidate = (base / f"{safe_id}.{ext}").resolve()
            if not str(candidate).startswith(str(base.resolve())):
                logger.warning(
                    "[rag_api] Path containment violation for client_id=%.20s",
                    client_id[:20],
                )
                continue
            if candidate.exists():
                return candidate
    return None


def _resolve_prompt_template(template_id: str) -> Optional[str]:
    """
    Load a prompt template by ID from the prompts store and return
    the system_instructions string for use as the LLM system prompt.
    Returns None if the template is not found or cannot be loaded.
    """
    import json as _json

    template_path = _PROMPTS_DIR / f"{template_id}.json"
    if not template_path.exists():
        logger.warning(
            "[rag_api] Prompt template '%s' not found at %s",
            template_id, template_path,
        )
        return None
    try:
        with template_path.open() as f:
            data = _json.load(f)
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
    Accepts either inline config dict OR a path to a config JSON/YAML file.
    """
    config_dict: Optional[Dict[str, Any]] = Field(
        None,
        description="Inline ClientConfig as JSON object."
    )
    config_file: Optional[str] = Field(
        None,
        description="Path to a ClientConfig JSON or YAML file (server-side path)."
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

    The pipeline must be built first via POST /api/v2/rag/pipeline/build
    OR it will be built on-the-fly if a config file exists at
    app/core/configs/{client_id}.json.

    The pipeline is cached after first build — subsequent queries are fast.
    """
    plog = PipelineLogger(request_path="rag_api", client_id=req.client_id)
    plog.info("RAG query received", query_length=len(req.query))

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

    pipeline = pipeline_factory.get_cached(req.client_id)
    if pipeline is None:
        config_path = _find_client_config(req.client_id)
        if config_path is None:
            searched_dirs = ", ".join(str(p) for p in _CONFIG_DIRS)
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No pipeline found for client_id='{req.client_id}' "
                    f"and no config file in: {searched_dirs}. "
                    f"Use POST /api/v2/rag/pipeline/build first."
                ),
            )
        try:
            if str(config_path).endswith(".yaml"):
                config = ClientConfig.from_yaml_file(config_path)
            else:
                config = ClientConfig.from_json_file(config_path)
            pipeline_factory.build(config)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to auto-build pipeline from config: {e}",
            )
        pipeline = pipeline_factory.get_cached(req.client_id)

    try:
        if not pipeline:
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline for '{req.client_id}' not in cache after build.",
            )

        plog = plog.bind(
            pipeline_id=req.client_id,
            embedder_model=getattr(getattr(pipeline, "embedder", None), "info", None) and pipeline.embedder.info.model,
            vectordb_backend=getattr(pipeline.vectordb, "kind", None),
        )

        # Resolve effective system prompt:
        # 1. Per-request override takes precedence.
        # 2. If not overridden, look up the prompt_template_id from the
        #    pipeline's retrieval config (saved in the client's config file).
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
                        template_id, req.client_id,
                    )

        result = pipeline.query(
            sanitized_query,
            metadata_filters=req.metadata_filters,
            top_k_retrieval=req.top_k_retrieval,
            top_k_final=req.top_k_final,
            system_prompt=effective_system_prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )

        plog.query_complete(
            chunks_used=len(result.context_chunks),
            reranked=result.reranked,
            trust_score=result.trust_score,
            total_ms=result.latency.get("total_ms", 0),
            token_usage=result.metadata.get("token_usage"),
        )

        # ── Shadow dual execution (observability only) ─────────────────────
        _shadow_dual_execute(
            client_id=req.client_id,
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
            client_id=req.client_id,
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
    Build (or rebuild) a client pipeline from config.
    Accepts inline config dict OR server-side config file path.
    """
    plog = PipelineLogger(request_path="rag_api")
    try:
        if req.config_dict:
            config = ClientConfig.from_dict(req.config_dict)
        elif req.config_file:
            p = Path(req.config_file)
            if not p.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Config file not found: {req.config_file}",
                )
            config = (
                ClientConfig.from_yaml_file(p)
                if str(p).endswith(".yaml")
                else ClientConfig.from_json_file(p)
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either 'config_dict' or 'config_file'.",
            )

        plog = plog.bind(client_id=config.client_id)
        plog.info("Pipeline build requested")

        # Invalidate existing cache first (force rebuild)
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
    pipeline_factory.invalidate(client_id)
    return {"status": "invalidated", "client_id": client_id}


@router.get("/pipeline/health")
async def pipeline_health(
    client_id: str = Query(..., description="Client ID to check"),
):
    """Per-client pipeline health check (VectorDB reachability, etc.)"""
    pipeline = pipeline_factory.get_cached(client_id)
    if pipeline is None:
        raise HTTPException(
            status_code=404,
            detail=f"No cached pipeline for client_id='{client_id}'.",
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
    import json as _json
    import time

    try:
        from app.core.migration.migration_controller import migration_controller

        if not migration_controller.should_run_shadow_dual_execution(client_id):
            return

        from app.core.config.client_config_resolver import (
            _find_config_path,
            _load_config_from_path,
        )

        config_path = _find_config_path(client_id)
        if config_path is None:
            return

        new_config = _load_config_from_path(client_id, config_path)
        new_pipeline = pipeline_factory.build(new_config)

        plog.info("Shadow dual execution: running new pipeline", client_id=client_id)

        t0 = time.perf_counter()
        new_result = new_pipeline.query(sanitized_query, **query_kwargs)
        new_latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        metrics = _evaluate_shadow_results(old_result, new_result, new_latency_ms)

        try:
            logger.info(
                "%s",
                _json.dumps({
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
