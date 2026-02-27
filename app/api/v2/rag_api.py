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

logger = logging.getLogger(__name__)

router = APIRouter()


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
    configs/{client_id}.json.

    The pipeline is cached after first build — subsequent queries are fast.
    """
    # Try to get cached pipeline first
    cached = pipeline_factory.list_cached()
    if req.client_id not in cached:
        # Try to auto-load from configs directory
        config_path = Path(f"configs/{req.client_id}.json")
        if not config_path.exists():
            config_path = Path(f"configs/{req.client_id}.yaml")

        if not config_path.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No pipeline found for client_id='{req.client_id}' "
                    f"and no config file at configs/{req.client_id}.json. "
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

    try:
        pipeline = pipeline_factory._cache.get(req.client_id)
        if not pipeline:
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline for '{req.client_id}' not in cache after build.",
            )

        result = pipeline.query(
            req.query,
            metadata_filters=req.metadata_filters,
            top_k_retrieval=req.top_k_retrieval,
            top_k_final=req.top_k_final,
            system_prompt=req.system_prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )

        return RAGQueryResponse(
            client_id=req.client_id,
            query=req.query,
            final_answer=result.final_answer,
            reranked=result.reranked,
            trust_score=result.trust_score,
            chunk_count=len(result.context_chunks),
            latency_ms=result.latency,
            metadata=result.metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[rag_api] Query failed for client '%s': %s", req.client_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipeline/build")
async def build_pipeline(req: BuildPipelineRequest):
    """
    Build (or rebuild) a client pipeline from config.
    Accepts inline config dict OR server-side config file path.
    """
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

        # Invalidate existing cache first (force rebuild)
        pipeline_factory.invalidate(config.client_id)
        pipeline = pipeline_factory.build(config)

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
        logger.error("[rag_api] Pipeline build failed: %s", e)
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
    cached = pipeline_factory.list_cached()
    if client_id not in cached:
        raise HTTPException(
            status_code=404,
            detail=f"No cached pipeline for client_id='{client_id}'.",
        )
    pipeline = pipeline_factory._cache[client_id]
    return pipeline.health_check()


@router.get("/pipeline/list")
async def list_pipelines():
    """List all currently cached client pipeline IDs."""
    return {
        "cached_pipelines": pipeline_factory.list_cached(),
        "count": len(pipeline_factory.list_cached()),
    }
