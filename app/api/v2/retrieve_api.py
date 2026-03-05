"""
================================================================================
Marketing Advantage AI — Retrieval API Router
File: app/api/v2/retrieve_api.py

Endpoints:
  POST /api/v2/retrieve/query   → Run retrieval query (embed + retrieve + rank)
  GET  /api/v2/retrieve/intents → List available retrieval intents
================================================================================
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session_v2 import get_db
from app.auth.guards import require_role
from app.services.ingestion.ingestion_service_v2 import get_embedder

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/retrieve")


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Natural language question")
    intent: str = Field("answer", description="Retrieval intent: answer | explore | audit")
    top_k: Optional[int] = Field(None, ge=1, le=50, description="Override max results (default from policy)")


class SignalDetail(BaseModel):
    semantic_score: Optional[float] = None
    tap_trust_score: Optional[float] = None
    agentic_validation_score: Optional[float] = None
    reasoning_quality_score: Optional[float] = None
    conflict_modifier: Optional[float] = None
    temporal_decay: Optional[float] = None


class ResultItem(BaseModel):
    rank: int
    chunk_id: str
    text: str
    score: float
    trust_decision: Optional[str] = None
    explanation: Dict[str, Any] = {}
    signals: Optional[SignalDetail] = None


class RetrieveResponse(BaseModel):
    query: str
    intent: str
    total_results: int
    total_dropped: int
    latency_ms: float
    results: List[ResultItem]


# ─────────────────────────────────────────────────────────────────────────────
# Pluggable embedder (uses MAI_EMBEDDER from .env — ollama/openai/huggingface)
# ─────────────────────────────────────────────────────────────────────────────

def _embed_query(query: str) -> List[float]:
    """Embed a query string → vector using the pluggable embedder."""
    embedder = get_embedder()
    result = embedder.encode(query, normalize_embeddings=True)
    return result.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/query", response_model=RetrieveResponse)
async def retrieve_query(
    req: RetrieveRequest,
    db: AsyncSession = Depends(get_db),
    _user=Depends(require_role("admin")),
):
    """
    Run an enterprise retrieval query.

    Flow: embed query → semantic recall → governance scoring → ranked results.
    Mirrors retrieve_cli.py but exposed as an HTTP API.
    """
    from app.retrieval.runtime import RetrievalRuntime
    from app.retrieval.policy import DEFAULT_POLICY_REGISTRY
    from app.retrieval.repository import RetrievalRepository
    from app.retrieval.types_retrieve import QueryContext, RetrievalIntent

    # Validate intent
    intent_map = {
        "answer": RetrievalIntent.ANSWER,
        "explore": RetrievalIntent.EXPLORE,
        "audit": RetrievalIntent.AUDIT,
    }
    intent_enum = intent_map.get(req.intent.lower())
    if intent_enum is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid intent '{req.intent}'. Must be one of: answer, explore, audit",
        )

    start = time.perf_counter()

    # 1. Embed query
    try:
        query_embedding = await _embed_in_thread(req.query)
    except Exception as e:
        logger.error(f"[RetrieveAPI] Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    # 2. Build runtime
    repository = RetrievalRepository(db_session=db)
    runtime = RetrievalRuntime(
        repository=repository,
        policy_registry=DEFAULT_POLICY_REGISTRY,
    )

    # Override max_results if user requested specific top_k
    if req.top_k is not None:
        policy = DEFAULT_POLICY_REGISTRY.resolve(intent_enum)
        policy.max_results = req.top_k

    # 3. Retrieve
    ctx = QueryContext(
        query=req.query,
        intent=intent_enum,
        requested_at=int(time.time()),
    )

    try:
        ranked_results, dropped = await runtime.retrieve(
            ctx=ctx,
            query_embedding=query_embedding,
        )
    except Exception as e:
        logger.error(f"[RetrieveAPI] Retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    # 4. Build response
    results: List[ResultItem] = []
    for idx, r in enumerate(ranked_results, start=1):
        signals = None
        sig_data = r.explanation.get("signals", {})
        if sig_data:
            signals = SignalDetail(
                semantic_score=sig_data.get("semantic_score"),
                tap_trust_score=sig_data.get("tap_trust_score"),
                agentic_validation_score=sig_data.get("agentic_validation_score"),
                reasoning_quality_score=sig_data.get("reasoning_quality_score"),
                conflict_modifier=sig_data.get("conflict_modifier"),
                temporal_decay=sig_data.get("temporal_decay"),
            )

        results.append(ResultItem(
            rank=idx,
            chunk_id=r.chunk_id,
            text=r.text,
            score=round(r.score, 6),
            trust_decision=r.trust_decision if isinstance(r.trust_decision, str) else (
                r.trust_decision.value if r.trust_decision else None
            ),
            explanation=r.explanation,
            signals=signals,
        ))

    logger.info(
        f"[RetrieveAPI] query='{req.query[:60]}' intent={req.intent} "
        f"results={len(results)} dropped={len(dropped)} latency={elapsed_ms}ms"
    )

    return RetrieveResponse(
        query=req.query,
        intent=req.intent,
        total_results=len(results),
        total_dropped=len(dropped),
        latency_ms=elapsed_ms,
        results=results,
    )


@router.get("/intents")
async def list_intents(_user=Depends(require_role("admin"))):
    """List available retrieval intents with descriptions."""
    return [
        {"value": "answer", "label": "Answer", "description": "Strict, high-trust retrieval for factual answers"},
        {"value": "explore", "label": "Explore", "description": "Broader recall for exploratory queries"},
        {"value": "audit", "label": "Audit", "description": "No filtering — full transparency for auditing"},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _embed_in_thread(query: str) -> List[float]:
    """Run embedding in thread pool to avoid blocking the event loop."""
    import asyncio
    return await asyncio.to_thread(_embed_query, query)
