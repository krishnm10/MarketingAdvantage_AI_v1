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

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session_v2 import get_db
from app.auth.guards import require_role
from app.services.ingestion.ingestion_service_v2 import get_embedder
from app.utils.pipeline_logger import PipelineLogger

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/retrieve")


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Natural language question")
    client_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="REQUIRED: Tenant / client identifier for tenant-scoped retrieval.",
    )
    intent: str = Field("answer", description="Retrieval intent: answer | explore | audit")
    top_k: Optional[int] = Field(None, ge=1, le=50, description="Override max results (default from policy)")
    search_mode: str = Field("semantic", description="Search mode: semantic | hybrid | keyword")
    enable_hyde: bool = Field(False, description="Enable HyDE query expansion (requires LLM)")
    hybrid_alpha: float = Field(0.7, ge=0.0, le=1.0, description="Semantic vs keyword weight (1.0=semantic, 0.0=keyword)")
    similarity_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity score filter")
    # LLM generation (optional — if true, retrieved context is sent to the configured LLM)
    generate_answer: bool = Field(False, description="Generate an LLM answer from retrieved context")
    max_context_chunks: int = Field(5, ge=1, le=15, description="Max retrieved chunks to pass to LLM")


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
    search_mode: str = "semantic"
    total_results: int
    total_dropped: int
    latency_ms: float
    results: List[ResultItem]
    # LLM generation (present only when generate_answer=True was requested)
    answer: Optional[str] = None
    answer_model: Optional[str] = None
    answer_latency_ms: Optional[float] = None
    # Explicit error message when generation was requested but failed
    answer_error: Optional[str] = None
    # Pipeline debug metadata (always populated — used by debug panel)
    debug_info: Optional[Dict[str, Any]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Pluggable embedder — uses ingestion pipeline embedder for the resolved client_id
# ─────────────────────────────────────────────────────────────────────────────

def _embed_query(query: str, business_id: Optional[str] = None) -> List[float]:
    """Embed a query string → vector using the pluggable embedder for this tenant."""
    embedder = get_embedder(business_id)
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
    from app.utils.tenant_validator import (
        validate_tenant_id_strict,
        get_storage_uuid_str,
        TenantValidationError,
    )
    from app.retrieval.components import (
        resolve_config_or_fail,
        resolve_runtime_components,
        resolve_runtime_components_legacy,
        instantiate_llm,
    )
    from app.retrieval.runtime import RetrievalRuntime
    from app.retrieval.repository import RetrievalRepository
    from app.retrieval.types_retrieve import QueryContext, RetrievalIntent
    from app.retrieval.policy import DEFAULT_POLICY_REGISTRY

    # Validate tenant with strict enforcement (uses TENANT_ENFORCEMENT_MODE)
    try:
        tenant_ctx = validate_tenant_id_strict(
            req.client_id,
            source="body",
            endpoint="retrieve_query",
        )
    except TenantValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    retrieve_cid = tenant_ctx.tenant_id
    # Storage UUID for vector/SQL filtering (Phase 2 will use this)
    _storage_uuid_str = get_storage_uuid_str(tenant_ctx)

    _cfg_for_authoritative, runtime_mode = resolve_config_or_fail(retrieve_cid)
    if _cfg_for_authoritative is not None:
        rc = resolve_runtime_components(_cfg_for_authoritative)
    else:
        rc = resolve_runtime_components_legacy()

    plog = PipelineLogger(
        request_path="retrieve_api",
        client_id=retrieve_cid,
        embedder_model=rc.embedder_type,
        vectordb_backend=rc.vectordb_type,
    )
    plog.info("Retrieve query received", query_length=len(req.query), intent=req.intent)

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

    # ── Security gate: PII redaction + prompt injection detection ─────────────
    # Must run before ANY external call (embed, LLM) so raw PII never leaves
    # the service boundary. Injection-detected queries are hard-blocked (400).
    from app.middleware.security_middleware import scan_text as _security_scan_text
    _scan_result = _security_scan_text(req.query, context="retrieve_query")
    if _scan_result.injection_detected:
        raise HTTPException(
            status_code=400,
            detail="Query rejected: potential prompt injection detected.",
        )
    # Use the PII-redacted version for all downstream processing
    embed_text = _scan_result.redacted_text

    def _hyde_llm_provider() -> str:
        p = (rc.llm_provider or "ollama").lower()
        return "gemini" if p == "google" else p

    # 1. Optional HyDE expansion — use LLM to generate hypothetical answer,
    #    then embed that instead of the raw query for better retrieval.
    if req.enable_hyde:
        try:
            if (rc.llm_provider or "").strip().lower() not in ("", "none"):
                llm_provider = _hyde_llm_provider()
                if llm_provider == "google":
                    llm_provider = "gemini"
                llm, _resolved = instantiate_llm(
                    llm_provider,
                    rc.llm_model,
                    api_key_env=rc.llm_api_key_env,
                    base_url=rc.llm_base_url,
                )
                hyde_prompt = (
                    "Write a short factual paragraph that would answer this question. "
                    "Do not say you don't know. Just give a plausible answer in 2-3 sentences.\n\n"
                    f"Question: {req.query}\n\nAnswer:"
                )
                resp = llm.generate(hyde_prompt, temperature=0.0, max_tokens=200)
                text = (resp.text or "").strip()
                if len(text) > 20:
                    embed_text = text
                    logger.info("[RetrieveAPI] HyDE: embedding hypothetical answer (%d chars)", len(text))
        except Exception as e:
            logger.warning("[RetrieveAPI] HyDE expansion failed, using raw query: %s", e)

    # 2. Embed query
    try:
        query_embedding = await _embed_in_thread(embed_text, retrieve_cid)
    except Exception as e:
        logger.error(f"[RetrieveAPI] Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    # Build repository aligned with merged Client JSON when authoritative
    if runtime_mode == "authoritative_config" and _cfg_for_authoritative is not None:
        from app.services.ingestion.ingestion_service_v2 import get_query_pipeline_for_client

        _pipe = get_query_pipeline_for_client(retrieve_cid)
        repository = RetrievalRepository(
            db_session=db,
            vectordb=_pipe.vectordb,
            collection=rc.collection,
        )
    else:
        repository = RetrievalRepository(db_session=db)

    runtime = RetrievalRuntime(
        repository=repository,
        policy_registry=DEFAULT_POLICY_REGISTRY,
    )

    # 3. Retrieve with tenant isolation
    ctx = QueryContext(
        query=req.query,
        intent=intent_enum,
        requested_at=int(time.time()),
    )

    try:
        ranked_results, dropped = await runtime.retrieve(
            ctx=ctx,
            query_embedding=query_embedding,
            max_results_override=req.top_k,
            tenant_id=retrieve_cid,           # Tenant slug for logging
            storage_uuid=_storage_uuid_str,   # Storage UUID for filtering
        )
    except Exception as e:
        logger.error(f"[RetrieveAPI] Retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    finally:
        repository.close()

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    # 4. Optional hybrid/keyword re-ranking via BM25 + RRF
    search_mode = req.search_mode.lower()
    if search_mode in ("hybrid", "keyword") and ranked_results:
        try:
            from app.core.search.bm25_index import BM25Index
            from app.core.search.rrf_fusion import reciprocal_rank_fusion

            # Build BM25 index from the retrieved chunks
            chunk_dicts = [
                {"id": r.chunk_id, "text": r.text, "score": r.score, "metadata": {}}
                for r in ranked_results
            ]
            bm25 = BM25Index()
            bm25.build(chunk_dicts)
            keyword_hits = bm25.search(req.query, k=len(ranked_results))
            keyword_dicts = [
                {"id": h.id, "text": h.text, "score": h.score, "metadata": h.metadata}
                for h in keyword_hits
            ]

            if search_mode == "hybrid":
                fused = reciprocal_rank_fusion(
                    vector_hits=chunk_dicts,
                    keyword_hits=keyword_dicts,
                    alpha=req.hybrid_alpha,
                    top_k=len(ranked_results),
                )
                # Re-order ranked_results based on RRF order
                fused_order = {f.id: i for i, f in enumerate(fused)}
                ranked_results.sort(
                    key=lambda r: fused_order.get(r.chunk_id, 999)
                )
                logger.info(
                    "[RetrieveAPI] Hybrid RRF applied | alpha=%.2f | %d results",
                    req.hybrid_alpha, len(ranked_results),
                )
            else:
                # keyword-only: re-order by BM25 rank
                bm25_order = {h.id: i for i, h in enumerate(keyword_hits)}
                ranked_results.sort(
                    key=lambda r: bm25_order.get(r.chunk_id, 999)
                )
                logger.info("[RetrieveAPI] Keyword BM25 re-rank applied | %d results", len(ranked_results))
        except Exception as e:
            logger.warning("[RetrieveAPI] Hybrid/keyword search failed, using semantic order: %s", e)

    # 4b. Optional reranker — stack-aware resolver + circuit breaker (non-fatal).
    _reranker_name = (rc.reranker_name or "none").strip().lower()
    reranker_used = "none"

    if (
        _reranker_name not in ("none", "", "disabled")
        and ranked_results
        and _cfg_for_authoritative is not None
    ):
        from app.core.rerankers.base import RerankCandidate as _RerankCandidate
        from app.retrieval.reranker_runtime import apply_reranker_with_fallback

        _rr_candidates = [
            _RerankCandidate(
                id=r.chunk_id,
                text=r.text,
                vector_score=r.score,
                metadata={},
            )
            for r in ranked_results
        ]
        _rr_top_k = req.top_k or 10
        _rr_scored, reranker_used, _fb_applied, _fb_reason = apply_reranker_with_fallback(
            config=_cfg_for_authoritative,
            query=req.query,
            candidates=_rr_candidates,
            top_k=_rr_top_k,
        )
        if reranker_used != "none":
            _rr_score_map = {c.id: (c.rerank_score or 0.0) for c in _rr_scored}
            ranked_results = [r for r in ranked_results if r.chunk_id in _rr_score_map]
            ranked_results.sort(
                key=lambda r: _rr_score_map.get(r.chunk_id, 0.0), reverse=True
            )
            logger.info(
                "[RetrieveAPI] Reranker '%s' applied | %d → %d candidates",
                reranker_used,
                len(_rr_candidates),
                len(ranked_results),
            )
        elif _fb_applied:
            logger.warning(
                "[RetrieveAPI] Reranker fallback exhausted: %s",
                _fb_reason,
            )

    # 5. Apply similarity_threshold filter
    if req.similarity_threshold > 0:
        ranked_results = [r for r in ranked_results if r.score >= req.similarity_threshold]

    # 6. Build response
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

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Optional LLM answer generation (RAG completion step)
    #    Uses the top-N chunks as context and calls the configured LLM.
    #    Supports all providers: openai, ollama, groq, gemini, anthropic.
    #    Failures surface as answer_error — never silently swallowed.
    # ─────────────────────────────────────────────────────────────────────────
    answer: Optional[str] = None
    answer_model: Optional[str] = None
    answer_latency_ms: Optional[float] = None
    answer_error: Optional[str] = None

    _llm_provider_name = (
        "gemini" if (rc.llm_provider or "openai").lower() == "google"
        else (rc.llm_provider or "openai").lower()
    )
    _rag_min_score = float(rc.rag_min_score)

    if req.generate_answer:
        if not results:
            answer_error = "No results retrieved — cannot generate a grounded answer without context."
        else:
            _max_score = max(r.score for r in results)
            if _max_score < _rag_min_score:
                answer_error = (
                    f"Retrieved context confidence is too low "
                    f"(best score={_max_score:.3f} < threshold={_rag_min_score:.2f}). "
                    "Try a more specific query or lower the 'Min Score' filter."
                )
            else:
                try:
                    _llm_gen_start = time.perf_counter()
                    _llm = None

                    if _llm_provider_name not in ("openai", "ollama", "groq", "grok", "gemini", "google", "anthropic"):
                        answer_error = (
                            f"LLM provider '{_llm_provider_name}' is not wired for answer generation. "
                            "Supported: openai, ollama, groq, gemini, anthropic. "
                            "Configure the LLM provider in Client JSON (RAG / pipeline-pluggable)."
                        )
                    else:
                        _prov = "gemini" if _llm_provider_name == "google" else _llm_provider_name
                        if _prov == "grok":
                            _prov = "groq"
                        _llm, answer_model = instantiate_llm(
                            _prov,
                            rc.llm_model,
                            api_key_env=rc.llm_api_key_env,
                            base_url=rc.llm_base_url,
                        )

                    if _llm is not None:
                        _top_chunks = results[: req.max_context_chunks]
                        _context_parts = []
                        for _ci, _chunk in enumerate(_top_chunks, start=1):
                            _context_parts.append(
                                f"[Source {_ci}] (relevance={_chunk.score:.3f})\n{_chunk.text.strip()}"
                            )
                        _context_str = "\n\n---\n\n".join(_context_parts)

                        # ── Context sanitization: PII redaction on retrieved text ─
                        _ctx_scan = _security_scan_text(
                            _context_str, context="retrieved_context",
                        )
                        _sanitized_context = _ctx_scan.redacted_text
                        if _sanitized_context != _context_str:
                            logger.info(
                                "[RetrieveAPI] Context sanitized before LLM — "
                                "PII redacted from retrieved chunks"
                            )
                            _context_str = _sanitized_context

                        # Enterprise-grade RAG prompt: strict grounding, citation, no hallucination
                        _rag_prompt = (
                            "You are a precise, grounded enterprise assistant.\n"
                            "Your task: answer the user's question using ONLY the retrieved passages below.\n\n"
                            "Rules:\n"
                            "  1. Cite every source you use with its number: [Source 1], [Source 2], etc.\n"
                            "  2. If the passages lack sufficient information to answer confidently, "
                            "respond EXACTLY with: "
                            "'I could not find a reliable answer in the available documents.'\n"
                            "  3. Never invent facts. Do not add information not present in the sources.\n"
                            "  4. Keep the answer factual, concise, and professional.\n\n"
                            f"QUESTION: {req.query}\n\n"
                            f"RETRIEVED PASSAGES:\n{_context_str}\n\n"
                            "ANSWER (grounded, with citations):"
                        )

                        _resp = _llm.generate(_rag_prompt, temperature=0.0, max_tokens=700)
                        _raw_answer = (_resp.text or "").strip()
                        if _raw_answer:
                            answer = _raw_answer
                        else:
                            answer_error = "LLM returned an empty response."

                        answer_latency_ms = round(
                            (time.perf_counter() - _llm_gen_start) * 1000, 2
                        )
                        logger.info(
                            "[RetrieveAPI] LLM answer | model=%s chunks=%d latency=%.0fms ok=%s",
                            answer_model,
                            len(_top_chunks),
                            answer_latency_ms,
                            answer is not None,
                        )

                except Exception as _gen_err:
                    logger.warning(
                        "[RetrieveAPI] LLM answer generation failed: %s", _gen_err,
                        exc_info=True,
                    )
                    answer_error = (
                        f"LLM generation error ({type(_gen_err).__name__}): "
                        f"{str(_gen_err)[:300]}"
                    )
                    answer = None

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Build debug metadata (always populated — used by the debug panel)
    # ─────────────────────────────────────────────────────────────────────────
    debug_info: Dict[str, Any] = {
        "client_id":               retrieve_cid,
        "embedder":                rc.embedder_type,
        "embedder_model":          rc.embedder_model or None,
        "llm_provider":            _llm_provider_name,
        "vectordb":                rc.vectordb_type,
        "chunking_strategy":       rc.chunking_strategy,
        "collection":              rc.collection,
        "search_mode":            search_mode,
        "intent":                 req.intent,
        "top_k_requested":        req.top_k,
        "top_k_returned":         len(results),
        "total_dropped":          len(dropped),
        "generate_answer":        req.generate_answer,
        "answer_generated":       answer is not None,
        "score_gate_threshold":   _rag_min_score,
        "max_score":              round(max(r.score for r in results), 4) if results else None,
        "min_score":              round(min(r.score for r in results), 4) if results else None,
        "hyde_enabled":           req.enable_hyde,
        "hybrid_alpha":           req.hybrid_alpha if search_mode == "hybrid" else None,
        "reranker_used":          reranker_used,
        "security_scan": {
            "pii_detected":       _scan_result.has_pii,
            "injection_detected": _scan_result.injection_detected,
        },
    }

    plog.query_complete(
        chunks_used=len(results),
        reranked=reranker_used != "none",
        total_ms=elapsed_ms,
        search_mode=search_mode,
        generate_answer=req.generate_answer,
        answer_model=answer_model,
    )

    return RetrieveResponse(
        query=req.query,
        intent=req.intent,
        search_mode=search_mode,
        total_results=len(results),
        total_dropped=len(dropped),
        latency_ms=elapsed_ms,
        results=results,
        answer=answer,
        answer_model=answer_model,
        answer_latency_ms=answer_latency_ms,
        answer_error=answer_error,
        debug_info=debug_info,
    )


@router.get("/intents")
async def list_intents(_user=Depends(require_role("admin"))):
    """List available retrieval intents with descriptions."""
    return [
        {"value": "answer", "label": "Answer", "description": "Strict, high-trust retrieval for factual answers"},
        {"value": "explore", "label": "Explore", "description": "Broader recall for exploratory queries"},
        {"value": "audit", "label": "Audit", "description": "No filtering — full transparency for auditing"},
    ]


@router.get("/pipeline-config")
async def retrieve_pipeline_config(
    client_id: Optional[str] = Query(
        None,
        description="Client / tenant id (defaults via MAI_DEFAULT_BUSINESS_ID or 'default').",
    ),
    _user=Depends(require_role("admin")),
):
    """
    Return active retrieval pipeline fields from merged Client JSON (resolver).
    Used by the Retrieve page debug panel.
    """
    import os as _os

    from app.middleware.security_middleware import validate_business_id
    from app.retrieval.components import (
        resolve_config_or_fail,
        resolve_runtime_components,
        resolve_runtime_components_legacy,
    )

    cid = validate_business_id(client_id or _os.getenv("MAI_DEFAULT_BUSINESS_ID"))
    cfg, mode = resolve_config_or_fail(cid)
    rc = resolve_runtime_components(cfg) if cfg is not None else resolve_runtime_components_legacy()

    llm_provider = (rc.llm_provider or "openai").lower()
    if llm_provider == "google":
        llm_provider = "gemini"

    llm_key_set = {
        "openai":    bool(_os.getenv("OPENAI_API_KEY")),
        "groq":      bool(_os.getenv("GROQ_API_KEY")),
        "grok":      bool(_os.getenv("GROQ_API_KEY")),
        "anthropic": bool(_os.getenv("ANTHROPIC_API_KEY")),
        "gemini":    bool(_os.getenv("GOOGLE_API_KEY") or _os.getenv("GEMINI_API_KEY")),
        "google":    bool(_os.getenv("GOOGLE_API_KEY") or _os.getenv("GEMINI_API_KEY")),
        "ollama":    True,
    }

    if rc.llm_api_key_env:
        llm_api_ok = bool(_os.getenv(rc.llm_api_key_env))
    else:
        llm_api_ok = llm_key_set.get(llm_provider, False)

    return {
        "client_id":            cid,
        "config_mode":          mode,
        "embedder":             rc.embedder_type,
        "embedder_model":       rc.embedder_model or "",
        "llm_provider":         llm_provider,
        "llm_model":            rc.llm_model or "unknown",
        "llm_api_key_set":      llm_api_ok,
        "vectordb":             rc.vectordb_type,
        "collection":           rc.collection,
        "chunking_strategy":    rc.chunking_strategy,
        "rag_min_score":        rc.rag_min_score,
        "supported_providers":  ["openai", "ollama", "groq", "gemini", "anthropic"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _embed_in_thread(query: str, business_id: Optional[str] = None) -> List[float]:
    """Run embedding in thread pool to avoid blocking the event loop."""
    import asyncio
    return await asyncio.to_thread(_embed_query, query, business_id)
