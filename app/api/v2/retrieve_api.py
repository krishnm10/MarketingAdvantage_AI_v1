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
from app.utils.pipeline_logger import PipelineLogger

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/retrieve")


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Natural language question")
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
    import os as _os_plog
    plog = PipelineLogger(
        request_path="retrieve_api",
        embedder_model=_os_plog.getenv("MAI_EMBEDDER", "unknown"),
        vectordb_backend=_os_plog.getenv("MAI_VECTORDB", "unknown"),
    )
    plog.info("Retrieve query received", query_length=len(req.query), intent=req.intent)

    from app.retrieval.runtime import RetrievalRuntime
    from app.retrieval.repository import RetrievalRepository
    from app.retrieval.types_retrieve import QueryContext, RetrievalIntent
    from app.retrieval.policy import DEFAULT_POLICY_REGISTRY

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

    import os as _os_module

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

    # 1. Optional HyDE expansion — use LLM to generate hypothetical answer,
    #    then embed that instead of the raw query for better retrieval.
    if req.enable_hyde:
        try:
            llm_provider = _os_module.getenv("MAI_LLM", "ollama").lower()
            llm = None
            if llm_provider == "ollama":
                from app.core.llms.ollama_v1 import OllamaLLM
                _ollama_model = _os_module.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")
                llm = OllamaLLM(model=_ollama_model)
            elif llm_provider == "openai":
                from app.core.llms.openai_v1 import OpenAILLM
                llm = OpenAILLM()
            elif llm_provider in ("groq", "grok"):
                from app.core.llms.groq_v1 import GroqLLM
                llm = GroqLLM()
            elif llm_provider in ("gemini", "google"):
                _api_key = _os_module.getenv("GEMINI_API_KEY", "")
                if _api_key:
                    from app.core.llms.gemini_v1 import GeminiLLM
                    llm = GeminiLLM(
                        model=_os_module.getenv("GEMINI_LLM_MODEL", "gemini-1.5-flash"),
                        api_key=_api_key,
                    )
            elif llm_provider == "anthropic":
                _api_key = _os_module.getenv("ANTHROPIC_API_KEY", "")
                if _api_key:
                    from app.core.llms.anthropic_v1 import AnthropicLLM
                    llm = AnthropicLLM(
                        model=_os_module.getenv("ANTHROPIC_LLM_MODEL", "claude-3-5-sonnet-20241022"),
                        api_key=_api_key,
                    )

            if llm is not None:
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
        query_embedding = await _embed_in_thread(embed_text)
    except Exception as e:
        logger.error(f"[RetrieveAPI] Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    # 2. Build runtime
    repository = RetrievalRepository(db_session=db)
    runtime = RetrievalRuntime(
        repository=repository,
        policy_registry=DEFAULT_POLICY_REGISTRY,
    )

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
            max_results_override=req.top_k,
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

    # 4b. Optional cross-encoder reranker — re-scores top-K candidates using a
    #     cross-attention model for higher precision before threshold filtering.
    #     Driven by MAI_RERANKER env var (default: none). Failures are non-fatal.
    _reranker_name = _os_module.getenv("MAI_RERANKER", "none").strip().lower()
    reranker_used = "none"

    if _reranker_name not in ("none", "", "disabled") and ranked_results:
        try:
            import app.core.rerankers.register as _rr_register  # noqa: F401 — side-effect registration
            from app.core.plugin_registry import reranker_registry as _rr_registry
            from app.core.rerankers.base import RerankCandidate as _RerankCandidate

            _rr_model = _os_module.getenv("MAI_RERANKER_MODEL", "").strip()
            _rr_build_kwargs: Dict[str, Any] = {}
            if _rr_model:
                _rr_build_kwargs["model_name"] = _rr_model

            _reranker = _rr_registry.build(_reranker_name, **_rr_build_kwargs)
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
            _rr_scored = _reranker.rerank(req.query, _rr_candidates, top_k=_rr_top_k)
            # Re-order ranked_results to match reranker's ordering
            _rr_score_map = {c.id: (c.rerank_score or 0.0) for c in _rr_scored}
            ranked_results = [r for r in ranked_results if r.chunk_id in _rr_score_map]
            ranked_results.sort(
                key=lambda r: _rr_score_map.get(r.chunk_id, 0.0), reverse=True
            )
            reranker_used = _reranker_name
            logger.info(
                "[RetrieveAPI] Reranker '%s' applied | %d → %d candidates",
                _reranker_name, len(_rr_candidates), len(ranked_results),
            )
        except Exception as _rr_err:
            logger.warning(
                "[RetrieveAPI] Reranker '%s' failed, using original order: %s",
                _reranker_name, _rr_err,
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

    _llm_provider_name = _os_module.getenv("MAI_LLM", "openai").lower()
    _rag_min_score = float(_os_module.getenv("RAG_ANSWER_MIN_SCORE", "0.25"))

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

                    if _llm_provider_name == "openai":
                        from app.core.llms.openai_v1 import OpenAILLM
                        _llm = OpenAILLM()
                        answer_model = _os_module.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

                    elif _llm_provider_name == "ollama":
                        from app.core.llms.ollama_v1 import OllamaLLM
                        answer_model = _os_module.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")
                        _llm = OllamaLLM(model=answer_model)

                    elif _llm_provider_name in ("groq", "grok"):
                        from app.core.llms.groq_v1 import GroqLLM
                        _llm = GroqLLM()
                        answer_model = _os_module.getenv("GROQ_LLM_MODEL", "llama-3.1-70b-versatile")

                    elif _llm_provider_name in ("gemini", "google"):
                        _gemini_key = _os_module.getenv("GEMINI_API_KEY", "")
                        if not _gemini_key:
                            answer_error = (
                                "LLM generation failed: GEMINI_API_KEY is not set. "
                                "Add it to your .env file."
                            )
                        else:
                            from app.core.llms.gemini_v1 import GeminiLLM
                            answer_model = _os_module.getenv("GEMINI_LLM_MODEL", "gemini-1.5-flash")
                            _llm = GeminiLLM(model=answer_model, api_key=_gemini_key)

                    elif _llm_provider_name == "anthropic":
                        _anthropic_key = _os_module.getenv("ANTHROPIC_API_KEY", "")
                        if not _anthropic_key:
                            answer_error = (
                                "LLM generation failed: ANTHROPIC_API_KEY is not set. "
                                "Add it to your .env file."
                            )
                        else:
                            from app.core.llms.anthropic_v1 import AnthropicLLM
                            answer_model = _os_module.getenv(
                                "ANTHROPIC_LLM_MODEL", "claude-3-5-sonnet-20241022"
                            )
                            _llm = AnthropicLLM(model=answer_model, api_key=_anthropic_key)

                    else:
                        answer_error = (
                            f"LLM provider '{_llm_provider_name}' is not wired for answer generation. "
                            "Supported: openai, ollama, groq, gemini, anthropic. "
                            "Update MAI_LLM in your .env file."
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
        "embedder":               _os_module.getenv("MAI_EMBEDDER", "unknown"),
        "llm_provider":           _llm_provider_name,
        "vectordb":               _os_module.getenv("MAI_VECTORDB", "unknown"),
        "chunking_strategy":      _os_module.getenv("CHUNKING_STRATEGY", "unknown"),
        "collection":             _os_module.getenv("MAI_COLLECTION", "ingested_content"),
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
async def retrieve_pipeline_config(_user=Depends(require_role("admin"))):
    """
    Return the active retrieval pipeline configuration from environment.
    Used by the Retrieve page debug panel to show what was active for each query.
    """
    import os as _os
    llm_provider = _os.getenv("MAI_LLM", "openai").lower()
    llm_key_set = {
        "openai":    bool(_os.getenv("OPENAI_API_KEY")),
        "groq":      bool(_os.getenv("GROQ_API_KEY")),
        "grok":      bool(_os.getenv("GROQ_API_KEY")),
        "anthropic": bool(_os.getenv("ANTHROPIC_API_KEY")),
        "gemini":    bool(_os.getenv("GEMINI_API_KEY")),
        "google":    bool(_os.getenv("GEMINI_API_KEY")),
        "ollama":    True,
    }
    llm_model_map = {
        "openai":    _os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
        "groq":      _os.getenv("GROQ_LLM_MODEL", "llama-3.1-70b-versatile"),
        "grok":      _os.getenv("GROQ_LLM_MODEL", "llama-3.1-70b-versatile"),
        "anthropic": _os.getenv("ANTHROPIC_LLM_MODEL", "claude-3-5-sonnet-20241022"),
        "gemini":    _os.getenv("GEMINI_LLM_MODEL", "gemini-1.5-flash"),
        "google":    _os.getenv("GEMINI_LLM_MODEL", "gemini-1.5-flash"),
        "ollama":    _os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b"),
    }
    return {
        "embedder":             _os.getenv("MAI_EMBEDDER", "unknown"),
        "embedder_model": (
            _os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
            if _os.getenv("MAI_EMBEDDER", "").lower() in ("google", "gemini")
            else _os.getenv("OPENAI_EMBED_MODEL", "")
            if _os.getenv("MAI_EMBEDDER", "").lower() == "openai"
            else _os.getenv("HF_EMBED_MODEL", "")
        ),
        "llm_provider":         llm_provider,
        "llm_model":            llm_model_map.get(llm_provider, "unknown"),
        "llm_api_key_set":      llm_key_set.get(llm_provider, False),
        "vectordb":             _os.getenv("MAI_VECTORDB", "unknown"),
        "collection":           _os.getenv("MAI_COLLECTION", "ingested_content"),
        "chunking_strategy":    _os.getenv("CHUNKING_STRATEGY", "unknown"),
        "rag_min_score":        float(_os.getenv("RAG_ANSWER_MIN_SCORE", "0.25")),
        "supported_providers":  ["openai", "ollama", "groq", "gemini", "anthropic"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _embed_in_thread(query: str) -> List[float]:
    """Run embedding in thread pool to avoid blocking the event loop."""
    import asyncio
    return await asyncio.to_thread(_embed_query, query)
