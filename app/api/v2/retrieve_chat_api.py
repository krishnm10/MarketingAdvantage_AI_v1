"""
================================================================================
Marketing Advantage AI — Chat Retrieval API
File: app/api/v2/retrieve_chat_api.py

ChatGPT-style multi-turn RAG retrieval with per-request LLM and reranker
selection.  Embedder stays fixed to the ingestion pipeline's configured
embedder so vector indices remain compatible.

Endpoints:
  POST /api/v2/retrieve/chat     → Multi-turn chat retrieval + LLM answer
  GET  /api/v2/models/llm        → Available LLM providers / models
  GET  /api/v2/models/reranker   → Available reranker plugins
================================================================================
"""

from __future__ import annotations

import logging
import os
import time
import uuid
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
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_SUPPORTED_LLM_PROVIDERS = ("openai", "ollama", "groq", "gemini", "anthropic", "google")
_MAX_HISTORY_TURNS = 20
_QUERY_REWRITE_MAX_TOKENS = 200


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' | 'assistant' | 'system'")
    content: str = Field(..., max_length=8000)


class ChatRetrieveRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Session UUID for grouping turns")
    messages: List[ChatMessage] = Field(..., min_length=1, description="Full conversation history for this session")
    # Per-request model overrides (None → use .env defaults)
    llm_provider: Optional[str] = Field(None, description="Override LLM provider")
    llm_model: Optional[str] = Field(None, description="Override LLM model name")
    reranker: Optional[str] = Field(None, description="Override reranker (none | crossencoder | bge_reranker | flashrank | cohere)")
    # Retrieval knobs
    intent: str = Field("answer", description="Retrieval intent: answer | explore | audit")
    top_k: Optional[int] = Field(None, ge=1, le=50)
    search_mode: str = Field("semantic", description="semantic | hybrid | keyword")
    similarity_threshold: float = Field(0.0, ge=0.0, le=1.0)
    enable_hyde: bool = Field(False)
    max_context_chunks: int = Field(5, ge=1, le=15)
    generate_answer: bool = Field(True)


class ChatResultItem(BaseModel):
    rank: int
    chunk_id: str
    text: str
    score: float
    trust_decision: Optional[str] = None
    trust_state: Optional[str] = None


class ChatRetrieveResponse(BaseModel):
    session_id: str
    query: str
    rewritten_query: Optional[str] = None
    intent: str
    search_mode: str
    total_results: int
    total_dropped: int
    latency_ms: float
    results: List[ChatResultItem]
    answer: Optional[str] = None
    answer_model: Optional[str] = None
    answer_latency_ms: Optional[float] = None
    answer_error: Optional[str] = None
    debug_info: Optional[Dict[str, Any]] = None


# ─────────────────────────────────────────────────────────────────────────────
# LLM Resolver (shared by rewrite + answer generation)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_llm(provider: str, model: Optional[str] = None):
    """
    Resolve an LLM connector instance for the given provider/model.
    Returns (llm_instance, resolved_model_name) or raises HTTPException.
    """
    provider = provider.lower()

    if provider == "openai":
        from app.core.llms.openai_v1 import OpenAILLM
        m = model or os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
        return OpenAILLM(), m

    if provider == "ollama":
        from app.core.llms.ollama_v1 import OllamaLLM
        m = model or os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")
        return OllamaLLM(model=m), m

    if provider in ("groq", "grok"):
        from app.core.llms.groq_v1 import GroqLLM
        m = model or os.getenv("GROQ_LLM_MODEL", "llama-3.1-70b-versatile")
        return GroqLLM(), m

    if provider in ("gemini", "google"):
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise HTTPException(400, "GEMINI_API_KEY is not set.")
        from app.core.llms.gemini_v1 import GeminiLLM
        m = model or os.getenv("GEMINI_LLM_MODEL", "gemini-1.5-flash")
        return GeminiLLM(model=m, api_key=api_key), m

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise HTTPException(400, "ANTHROPIC_API_KEY is not set.")
        from app.core.llms.anthropic_v1 import AnthropicLLM
        m = model or os.getenv("ANTHROPIC_LLM_MODEL", "claude-3-5-sonnet-20241022")
        return AnthropicLLM(model=m, api_key=api_key), m

    raise HTTPException(400, f"Unsupported LLM provider: '{provider}'. Supported: {', '.join(_SUPPORTED_LLM_PROVIDERS)}")


# ─────────────────────────────────────────────────────────────────────────────
# Query Rewriter (multi-turn → standalone question)
# ─────────────────────────────────────────────────────────────────────────────

def _rewrite_query(messages: List[ChatMessage], llm, llm_model: str) -> str:
    """
    Given a multi-turn conversation, rewrite the last user message into a
    self-contained query suitable for embedding search.
    """
    if len(messages) <= 1:
        return messages[-1].content

    history_lines = []
    for msg in messages[-_MAX_HISTORY_TURNS:]:
        tag = msg.role.upper()
        history_lines.append(f"{tag}: {msg.content}")
    history_str = "\n".join(history_lines)

    prompt = (
        "You are a search-query rewriter. Given the conversation below, "
        "rewrite the LAST user message into a single, self-contained search query "
        "that captures the full intent including any context from earlier turns. "
        "Return ONLY the rewritten query, nothing else.\n\n"
        f"CONVERSATION:\n{history_str}\n\n"
        "REWRITTEN QUERY:"
    )

    try:
        resp = llm.generate(prompt, temperature=0.0, max_tokens=_QUERY_REWRITE_MAX_TOKENS)
        rewritten = (resp.text or "").strip()
        if len(rewritten) > 5:
            return rewritten
    except Exception as e:
        logger.warning("[ChatRetrieve] Query rewrite failed, using raw query: %s", e)

    return messages[-1].content


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helper
# ─────────────────────────────────────────────────────────────────────────────

def _embed_query(query: str) -> List[float]:
    embedder = get_embedder()
    result = embedder.encode(query, normalize_embeddings=True)
    return result.tolist()


async def _embed_in_thread(query: str) -> List[float]:
    import asyncio
    return await asyncio.to_thread(_embed_query, query)


# ─────────────────────────────────────────────────────────────────────────────
# Main Chat Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatRetrieveResponse)
async def chat_retrieve(
    req: ChatRetrieveRequest,
    db: AsyncSession = Depends(get_db),
    _user=Depends(require_role("admin")),
):
    """
    Multi-turn RAG chat: rewrite query → embed → retrieve → (optional rerank)
    → generate LLM answer from grounded context.

    Embedder is always the ingestion-pipeline embedder (MAI_EMBEDDER).
    LLM and reranker can be overridden per request.
    """
    from app.retrieval.runtime import RetrievalRuntime
    from app.retrieval.repository import RetrievalRepository
    from app.retrieval.types_retrieve import QueryContext, RetrievalIntent
    from app.retrieval.policy import DEFAULT_POLICY_REGISTRY
    from app.middleware.security_middleware import scan_text as _security_scan_text

    start = time.perf_counter()

    # ── Validate intent ──────────────────────────────────────────────────────
    intent_map = {
        "answer": RetrievalIntent.ANSWER,
        "explore": RetrievalIntent.EXPLORE,
        "audit": RetrievalIntent.AUDIT,
    }
    intent_enum = intent_map.get(req.intent.lower())
    if not intent_enum:
        raise HTTPException(400, f"Invalid intent '{req.intent}'")

    # ── Resolve LLM (for rewrite + answer) ───────────────────────────────────
    llm_provider = (req.llm_provider or os.getenv("MAI_LLM", "openai")).lower()
    llm_model_name: Optional[str] = req.llm_model
    try:
        llm, llm_model_name = _resolve_llm(llm_provider, llm_model_name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Failed to initialize LLM ({llm_provider}): {e}")

    # ── Security scan on last user message ───────────────────────────────────
    last_user_msg = req.messages[-1].content
    scan_result = _security_scan_text(last_user_msg, context="chat_retrieve")
    if scan_result.injection_detected:
        raise HTTPException(400, "Query rejected: potential prompt injection detected.")

    # ── Query rewrite (multi-turn → standalone) ──────────────────────────────
    raw_query = scan_result.redacted_text
    rewritten_query: Optional[str] = None
    if len(req.messages) > 1:
        rewritten_query = _rewrite_query(req.messages, llm, llm_model_name)
        # Security scan on rewritten query too
        rw_scan = _security_scan_text(rewritten_query, context="chat_rewrite")
        if rw_scan.injection_detected:
            rewritten_query = raw_query
        else:
            rewritten_query = rw_scan.redacted_text

    embed_text = rewritten_query or raw_query

    # ── Optional HyDE ────────────────────────────────────────────────────────
    if req.enable_hyde:
        try:
            hyde_prompt = (
                "Write a short factual paragraph that would answer this question. "
                "Do not say you don't know. Just give a plausible answer in 2-3 sentences.\n\n"
                f"Question: {embed_text}\n\nAnswer:"
            )
            resp = llm.generate(hyde_prompt, temperature=0.0, max_tokens=200)
            text = (resp.text or "").strip()
            if len(text) > 20:
                embed_text = text
                logger.info("[ChatRetrieve] HyDE expansion applied (%d chars)", len(text))
        except Exception as e:
            logger.warning("[ChatRetrieve] HyDE failed: %s", e)

    # ── Embed ────────────────────────────────────────────────────────────────
    try:
        query_embedding = await _embed_in_thread(embed_text)
    except Exception as e:
        raise HTTPException(500, f"Embedding failed: {e}")

    # ── Retrieve ─────────────────────────────────────────────────────────────
    repository = RetrievalRepository(db_session=db)
    runtime = RetrievalRuntime(repository=repository, policy_registry=DEFAULT_POLICY_REGISTRY)

    ctx = QueryContext(
        query=raw_query,
        intent=intent_enum,
        requested_at=int(time.time()),
    )

    try:
        ranked_results, dropped = await runtime.retrieve(
            ctx=ctx, query_embedding=query_embedding, max_results_override=req.top_k,
        )
    except Exception as e:
        raise HTTPException(500, f"Retrieval failed: {e}")
    finally:
        repository.close()

    # ── Optional reranker ────────────────────────────────────────────────────
    reranker_name = (req.reranker or os.getenv("MAI_RERANKER", "none")).strip().lower()
    reranker_used = "none"

    if reranker_name not in ("none", "", "disabled") and ranked_results:
        try:
            import app.core.rerankers.register as _rr_reg  # noqa: F401
            from app.core.plugin_registry import reranker_registry
            from app.core.rerankers.base import RerankCandidate

            rr_model = os.getenv("MAI_RERANKER_MODEL", "").strip()
            build_kw: Dict[str, Any] = {}
            if rr_model:
                build_kw["model_name"] = rr_model

            reranker = reranker_registry.build(reranker_name, **build_kw)
            candidates = [
                RerankCandidate(id=r.chunk_id, text=r.text, vector_score=r.score, metadata={})
                for r in ranked_results
            ]
            rr_top_k = req.top_k or 10
            scored = reranker.rerank(raw_query, candidates, top_k=rr_top_k)
            score_map = {c.id: (c.rerank_score or 0.0) for c in scored}
            ranked_results = [r for r in ranked_results if r.chunk_id in score_map]
            ranked_results.sort(key=lambda r: score_map.get(r.chunk_id, 0.0), reverse=True)
            reranker_used = reranker_name
            logger.info("[ChatRetrieve] Reranker '%s' applied | %d candidates", reranker_name, len(scored))
        except Exception as e:
            logger.warning("[ChatRetrieve] Reranker '%s' failed: %s", reranker_name, e)

    # ── BM25 / Hybrid re-ranking ────────────────────────────────────────────
    search_mode = req.search_mode.lower()
    if search_mode in ("hybrid", "keyword") and ranked_results:
        try:
            from app.core.search.bm25_index import BM25Index
            from app.core.search.rrf_fusion import reciprocal_rank_fusion

            chunk_dicts = [{"id": r.chunk_id, "text": r.text, "score": r.score, "metadata": {}} for r in ranked_results]
            bm25 = BM25Index()
            bm25.build(chunk_dicts)
            kw_hits = bm25.search(raw_query, k=len(ranked_results))
            kw_dicts = [{"id": h.id, "text": h.text, "score": h.score, "metadata": h.metadata} for h in kw_hits]

            if search_mode == "hybrid":
                fused = reciprocal_rank_fusion(vector_hits=chunk_dicts, keyword_hits=kw_dicts, alpha=0.7, top_k=len(ranked_results))
                fused_order = {f.id: i for i, f in enumerate(fused)}
                ranked_results.sort(key=lambda r: fused_order.get(r.chunk_id, 999))
            else:
                bm25_order = {h.id: i for i, h in enumerate(kw_hits)}
                ranked_results.sort(key=lambda r: bm25_order.get(r.chunk_id, 999))
        except Exception as e:
            logger.warning("[ChatRetrieve] Hybrid/keyword search failed: %s", e)

    # ── Threshold filter ─────────────────────────────────────────────────────
    if req.similarity_threshold > 0:
        ranked_results = [r for r in ranked_results if r.score >= req.similarity_threshold]

    # ── Build results ────────────────────────────────────────────────────────
    results: List[ChatResultItem] = []
    for idx, r in enumerate(ranked_results, start=1):
        trust_state = None
        sig = r.explanation.get("interpretation", {})
        trust_state = sig.get("trust_state", "validated")

        results.append(ChatResultItem(
            rank=idx,
            chunk_id=r.chunk_id,
            text=r.text,
            score=round(r.score, 6),
            trust_decision=r.trust_decision.value if hasattr(r.trust_decision, "value") else (r.trust_decision if isinstance(r.trust_decision, str) else None),
            trust_state=trust_state,
        ))

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    # ── LLM answer generation ────────────────────────────────────────────────
    answer: Optional[str] = None
    answer_model: Optional[str] = llm_model_name
    answer_latency_ms: Optional[float] = None
    answer_error: Optional[str] = None

    rag_min_score = float(os.getenv("RAG_ANSWER_MIN_SCORE", "0.25"))

    if req.generate_answer:
        if not results:
            answer_error = "No results retrieved — cannot generate a grounded answer."
        else:
            max_score = max(r.score for r in results)
            if max_score < rag_min_score:
                answer_error = (
                    f"Retrieved context confidence too low (best={max_score:.3f} < threshold={rag_min_score:.2f}). "
                    "Try a more specific query."
                )
            else:
                try:
                    gen_start = time.perf_counter()
                    top_chunks = results[:req.max_context_chunks]
                    context_parts = [
                        f"[Source {i}] (score={c.score:.3f})\n{c.text.strip()}"
                        for i, c in enumerate(top_chunks, 1)
                    ]
                    context_str = "\n\n---\n\n".join(context_parts)

                    # ── Context sanitization: PII redaction on retrieved text ─
                    _ctx_scan = _security_scan_text(
                        context_str, context="retrieved_context",
                    )
                    _sanitized_context = _ctx_scan.redacted_text
                    if _sanitized_context != context_str:
                        logger.info(
                            "[ChatRetrieve] Context sanitized before LLM — "
                            "PII redacted from retrieved chunks"
                        )
                        context_str = _sanitized_context

                    rag_prompt = (
                        "You are a precise, grounded enterprise assistant.\n"
                        "Answer the user's question using ONLY the retrieved passages below.\n\n"
                        "Rules:\n"
                        "  1. Cite every source: [Source 1], [Source 2], etc.\n"
                        "  2. If the passages lack sufficient information, respond EXACTLY with: "
                        "'I could not find a reliable answer in the available documents.'\n"
                        "  3. Never invent facts.\n"
                        "  4. Be factual, concise, and professional.\n\n"
                        f"QUESTION: {raw_query}\n\n"
                        f"RETRIEVED PASSAGES:\n{context_str}\n\n"
                        "ANSWER (grounded, with citations):"
                    )

                    resp = llm.generate(rag_prompt, temperature=0.0, max_tokens=700)
                    raw_answer = (resp.text or "").strip()
                    if raw_answer:
                        answer = raw_answer
                    else:
                        answer_error = "LLM returned an empty response."

                    answer_latency_ms = round((time.perf_counter() - gen_start) * 1000, 2)
                except Exception as e:
                    logger.warning("[ChatRetrieve] LLM answer failed: %s", e, exc_info=True)
                    answer_error = f"LLM generation error: {str(e)[:300]}"

    # ── Debug info ───────────────────────────────────────────────────────────
    debug_info: Dict[str, Any] = {
        "embedder": os.getenv("MAI_EMBEDDER", "unknown"),
        "llm_provider": llm_provider,
        "llm_model": llm_model_name,
        "vectordb": os.getenv("MAI_VECTORDB", "unknown"),
        "collection": os.getenv("MAI_COLLECTION", "ingested_content"),
        "chunking_strategy": os.getenv("CHUNKING_STRATEGY", "unknown"),
        "search_mode": search_mode,
        "intent": req.intent,
        "reranker_used": reranker_used,
        "score_gate_threshold": rag_min_score,
        "max_score": round(max(r.score for r in results), 4) if results else None,
        "hyde_enabled": req.enable_hyde,
        "query_rewritten": rewritten_query is not None,
        "security_scan": {
            "pii_detected": scan_result.has_pii,
            "injection_detected": scan_result.injection_detected,
        },
    }

    logger.info(
        "[ChatRetrieve] session=%s query='%s' rewritten=%s results=%d latency=%.0fms",
        req.session_id, raw_query[:60], rewritten_query is not None, len(results), elapsed_ms,
    )

    return ChatRetrieveResponse(
        session_id=req.session_id,
        query=raw_query,
        rewritten_query=rewritten_query,
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
