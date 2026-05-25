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

import json
import logging
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.observability.rag_chat_trace import (
    chunk_text_digest,
    llm_usage_payload,
    maybe_start_rag_chat_trace,
    rag_chat_trace_include_prompt_hash,
    text_digest_utf8,
    vector_l2_norm,
)
from app.db.session_v2 import get_db
from app.auth.guards import require_role
from app.services.ingestion.ingestion_service_v2 import get_embedder
from app.services.query_routing import build_direct_response, get_orchestrator
from app.services.query_routing.types import QueryRoute, RouteDecision
from app.retrieval.types_retrieve import RankedResult
from app.core.prompts.library_loader import strip_and_verify
from app.services.faithfulness_verifier import verify_or_refuse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/retrieve")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_SUPPORTED_LLM_PROVIDERS = ("openai", "ollama", "groq", "gemini", "anthropic", "google")
_MAX_HISTORY_TURNS = 20
_QUERY_REWRITE_MAX_TOKENS = 200

# Must match wording in rag_prompt Rule 3 (used to detect contradictory model hedges).
_GROUNDING_REFUSAL_PHRASE = (
    "I could not find a reliable answer in the available documents."
)


def _chat_rag_compliance_block() -> str:
    """
    Default enterprise RAG instructions and citation rules (unchanged when no Prompt Library prefix).
    Tenant `system_instructions` from the library are prepended before this block when configured.
    """
    return (
        "You are a precise, grounded enterprise assistant.\n"
        "Answer using ONLY the retrieved passages below.\n\n"
        "Rules:\n"
        "  1. Cite every factual claim using [Source 1], [Source 2], etc.\n"
        "  2. If the passages clearly contain facts that answer the user message "
        "(including partial lists — e.g. some order IDs when the user asks for orders), "
        "summarize those facts with citations. Do not refuse when relevant text exists.\n"
        "  3. If NONE of the passages are relevant or they contain zero usable facts "
        "for the user message, respond EXACTLY with ONE line ONLY: "
        f"'{_GROUNDING_REFUSAL_PHRASE}'\n"
        "  4. NEVER output both extracted facts/table AND the refusal from rule 3. "
        "Choose one: grounded answer OR refusal — never both in the same response.\n"
        "  5. When the user asks for multiple identifiers (order numbers, invoice numbers, IDs), "
        "quote them exactly as shown in passages (preserve INV-..., ABC-NNNN patterns).\n"
        "  6. Never invent facts not present in the passages.\n"
        "  7. When listing many identifiers, use a Markdown bullet list.\n"
        "  8. Deduplicate identifiers: include each unique order/invoice/token string at most ONCE "
        "(the same ID may appear in multiple [Source] blocks — merge repeats; "
        'do NOT add labels like "(again)". For one merged line cite every source that '
        "contained it, e.g. `MME-0099 — [Source 1], [Source 3]`).\n"
        "  9. Do not use outside knowledge, public anecdotes, companies, dates, products, "
        "or scenarios that are NOT literally supported by the passages. "
        "If the passages do not name it, omit it entirely.\n\n"
    )


# #region agent log
def _agent_debug_ndjson(
    *,
    hypothesis_id: str,
    location: str,
    message: str,
    data: Optional[Dict[str, Any]] = None,
    run_id: str = "retrieve_chat",
) -> None:
    try:
        payload: Dict[str, Any] = {
            "sessionId": "e855ab",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        with open("debug-e855ab.log", "a", encoding="utf-8") as _f:
            _f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


# #endregion


def _chat_llm_transport_exc(exc: BaseException) -> bool:
    """True when the sync LLM HTTP client hit timeout or connection-layer failure."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    try:
        import httpx

        return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))
    except ImportError:
        return False


def _normalize_line_for_refusal_compare(line: str) -> str:
    return (
        line.strip()
        .strip('"')
        .strip("'")
        .rstrip(".")
        .lower()
    )


def _strip_contradictory_refusal(answer: str) -> tuple[str, bool]:
    """
    Small local LLMs sometimes emit the exact refusal sentence after a good partial answer.
    When we see grounded signals (citations / invoice-style ids), drop refusal-only lines.
    """
    refusal = _GROUNDING_REFUSAL_PHRASE
    if not answer or refusal not in answer:
        return answer, False
    lo = answer.lower()
    hyphen_id = bool(re.search(r"[a-z]{2,}-[a-z0-9]{3,}", lo))
    grounded = (
        "[source " in lo
        or "inv-" in lo
        or "invoice number" in lo
        or "order number" in lo
        or hyphen_id
    )
    if not grounded:
        return answer, False
    lines_out: List[str] = []
    for line in answer.splitlines():
        norm = _normalize_line_for_refusal_compare(line)
        if norm == refusal.lower():
            continue
        if refusal.lower() in line.lower() and len(line.strip()) <= len(refusal) + 8:
            continue
        lines_out.append(line)
    cleaned = "\n".join(lines_out).strip()
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    if not cleaned:
        return answer, False
    return cleaned, cleaned != answer


# ── P2/P3 production helpers — structured identifiers & detail-query grounding ──

# Invoice / order style tokens (INV-6640, MME-0099, SRG-1144, PCR-7723, RG-2024, …)
_STRUCTURED_ID_RE = re.compile(r"\b[A-Z]{2,}-[A-Z0-9]{3,}\b")
_DETAIL_QUERY_HINT_RE = re.compile(
    r"(?is)\b(?:give|show)\s+me\s+(?:the\s+)?details?\s+(?:of|for|about)\b|\bdetails?\s+(?:of|for|about)\b|\bwhat\s+(?:are\s+)?the\s+details?\s+(?:of|for)\b"
)
# "Order Number 67890", "invoice # 12345", short "order 67890"
_ORDER_NUMBER_DIGITS_RE = re.compile(
    r"(?i)\b(?:order|invoice)\s*(?:number|no\.?|#)?\s*[:#\s]*(\d{4,})\b|\border\s+#?\s*(\d{4,})\b"
)


def _all_structured_ids_upper(blob: str) -> List[str]:
    """All ABC-XYZ style tokens (deduped, stable order)."""
    if not blob:
        return []
    out: List[str] = []
    seen: set[str] = set()
    for m in _STRUCTURED_ID_RE.findall(blob.upper()):
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _extract_numeric_order_focus_token(*query_parts: str) -> Optional[str]:
    """
    Explicit numeric order / invoice refs (e.g. 67890) when user pins a detail/order query.
    """
    blob = " ".join(p for p in query_parts if p).strip()
    if not blob:
        return None
    m = _ORDER_NUMBER_DIGITS_RE.search(blob)
    if m:
        return (m.group(1) or m.group(2) or "").strip() or None
    low = blob.lower()
    if _DETAIL_QUERY_HINT_RE.search(low):
        long_nums = re.findall(r"\b(\d{5,})\b", blob)
        if len(long_nums) == 1:
            return long_nums[0]
    return None


def _strict_identifier_grounding_query(raw: str, rewritten: str) -> bool:
    blob = f"{raw} {rewritten}".strip()
    if not blob:
        return False
    if _DETAIL_QUERY_HINT_RE.search(blob.lower()):
        return True
    if _ORDER_NUMBER_DIGITS_RE.search(blob):
        return True
    return False


def _strict_detail_grounding_keys(raw: str, rewritten: str) -> List[str]:
    """
    Tokens the user explicitly asked about in a detail/order query; all must appear in passages
    or we refuse (prevents small-LLM hallucinations when retrieval misses).
    """
    if not _strict_identifier_grounding_query(raw, rewritten):
        return []
    blob = f"{raw} {rewritten}".strip()
    keys: List[str] = []
    seen: set[str] = set()
    for sid in _all_structured_ids_upper(blob):
        if sid not in seen:
            keys.append(sid)
            seen.add(sid)
    num = _extract_numeric_order_focus_token(raw, rewritten)
    if num and num not in seen:
        keys.append(num)
        seen.add(num)
    return keys


def _token_appears_in_passages(passages: str, token: str) -> bool:
    if not token or not passages:
        return False
    if token.isdigit():
        return re.search(rf"(?<!\d){re.escape(token)}(?!\d)", passages) is not None
    return token.upper() in passages.upper()


def _extract_detail_focus_token(*query_parts: str) -> Optional[str]:
    """Pick a canonical focus token: structured ID (MME-0099) or explicit numeric order ref."""
    blob = " ".join(p for p in query_parts if p).strip()
    if not blob:
        return None
    matches = _STRUCTURED_ID_RE.findall(blob.upper())
    best: Optional[str] = None
    if matches:
        uniq = sorted(set(matches), key=len, reverse=True)
        low_join = blob.lower()
        best_pri = -1
        for m in uniq:
            pri = len(m)
            if _DETAIL_QUERY_HINT_RE.search(low_join) and m.lower() in low_join:
                pri += 50
            pri += low_join.count(m.lower())
            if pri > best_pri:
                best_pri = pri
                best = m
    if best:
        return best
    return _extract_numeric_order_focus_token(*query_parts)


def _prioritize_ranked_for_identifier(
    ranked: List[RankedResult],
    token: Optional[str],
) -> List[RankedResult]:
    """Boost chunks whose text mentions the identifier (P3 detail queries)."""
    if not ranked or not token:
        return ranked
    needle = token.strip().upper()
    if len(needle) < 4:
        return ranked
    if not any(needle in r.text.upper() for r in ranked):
        return ranked
    return sorted(ranked, key=lambda r: (needle not in r.text.upper(), -float(r.score)))


_AMOUNT_CONFLICT_RE = re.compile(r"\$[\d,]+(?:\.\d{2})?")


def _normalize_focus_id(focus_id: Optional[str]) -> str:
    if not focus_id:
        return ""
    return focus_id.replace("-", "").replace(" ", "").lower()


def _chunk_mentions_focus(chunk: RankedResult, focus_id: Optional[str]) -> bool:
    if not focus_id:
        return False
    blob = _normalize_focus_id(chunk.text or "")
    needle = _normalize_focus_id(focus_id)
    return bool(needle and needle in blob)


def _filter_to_dominant_file(
    chunks: List[RankedResult],
    focus_id: Optional[str],
    max_chunks: int = 2,
) -> List[RankedResult]:
    """
    STRUCTURED route only: keep chunks from the dominant file_id that mentions focus_id.
    Safe fallbacks when file_id or focus matches are missing.
    """
    if not chunks:
        return chunks

    if focus_id:
        matching = [c for c in chunks if _chunk_mentions_focus(c, focus_id)]
    else:
        matching = list(chunks)

    if not matching:
        logger.warning(
            "[ChatRetrieve] No chunk mentions focus_id=%r; falling back to top-1",
            focus_id,
        )
        return chunks[:1]

    file_counts = Counter(c.file_id for c in matching if c.file_id)
    if not file_counts:
        logger.warning(
            "[ChatRetrieve] file_id missing on matching chunks; falling back to top-%d",
            max_chunks,
        )
        return chunks[:max_chunks]

    dominant_file_id = file_counts.most_common(1)[0][0]
    filtered = [c for c in chunks if c.file_id == dominant_file_id]
    return filtered[:max_chunks]


def _detect_amount_conflicts(chunks: List[RankedResult]) -> bool:
    """True when two+ chunks contain disjoint dollar amount sets (STRUCTURED pre-LLM)."""
    amount_sets: List[set] = []
    for chunk in chunks:
        found = set(_AMOUNT_CONFLICT_RE.findall(chunk.text or ""))
        if found:
            amount_sets.append(found)
    if len(amount_sets) < 2:
        return False
    for i in range(len(amount_sets)):
        for j in range(i + 1, len(amount_sets)):
            left, right = amount_sets[i], amount_sets[j]
            if left and right and left.isdisjoint(right):
                return True
    return False


def _dedupe_identifier_lines(answer: str) -> Tuple[str, bool]:
    """
    Collapse duplicate list lines keyed by structured IDs (P2).
    Removes trivial '(again)' hedges from small LLMs.
    """
    if not answer:
        return answer, False
    lines = answer.splitlines()
    seen_keys: set[Tuple[str, ...]] = set()
    changed = False
    out: List[str] = []
    for line in lines:
        stripped = re.sub(r"\s*\(?again\)?\.?$", "", line, flags=re.IGNORECASE).strip()
        ids = tuple(sorted(set(_STRUCTURED_ID_RE.findall(stripped.upper()))))
        if ids:
            if ids in seen_keys:
                changed = True
                continue
            seen_keys.add(ids)
            cleaned = re.sub(r"\(?again\)?", "", stripped, flags=re.IGNORECASE).strip()
            if cleaned != line.strip():
                changed = True
            out.append(cleaned)
        else:
            out.append(line)
    merged = "\n".join(out).strip()
    return merged, changed or merged != answer.strip()


def _build_focus_fallback_answer(context_str: str, token: str) -> Optional[str]:
    """Deterministic excerpts when the LLM refuses but passages contain the token (P3)."""
    if not token or not context_str:
        return None
    if token.upper() not in context_str.upper():
        return None
    blocks = re.split(r"\n-{3,}\n", context_str)
    kept: List[str] = []
    for b in blocks:
        chunk = b.strip()
        if token.upper() not in chunk.upper():
            continue
        if len(chunk) > 1400:
            chunk = chunk[:1400].rsplit(" ", 1)[0] + " …"
        kept.append(chunk)
    if not kept:
        return None
    header = (
        f"Here is what the retrieved passages say about **{token}** "
        "(verbatim excerpts; cite sources by section header):\n\n"
    )
    return header + "\n\n".join(kept)


def _maybe_focus_fallback_answer(
    answer: Optional[str],
    context_str: str,
    token: Optional[str],
) -> Tuple[Optional[str], bool]:
    """Replace bare refusal with deterministic excerpts when grounded text exists."""
    if not token:
        return answer, False
    fb = _build_focus_fallback_answer(context_str, token)
    if not fb:
        return answer, False
    if not answer or not answer.strip():
        return fb, True
    low = answer.strip().lower()
    refusal = _GROUNDING_REFUSAL_PHRASE.lower()
    if refusal == low:
        return fb, True
    if refusal in low and "[source" not in low:
        return fb, True
    return answer, False


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' | 'assistant' | 'system'")
    content: str = Field(..., max_length=8000)


class ChatRetrieveRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Session UUID for grouping turns")
    messages: List[ChatMessage] = Field(..., min_length=1, description="Full conversation history for this session")
    client_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="REQUIRED: Tenant / client identifier for tenant-scoped retrieval.",
    )
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
    rewrite_enabled: Optional[bool] = Field(
        None,
        description="Override tenant rewrite toggle for this request. None uses route + tenant config.",
    )
    system_prompt_override: Optional[str] = Field(
        None,
        description="Session-only system instruction text override (not persisted).",
    )


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


@dataclass
class _ChatSessionCtx:
    chat_history: List[ChatMessage]


def _build_direct_chat_response(
    *,
    req: ChatRetrieveRequest,
    raw_query: str,
    scan_result: Any,
    route_decision: RouteDecision,
    direct_answer: str,
    retrieve_cid: str,
    rc: Any,
    trace: Any,
    start: float,
    search_mode: str,
) -> ChatRetrieveResponse:
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    _, _direct_verification = verify_or_refuse(
        answer=direct_answer or "",
        source_texts=[],
        focus_id=None,
        route=route_decision.route.value,
    )
    debug_info: Dict[str, Any] = {
        "client_id": retrieve_cid,
        "embedder": rc.embedder_type,
        "vectordb": rc.vectordb_type,
        "collection": rc.collection,
        "search_mode": search_mode,
        "intent": req.intent,
        "route_decision": route_decision.to_trace_dict(),
        "retrieval_skipped": True,
        "query_route": route_decision.route.value,
        "route_layer": route_decision.layer_used.value,
        "estimated_tokens_saved": route_decision.estimated_tokens_saved,
        "rag_tracing": trace is not None,
        "rag_trace_id": trace.trace_id if trace else None,
        "security_scan": {
            "pii_detected": scan_result.has_pii,
            "injection_detected": scan_result.injection_detected,
        },
        "faithfulness_verifier": {
            "status": _direct_verification.status.value,
            "checks": [
                {
                    "id": c.check_id,
                    "status": c.status.value,
                    "detail": c.detail,
                }
                for c in _direct_verification.checks
            ],
            "fail_reason": _direct_verification.fail_reason,
        },
    }
    logger.info(
        "[ChatRetrieve] L0 direct response route=%s session=%s latency=%.0fms",
        route_decision.route.value,
        req.session_id,
        elapsed_ms,
    )
    return ChatRetrieveResponse(
        session_id=req.session_id,
        query=raw_query,
        rewritten_query=None,
        intent=req.intent,
        search_mode=search_mode,
        total_results=0,
        total_dropped=0,
        latency_ms=elapsed_ms,
        results=[],
        answer=direct_answer,
        answer_model=None,
        answer_latency_ms=0.0,
        answer_error=None,
        debug_info=debug_info,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM Resolver (shared by rewrite + answer generation)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_llm(
    provider: str,
    model: Optional[str],
    *,
    api_key_env: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """
    Resolve an LLM connector instance for the given provider/model.
    Returns (llm_instance, resolved_model_name) or raises HTTPException.
    """
    from app.retrieval.components import instantiate_llm

    provider = provider.lower()
    if provider == "google":
        provider = "gemini"
    if provider == "grok":
        provider = "groq"

    try:
        eff_model = model or ""
        llm, resolved = instantiate_llm(
            provider,
            eff_model,
            api_key_env=api_key_env,
            base_url=base_url,
        )
        return llm, resolved
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


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

def _embed_query(query: str, business_id: Optional[str] = None) -> List[float]:
    embedder = get_embedder(business_id)
    result = embedder.encode(query, normalize_embeddings=True)
    return result.tolist()


async def _embed_in_thread(query: str, business_id: Optional[str] = None) -> List[float]:
    import asyncio
    return await asyncio.to_thread(_embed_query, query, business_id)


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

    Embedder and defaults follow merged Client JSON for the resolved tenant.
    LLM and reranker can be overridden per request.
    """
    from app.utils.tenant_validator import (
        validate_tenant_id_strict,
        get_storage_uuid_str,
        TenantValidationError,
    )
    from app.retrieval.components import (
        log_runtime_telemetry,
        resolve_config_or_fail,
        resolve_runtime_components,
        resolve_runtime_components_legacy,
    )
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

    # Validate tenant with strict enforcement (uses TENANT_ENFORCEMENT_MODE)
    try:
        tenant_ctx = validate_tenant_id_strict(
            req.client_id,
            source="body",
            endpoint="chat_retrieve",
        )
    except TenantValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    retrieve_cid = tenant_ctx.tenant_id
    # Storage UUID for vector/SQL filtering (Phase 2 will use this)
    _storage_uuid_str = get_storage_uuid_str(tenant_ctx)

    trace = maybe_start_rag_chat_trace(req.session_id, retrieve_cid)
    _trace_prev = time.perf_counter()

    def _trace_lap() -> float:
        nonlocal _trace_prev
        now = time.perf_counter()
        delta_ms = round((now - _trace_prev) * 1000, 3)
        _trace_prev = now
        return delta_ms

    try:
        _trace_prev = time.perf_counter()

        _cfg_chat, runtime_mode_chat = resolve_config_or_fail(retrieve_cid)
        if _cfg_chat is not None:
            rc = resolve_runtime_components(_cfg_chat)
        else:
            rc = resolve_runtime_components_legacy()

        _active_template_id: Optional[str] = None
        if _cfg_chat is not None and _cfg_chat.retrieval:
            _active_template_id = _cfg_chat.retrieval.prompt_template_id
        effective_max_context_chunks = req.max_context_chunks
        matched_pattern: Optional[str] = None

        try:
            log_runtime_telemetry(rc, "/api/v2/retrieve/chat")
        except Exception:
            logger.debug("[ChatRetrieve] log_runtime_telemetry failed", exc_info=True)

        # ── Security scan on last user message ───────────────────────────────────
        last_user_msg = req.messages[-1].content
        scan_result = _security_scan_text(last_user_msg, context="chat_retrieve")
        if scan_result.injection_detected:
            raise HTTPException(400, "Query rejected: potential prompt injection detected.")

        raw_query = scan_result.redacted_text
        search_mode = req.search_mode.lower()
        effective_top_k = req.top_k if req.top_k is not None else rc.top_k_final

        # ── L0 Query Router (before embed / retrieve / LLM) ─────────────────────
        session_ctx = _ChatSessionCtx(chat_history=req.messages)
        tenant_config = _cfg_chat
        _route_decision = await get_orchestrator().route(
            raw_query=raw_query,
            top_k=effective_top_k,
            session_ctx=session_ctx,
            tenant_config=tenant_config,
        )
        if trace:
            trace.route_decision = _route_decision.to_trace_dict()
            trace.retrieval_skipped = not _route_decision.retrieval_allowed
            trace.route_latency_ms = _route_decision.route_latency_ms
            trace.add_event(
                "L0",
                "query_route",
                _trace_lap(),
                _route_decision.to_trace_dict(),
            )

        if not _route_decision.retrieval_allowed:
            _dr = build_direct_response(_route_decision, raw_query, tenant_config)
            return _build_direct_chat_response(
                req=req,
                raw_query=raw_query,
                scan_result=scan_result,
                route_decision=_route_decision,
                direct_answer=_dr.answer,
                retrieve_cid=retrieve_cid,
                rc=rc,
                trace=trace,
                start=start,
                search_mode=search_mode,
            )

        matched_pattern = _route_decision.matched_pattern
        if _route_decision.route == QueryRoute.STRUCTURED:
            effective_max_context_chunks = 2

        _recall_limit = _route_decision.max_recall_candidates
        _rewrite_enabled = _route_decision.rewrite_allowed
        if req.rewrite_enabled is False:
            _rewrite_enabled = False
        elif req.rewrite_enabled is True:
            _rewrite_enabled = _route_decision.rewrite_allowed
        _hyde_enabled = _route_decision.hyde_allowed
        _rerank_enabled = _route_decision.rerank_allowed

        cfg_llm = (rc.llm_provider or "openai").lower()
        if cfg_llm == "google":
            cfg_llm = "gemini"

        # ── Resolve LLM (for rewrite + answer) ───────────────────────────────────
        llm_provider = (req.llm_provider or cfg_llm).lower()
        llm_model_name: Optional[str] = req.llm_model or rc.llm_model
        try:
            llm, llm_model_name = _resolve_llm(
                llm_provider,
                llm_model_name,
                api_key_env=rc.llm_api_key_env,
                base_url=rc.llm_base_url,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"Failed to initialize LLM ({llm_provider}): {e}")

        # ── Query rewrite (multi-turn → standalone) ──────────────────────────────
        rewritten_query: Optional[str] = None
        if _rewrite_enabled and len(req.messages) > 1:
            rewritten_query = _rewrite_query(req.messages, llm, llm_model_name)
            # Security scan on rewritten query too
            rw_scan = _security_scan_text(rewritten_query, context="chat_rewrite")
            if rw_scan.injection_detected:
                rewritten_query = raw_query
            else:
                rewritten_query = rw_scan.redacted_text
    
        embed_text = rewritten_query or raw_query
        if _route_decision.route == QueryRoute.STRUCTURED:
            if matched_pattern:
                embed_text = matched_pattern
            else:
                logger.warning(
                    "[ChatRetrieve] STRUCTURED route without matched_pattern; "
                    "embed_text falls back to raw_query",
                )
                embed_text = raw_query

        detail_focus_token = _extract_detail_focus_token(raw_query, rewritten_query or "")
        strict_detail_grounding_keys = _strict_detail_grounding_keys(raw_query, rewritten_query or "")
        if (
            not strict_detail_grounding_keys
            and _route_decision.route == QueryRoute.STRUCTURED
            and matched_pattern
        ):
            strict_detail_grounding_keys = [matched_pattern]
        hyde_skipped_for_identifier = False
    
        # ── Optional HyDE ────────────────────────────────────────────────────────
        if _hyde_enabled and req.enable_hyde:
            if detail_focus_token:
                hyde_skipped_for_identifier = True
                logger.info(
                    "[ChatRetrieve] HyDE skipped — explicit identifier / order pin in query (retrieval-aligned embed)",
                )
            else:
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
    
        if trace:
            trace.add_event(
                "L1",
                "preprocess",
                _trace_lap(),
                {
                    "hash_raw_query": text_digest_utf8(raw_query),
                    "hash_embed_text": text_digest_utf8(embed_text),
                    "raw_query_chars": len(raw_query),
                    "embed_text_chars": len(embed_text),
                    "query_rewritten": rewritten_query is not None,
                    "hyde_requested": bool(req.enable_hyde),
                    "hyde_skipped_for_identifier": hyde_skipped_for_identifier,
                    "detail_focus_token": detail_focus_token,
                    "strict_detail_grounding_keys": strict_detail_grounding_keys,
                    "embed_focus_differs_from_raw": embed_text.strip() != raw_query.strip(),
                    "prompt_locked": True,
                    "prompt_template_id": _active_template_id,
                    "prompt_source": "tenant_config",
                    "matched_pattern": matched_pattern,
                    "effective_max_context_chunks": effective_max_context_chunks,
                },
            )
    
        # ── Embed ────────────────────────────────────────────────────────────────
        try:
            query_embedding = await _embed_in_thread(embed_text, retrieve_cid)
        except Exception as e:
            raise HTTPException(500, f"Embedding failed: {e}")
    
        if trace:
            _vn = vector_l2_norm(query_embedding)
            trace.add_event(
                "L1",
                "embed_query",
                _trace_lap(),
                {
                    "embedding_model_id": f"{rc.embedder_type}:{rc.embedder_model}",
                    "vector_dimension": len(query_embedding),
                    "vector_l2_norm": round(float(_vn), 8) if _vn == _vn else None,
                },
            )
    
        # ── Retrieve ─────────────────────────────────────────────────────────────
        if runtime_mode_chat == "authoritative_config" and _cfg_chat is not None:
            from app.services.ingestion.ingestion_service_v2 import get_query_pipeline_for_client
    
            _pipe_c = get_query_pipeline_for_client(retrieve_cid)
            repository = RetrievalRepository(
                db_session=db,
                vectordb=_pipe_c.vectordb,
                collection=rc.collection,
            )
        else:
            repository = RetrievalRepository(db_session=db)
        runtime = RetrievalRuntime(repository=repository, policy_registry=DEFAULT_POLICY_REGISTRY)
    
        ctx = QueryContext(
            query=raw_query,
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
                recall_limit_override=_recall_limit,
            )
        except Exception as e:
            raise HTTPException(500, f"Retrieval failed: {e}")
        finally:
            repository.close()
    
        # ── Optional reranker (stack-aware resolver + circuit breaker) ─────────
        _default_rr = (rc.reranker_name or "none").strip().lower()
        reranker_name = (req.reranker or _default_rr).strip().lower()
        reranker_used = "none"
        reranker_fallback_applied = False
        reranker_fallback_reason: Optional[str] = None

        if (
            _rerank_enabled
            and reranker_name not in ("none", "", "disabled")
            and ranked_results
            and _cfg_chat is not None
        ):
            from app.core.rerankers.base import RerankCandidate
            from app.retrieval.reranker_runtime import apply_reranker_with_fallback

            rr_candidates = [
                RerankCandidate(
                    id=r.chunk_id,
                    text=r.text,
                    vector_score=r.score,
                    metadata={},
                )
                for r in ranked_results
            ]
            rr_top_k = req.top_k or 10
            scored, reranker_used, reranker_fallback_applied, reranker_fallback_reason = (
                apply_reranker_with_fallback(
                    config=_cfg_chat,
                    query=raw_query,
                    candidates=rr_candidates,
                    top_k=rr_top_k,
                    request_plugin_override=req.reranker,
                )
            )
            if reranker_used != "none":
                score_map = {c.id: (c.rerank_score or 0.0) for c in scored}
                ranked_results = [r for r in ranked_results if r.chunk_id in score_map]
                ranked_results.sort(
                    key=lambda r: score_map.get(r.chunk_id, 0.0), reverse=True
                )
                logger.info(
                    "[ChatRetrieve] Reranker '%s' applied | %d candidates",
                    reranker_used,
                    len(scored),
                )
    
        # ── BM25 / Hybrid re-ranking ────────────────────────────────────────────
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
    
        # ── P3: surface chunks that mention the explicit identifier first ─────────
        if detail_focus_token:
            ranked_results = _prioritize_ranked_for_identifier(ranked_results, detail_focus_token)
            logger.info(
                "[ChatRetrieve] Identifier-focused reorder applied | token=%s",
                detail_focus_token,
            )

        if _route_decision.route == QueryRoute.STRUCTURED:
            ranked_results = _filter_to_dominant_file(
                ranked_results,
                focus_id=matched_pattern or detail_focus_token,
                max_chunks=effective_max_context_chunks,
            )
            logger.info(
                "[ChatRetrieve] STRUCTURED dominant-file filter | focus=%s | chunks=%d",
                matched_pattern or detail_focus_token,
                len(ranked_results),
            )
    
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
    
        if trace:
            trace.add_event(
                "L2",
                "retrieve_pipeline",
                _trace_lap(),
                {
                    "ranked_count": len(ranked_results),
                    "dropped_count": len(dropped),
                    "search_mode": search_mode,
                    "reranker_used": reranker_used,
                    "reranker_fallback_applied": reranker_fallback_applied,
                    "reranker_fallback_reason": reranker_fallback_reason,
                    "similarity_threshold": req.similarity_threshold,
                    "request_top_k": req.top_k,
                    "results_count": len(results),
                    "detail_focus_token": detail_focus_token,
                    "structured_file_filter_applied": (
                        _route_decision.route == QueryRoute.STRUCTURED
                    ),
                    "chunk_file_ids": [
                        r.file_id for r in ranked_results[:effective_max_context_chunks]
                    ],
                },
            )
    
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    
        # ── LLM answer generation ────────────────────────────────────────────────
        answer: Optional[str] = None
        answer_model: Optional[str] = llm_model_name
        answer_latency_ms: Optional[float] = None
        answer_error: Optional[str] = None
        transport_exc_name: Optional[str] = None
        prompt_template_id_effective: Optional[str] = None
        prompt_template_resolved: bool = False
        prompt_template_source: str = "default"
    
        rag_min_score = float(rc.rag_min_score)
    
        identifier_dedupe_applied = False
        focus_fallback_used = False
        context_chunks_sent_to_llm = 0
        skipped_llm_grounding_miss = False
        tokens_missing_from_passages: List[str] = []
        _verification = None
    
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

                        tenant_prompt_prefix = ""
                        _structured_system_instructions = ""
                        if _cfg_chat is not None:
                            from app.core.prompts.ssot import resolve_prompt_ssot

                            _ps = resolve_prompt_ssot(_cfg_chat)
                            prompt_template_id_effective = (
                                _active_template_id or _ps.effective_template_id
                            )
                            if _ps.instructions:
                                _structured_system_instructions = _ps.instructions
                                instruction_text = _ps.instructions
                                if req.system_prompt_override:
                                    instruction_text = req.system_prompt_override
                                tenant_prompt_prefix = (
                                    strip_and_verify(instruction_text) + "\n\n---\n\n"
                                )
                                prompt_template_resolved = _ps.library_found or _ps.source in (
                                    "legacy_inline",
                                    "preset_mapped",
                                    "default_builtin",
                                )
                                prompt_template_source = (
                                    "config"
                                    if _ps.source in ("library", "preset_mapped")
                                    else _ps.source
                                )
                            elif req.system_prompt_override:
                                tenant_prompt_prefix = (
                                    strip_and_verify(req.system_prompt_override)
                                    + "\n\n---\n\n"
                                )
                                prompt_template_source = "session_override_text"
                            elif prompt_template_id_effective:
                                logger.warning(
                                    "[ChatRetrieve] prompt SSOT id=%r not resolved; "
                                    "using default compliance block only.",
                                    prompt_template_id_effective,
                                )

                        top_chunks = results[:effective_max_context_chunks]
                        context_chunks_sent_to_llm = len(top_chunks)
                        llm_ranked = ranked_results[:effective_max_context_chunks]
                        context_parts = [
                            f"[Source {i}] (score={c.score:.3f})\n{c.text.strip()}"
                            for i, c in enumerate(top_chunks, 1)
                        ]
                        context_str = "\n\n---\n\n".join(context_parts)

                        if (
                            _route_decision.route == QueryRoute.STRUCTURED
                            and _detect_amount_conflicts(llm_ranked)
                        ):
                            context_str = (
                                "WARNING: Retrieved passages contain potentially conflicting "
                                "amounts. Only report values that appear in the passage "
                                "explicitly cited. Do not blend values across passages.\n\n"
                            ) + context_str
    
                        # ── Context sanitization: PII redaction on retrieved text ─
                        _ctx_scan = _security_scan_text(
                            context_str, context="retrieved_context",
                        )
                        _sanitized_context = _ctx_scan.redacted_text
                        _ctx_pii_redacted = _sanitized_context != context_str
                        if _sanitized_context != context_str:
                            logger.info(
                                "[ChatRetrieve] Context sanitized before LLM — "
                                "PII redacted from retrieved chunks"
                            )
                            context_str = _sanitized_context
    
                        embed_focus = embed_text.strip()
                        raw_trim = raw_query.strip()
    
                        tokens_missing_from_passages = [
                            k for k in strict_detail_grounding_keys
                            if not _token_appears_in_passages(context_str, k)
                        ]
                        skipped_llm_grounding_miss = bool(
                            strict_detail_grounding_keys and tokens_missing_from_passages,
                        )
    
                        identifier_focus_block = ""
                        if detail_focus_token:
                            identifier_focus_block = (
                                f"PRIMARY IDENTIFIER IN FOCUS: {detail_focus_token}\n"
                                "Prioritize sentences that mention this exact token verbatim. "
                                "Every factual claim about this identifier must cite [Source n]. "
                                "Do not refuse when any passage contains this token.\n\n"
                            )
                        if embed_focus != raw_trim:
                            question_block = identifier_focus_block + (
                                f"USER MESSAGE (latest):\n{raw_trim}\n\n"
                                f"RETRIEVAL FOCUS (embeddings keyword search targeted this):\n{embed_focus}\n"
                                "Answer the USER MESSAGE using ONLY the passages. "
                                "Use the retrieval focus as context for why these passages appear.\n"
                            )
                        else:
                            question_block = identifier_focus_block + f"USER MESSAGE:\n{raw_trim}\n\n"
    
                        if trace:
                            _chunks_meta: List[Dict[str, Any]] = [
                                {
                                    "rank": c.rank,
                                    "chunk_id": c.chunk_id,
                                    "score": c.score,
                                    "text_chars": len(c.text or ""),
                                    "text_digest": chunk_text_digest(c.text or ""),
                                }
                                for c in top_chunks
                            ]
                            trace.add_event(
                                "L2",
                                "context_injection",
                                _trace_lap(),
                                {
                                    "max_context_chunks": effective_max_context_chunks,
                                    "chunks_sent": len(top_chunks),
                                    "hash_context_post_pii": text_digest_utf8(context_str),
                                    "pii_redacted_from_context": _ctx_pii_redacted,
                                    "retrieval_focus_differs_from_raw": embed_focus != raw_trim,
                                    "chunks": _chunks_meta,
                                    "strict_detail_grounding_keys": strict_detail_grounding_keys,
                                    "tokens_missing_from_passages": tokens_missing_from_passages,
                                    "skipped_llm_grounding_miss": skipped_llm_grounding_miss,
                                    "request_top_k": req.top_k,
                                    "max_semantic_score": round(max_score, 5),
                                    "chunk_file_ids": [r.file_id for r in llm_ranked],
                                },
                            )
    
                        # Passages were selected using `embed_text` (rewrite / HyDE), but we used to ask
                        # the LLM only `raw_query` — small models often refuse when those diverge.
    
                        refusal_scrubbed = False
                        raw_answer = ""
                        rag_llm_resp: Any = None
                        rag_prompt = ""
                        if skipped_llm_grounding_miss:
                            answer = _GROUNDING_REFUSAL_PHRASE
                            logger.info(
                                "[ChatRetrieve] Grounding gate — token(s) not in retrieved passages: %s "
                                "| top_k=%s chunks_to_llm=%d",
                                tokens_missing_from_passages,
                                req.top_k,
                                len(top_chunks),
                            )
                        else:
                            rag_prompt = (
                                f"{tenant_prompt_prefix}"
                                f"{_chat_rag_compliance_block()}"
                                f"{question_block}\n"
                                f"RETRIEVED PASSAGES:\n{context_str}\n\n"
                                "ANSWER (grounded, with citations):"
                            )

                            # #region agent log
                            _agent_debug_ndjson(
                                hypothesis_id="H1",
                                location="retrieve_chat_api:pre_rag_generate",
                                message="before_llm_generate",
                                data={
                                    "has_focus_token": bool(detail_focus_token),
                                    "context_chars": len(context_str or ""),
                                },
                            )
                            # #endregion

                            resp = None
                            try:
                                resp = llm.generate(rag_prompt, temperature=0.0, max_tokens=1200)
                            except BaseException as _gen_exc:
                                # Transport timeouts / disconnects → deterministic excerpts when possible.
                                if (
                                    _chat_llm_transport_exc(_gen_exc)
                                    and detail_focus_token
                                    and context_str.strip()
                                ):
                                    transport_exc_name = type(_gen_exc).__name__
                                    logger.warning(
                                        "[ChatRetrieve] RAG llm.generate transport failure "
                                        "(attempting Layer 5 focus fallback): %s",
                                        transport_exc_name,
                                        exc_info=True,
                                    )
                                    # #region agent log
                                    _agent_debug_ndjson(
                                        hypothesis_id="H2",
                                        location="retrieve_chat_api:ragen_transport_exc",
                                        message="transport_failure_focus_path",
                                        data={"exc_type": transport_exc_name},
                                    )
                                    # #endregion
                                    fb_exc = _build_focus_fallback_answer(
                                        context_str, detail_focus_token
                                    )
                                    if fb_exc:
                                        answer = fb_exc
                                        focus_fallback_used = True
                                        raw_answer = ""
                                        rag_llm_resp = None
                                        answer, identifier_dedupe_applied = _dedupe_identifier_lines(answer)
                                        # #region agent log
                                        _agent_debug_ndjson(
                                            hypothesis_id="H3",
                                            location="retrieve_chat_api:after_timeout_fallback",
                                            message="focus_fallback_answer_set",
                                            data={
                                                "answer_chars": len(answer or ""),
                                                "focus_fallback_used": True,
                                            },
                                        )
                                        # #endregion
                                        logger.info(
                                            "[ChatRetrieve] Transport failure — excerpts from "
                                            "passages | token=%s",
                                            detail_focus_token,
                                        )
                                    else:
                                        answer_error = (
                                            f"LLM transport failure ({transport_exc_name}); "
                                            "could not build deterministic excerpts for focus token."
                                        )
                                        raw_answer = ""
                                        rag_llm_resp = None
                                else:
                                    raise _gen_exc

                            if transport_exc_name is None and resp is not None:
                                rag_llm_resp = resp
                                raw_answer = (resp.text or "").strip()
                                if raw_answer:
                                    sanitized, refusal_scrubbed = _strip_contradictory_refusal(raw_answer)
                                    answer = sanitized
                                    if refusal_scrubbed:
                                        logger.info(
                                            "[ChatRetrieve] Stripped contradictory refusal hedge "
                                            "after grounded excerpts (small-LLM pattern)"
                                        )
                                    if answer is not None:
                                        answer, identifier_dedupe_applied = _dedupe_identifier_lines(answer)
                                        answer, focus_fallback_used = _maybe_focus_fallback_answer(
                                            answer, context_str, detail_focus_token
                                        )
                                        if focus_fallback_used and answer:
                                            answer, _extra_dedupe = _dedupe_identifier_lines(answer)
                                            identifier_dedupe_applied = (
                                                identifier_dedupe_applied or _extra_dedupe
                                            )
                                        if identifier_dedupe_applied:
                                            logger.info(
                                                "[ChatRetrieve] P2 identifier line dedupe applied"
                                            )
                                        if focus_fallback_used:
                                            logger.info(
                                                "[ChatRetrieve] P3 focus fallback applied | "
                                                "token=%s",
                                                detail_focus_token,
                                            )
                                else:
                                    answer_error = "LLM returned an empty response."

                        final_answer = answer if answer is not None else ""
                        _llm_answer, _verification = verify_or_refuse(
                            answer=final_answer,
                            source_texts=[c.text for c in top_chunks if c.text],
                            focus_id=matched_pattern,
                            route=_route_decision.route.value,
                        )
                        answer = _llm_answer
    
                        if trace:
                            if skipped_llm_grounding_miss:
                                trace.add_event(
                                    "L3",
                                    "llm_skipped_grounding_gate",
                                    _trace_lap(),
                                    {
                                        "tokens_missing_from_passages": tokens_missing_from_passages,
                                        "prompt_template_source": prompt_template_source,
                                        "prompt_template_id_effective": prompt_template_id_effective,
                                    },
                                )
                            else:
                                l3_payload: Dict[str, Any] = {
                                    **llm_usage_payload(rag_llm_resp),
                                    "llm_provider": llm_provider,
                                    "llm_model": llm_model_name,
                                    "max_tokens": 1200,
                                    "temperature": 0.0,
                                    "prompt_template_source": prompt_template_source,
                                    "prompt_template_id_effective": prompt_template_id_effective,
                                }
                                if transport_exc_name:
                                    l3_payload["transport_recovery"] = transport_exc_name
                                    l3_payload["focus_fallback_after_transport"] = bool(
                                        focus_fallback_used
                                    )
                                if rag_chat_trace_include_prompt_hash() and rag_prompt:
                                    l3_payload["hash_rag_prompt"] = text_digest_utf8(rag_prompt)
                                    l3_payload["rag_prompt_chars"] = len(rag_prompt)
                                _ev = (
                                    "llm_generate_transport_recovery"
                                    if transport_exc_name
                                    else "llm_generate"
                                )
                                trace.add_event(
                                    "L3",
                                    _ev,
                                    _trace_lap(),
                                    l3_payload,
                                )
                            trace.add_event(
                                "L4",
                                "post_process",
                                _trace_lap(),
                                {
                                    "refusal_scrubbed": refusal_scrubbed,
                                    "identifier_dedupe_applied": identifier_dedupe_applied,
                                    "hash_answer": text_digest_utf8(answer or ""),
                                    "answer_chars": len(answer or ""),
                                },
                            )
                            trace.add_event(
                                "L5",
                                "focus_fallback",
                                _trace_lap(),
                                {
                                    "focus_fallback_used": focus_fallback_used,
                                    "detail_focus_token": detail_focus_token,
                                },
                            )
                            if _verification is not None:
                                trace.add_event(
                                    "L5",
                                    "faithfulness_verifier",
                                    _trace_lap(),
                                    {
                                        "status": _verification.status.value,
                                        "checks": [
                                            {
                                                "id": c.check_id,
                                                "status": c.status.value,
                                                "detail": c.detail,
                                            }
                                            for c in _verification.checks
                                        ],
                                        "fail_reason": _verification.fail_reason,
                                    },
                                )
    
                        answer_latency_ms = round((time.perf_counter() - gen_start) * 1000, 2)
                    except Exception as e:
                        logger.warning("[ChatRetrieve] LLM answer failed: %s", e, exc_info=True)
                        answer_error = f"LLM generation error: {str(e)[:300]}"
    
        # ── Debug info ───────────────────────────────────────────────────────────
        debug_info: Dict[str, Any] = {
            "client_id": retrieve_cid,
            "embedder": rc.embedder_type,
            "llm_provider": llm_provider,
            "llm_model": llm_model_name,
            "vectordb": rc.vectordb_type,
            "collection": rc.collection,
            "chunking_strategy": rc.chunking_strategy,
            "search_mode": search_mode,
            "intent": req.intent,
            "reranker_used": reranker_used,
            "score_gate_threshold": rag_min_score,
            "max_score": round(max(r.score for r in results), 4) if results else None,
            "hyde_enabled": req.enable_hyde,
            "hyde_skipped_for_identifier": hyde_skipped_for_identifier,
            "detail_focus_token": detail_focus_token,
            "strict_detail_grounding_keys": strict_detail_grounding_keys,
            "request_top_k": req.top_k,
            "max_context_chunks": req.max_context_chunks,
            "effective_max_context_chunks": effective_max_context_chunks,
            "prompt_locked": True,
            "prompt_template_id_locked": _active_template_id,
            "matched_pattern": matched_pattern,
            "ranked_results_count": len(results),
            "context_chunks_sent_to_llm": context_chunks_sent_to_llm,
            "skipped_llm_grounding_miss": skipped_llm_grounding_miss,
            "tokens_missing_from_passages": tokens_missing_from_passages,
            "identifier_dedupe_applied": identifier_dedupe_applied,
            "focus_fallback_used": focus_fallback_used,
            "faithfulness_verifier": (
                {
                    "status": _verification.status.value,
                    "checks": [
                        {
                            "id": c.check_id,
                            "status": c.status.value,
                            "detail": c.detail,
                        }
                        for c in _verification.checks
                    ],
                    "fail_reason": _verification.fail_reason,
                }
                if _verification is not None
                else None
            ),
            "llm_transport_error": transport_exc_name,
            "prompt_template_id_effective": prompt_template_id_effective,
            "prompt_template_resolved": prompt_template_resolved,
            "prompt_template_source": prompt_template_source,
            "query_rewritten": rewritten_query is not None,
            "route_decision": _route_decision.to_trace_dict(),
            "retrieval_skipped": False,
            "query_route": _route_decision.route.value,
            "route_layer": _route_decision.layer_used.value,
            "rag_tracing": trace is not None,
            "rag_trace_id": trace.trace_id if trace else None,
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
    except HTTPException as e:
        if trace:
            det = e.detail
            if isinstance(det, str):
                msg = det[:500]
            else:
                import json as _json
                try:
                    msg = _json.dumps(det, ensure_ascii=False)[:500]
                except Exception:
                    msg = repr(det)[:500]
            trace.error_state = {
                "stage": "http",
                "type": "HTTPException",
                "message": msg,
            }
        raise
    except Exception as e:
        if trace and trace.error_state is None:
            trace.set_error("exception", e)
        raise
    finally:
        if trace:
            trace.emit_final()
