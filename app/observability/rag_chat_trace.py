"""
Structured end-to-end RAG chat trace (layers L1–L5) for POST /api/v2/retrieve/chat.

Env:
  RAG_CHAT_TRACE_ENABLED       — if true, emit one composite RAG_CHAT_TRACE JSON log per request
  RAG_CHAT_TRACE_PER_LAYER     — if true, also emit one log line per layer event (same trace_id)
  RAG_CHAT_TRACE_INCLUDE_PROMPT_HASH — if false, omit hash_rag_prompt from payloads (default true)

Payloads are metadata-only (lengths, hashes, ids, scores) to stay under cloud log line limits.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

_LOGGER = logging.getLogger(__name__)

EVENT_NAME = "RAG_CHAT_TRACE"


def rag_chat_trace_enabled() -> bool:
    return os.getenv("RAG_CHAT_TRACE_ENABLED", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def rag_chat_trace_per_layer() -> bool:
    return os.getenv("RAG_CHAT_TRACE_PER_LAYER", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def rag_chat_trace_include_prompt_hash() -> bool:
    v = os.getenv("RAG_CHAT_TRACE_INCLUDE_PROMPT_HASH", "true").strip().lower()
    return v not in ("0", "false", "no", "off")


def text_digest_utf8(text: str) -> str:
    """First 16 hex chars of SHA-256 (UTF-8). Empty string -> empty digest."""
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def chunk_text_digest(text: str) -> str:
    """Optional per-chunk content digest without logging body (metadata-only trace)."""
    return text_digest_utf8(text)


def vector_l2_norm(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sum(x * x for x in values)
    if math.isnan(s) or s < 0:
        return float("nan")
    return math.sqrt(s)


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class RagChatTrace:
    """In-memory collector; call emit_final() in a finally block."""

    trace_id: str
    session_id: str
    client_id: str
    started_perf: float
    started_utc: str
    events: List[Dict[str, Any]] = field(default_factory=list)
    error_state: Optional[Dict[str, Any]] = None
    request_path: str = "/api/v2/retrieve/chat"
    route_decision: Optional[Dict[str, Any]] = None
    retrieval_skipped: bool = False
    route_latency_ms: float = 0.0

    def add_event(
        self,
        layer: str,
        stage: str,
        duration_ms: float,
        payload: Dict[str, Any],
    ) -> None:
        row = {
            "layer": layer,
            "stage": stage,
            "ts_utc": _utc_iso(time.time()),
            "duration_ms": round(float(duration_ms), 3),
            "payload": payload,
        }
        self.events.append(row)
        if rag_chat_trace_per_layer():
            try:
                line = json.dumps(
                    {
                        "event": f"{EVENT_NAME}_LAYER",
                        "trace_id": self.trace_id,
                        "trace_id_short": self.trace_id[:8],
                        "session_id": self.session_id,
                        "client_id": self.client_id,
                        "layer": layer,
                        "stage": stage,
                        "row": row,
                    },
                    ensure_ascii=False,
                )
                _LOGGER.info(line)
            except Exception:
                _LOGGER.debug("[RagChatTrace] per-layer emit failed", exc_info=True)

    def set_error(
        self,
        stage: str,
        exc: BaseException,
        *,
        err_type: Optional[str] = None,
    ) -> None:
        msg = str(exc)[:500]
        self.error_state = {
            "stage": stage,
            "type": err_type or type(exc).__name__,
            "message": msg,
        }

    def emit_final(self) -> None:
        if not rag_chat_trace_enabled():
            return
        total_ms = round((time.perf_counter() - self.started_perf) * 1000, 2)
        blob = {
            "event": EVENT_NAME,
            "trace_id": self.trace_id,
            "trace_id_short": self.trace_id[:8],
            "session_id": self.session_id,
            "client_id": self.client_id,
            "request_path": self.request_path,
            "started_utc": self.started_utc,
            "total_duration_ms": total_ms,
            "layers": self.events,
            "error_state": self.error_state,
            "route_decision": self.route_decision,
            "retrieval_skipped": self.retrieval_skipped,
            "route_latency_ms": round(float(self.route_latency_ms), 3),
        }
        try:
            line = json.dumps(blob, ensure_ascii=False)
            if len(line) > 200_000:
                _LOGGER.warning(
                    "[RagChatTrace] trace payload very large (%d bytes); consider trimming",
                    len(line),
                )
            _LOGGER.info(line)
        except Exception:
            _LOGGER.exception("[RagChatTrace] emit_final failed")


def maybe_start_rag_chat_trace(session_id: str, client_id: str) -> Optional[RagChatTrace]:
    if not rag_chat_trace_enabled():
        return None
    tid = str(uuid.uuid4())
    return RagChatTrace(
        trace_id=tid,
        session_id=session_id,
        client_id=client_id,
        started_perf=time.perf_counter(),
        started_utc=_utc_iso(time.time()),
    )


def _usage_from_ollama_raw(raw: Any) -> Optional[Tuple[int, int, int]]:
    """Re-read Ollama native counts from stored SDK response when LLMResponse tokens are zero."""
    if raw is None:
        return None

    def _get(key: str) -> int:
        if hasattr(raw, key):
            val = getattr(raw, key, 0)
        elif isinstance(raw, dict):
            val = raw.get(key, 0)
        else:
            val = 0
        try:
            return int(val or 0)
        except (TypeError, ValueError):
            return 0

    usage = None
    if hasattr(raw, "usage"):
        usage = getattr(raw, "usage", None)
    elif isinstance(raw, dict):
        usage = raw.get("usage")

    if isinstance(usage, dict) and usage:
        pt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        ct = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        tt = int(usage.get("total_tokens") or 0)
        if tt <= 0 and (pt > 0 or ct > 0):
            tt = pt + ct
        if pt > 0 or ct > 0 or tt > 0:
            return pt, ct, tt

    pt = _get("prompt_eval_count")
    ct = _get("eval_count")
    if pt > 0 or ct > 0:
        return pt, ct, pt + ct
    return None


def llm_usage_payload(resp: Any) -> Dict[str, Any]:
    """Normalize usage from app.core.llms.base.LLMResponse or duck-typed."""
    if resp is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "finish_reason": None,
            "provider_usage_available": False,
        }
    pt = getattr(resp, "prompt_tokens", None)
    ct = getattr(resp, "completion_tokens", None)
    tt = getattr(resp, "total_tokens", None)
    fr = getattr(resp, "finish_reason", None)
    try:
        pt_i = int(pt or 0)
        ct_i = int(ct or 0)
        tt_i = int(tt or 0)
    except (TypeError, ValueError):
        pt_i = ct_i = tt_i = 0

    if pt_i <= 0 and ct_i <= 0 and tt_i <= 0:
        raw_usage = _usage_from_ollama_raw(getattr(resp, "raw", None))
        if raw_usage is not None:
            pt_i, ct_i, tt_i = raw_usage

    if tt_i <= 0 and (pt_i > 0 or ct_i > 0):
        tt_i = pt_i + ct_i

    avail = (pt_i > 0) or (ct_i > 0) or (tt_i > 0)
    return {
        "prompt_tokens": pt_i,
        "completion_tokens": ct_i,
        "total_tokens": tt_i,
        "finish_reason": str(fr) if fr is not None else None,
        "provider_usage_available": bool(avail),
    }


def aggregate_query_token_usage(
    steps: List[Optional[Tuple[str, Any]]],
) -> Dict[str, Any]:
    """
    Sum provider-reported tokens across executed LLM steps (rewrite, hyde, answer).

    Steps with unavailable usage are included in the breakdown but excluded from totals.
    """
    step_entries: List[Dict[str, Any]] = []
    total_pt = 0
    total_ct = 0
    total_tt = 0
    any_available = False

    for item in steps:
        if item is None:
            continue
        step_name, resp = item
        usage = llm_usage_payload(resp)
        entry: Dict[str, Any] = {
            "step": step_name,
            "executed": True,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "provider_usage_available": usage["provider_usage_available"],
        }
        if usage.get("finish_reason") is not None:
            entry["finish_reason"] = usage["finish_reason"]
        step_entries.append(entry)

        if usage["provider_usage_available"]:
            any_available = True
            total_pt += int(usage["prompt_tokens"])
            total_ct += int(usage["completion_tokens"])
            step_tt = int(usage["total_tokens"])
            if step_tt <= 0:
                step_tt = int(usage["prompt_tokens"]) + int(usage["completion_tokens"])
            total_tt += step_tt

    return {
        "label": "total_query_tokens",
        "provider_usage_available": any_available,
        "prompt_tokens": total_pt,
        "completion_tokens": total_ct,
        "total_tokens": total_tt,
        "finish_reason": None,
        "steps": step_entries,
    }
