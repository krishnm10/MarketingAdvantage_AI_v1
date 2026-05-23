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
from typing import Any, Dict, List, Optional

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
    avail = (pt_i > 0) or (ct_i > 0) or (tt_i > 0) or (fr is not None and str(fr).strip() != "")
    return {
        "prompt_tokens": pt_i,
        "completion_tokens": ct_i,
        "total_tokens": tt_i,
        "finish_reason": str(fr) if fr is not None else None,
        "provider_usage_available": bool(avail),
    }
