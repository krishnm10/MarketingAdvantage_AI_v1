"""
Unified runtime telemetry emitter.

Single entry point for emitting structured telemetry events from any
pipeline stage. Normalizes field names to the canonical schema, strips
empty/None values to keep log volume stable, and delegates to the
existing PipelineLogger / stdlib logging infrastructure.

Usage:
    from app.core.runtime.runtime_telemetry import emit_runtime_event

    emit_runtime_event(
        event="QUERY_START",
        request_id=ctx.request_id,
        tenant_id=ctx.tenant_id,
        stack_used="rag_pipeline",
        latency_ms={"embed_ms": 12.3},
    )
"""

from __future__ import annotations

import json as _json
import logging
from dataclasses import asdict
from typing import Any, Dict, Optional, Union

from app.core.runtime.telemetry_schema import (
    TelemetryRecord,
    F_REQUEST_ID,
    F_TRACE_ID,
    F_TENANT_ID,
    F_CLIENT_ID,
    F_PIPELINE_ID,
    F_STACK_USED,
    F_RUNTIME_AUTHORITY,
)

_logger = logging.getLogger("mai.runtime.telemetry")

_CANONICAL_FIELDS = frozenset({
    "event",
    "request_id", "trace_id", "tenant_id", "client_id", "pipeline_id",
    "stack_used", "runtime_authority",
    "vectordb_backend", "embedder_provider", "embedder_model",
    "reranker_provider", "reranker_model", "llm_provider", "llm_model",
    "retrieval_mode", "search_mode",
    "latency_ms", "token_usage",
    "retrieved_count", "reranked_count", "final_count",
    "cache_hits", "cache_misses",
    "threshold_gate", "token_budget", "security_events",
})


def emit_runtime_event(
    event: str,
    *,
    record: Optional[TelemetryRecord] = None,
    level: int = logging.INFO,
    **kwargs: Any,
) -> None:
    """
    Emit a single structured telemetry event.

    Args:
        event:   Event name (e.g. "QUERY_START", "RETRIEVAL_COMPLETE").
        record:  Optional TelemetryRecord — its fields are merged into
                 the payload. Explicit kwargs override record fields.
        level:   Log level. Defaults to INFO.
        **kwargs: Additional telemetry fields. Non-canonical keys are
                  placed under an "extra" sub-dict to keep the top-level
                  schema stable.
    """
    if not _logger.isEnabledFor(level):
        return

    payload: Dict[str, Any] = {"event": event}

    if record is not None:
        for k, v in asdict(record).items():
            if _is_populated(v):
                payload[k] = v

    for k, v in kwargs.items():
        if _is_populated(v):
            if k in _CANONICAL_FIELDS:
                payload[k] = v
            else:
                payload.setdefault("extra", {})[k] = v

    _logger.log(level, "%s", _json.dumps(payload, default=str))


def emit_from_dict(
    event: str,
    data: Dict[str, Any],
    *,
    level: int = logging.INFO,
) -> None:
    """
    Emit a telemetry event from a pre-built dict.

    Normalizes keys: strips None/empty values and separates non-canonical
    fields into "extra".
    """
    if not _logger.isEnabledFor(level):
        return

    payload: Dict[str, Any] = {"event": event}
    extras: Dict[str, Any] = {}

    for k, v in data.items():
        if not _is_populated(v):
            continue
        if k in _CANONICAL_FIELDS:
            payload[k] = v
        else:
            extras[k] = v

    if extras:
        payload["extra"] = extras

    _logger.log(level, "%s", _json.dumps(payload, default=str))


def _is_populated(v: Any) -> bool:
    """Return False for None, empty strings, empty dicts/lists, and zero counts."""
    if v is None:
        return False
    if isinstance(v, str) and not v:
        return False
    if isinstance(v, (dict, list)) and not v:
        return False
    return True
