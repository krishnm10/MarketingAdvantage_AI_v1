"""
Structured ingestion failure context for DLQ recording (no exception mutation).
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class IngestionFailureContext:
    """Transient failure metadata set before re-raise from streaming windows."""

    stage: str
    window_index: Optional[int] = None
    pipeline_mode: str = "streaming"
    extra: Optional[Dict[str, Any]] = None


_ingestion_failure_ctx: ContextVar[Optional[IngestionFailureContext]] = ContextVar(
    "ingestion_failure_ctx",
    default=None,
)


def set_ingestion_failure_context(ctx: IngestionFailureContext) -> None:
    _ingestion_failure_ctx.set(ctx)


def pop_ingestion_failure_context() -> Optional[IngestionFailureContext]:
    """Return and clear the current failure context (one-shot for DLQ write)."""
    ctx = _ingestion_failure_ctx.get()
    _ingestion_failure_ctx.set(None)
    return ctx
