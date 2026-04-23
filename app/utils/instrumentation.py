# =============================================
# instrumentation.py — Structured Logging, Timing, and Request Tracing
#
# Provides:
#   timed_stage()       — async context manager for per-stage latency tracking
#   IngestionLogger     — structured logging with file_id, business_id, stage fields
#   RequestIDMiddleware — propagates X-Request-ID through every HTTP request
#
# Usage in ingestion pipeline:
#   async with timed_stage("chunking", file_id=file_id, business_id=bid):
#       chunks = await _chunk_text_with_strategy(...)
# =============================================

from __future__ import annotations

import logging
import time
import uuid as _uuid_module
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURED INGESTION LOGGER
# Wraps Python's stdlib logger with structured `extra` fields so log
# aggregators (Datadog, CloudWatch, Splunk) can index on file_id / business_id.
# ─────────────────────────────────────────────────────────────────────────────

class IngestionLogger:
    """
    Structured logger with stable, parseable extra fields.

    Instead of:
        log_info(f"[IngestionV2] Starting ingestion for {file_id}")

    Use:
        ing_log = IngestionLogger("ingestion_service_v2")
        ing_log.info("Starting ingestion", file_id=file_id, stage="start")

    This produces:
        {"file_id": "...", "business_id": "...", "stage": "start", "message": "..."}
    which log aggregators can index, filter, and alert on.
    """

    def __init__(self, name: str):
        self._log = logging.getLogger(name)

    def _emit(
        self,
        level: int,
        msg: str,
        *,
        file_id: Optional[str] = None,
        business_id: Optional[str] = None,
        stage: Optional[str] = None,
        duration_ms: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        extra: dict = {}
        if file_id:
            extra["file_id"] = file_id
        if business_id:
            extra["business_id"] = business_id
        if stage:
            extra["stage"] = stage
        if duration_ms is not None:
            extra["duration_ms"] = round(duration_ms, 2)
        extra.update(kwargs)
        self._log.log(level, msg, extra=extra)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.ERROR, msg, **kwargs)

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.DEBUG, msg, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# TIMED STAGE — async context manager
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def timed_stage(
    stage_name: str,
    *,
    file_id: Optional[str] = None,
    business_id: Optional[str] = None,
    log: Optional[logging.Logger] = None,
) -> AsyncGenerator[None, None]:
    """
    Async context manager that times a pipeline stage and emits a structured
    log entry on completion.

    Usage::

        async with timed_stage("chunking", file_id=file_id, business_id=bid):
            chunks = await _chunk_text_with_strategy(text, ...)

    Emits::

        INFO stage_completed stage=chunking file_id=... duration_ms=42.3
    """
    _log = log or logger
    t0 = time.monotonic()
    try:
        yield
    finally:
        elapsed_ms = (time.monotonic() - t0) * 1000
        extra: dict = {
            "stage": stage_name,
            "duration_ms": round(elapsed_ms, 2),
        }
        if file_id:
            extra["file_id"] = file_id
        if business_id:
            extra["business_id"] = business_id
        _log.info("Stage completed: %s (%.1fms)", stage_name, elapsed_ms, extra=extra)


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST ID MIDDLEWARE
# Propagates a stable X-Request-ID header for distributed tracing.
# If the caller supplies one we reuse it; otherwise we generate a UUID v4.
# ─────────────────────────────────────────────────────────────────────────────

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Attach X-Request-ID to every request and response.

    - Reads inbound X-Request-ID header if present (caller-supplied correlation)
    - Falls back to a freshly generated UUID v4
    - Writes the ID into request.state.request_id (accessible from route handlers)
    - Copies the ID into the response X-Request-ID header

    Register BEFORE all other middleware so the ID is available everywhere::

        app.add_middleware(RequestIDMiddleware)
    """

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        request_id = (
            request.headers.get("X-Request-ID")
            or str(_uuid_module.uuid4())
        )
        request.state.request_id = request_id
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
