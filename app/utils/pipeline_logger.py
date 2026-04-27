# =============================================================================
# pipeline_logger.py — Enterprise Structured Pipeline Logger
#
# Provides a single, canonical structured logging interface for ALL data-plane
# operations: ingestion, RAG queries, and retrieval. Every log entry carries
# a stable set of indexable fields that log aggregators (Datadog, Splunk,
# CloudWatch, OpenSearch) can filter, group, and alert on.
#
# Required fields (populated where available, None otherwise):
#   client_id         — tenant / business identifier
#   pipeline_id       — cached pipeline key or request correlation id
#   embedder_model    — active embedding model name
#   vectordb_backend  — active VectorDB kind (chroma, qdrant, pinecone, …)
#   request_path      — logical origin: rag_api | retrieve_api | ingestion
#   token_usage       — embedding or LLM token count (when available)
#
# Usage:
#   from app.utils.pipeline_logger import PipelineLogger
#
#   plog = PipelineLogger(request_path="rag_api")
#   plog.info("Query started", client_id="acme", embedder_model="nomic-embed-text")
#   plog.info("Embedding complete", client_id="acme", token_usage=342, duration_ms=12.4)
#
# The logger is intentionally lightweight — it delegates to stdlib logging with
# structured `extra` fields.  When LOG_FORMAT=json is active (see logger.py),
# structlog serialises these fields as top-level JSON keys automatically.
# =============================================================================

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Union


_DEFAULT_LOGGER_NAME = "mai.pipeline"

# Env-configurable minimum level for pipeline logs (default: INFO).
_PIPELINE_LOG_LEVEL = getattr(
    logging,
    os.getenv("PIPELINE_LOG_LEVEL", "INFO").upper(),
    logging.INFO,
)


class PipelineLogger:
    """
    Enterprise structured logger for all pipeline data-plane operations.

    Each instance is bound to a ``request_path`` (rag_api, retrieve_api,
    ingestion, etc.) so downstream log entries inherit the origin without
    requiring callers to repeat it on every call.

    All keyword arguments passed to info/warning/error/debug are emitted as
    structured ``extra`` fields on the stdlib LogRecord.

    Thread-safe: stdlib Logger handles concurrency internally.
    """

    __slots__ = ("_log", "_defaults")

    def __init__(
        self,
        request_path: str,
        *,
        logger_name: str = _DEFAULT_LOGGER_NAME,
        client_id: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        embedder_model: Optional[str] = None,
        vectordb_backend: Optional[str] = None,
    ) -> None:
        self._log = logging.getLogger(logger_name)
        self._defaults: Dict[str, Any] = {
            "request_path": request_path,
        }
        if client_id is not None:
            self._defaults["client_id"] = client_id
        if pipeline_id is not None:
            self._defaults["pipeline_id"] = pipeline_id
        if embedder_model is not None:
            self._defaults["embedder_model"] = embedder_model
        if vectordb_backend is not None:
            self._defaults["vectordb_backend"] = vectordb_backend

    # ------------------------------------------------------------------
    # Bind additional defaults after construction (e.g. after pipeline
    # resolution when the embedder model becomes known).
    # ------------------------------------------------------------------

    def bind(self, **kwargs: Any) -> "PipelineLogger":
        """Return a *new* logger with additional default fields merged in."""
        merged = {**self._defaults, **kwargs}
        child = PipelineLogger.__new__(PipelineLogger)
        child._log = self._log
        child._defaults = merged
        return child

    # ------------------------------------------------------------------
    # Core emit — merges defaults + per-call fields into ``extra``
    # ------------------------------------------------------------------

    def _emit(self, level: int, msg: str, **kwargs: Any) -> None:
        if not self._log.isEnabledFor(level):
            return
        extra = {**self._defaults}
        extra.update(kwargs)
        # Ensure the canonical field set is always present (as None when unset)
        # so log schemas stay consistent across all entries.
        for key in (
            "client_id",
            "pipeline_id",
            "embedder_model",
            "vectordb_backend",
            "request_path",
            "token_usage",
        ):
            extra.setdefault(key, None)
        self._log.log(level, msg, extra=extra)

    # ------------------------------------------------------------------
    # Public convenience methods
    # ------------------------------------------------------------------

    def info(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.ERROR, msg, **kwargs)

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.DEBUG, msg, **kwargs)

    # ------------------------------------------------------------------
    # Helpers for common patterns
    # ------------------------------------------------------------------

    def stage_start(self, stage: str, **kwargs: Any) -> None:
        """Log the start of a named pipeline stage."""
        self._emit(logging.INFO, "stage_start: %s", stage=stage, **kwargs)

    def stage_end(
        self,
        stage: str,
        *,
        duration_ms: float,
        **kwargs: Any,
    ) -> None:
        """Log the completion of a named pipeline stage with latency."""
        self._emit(
            logging.INFO,
            "stage_end: %s (%.1fms)",
            stage=stage,
            duration_ms=round(duration_ms, 2),
            **kwargs,
        )

    def query_complete(
        self,
        *,
        chunks_used: int,
        reranked: bool,
        trust_score: Optional[float] = None,
        total_ms: float,
        **kwargs: Any,
    ) -> None:
        """Structured log for RAG/retrieval query completion."""
        self._emit(
            logging.INFO,
            "query_complete: chunks=%d reranked=%s trust=%s total_ms=%.1f",
            chunks_used=chunks_used,
            reranked=reranked,
            trust_score=trust_score,
            total_ms=round(total_ms, 2),
            **kwargs,
        )

    def ingestion_complete(
        self,
        *,
        file_id: str,
        total_chunks: int,
        unique_chunks: int,
        duration_ms: float,
        **kwargs: Any,
    ) -> None:
        """Structured log for ingestion pipeline completion."""
        self._emit(
            logging.INFO,
            "ingestion_complete: file=%s total=%d unique=%d (%.1fms)",
            file_id=file_id,
            total_chunks=total_chunks,
            unique_chunks=unique_chunks,
            duration_ms=round(duration_ms, 2),
            **kwargs,
        )
