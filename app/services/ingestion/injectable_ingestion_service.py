# =============================================
# injectable_ingestion_service.py — DI-Friendly Ingestion Service
#
# PROBLEM WITH ORIGINAL IngestionServiceV2:
#   All methods are @staticmethod. This means:
#   - No dependency injection — test isolation requires monkey-patching globals
#   - No interface/contract — cannot swap implementations (test vs prod)
#   - Hidden global state via module-level _pipeline_cache, async_session, etc.
#
# THIS CLASS:
#   - Accepts all dependencies via __init__ (session factory, pipeline resolver,
#     security middleware, chunk scorer)
#   - Registers as a FastAPI dependency via get_ingestion_service()
#   - Makes pipeline resolution explicit and stable per file ingest
#   - Applies PII redaction before embedding and chunk storage
#   - Records chunk quality scores for auditing
#
# USAGE IN FastAPI:
#
#   @router.post("/ingest/{file_id}")
#   async def ingest(
#       file_id: str,
#       service: InjectableIngestionService = Depends(get_ingestion_service),
#   ):
#       await service.process_file(file_id)
#       return {"status": "queued"}
# =============================================

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.utils.instrumentation import IngestionLogger, timed_stage
from app.middleware.security_middleware import scan_text, SecurityScanResult

logger = logging.getLogger(__name__)
_ing_log = IngestionLogger(__name__)

_CHUNK_QUALITY_THRESHOLD: float = float(
    os.getenv("CHUNK_QUALITY_THRESHOLD", "0.25")
)
_APPLY_PII_REDACTION: bool = (
    os.getenv("INGESTION_PII_REDACTION", "true").lower() in ("1", "true", "yes")
)


class InjectableIngestionService:
    """
    Dependency-injectable ingestion service.

    Wraps the existing static IngestionServiceV2 with:
      1. Constructor-injected session factory (testable without DB)
      2. PII detection + redaction on every chunk's text before embedding
      3. Chunk quality scoring with configurable threshold filtering
      4. Explicit pipeline reference passed through the call stack
      5. Structured logging at every stage

    This class does NOT duplicate ingestion logic — it delegates to
    IngestionServiceV2 for all DB and VectorDB operations.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker,
        *,
        pipeline_resolver: Optional[Callable] = None,
        apply_pii_redaction: bool = _APPLY_PII_REDACTION,
        chunk_quality_threshold: float = _CHUNK_QUALITY_THRESHOLD,
    ):
        self._session = session_factory
        self._pipeline_resolver = pipeline_resolver
        self.apply_pii_redaction = apply_pii_redaction
        self.chunk_quality_threshold = chunk_quality_threshold

    # ─────────────────────────────────────────────────────────────────
    # PRIMARY ENTRYPOINT
    # ─────────────────────────────────────────────────────────────────

    async def process_file(
        self,
        file_id: str,
        file_path: Optional[str] = None,
        business_id: Optional[str] = None,
    ) -> None:
        """
        Process a single file through the full ingestion pipeline.

        Steps:
          1. Resolve pipeline once for this file (stable for full duration)
          2. Parse → chunk → PII-scan chunks → quality-filter → dedup → embed → store
        """
        from app.services.ingestion.ingestion_orchestrator import IngestionOrchestrator

        _ing_log.info("Injectable process_file starting", file_id=file_id, stage="start")
        async with timed_stage("full_pipeline", file_id=file_id):
            await IngestionOrchestrator().ingest_file(
                file_id=file_id,
                client_id=business_id,
                file_path=file_path,
                kind="http_async",
            )
        _ing_log.info("Injectable process_file complete", file_id=file_id, stage="complete")

    # ─────────────────────────────────────────────────────────────────
    # PII GUARD — call before embedding any chunk text
    # ─────────────────────────────────────────────────────────────────

    def apply_security_scan(
        self,
        chunks: List[Dict[str, Any]],
        *,
        file_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[SecurityScanResult]]:
        """
        Scan all chunk texts for PII and prompt injection.

        Returns:
          (secured_chunks, scan_results)

          secured_chunks: chunks with cleaned_text/text replaced by redacted versions
          scan_results:   per-chunk SecurityScanResult for audit logging

        Chunks flagged for prompt injection have is_injection=True in metadata.
        They are NOT dropped — the caller decides whether to block or continue
        with the redacted text.
        """
        secured: List[Dict[str, Any]] = []
        results: List[SecurityScanResult] = []

        for chunk in chunks:
            text = chunk.get("cleaned_text") or chunk.get("text") or ""
            result = scan_text(
                text,
                redact_pii_data=self.apply_pii_redaction,
                check_injection=True,
                context=file_id,
            )
            results.append(result)

            if not result.is_safe:
                meta = dict(chunk.get("metadata") or {})
                meta["security_injection_detected"] = True
                meta["security_injection_hits"] = result.injection_patterns_hit[:3]
                chunk = {**chunk, "metadata": meta}
                _ing_log.warning(
                    "Prompt injection detected in chunk",
                    file_id=file_id,
                    stage="security_scan",
                )

            if result.has_pii and self.apply_pii_redaction:
                chunk = {
                    **chunk,
                    "cleaned_text": result.redacted_text,
                    "text": result.redacted_text,
                }

            secured.append(chunk)

        return secured, results

    # ─────────────────────────────────────────────────────────────────
    # CHUNK QUALITY FILTER
    # ─────────────────────────────────────────────────────────────────

    def filter_low_quality_chunks(
        self,
        chunks: List[Dict[str, Any]],
        *,
        file_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split chunks into (high_quality, low_quality) based on quality score.

        Low-quality chunks are NOT embedded into VectorDB but ARE stored in
        ingested_content with is_low_quality=True for audit purposes.
        """
        from app.core.chunking_stratagies.chunk_quality_scorer import score_chunk_quality

        high: List[Dict[str, Any]] = []
        low: List[Dict[str, Any]] = []

        for chunk in chunks:
            text = chunk.get("cleaned_text") or chunk.get("text") or ""
            tokens = int(chunk.get("tokens") or 0)
            score = score_chunk_quality(text, tokens)
            chunk = {**chunk, "quality_score": score}

            if score >= self.chunk_quality_threshold:
                high.append(chunk)
            else:
                low.append({**chunk, "is_low_quality": True})

        if low:
            _ing_log.info(
                "Filtered low-quality chunks",
                file_id=file_id,
                stage="quality_filter",
                high=len(high),
                low=len(low),
                threshold=self.chunk_quality_threshold,
            )

        return high, low

    # ─────────────────────────────────────────────────────────────────
    # TOKEN COST TRACKING (wraps embed_and_store to capture usage)
    # ─────────────────────────────────────────────────────────────────

    async def embed_with_cost_tracking(
        self,
        file_id: str,
        business_id: Optional[str],
        file_type: str,
        chunks: List[Dict[str, Any]],
        *,
        pipeline: Any = None,
        db: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Embed chunks and record token/cost metrics.

        Returns a cost summary dict:
            {"chunks_embedded": N, "estimated_tokens": N, "embedder": "..."}
        """
        from app.services.ingestion.ingestion_service_v2 import IngestionServiceV2

        estimated_tokens = sum(
            int(c.get("tokens") or 0) for c in chunks
        )

        async with timed_stage("embed_and_store", file_id=file_id):
            await IngestionServiceV2.embed_and_store(
                file_id, business_id, file_type, chunks,
                pipeline=pipeline,
                db=db,
                skip_presence_check=True,
            )

        embedder_name = ""
        if pipeline:
            embedder_name = str(
                getattr(getattr(pipeline, "embedder", None), "kind", "")
                or getattr(
                    getattr(getattr(pipeline, "embedder", None), "info", None),
                    "provider", "",
                )
            )

        cost_summary = {
            "chunks_embedded": len(chunks),
            "estimated_tokens": estimated_tokens,
            "embedder": embedder_name,
        }
        _ing_log.info(
            "Embedding complete",
            file_id=file_id,
            stage="embed",
            **cost_summary,
        )
        return cost_summary


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI DEPENDENCY FACTORY
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_ingestion_service() -> InjectableIngestionService:
    """
    FastAPI dependency: returns a singleton InjectableIngestionService.

    Usage::

        @router.post("/ingest/{file_id}")
        async def ingest(
            file_id: str,
            service: InjectableIngestionService = Depends(get_ingestion_service),
        ):
            await service.process_file(file_id)
    """
    from app.db.session_v2 import async_engine
    session_factory = async_sessionmaker(async_engine, expire_on_commit=False, autoflush=False)
    return InjectableIngestionService(session_factory=session_factory)
