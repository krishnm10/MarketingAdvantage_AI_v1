"""
================================================================================
Ingestion Orchestrator — Single entry point for all ingestion operations.

ARCHITECTURE:
    External callers (API, worker, connectors) → IngestionOrchestrator
        → _enforce_embedding_policy  (sensitivity gate — fail fast)
        → sanitize_ingestion_chunks  (PII gate)
        → IngestionServiceV2         (chunking, dedup, storage, embedding)

PURPOSE:
    1. Enforces the non-negotiable EMBEDDING POLICY: clients with
       data_sensitivity="high" are restricted to local-only embedders
       (huggingface, ollama). The check runs BEFORE any data is parsed,
       chunked, or processed — fail fast, fail loud.

    2. Enforces the non-negotiable SECURITY GATE: ALL data entering the
       ingestion pipeline passes through PII sanitization before it
       reaches embedders, vector stores, or any external API.

    The PII hook is defined ONCE here and injected into every
    IngestionServiceV2 call via the pre_embed_hook parameter. The service
    itself is agnostic to what the hook does — clean separation of concerns.

USAGE:
    orchestrator = IngestionOrchestrator(client_config=cfg)
    await orchestrator.ingest_file(file_id="...", client_id="...")
    await orchestrator.ingest_parsed_output(file_id="...", parsed={...}, client_id="...")
================================================================================
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, FrozenSet, List, Optional

from app.utils.pipeline_logger import PipelineLogger
from app.services.security.ingestion_security import sanitize_ingestion_chunks

logger = logging.getLogger(__name__)

# Providers that run entirely on local hardware — no data crosses the network.
_LOCAL_EMBEDDER_PROVIDERS: FrozenSet[str] = frozenset({
    "huggingface",
    "ollama",
})


def _fire_pii_audit(
    *,
    client_id: Optional[str],
    file_id: str,
    pii_meta: Dict[str, Any],
) -> None:
    """
    Best-effort async audit log write for PII detected during ingestion.
    Never blocks the ingestion pipeline — failures are logged and swallowed.
    """
    try:
        from app.services.security.security_audit_service import SecurityAuditService

        svc = SecurityAuditService()
        coro = svc.log_pii_event(
            business_id=client_id or "unknown",
            pipeline_id=file_id,
            node_id="ingestion_orchestrator",
            position="pre_embedding",
            entities_detected=pii_meta.get("pii_types_found", []),
            action_taken="REDACT",
            severity_max=pii_meta.get("severity_max"),
            chunk_ids_affected=pii_meta.get("chunk_ids_affected", []),
            latency_ms=pii_meta.get("latency_ms"),
            meta_data={
                "total_chunks": pii_meta.get("total_chunks", 0),
                "chunks_with_pii": pii_meta.get("chunks_with_pii", 0),
                "source": "ingestion_orchestrator",
            },
        )
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            asyncio.run(coro)
    except Exception:
        logger.warning(
            "Failed to fire PII audit log for file_id=%s — non-fatal, ingestion continues",
            file_id,
        )


class IngestionOrchestrator:
    """
    Orchestrates ingestion with mandatory security sanitization.

    Every ingestion path — file upload, pre-parsed RSS/API, connector
    output — MUST go through this class to guarantee PII redaction
    before data leaves the service boundary.
    """

    def __init__(self, *, client_config: Optional[Any] = None) -> None:
        """
        Args:
            client_config: Optional ClientConfig instance. When provided,
                           PII middleware settings (custom patterns, severity
                           blocking, audit logging) are read from
                           config.security.pii_middleware.
                           Embedding policy is read from
                           config.security.data_sensitivity.
        """
        self._client_config = client_config

    # ------------------------------------------------------------------
    # Embedding policy — fail fast before any data is processed
    # ------------------------------------------------------------------

    def _enforce_embedding_policy(self, *, plog: PipelineLogger) -> None:
        """
        Validate that the configured embedder is compatible with the
        client's data sensitivity tier.

        Policy:
            high   → local embedders only (huggingface, ollama). Remote
                     providers are rejected because raw text (even after PII
                     redaction) must never leave the service boundary.
            medium → cloud embedders are permitted; PII is sanitized by the
                     pre_embed_hook before reaching the embedder (default).
            low    → no restrictions.

        Raises:
            ValueError: if a remote embedder is configured for a
                        high-sensitivity client. Fails fast before ingestion.
        """
        cfg = self._client_config
        sensitivity = "medium"
        if cfg is not None:
            sensitivity = getattr(
                getattr(cfg, "security", None), "data_sensitivity", "medium"
            )

        embedder_provider = self._resolve_embedder_provider()

        is_local = embedder_provider in _LOCAL_EMBEDDER_PROVIDERS
        plog.info(
            "Embedding policy applied",
            sensitivity=sensitivity,
            embedder_provider=embedder_provider,
            embedder_is_local=is_local,
        )

        if sensitivity == "high" and not is_local:
            raise ValueError(
                f"Embedding policy violation: remote embedder '{embedder_provider}' "
                f"is not allowed for high-sensitivity clients. "
                f"Configure a local embedder (huggingface, ollama) or "
                f"lower data_sensitivity in ClientConfig.security."
            )

    def _resolve_embedder_provider(self) -> str:
        """
        Determine the active embedder provider string.

        Resolution order:
          1. ClientConfig → pipeline_factory.build() → pipeline.embedder.info.provider
          2. Fallback to MAI_EMBEDDER env var (for legacy/env-driven pipelines)
        """
        cfg = self._client_config
        if cfg is not None:
            try:
                from app.core.pipeline_factory import pipeline_factory
                pipeline = pipeline_factory.get_cached(cfg.client_id)
                if pipeline is None:
                    pipeline = pipeline_factory.build(cfg)
                return pipeline.embedder.info.provider.lower()
            except Exception as e:
                logger.debug(
                    "Could not resolve embedder from ClientConfig: %s — "
                    "falling back to MAI_EMBEDDER env var",
                    e,
                )

        import os
        return os.getenv("MAI_EMBEDDER", "unknown").lower()

    # ------------------------------------------------------------------
    # PII hook — defined once, reused across all ingestion paths
    # ------------------------------------------------------------------

    def _make_pii_hook(
        self,
        *,
        file_id: str,
        client_id: Optional[str],
        plog: PipelineLogger,
    ):
        """
        Build a pre_embed_hook closure that sanitizes chunks via the
        central security layer and logs PII findings.

        The returned callable has the signature expected by
        IngestionServiceV2._run_pipeline(pre_embed_hook=...):
            (chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]
        """
        client_config = self._client_config

        def _sanitize_hook(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            sanitized, pii_meta = sanitize_ingestion_chunks(
                chunks, client_config=client_config,
            )
            if pii_meta.get("chunks_with_pii"):
                plog.info(
                    "PII sanitized before embedding",
                    file_id=file_id,
                    client_id=client_id,
                    pii_types=pii_meta["pii_types_found"],
                    chunks_affected=pii_meta["chunks_with_pii"],
                    severity_max=pii_meta["severity_max"],
                    latency_ms=pii_meta["latency_ms"],
                )
                _fire_pii_audit(
                    client_id=client_id,
                    file_id=file_id,
                    pii_meta=pii_meta,
                )
            return sanitized

        return _sanitize_hook

    # ------------------------------------------------------------------
    # File ingestion (UI upload, bulk, watcher)
    # ------------------------------------------------------------------

    async def ingest_file(
        self,
        file_id: str,
        client_id: Optional[str] = None,
        *,
        file_path: Optional[str] = None,
    ) -> None:
        """
        Ingest a file through the full pipeline with PII sanitization.

        Delegates to IngestionServiceV2.process_file() with a pre_embed_hook
        that redacts PII before chunks reach dedup/storage/embedding.
        """
        from app.services.ingestion.ingestion_service_v2 import IngestionServiceV2

        plog = PipelineLogger(
            request_path="ingestion",
            client_id=client_id,
        )
        plog.info("Orchestrator: file ingestion started", file_id=file_id)

        self._enforce_embedding_policy(plog=plog)

        hook = self._make_pii_hook(
            file_id=file_id,
            client_id=client_id,
            plog=plog,
        )

        await IngestionServiceV2.process_file(
            file_id=file_id,
            file_path=file_path,
            business_id=client_id,
            pre_embed_hook=hook,
        )

        plog.info("Orchestrator: file ingestion completed", file_id=file_id)

    # ------------------------------------------------------------------
    # Pre-parsed ingestion (RSS, API connector, Kafka)
    # ------------------------------------------------------------------

    async def ingest_parsed_output(
        self,
        file_id: str,
        parsed: Dict[str, Any],
        client_id: Optional[str] = None,
    ) -> None:
        """
        Ingest pre-parsed content with PII sanitization.

        Delegates to IngestionServiceV2.ingest_parsed_output() with a
        pre_embed_hook that redacts PII before chunks reach embedding.
        """
        from app.services.ingestion.ingestion_service_v2 import IngestionServiceV2

        plog = PipelineLogger(
            request_path="ingestion",
            client_id=client_id,
        )
        plog.info("Orchestrator: parsed ingestion started", file_id=file_id)

        self._enforce_embedding_policy(plog=plog)

        hook = self._make_pii_hook(
            file_id=file_id,
            client_id=client_id,
            plog=plog,
        )

        await IngestionServiceV2.ingest_parsed_output(
            file_id=file_id,
            parsed_output=parsed,
            pre_embed_hook=hook,
        )

        plog.info("Orchestrator: parsed ingestion completed", file_id=file_id)
