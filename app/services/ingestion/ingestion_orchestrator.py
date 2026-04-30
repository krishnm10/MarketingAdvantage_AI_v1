"""
================================================================================
Ingestion Orchestrator — Single entry point for all ingestion operations.

ARCHITECTURE:
    External callers (API, worker, connectors) → IngestionOrchestrator
        → get_client_config       (authoritative config resolution)
        → _enforce_embedding_policy  (sensitivity gate — fail fast)
        → sanitize_ingestion_chunks  (PII gate)
        → IngestionServiceV2         (chunking, dedup, storage, embedding)

NON-NEGOTIABLES:
    1. ClientConfig is resolved via get_client_config() at the START of
       every entry point. No env-driven fallback. No optional configs.
    2. Embedding policy is enforced from the resolved config ONLY.
    3. PII sanitization runs on EVERY ingestion path.

USAGE:
    orchestrator = IngestionOrchestrator()
    await orchestrator.ingest_file(file_id="...", client_id="acme")
    await orchestrator.ingest_parsed_output(file_id="...", parsed={...}, client_id="acme")
================================================================================
"""

from __future__ import annotations

import asyncio
import json as _json_module
import logging
from typing import Any, Dict, FrozenSet, List, Optional

from app.core.config.client_config_schema import ClientConfig
from app.core.config.client_config_resolver import get_config_fingerprint
from app.utils.pipeline_logger import PipelineLogger
from app.services.security.ingestion_security import sanitize_ingestion_chunks

logger = logging.getLogger(__name__)

_LOCAL_EMBEDDER_PROVIDERS: FrozenSet[str] = frozenset({
    "huggingface",
    "ollama",
})


def _fire_pii_audit(
    *,
    client_id: str,
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
            business_id=client_id,
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


def _resolve_config(client_id: str) -> ClientConfig:
    """
    Authoritative config resolution. Every ingestion path starts here.

    Raises ConfigValidationError or FileNotFoundError on bad config.
    """
    from app.core.config.client_config_resolver import get_client_config
    return get_client_config(client_id)


def _log_resolved_config(
    config: ClientConfig,
    *,
    plog: PipelineLogger,
) -> None:
    """Emit a structured JSON log confirming config resolution for ingestion."""
    embedder_model = None
    sub = getattr(config.embedder, config.embedder.type.value, None)
    if sub is not None:
        embedder_model = getattr(sub, "model", None)

    llm_model = None
    if config.llm and config.llm.single:
        llm_model = config.llm.single.model

    fingerprint = get_config_fingerprint(config)

    try:
        logger.info(
            "%s",
            _json_module.dumps({
                "event": "INGESTION_CONFIG_RESOLVED",
                "client_id": config.client_id,
                "config_fingerprint": fingerprint,
                "embedder_type": config.embedder.type.value,
                "embedder_model": embedder_model,
                "vectordb_backend": config.vectordb.type.value,
                "vectordb_collection": config.vectordb.collection,
                "llm_model": llm_model,
                "reranker": config.reranker.type.value if config.reranker else "none",
                "data_sensitivity": config.security.data_sensitivity,
                "pii_enabled": config.security.pii_middleware.enabled,
            }, default=str),
        )
    except Exception:
        pass

    plog.info(
        "Resolved authoritative ClientConfig",
        client_id=config.client_id,
        config_fingerprint=fingerprint,
        embedder=config.embedder.type.value,
        vectordb=config.vectordb.type.value,
    )


class IngestionOrchestrator:
    """
    Orchestrates ingestion with mandatory security sanitization.

    Every ingestion path — file upload, pre-parsed RSS/API, connector
    output — MUST go through this class to guarantee:
        1. Authoritative ClientConfig resolution via get_client_config().
        2. Embedding policy enforcement from the resolved config.
        3. PII redaction before data leaves the service boundary.
    """

    # ------------------------------------------------------------------
    # Embedding policy — fail fast before any data is processed
    # ------------------------------------------------------------------

    @staticmethod
    def _enforce_embedding_policy(
        config: ClientConfig,
        *,
        plog: PipelineLogger,
    ) -> None:
        """
        Validate that the configured embedder is compatible with the
        client's data sensitivity tier.

        All decisions are driven by the resolved ClientConfig — no env
        fallback. The embedder type is read directly from config.embedder.type.

        Policy:
            high   → local embedders only (huggingface, ollama).
            medium → cloud embedders permitted; PII sanitized first.
            low    → no restrictions.

        Raises:
            ValueError: remote embedder configured for high-sensitivity client.
        """
        sensitivity = config.security.data_sensitivity
        embedder_provider = config.embedder.type.value.lower()
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

    # ------------------------------------------------------------------
    # PII hook — defined once, reused across all ingestion paths
    # ------------------------------------------------------------------

    @staticmethod
    def _make_pii_hook(
        config: ClientConfig,
        *,
        file_id: str,
        client_id: str,
        plog: PipelineLogger,
    ):
        """
        Build a pre_embed_hook closure that sanitizes chunks via the
        central security layer and logs PII findings.

        The returned callable has the signature expected by
        IngestionServiceV2._run_pipeline(pre_embed_hook=...):
            (chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]
        """

        def _sanitize_hook(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            sanitized, pii_meta = sanitize_ingestion_chunks(
                chunks, client_config=config,
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
        client_id: str = "default",
        *,
        file_path: Optional[str] = None,
    ) -> None:
        """
        Ingest a file through the full pipeline with PII sanitization.

        Flow:
            1. Resolve authoritative ClientConfig.
            2. Enforce embedding policy (fail fast).
            3. Build PII hook from resolved config.
            4. Delegate to IngestionServiceV2.process_file().
        """
        from app.services.ingestion.ingestion_service_v2 import IngestionServiceV2

        plog = PipelineLogger(
            request_path="ingestion",
            client_id=client_id,
        )
        plog.info("Orchestrator: file ingestion started", file_id=file_id)

        config = _resolve_config(client_id)
        _log_resolved_config(config, plog=plog)

        self._enforce_embedding_policy(config, plog=plog)

        hook = self._make_pii_hook(
            config,
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
        client_id: str = "default",
    ) -> None:
        """
        Ingest pre-parsed content with PII sanitization.

        Flow:
            1. Resolve authoritative ClientConfig.
            2. Enforce embedding policy (fail fast).
            3. Build PII hook from resolved config.
            4. Delegate to IngestionServiceV2.ingest_parsed_output().
        """
        from app.services.ingestion.ingestion_service_v2 import IngestionServiceV2

        plog = PipelineLogger(
            request_path="ingestion",
            client_id=client_id,
        )
        plog.info("Orchestrator: parsed ingestion started", file_id=file_id)

        config = _resolve_config(client_id)
        _log_resolved_config(config, plog=plog)

        self._enforce_embedding_policy(config, plog=plog)

        hook = self._make_pii_hook(
            config,
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
