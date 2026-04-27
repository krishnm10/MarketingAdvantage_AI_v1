"""
================================================================================
Ingestion Security Gate — Pre-embedding PII sanitization for ingested chunks.

ARCHITECTURE POSITION:
    Parser → Chunker → **ingestion_security.sanitize_ingestion_chunks()** → Embedder → VectorDB

PURPOSE:
    Ensures no raw PII reaches third-party embedding APIs or is persisted in
    vector stores. This closes the critical security gap where ingestion
    bypassed the central security layer that was already protecting retrieval
    and RAG query paths.

DESIGN DECISIONS:
    - Delegates ALL PII detection/redaction to the existing RegexPIIMiddleware
      (app.core.pipeline_nodes.pii_middleware) to keep security logic in one
      place. No pattern duplication.
    - Respects ClientConfig.security.pii_middleware settings (custom patterns,
      action, block_on_severity, audit_log_enabled) when a ClientConfig is
      provided. Falls back to safe defaults when called without config.
    - Does NOT modify the chunk dict schema — only mutates the "text" and
      "cleaned_text" values in-place on shallow copies. All other fields
      (metadata, semantic_hash, embedding_model, page_number, etc.) are
      preserved verbatim.
    - Returns a separate pii_metadata summary for the caller to use for
      audit logging, metrics, or conditional pipeline logic (e.g. blocking
      the file if CRITICAL PII is found).

THREAD SAFETY:
    Stateless function; safe to call from any async context.
================================================================================
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from app.middleware.security_middleware import SecurityScanResult  # noqa: F401 — re-exported for callers

logger = logging.getLogger(__name__)


def _build_middleware_config(
    client_config: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Extract PII middleware config dict from a ClientConfig, if provided.
    Returns a config dict consumable by RegexPIIMiddleware.__init__().
    Falls back to safe production defaults when no config is available.
    """
    defaults: Dict[str, Any] = {
        "position": ["pre_embedding"],
        "action": "REDACT",
        "audit_log_enabled": False,
        "custom_patterns": [],
    }

    if client_config is None:
        return defaults

    security = getattr(client_config, "security", None)
    if security is None:
        return defaults

    pii_cfg = getattr(security, "pii_middleware", None)
    if pii_cfg is None:
        return defaults

    return {
        "position": getattr(pii_cfg, "positions", defaults["position"]),
        "action": getattr(pii_cfg, "action", defaults["action"]),
        "block_on_severity": getattr(pii_cfg, "block_on_severity", None),
        "trust_score_penalty": getattr(pii_cfg, "trust_score_penalty", 0.0),
        "audit_log_enabled": getattr(pii_cfg, "audit_log_enabled", False),
        "custom_patterns": [
            {"name": p.name, "pattern": p.pattern, "severity": p.severity}
            for p in getattr(pii_cfg, "custom_patterns", [])
        ],
    }


def sanitize_ingestion_chunks(
    chunks: List[Dict[str, Any]],
    client_config: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Sanitize ingestion chunks by redacting PII before embedding/storage.

    Args:
        chunks:        List of chunk dicts as produced by the chunking stage.
                       Each chunk MUST have a "text" key. May also have
                       "cleaned_text", "metadata", and other fields.
        client_config: Optional ClientConfig instance. When provided, PII
                       middleware settings (custom patterns, severity blocking,
                       audit logging) are read from config.security.pii_middleware.
                       When None, safe defaults are used (regex REDACT, no audit).

    Returns:
        (sanitized_chunks, pii_metadata)

        sanitized_chunks: New list of chunk dicts with PII-redacted text.
                          Chunk schema is unchanged — only "text" and
                          "cleaned_text" values are replaced with redacted
                          versions. All other fields are preserved.

        pii_metadata:     Summary dict suitable for logging/audit:
                          {
                              "total_chunks":     int,
                              "chunks_with_pii":  int,
                              "pii_types_found":  List[str],
                              "blocked":          bool,
                              "severity_max":     Optional[str],
                              "latency_ms":       float,
                              "chunk_ids_affected": List[str],
                          }
    """
    if not chunks:
        return [], _empty_pii_metadata()

    t0 = time.perf_counter()

    from app.core.pipeline_nodes.pii_middleware import RegexPIIMiddleware

    middleware_cfg = _build_middleware_config(client_config)
    middleware = RegexPIIMiddleware(config=middleware_cfg)

    sanitized, scan_result = middleware.scan_chunks(
        chunks, position="pre_embedding",
    )

    # scan_chunks returns shallow copies with "text" (and "content") already
    # redacted. We also need to redact "cleaned_text" if it exists, since
    # both fields flow into DB storage.
    if scan_result.entities_found:
        for orig, sanitized_chunk in zip(chunks, sanitized):
            cleaned = orig.get("cleaned_text")
            if cleaned and isinstance(cleaned, str) and isinstance(sanitized_chunk, dict):
                cleaned_scan = middleware.scan_text(cleaned, position="pre_embedding")
                if cleaned_scan.entities_found:
                    sanitized_chunk["cleaned_text"] = cleaned_scan.redacted_text

    total_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Build affected chunk IDs by comparing original vs sanitized text.
    # Avoids a redundant scan pass — if text changed, PII was found.
    chunk_ids_affected: List[str] = []
    if scan_result.entities_found:
        for orig, clean in zip(chunks, sanitized):
            if isinstance(clean, dict) and orig.get("text") != clean.get("text"):
                cid = orig.get("semantic_hash") or orig.get("id")
                if cid:
                    chunk_ids_affected.append(str(cid))

    pii_metadata: Dict[str, Any] = {
        "total_chunks": len(chunks),
        "chunks_with_pii": len(chunk_ids_affected),
        "pii_types_found": scan_result.entities_found,
        "blocked": scan_result.blocked,
        "severity_max": (
            scan_result.severity_max.value if scan_result.severity_max else None
        ),
        "latency_ms": total_ms,
        "chunk_ids_affected": chunk_ids_affected,
    }

    if scan_result.entities_found:
        logger.info(
            "Ingestion PII scan complete: chunks=%d pii_found=%s severity=%s blocked=%s latency=%.1fms",
            len(chunks),
            scan_result.entities_found,
            pii_metadata["severity_max"],
            scan_result.blocked,
            total_ms,
        )
    else:
        logger.debug(
            "Ingestion PII scan clean: chunks=%d latency=%.1fms",
            len(chunks), total_ms,
        )

    return sanitized, pii_metadata


def _empty_pii_metadata() -> Dict[str, Any]:
    """Return a pii_metadata dict for empty input — consistent schema."""
    return {
        "total_chunks": 0,
        "chunks_with_pii": 0,
        "pii_types_found": [],
        "blocked": False,
        "severity_max": None,
        "latency_ms": 0.0,
        "chunk_ids_affected": [],
    }
