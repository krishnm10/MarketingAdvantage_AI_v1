"""
Tenant Isolation Audit & Telemetry — Centralized Instrumentation

This module is the SINGLE source of structured telemetry for all tenant
isolation operations. Every component that enforces, validates, or operates
on tenant boundaries MUST emit events through this module.

Coverage:
  - VectorDB searches (tenant-filtered queries)
  - Ingestion writes (tenant-scoped vector upserts)
  - Retrieval filtering (post-processor, reranker, context window)
  - Tenant validation failures (API boundary rejections)
  - Cross-tenant override attempts (security violations)
  - Degraded isolation fallback paths

Safety guarantees (NEVER logged):
  - Raw document content or chunk text
  - PII values (names, emails, phone numbers, IDs)
  - Embedding vectors (float arrays)
  - API keys or secrets

All events are JSON-structured with stable field names for log aggregation,
alerting rules, and compliance dashboards.
"""

from __future__ import annotations

import json
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tenant_audit")


class AuditSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    SECURITY = "security"
    CRITICAL = "critical"


class AuditCategory(str, Enum):
    VECTORDB_SEARCH = "vectordb_search"
    INGESTION_WRITE = "ingestion_write"
    RETRIEVAL_FILTER = "retrieval_filter"
    TENANT_VALIDATION = "tenant_validation"
    CROSS_TENANT_ATTEMPT = "cross_tenant_attempt"
    DEGRADED_ISOLATION = "degraded_isolation"


# ─────────────────────────────────────────────────────────────────────────────
# Core emit function — all events flow through here
# ─────────────────────────────────────────────────────────────────────────────

def _emit(
    *,
    event: str,
    category: AuditCategory,
    severity: AuditSeverity,
    tenant_id: str,
    payload: Dict[str, Any],
) -> None:
    """
    Emit a single structured audit event.

    All values in payload are sanitized to prevent PII/content leakage.
    """
    record = {
        "event": event,
        "category": category.value,
        "severity": severity.value,
        "tenant_id": _safe_tenant(tenant_id),
        "timestamp_ms": int(time.time() * 1000),
        **{k: _sanitize_value(k, v) for k, v in payload.items()},
    }

    log_line = json.dumps(record, default=str, separators=(",", ":"))

    if severity == AuditSeverity.CRITICAL:
        logger.critical(log_line)
    elif severity == AuditSeverity.SECURITY:
        logger.warning(log_line)
    elif severity == AuditSeverity.WARNING:
        logger.warning(log_line)
    else:
        logger.info(log_line)


# ─────────────────────────────────────────────────────────────────────────────
# VectorDB Search Telemetry
# ─────────────────────────────────────────────────────────────────────────────

def log_vectordb_search(
    *,
    tenant_id: str,
    collection: str,
    search_mode: str = "vector",
    backend: str = "unknown",
    filter_count: int = 0,
    retrieval_count: int = 0,
    latency_ms: float = 0.0,
    isolation_enforced: bool = True,
) -> None:
    """Emit telemetry for a tenant-scoped VectorDB search operation."""
    _emit(
        event="TENANT_VECTORDB_SEARCH",
        category=AuditCategory.VECTORDB_SEARCH,
        severity=AuditSeverity.INFO,
        tenant_id=tenant_id,
        payload={
            "collection": collection,
            "search_mode": search_mode,
            "backend": backend,
            "filter_count": filter_count,
            "retrieval_count": retrieval_count,
            "latency_ms": round(latency_ms, 2),
            "isolation_enforced": isolation_enforced,
            "filter_summary": f"tenant={tenant_id},filters={filter_count}",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion Write Telemetry
# ─────────────────────────────────────────────────────────────────────────────

def log_ingestion_write(
    *,
    tenant_id: str,
    file_id: str,
    collection: str = "",
    vector_count: int = 0,
    source_type: str = "file",
    dedup_skipped: int = 0,
    metadata_valid: bool = True,
) -> None:
    """Emit telemetry for a tenant-scoped ingestion vector write."""
    _emit(
        event="TENANT_INGESTION_WRITE",
        category=AuditCategory.INGESTION_WRITE,
        severity=AuditSeverity.INFO,
        tenant_id=tenant_id,
        payload={
            "file_id": file_id,
            "collection": collection,
            "vector_count": vector_count,
            "source_type": source_type,
            "dedup_skipped": dedup_skipped,
            "metadata_valid": metadata_valid,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Filtering Telemetry
# ─────────────────────────────────────────────────────────────────────────────

def log_retrieval_filter(
    *,
    tenant_id: str,
    stage: str,
    input_count: int = 0,
    output_count: int = 0,
    filtered_count: int = 0,
    filter_reason: str = "",
    search_mode: str = "semantic",
    collection: str = "",
) -> None:
    """Emit telemetry for retrieval-time filtering (post-processor, reranker, CWM)."""
    _emit(
        event="TENANT_RETRIEVAL_FILTER",
        category=AuditCategory.RETRIEVAL_FILTER,
        severity=AuditSeverity.INFO,
        tenant_id=tenant_id,
        payload={
            "stage": stage,
            "input_count": input_count,
            "output_count": output_count,
            "filtered_count": filtered_count,
            "filter_reason": filter_reason,
            "search_mode": search_mode,
            "collection": collection,
            "filter_summary": f"stage={stage},in={input_count},out={output_count}",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tenant Validation Failure Telemetry
# ─────────────────────────────────────────────────────────────────────────────

def log_tenant_validation_failure(
    *,
    tenant_id: str,
    endpoint: str,
    reason: str,
    source: str = "unknown",
    raw_value_hint: str = "",
) -> None:
    """Emit audit event for a rejected tenant validation attempt."""
    _emit(
        event="TENANT_VALIDATION_FAILURE",
        category=AuditCategory.TENANT_VALIDATION,
        severity=AuditSeverity.SECURITY,
        tenant_id=tenant_id or "NONE",
        payload={
            "endpoint": endpoint,
            "reason": reason,
            "source": source,
            "raw_value_hint": _truncate(raw_value_hint, 30),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Tenant Override Attempt Telemetry
# ─────────────────────────────────────────────────────────────────────────────

def log_cross_tenant_attempt(
    *,
    tenant_id: str,
    attempted_target: str,
    endpoint: str = "",
    caller: str = "",
    collection: str = "",
) -> None:
    """Emit critical audit event for a cross-tenant access attempt."""
    _emit(
        event="CROSS_TENANT_OVERRIDE_ATTEMPT",
        category=AuditCategory.CROSS_TENANT_ATTEMPT,
        severity=AuditSeverity.CRITICAL,
        tenant_id=tenant_id,
        payload={
            "attempted_target": _safe_tenant(attempted_target),
            "endpoint": endpoint,
            "caller": caller,
            "collection": collection,
            "blocked": True,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Degraded Isolation Fallback Telemetry
# ─────────────────────────────────────────────────────────────────────────────

def log_degraded_isolation(
    *,
    tenant_id: str,
    reason: str,
    component: str,
    fallback_action: str = "bypass_disabled",
    collection: str = "",
) -> None:
    """Emit warning for degraded tenant isolation (single-tenant bypass, etc.)."""
    _emit(
        event="TENANT_ISOLATION_DEGRADED",
        category=AuditCategory.DEGRADED_ISOLATION,
        severity=AuditSeverity.WARNING,
        tenant_id=tenant_id or "NONE",
        payload={
            "reason": reason,
            "component": component,
            "fallback_action": fallback_action,
            "collection": collection,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Missing Tenant Context Telemetry
# ─────────────────────────────────────────────────────────────────────────────

def log_missing_tenant_context(
    *,
    component: str,
    operation: str,
    fallback_used: str = "default",
) -> None:
    """Emit warning when a component operates without explicit tenant context."""
    _emit(
        event="TENANT_CONTEXT_MISSING",
        category=AuditCategory.DEGRADED_ISOLATION,
        severity=AuditSeverity.WARNING,
        tenant_id="MISSING",
        payload={
            "component": component,
            "operation": operation,
            "fallback_used": fallback_used,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Safety helpers — NEVER log PII, content, or embeddings
# ─────────────────────────────────────────────────────────────────────────────

_FORBIDDEN_KEYS = frozenset({
    "text", "content", "chunk_text", "raw_text", "cleaned_text",
    "document", "body", "message", "embedding", "embeddings",
    "vector", "vectors", "query_text", "answer",
    "email", "phone", "ssn", "address", "name", "password",
    "api_key", "secret", "token", "credential",
})

_TRUNCATE_KEYS = frozenset({
    "raw_value", "raw_value_hint", "detail", "error",
})


def _sanitize_value(key: str, value: Any) -> Any:
    """
    Sanitize a value before logging.
    Strips content, PII, and embedding data from log payloads.
    """
    key_lower = key.lower()

    if key_lower in _FORBIDDEN_KEYS:
        return "[REDACTED]"

    if key_lower in _TRUNCATE_KEYS:
        return _truncate(str(value), 60)

    if isinstance(value, (list, tuple)) and len(value) > 0:
        if isinstance(value[0], float):
            return f"[vector_dim={len(value)}]"
        if len(value) > 20:
            return f"[list_len={len(value)}]"

    if isinstance(value, str) and len(value) > 200:
        return value[:200] + "..."

    return value


def _safe_tenant(tenant_id: Optional[str]) -> str:
    """Ensure tenant_id is safe to log (no PII, bounded length)."""
    if not tenant_id:
        return "NONE"
    safe = str(tenant_id).strip()[:64]
    if any(c in safe for c in ('\n', '\r', '"', '\\')):
        return safe.replace('\n', '').replace('\r', '').replace('"', '').replace('\\', '')[:64]
    return safe


def _truncate(value: str, max_len: int = 60) -> str:
    """Truncate a string for safe logging."""
    if len(value) <= max_len:
        return value
    return value[:max_len] + "..."
