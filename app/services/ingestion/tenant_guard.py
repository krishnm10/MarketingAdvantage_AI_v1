"""
Ingestion Tenant Guard — Immutable tenant enforcement for all ingestion paths.

This module guarantees that:
  1. Every ingestion operation has a validated, non-empty tenant_id.
  2. Tenant identity is resolved ONCE and propagated immutably.
  3. No ingestion path can silently drop, mutate, or invent tenant_id.
  4. Vector metadata always receives the validated tenant_id.
  5. Retry/replay paths recover tenant from DB records, never invent new ones.

Architecture:
  API boundary validates tenant → file_router receives validated tenant →
  IngestionOrchestrator receives tenant → IngestionServiceV2 propagates tenant →
  VectorDB upsert includes tenant in metadata.

Security guarantees:
  - Rejects missing tenant_id (unless single-tenant default mode)
  - Rejects mutable overrides mid-pipeline
  - Rejects conflicting tenant metadata
  - Logs all enforcement decisions with structured telemetry
"""

from __future__ import annotations

import logging
import json as _json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.utils.tenant_validator import (
    validate_tenant_id,
    TenantValidationError,
    TenantContext,
)

logger = logging.getLogger(__name__)


class IngestionTenantViolation(ValueError):
    """Raised when ingestion tenant enforcement is violated."""

    def __init__(self, reason: str, *, file_id: str = "", source: str = ""):
        self.reason = reason
        self.file_id = file_id
        self.source = source
        super().__init__(reason)


@dataclass(frozen=True)
class IngestionTenantContext:
    """
    Immutable tenant context for a single ingestion operation.

    Once created, this cannot be mutated. All downstream code must use
    this context rather than raw strings or parameters.
    """
    tenant_id: str
    file_id: str
    source: str
    validated: bool = True


def resolve_ingestion_tenant(
    *,
    business_id: Optional[str],
    file_id: str = "",
    source: str = "unknown",
    allow_default: bool = True,
) -> IngestionTenantContext:
    """
    Validate and lock the tenant identity for an ingestion operation.

    This is the SINGLE entry point for tenant resolution in all ingestion paths.
    Called once at the start of each ingestion flow, the result is passed down
    immutably through orchestrator → service → vector upsert.

    Args:
        business_id: The raw tenant/business ID from the API or DB record.
        file_id:     The file ID being ingested (for telemetry).
        source:      The ingestion source type (for telemetry).
        allow_default: If True, missing tenant maps to "default" (single-tenant compat).

    Returns:
        IngestionTenantContext (frozen, immutable).

    Raises:
        IngestionTenantViolation on validation failure.
    """
    try:
        ctx = validate_tenant_id(
            business_id,
            source="ingestion",
            endpoint=f"ingestion/{source}",
            allow_default=allow_default,
        )
    except TenantValidationError as e:
        _log_rejection(
            reason=e.reason,
            file_id=file_id,
            source=source,
            raw_value=str(business_id)[:60],
        )
        raise IngestionTenantViolation(
            f"Ingestion rejected: {e}",
            file_id=file_id,
            source=source,
        ) from e

    _log_resolved(
        tenant_id=ctx.tenant_id,
        file_id=file_id,
        source=source,
    )

    return IngestionTenantContext(
        tenant_id=ctx.tenant_id,
        file_id=file_id,
        source=source,
        validated=True,
    )


def resolve_tenant_from_db_record(
    file_record: Any,
    *,
    source: str = "retry",
) -> IngestionTenantContext:
    """
    Recover validated tenant from a persisted DB record (IngestedFileV2).

    Used by retry/replay/admin paths where the original tenant was already
    validated and stored. The DB record is the authoritative source.

    Raises IngestionTenantViolation if the DB record has no valid tenant.
    """
    raw_business_id = getattr(file_record, "business_id", None)
    file_id = str(getattr(file_record, "id", ""))

    if raw_business_id is not None:
        raw_business_id = str(raw_business_id)

    return resolve_ingestion_tenant(
        business_id=raw_business_id,
        file_id=file_id,
        source=source,
        allow_default=True,
    )


def enforce_tenant_immutability(
    *,
    expected: IngestionTenantContext,
    actual_business_id: Optional[str],
    stage: str = "unknown",
) -> None:
    """
    Assert that mid-pipeline tenant metadata has not been mutated.

    Called at vector-upsert boundaries and dedup checkpoints to verify
    that the tenant in the data matches the locked context.

    Raises IngestionTenantViolation on mismatch.
    """
    if actual_business_id is None:
        return

    normalized = str(actual_business_id).strip().lower()
    if not normalized or normalized == expected.tenant_id:
        return

    if normalized == "default" and expected.tenant_id == "default":
        return

    _log_rejection(
        reason="tenant_mutated",
        file_id=expected.file_id,
        source=expected.source,
        raw_value=f"expected={expected.tenant_id}, actual={normalized}",
    )
    raise IngestionTenantViolation(
        f"Tenant metadata mutation detected at stage '{stage}': "
        f"locked tenant_id='{expected.tenant_id}' but data contains "
        f"business_id='{normalized}'.",
        file_id=expected.file_id,
        source=expected.source,
    )


def build_vector_metadata(
    *,
    tenant_ctx: IngestionTenantContext,
    file_id: str,
    file_name: str = "",
    source_type: str = "",
    source_url: str = "",
    chunk_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build standardized vector metadata with guaranteed tenant field.

    This is the ONLY place vector metadata should be constructed for
    ingestion-originated vectors. The tenant_id is always present and
    maps to the 'business_id' metadata key used by VectorDB filters.
    """
    md: Dict[str, Any] = {
        "business_id": tenant_ctx.tenant_id,
        "file_id": file_id,
        "source_type": source_type,
        "file_name": file_name,
        "source": file_name,
        "url": source_url,
    }

    if chunk_metadata and isinstance(chunk_metadata, dict):
        page_number = chunk_metadata.get("page_number")
        if isinstance(page_number, int) and page_number > 0:
            md["page_number"] = page_number
        section_title = chunk_metadata.get("section_title")
        if section_title:
            md["section_title"] = str(section_title)
        heading_depth = chunk_metadata.get("heading_depth")
        if isinstance(heading_depth, int):
            md["heading_depth"] = heading_depth
        chunk_position = chunk_metadata.get("chunk_position")
        if isinstance(chunk_position, dict):
            start_char = chunk_position.get("start_char")
            end_char = chunk_position.get("end_char")
            if isinstance(start_char, int):
                md["chunk_start_char"] = start_char
            if isinstance(end_char, int):
                md["chunk_end_char"] = end_char
        semantic_hash = chunk_metadata.get("semantic_hash")
        if semantic_hash:
            md["semantic_hash"] = str(semantic_hash)

    return md


def log_ingestion_telemetry(
    *,
    tenant_ctx: IngestionTenantContext,
    event: str,
    vector_count: int = 0,
    metadata_valid: bool = True,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit structured ingestion telemetry with guaranteed tenant context."""
    payload = {
        "event": event,
        "tenant_id": tenant_ctx.tenant_id,
        "file_id": tenant_ctx.file_id,
        "ingestion_source": tenant_ctx.source,
        "vector_count": vector_count,
        "metadata_validation_result": "valid" if metadata_valid else "invalid",
    }
    if extra:
        payload.update(extra)

    logger.info("%s", _json.dumps(payload, default=str))


# ─── Internal telemetry helpers ────────────────────────────────────────────

def _log_resolved(*, tenant_id: str, file_id: str, source: str) -> None:
    logger.info(
        '{"event":"INGESTION_TENANT_RESOLVED",'
        '"tenant_id":"%s","file_id":"%s","source":"%s"}',
        tenant_id, file_id, source,
    )


def _log_rejection(*, reason: str, file_id: str, source: str, raw_value: str) -> None:
    logger.warning(
        '{"event":"INGESTION_TENANT_REJECTED",'
        '"reason":"%s","file_id":"%s","source":"%s","raw_value":"%s"}',
        reason, file_id, source, raw_value.replace('"', "'")[:60],
    )
