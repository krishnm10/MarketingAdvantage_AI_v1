"""
Centralized tenant ID validation and normalization.

Every API entry point must validate tenant/client/business IDs through
this module BEFORE any pipeline, config resolution, or database operation.
This is the single source of truth for what constitutes a valid tenant
identifier in the system.

Security guarantees:
  - Empty / whitespace-only IDs are rejected
  - Path traversal characters (.., /, \\) are rejected
  - Reserved internal names are rejected
  - Wildcard patterns are rejected
  - IDs are normalized to lowercase alphanumeric + underscore/hyphen
  - Max length enforced (64 chars)
  - Immutable TenantContext prevents post-validation mutation
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_SAFE_TENANT_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,62}[a-z0-9]$|^[a-z0-9]$")
_STRIP_CHARS_RE = re.compile(r"[^a-z0-9_\-]")

_RESERVED_NAMES = frozenset({
    "__system__", "__internal__", "__admin__", "__test__",
    "system", "internal", "admin", "root", "superuser",
    "null", "none", "undefined", "void",
})

_WILDCARD_PATTERNS = frozenset({
    "*", "%", ".*", "**", "__all__", "any", "all",
})

_MAX_LENGTH = 64


class TenantValidationError(ValueError):
    """Raised when a tenant ID fails validation."""

    def __init__(self, reason: str, *, raw_value: str = "", endpoint: str = ""):
        self.reason = reason
        self.raw_value = raw_value
        self.endpoint = endpoint
        super().__init__(reason)


@dataclass(frozen=True)
class TenantContext:
    """
    Immutable, request-scoped tenant identity.

    Once created by validate_tenant_id(), this object cannot be mutated.
    All downstream code receives this instead of raw strings.
    """
    tenant_id: str
    raw_input: str
    source: str
    endpoint: str


def validate_tenant_id(
    raw_id: Optional[str],
    *,
    source: str = "body",
    endpoint: str = "unknown",
    allow_default: bool = False,
) -> TenantContext:
    """
    Validate, normalize, and wrap a tenant ID into an immutable TenantContext.

    Args:
        raw_id:        The raw tenant/client/business ID from the request.
        source:        Where the ID came from: "body", "path", "query", "header".
        endpoint:      The API endpoint name (for telemetry).
        allow_default: If True, None/empty maps to "default" (single-tenant compat).
                       If False, None/empty raises TenantValidationError.

    Returns:
        TenantContext with the validated, normalized tenant_id.

    Raises:
        TenantValidationError on any validation failure.
    """
    _raw_for_log = repr(raw_id)[:80] if raw_id else "None"

    # ── Rule 1: Handle missing/empty ──────────────────────────────────
    if raw_id is None or not str(raw_id).strip():
        if allow_default:
            logger.debug(
                '{"event":"TENANT_VALIDATED",'
                '"tenant_id":"default","source":"%s",'
                '"endpoint":"%s","defaulted":true}',
                source, endpoint,
            )
            return TenantContext(
                tenant_id="default",
                raw_input=raw_id or "",
                source=source,
                endpoint=endpoint,
            )
        _reject(
            reason="missing_tenant_id",
            detail="Tenant ID is required but was not provided.",
            raw_value=_raw_for_log,
            endpoint=endpoint,
        )

    raw = str(raw_id).strip()

    # ── Rule 2: Path traversal ────────────────────────────────────────
    if ".." in raw or "/" in raw or "\\" in raw:
        _reject(
            reason="path_traversal",
            detail="Tenant ID contains path traversal characters.",
            raw_value=_raw_for_log,
            endpoint=endpoint,
        )

    # ── Rule 3: Wildcard patterns (check raw before normalization) ────
    if raw.lower() in _WILDCARD_PATTERNS:
        _reject(
            reason="wildcard_pattern",
            detail=f"Tenant ID '{raw}' matches a wildcard pattern.",
            raw_value=_raw_for_log,
            endpoint=endpoint,
        )

    # ── Rule 4: Normalize ─────────────────────────────────────────────
    normalized = _STRIP_CHARS_RE.sub("", raw.lower())

    if not normalized:
        _reject(
            reason="empty_after_normalization",
            detail="Tenant ID contains no valid characters after normalization.",
            raw_value=_raw_for_log,
            endpoint=endpoint,
        )

    # ── Rule 4: Length limit ──────────────────────────────────────────
    if len(normalized) > _MAX_LENGTH:
        normalized = normalized[:_MAX_LENGTH]

    # ── Rule 5: Wildcard patterns (post-normalization check) ───────────
    if normalized in _WILDCARD_PATTERNS:
        _reject(
            reason="wildcard_pattern",
            detail=f"Tenant ID '{normalized}' matches a wildcard pattern.",
            raw_value=_raw_for_log,
            endpoint=endpoint,
        )

    # ── Rule 6: Reserved names ────────────────────────────────────────
    if normalized in _RESERVED_NAMES:
        _reject(
            reason="reserved_name",
            detail=f"Tenant ID '{normalized}' is a reserved internal name.",
            raw_value=_raw_for_log,
            endpoint=endpoint,
        )

    # ── Rule 7: Format validation ─────────────────────────────────────
    if not _SAFE_TENANT_RE.match(normalized):
        _reject(
            reason="invalid_format",
            detail=(
                f"Tenant ID '{normalized}' does not match the required format "
                "(lowercase alphanumeric, underscore, hyphen; must start/end "
                "with alphanumeric)."
            ),
            raw_value=_raw_for_log,
            endpoint=endpoint,
        )

    # ── Success ───────────────────────────────────────────────────────
    logger.info(
        '{"event":"TENANT_VALIDATED",'
        '"tenant_id":"%s","source":"%s",'
        '"endpoint":"%s","normalized":%s}',
        normalized, source, endpoint,
        str(normalized != raw).lower(),
    )

    return TenantContext(
        tenant_id=normalized,
        raw_input=raw,
        source=source,
        endpoint=endpoint,
    )


def validate_business_id(
    raw_id: Optional[str],
    *,
    endpoint: str = "unknown",
    allow_default: bool = True,
) -> TenantContext:
    """
    Convenience wrapper for ingestion APIs that use 'business_id'
    instead of 'client_id'. Same validation rules apply.
    """
    return validate_tenant_id(
        raw_id,
        source="form",
        endpoint=endpoint,
        allow_default=allow_default,
    )


def _reject(
    *,
    reason: str,
    detail: str,
    raw_value: str,
    endpoint: str,
) -> None:
    """Log structured rejection telemetry and raise."""
    logger.warning(
        '{"event":"TENANT_VALIDATION_REJECTED",'
        '"reason":"%s",'
        '"raw_value":"%s",'
        '"endpoint":"%s",'
        '"validation_result":"rejected"}',
        reason,
        raw_value.replace('"', "'")[:80],
        endpoint,
    )
    try:
        from app.services.security.tenant_audit import log_tenant_validation_failure
        log_tenant_validation_failure(
            tenant_id=raw_value[:30],
            endpoint=endpoint,
            reason=reason,
            source="api_boundary",
            raw_value_hint=raw_value[:30],
        )
    except Exception:
        pass
    raise TenantValidationError(
        detail, raw_value=raw_value, endpoint=endpoint,
    )
