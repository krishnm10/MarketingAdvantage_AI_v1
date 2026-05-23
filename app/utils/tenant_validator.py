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

Tenant Identity Model:
======================
This system uses a two-key tenant identity model:

1. **Tenant Slug** (e.g. "acme_corp", "default")
   - Human-readable identifier used in APIs, configs, and UI
   - Validated and normalized by validate_tenant_id()
   - Used for: config resolution, pipeline building, logging

2. **Storage UUID** (derived from slug via uuid5)
   - Deterministic UUID derived from tenant slug
   - Stored in: IngestedFileV2.business_id, IngestedContentV2.business_id,
     vector metadata "business_id" field
   - Used for: DB filtering, vector search tenant isolation
   - Derived via: storage_business_uuid_for_tenant(slug)

Canonical key for filtering:
  - VectorDB searches filter on "business_id" = str(storage_uuid)
  - SQL queries filter on business_id = storage_uuid (UUID column)
  - Always derive storage_uuid from validated tenant slug using get_storage_uuid()

Enforcement Modes:
==================
Controlled by TENANT_ENFORCEMENT_MODE env var:
  - "off"    : Legacy behavior, allow missing tenant_id (dev only)
  - "warn"   : Log warnings for missing/invalid tenant but continue
  - "strict" : Reject requests without valid tenant_id (production)
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.auth.deps import UserPayload

logger = logging.getLogger(__name__)


# =============================================================================
# Enforcement Mode Configuration
# =============================================================================

class TenantEnforcementMode(str, Enum):
    """Controls strictness of tenant validation."""
    OFF = "off"       # Legacy: allow missing tenant, default to "default"
    WARN = "warn"     # Log warnings but allow operation to continue
    STRICT = "strict" # Reject requests without valid tenant


def get_enforcement_mode() -> TenantEnforcementMode:
    """
    Get current tenant enforcement mode from environment.
    
    Set via TENANT_ENFORCEMENT_MODE env var.
    Default: "strict" (production-safe; rejects missing tenant_id).
    """
    raw = os.getenv("TENANT_ENFORCEMENT_MODE", "strict").strip().lower()
    try:
        return TenantEnforcementMode(raw)
    except ValueError:
        logger.warning(
            "Invalid TENANT_ENFORCEMENT_MODE '%s', defaulting to 'strict'", raw
        )
        return TenantEnforcementMode.STRICT


# Fixed namespace UUID for tenant → storage UUID derivation (RFC 4122 uuid5)
MAI_TENANT_NAMESPACE = uuid.UUID("321e4567-e89b-12d3-a456-426614174001")

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


# =============================================================================
# Strict Validation (Phase 1: Mandatory tenant enforcement)
# =============================================================================

def validate_tenant_id_strict(
    raw_id: Optional[str],
    *,
    source: str = "body",
    endpoint: str = "unknown",
) -> TenantContext:
    """
    Validate tenant ID with strict enforcement - NEVER allows default fallback.
    
    Use this for all tenant-scoped operations where a specific tenant is required:
    - Retrieval queries
    - Chat endpoints
    - File uploads
    - Ingestion admin operations
    
    Behavior depends on TENANT_ENFORCEMENT_MODE:
    - "strict": Raises TenantValidationError if tenant_id is missing/invalid
    - "warn": Logs warning, falls back to "default" (allows operation to continue)
    - "off": Falls back to "default" silently (legacy compatibility)
    
    Args:
        raw_id:   The raw tenant/client/business ID from the request.
        source:   Where the ID came from: "body", "path", "query", "header".
        endpoint: The API endpoint name (for telemetry).
    
    Returns:
        TenantContext with the validated, normalized tenant_id.
    
    Raises:
        TenantValidationError if tenant_id is missing/invalid and mode is "strict".
    """
    mode = get_enforcement_mode()
    
    # Check if tenant_id is missing/empty
    if raw_id is None or not str(raw_id).strip():
        if mode == TenantEnforcementMode.STRICT:
            _reject(
                reason="missing_tenant_id_strict",
                detail=(
                    "Tenant ID (client_id) is required for this operation. "
                    "Please provide a valid client_id in your request."
                ),
                raw_value="None",
                endpoint=endpoint,
            )
        elif mode == TenantEnforcementMode.WARN:
            logger.warning(
                '{"event":"TENANT_MISSING_WARN",'
                '"endpoint":"%s","source":"%s",'
                '"message":"tenant_id missing, falling back to default"}',
                endpoint, source,
            )
        # For WARN and OFF modes, fall through to allow_default=True
        return validate_tenant_id(
            raw_id,
            source=source,
            endpoint=endpoint,
            allow_default=True,
        )
    
    # Tenant was provided, validate normally (no default fallback)
    return validate_tenant_id(
        raw_id,
        source=source,
        endpoint=endpoint,
        allow_default=False,
    )


# =============================================================================
# Storage UUID Derivation (for DB and vector filtering)
# =============================================================================

def get_storage_uuid(tenant_ctx: TenantContext) -> uuid.UUID:
    """
    Derive the deterministic storage UUID for a validated tenant.
    
    This UUID is used for:
    - IngestedFileV2.business_id (UUID column)
    - IngestedContentV2.business_id (stored as string)
    - Vector metadata "business_id" field
    - All tenant-scoped DB and vector search filters
    
    Args:
        tenant_ctx: Validated TenantContext from validate_tenant_id*()
    
    Returns:
        UUID derived from the tenant slug via uuid5.
    """
    slug = (tenant_ctx.tenant_id or "default").strip().lower()
    if not slug:
        slug = "default"
    return uuid.uuid5(MAI_TENANT_NAMESPACE, f"mai:tenant:{slug}")


def get_storage_uuid_str(tenant_ctx: TenantContext) -> str:
    """
    Get the storage UUID as a string for use in vector metadata filters.
    
    Args:
        tenant_ctx: Validated TenantContext from validate_tenant_id*()
    
    Returns:
        String representation of the storage UUID (e.g. for Chroma where clause).
    """
    return str(get_storage_uuid(tenant_ctx))


# =============================================================================
# Tenant Access Authorization (Phase 1.3: User ↔ Tenant binding)
# =============================================================================

class TenantAccessDeniedError(Exception):
    """Raised when a user attempts to access a tenant they are not authorized for."""
    
    def __init__(self, user_id: str, tenant_id: str, reason: str = ""):
        self.user_id = user_id
        self.tenant_id = tenant_id
        self.reason = reason
        super().__init__(
            f"User '{user_id}' is not authorized to access tenant '{tenant_id}'"
            + (f": {reason}" if reason else "")
        )


def enforce_tenant_access(
    user: "UserPayload",
    tenant_ctx: TenantContext,
    *,
    endpoint: str = "unknown",
) -> None:
    """
    Verify that a user is authorized to access the requested tenant.
    
    Current implementation:
    - Admin users can access all tenants
    - Non-admin users can only access tenants they are explicitly assigned to
      (via user.allowed_tenants list, if present)
    
    Future enhancements:
    - Add tenant ↔ user mapping table in DB
    - Support team/org-based tenant access
    - Add audit logging for access attempts
    
    Args:
        user:       The authenticated user (from require_role dependency).
        tenant_ctx: The validated tenant context from the request.
        endpoint:   API endpoint name for logging.
    
    Raises:
        TenantAccessDeniedError if user cannot access the tenant.
    """
    # Admin users can access all tenants (current behavior)
    if hasattr(user, "role") and user.role == "admin":
        logger.debug(
            '{"event":"TENANT_ACCESS_GRANTED",'
            '"user":"%s","tenant":"%s","reason":"admin_role","endpoint":"%s"}',
            getattr(user, "sub", "unknown"),
            tenant_ctx.tenant_id,
            endpoint,
        )
        return
    
    # Check if user has explicit tenant access list
    allowed_tenants = getattr(user, "allowed_tenants", None)
    
    if allowed_tenants is None:
        # No tenant restriction configured - allow access (backward compatibility)
        logger.debug(
            '{"event":"TENANT_ACCESS_GRANTED",'
            '"user":"%s","tenant":"%s","reason":"no_tenant_restrictions","endpoint":"%s"}',
            getattr(user, "sub", "unknown"),
            tenant_ctx.tenant_id,
            endpoint,
        )
        return
    
    # Check if requested tenant is in user's allowed list
    if tenant_ctx.tenant_id in allowed_tenants:
        logger.debug(
            '{"event":"TENANT_ACCESS_GRANTED",'
            '"user":"%s","tenant":"%s","reason":"in_allowed_list","endpoint":"%s"}',
            getattr(user, "sub", "unknown"),
            tenant_ctx.tenant_id,
            endpoint,
        )
        return
    
    # Access denied - log and raise
    logger.warning(
        '{"event":"TENANT_ACCESS_DENIED",'
        '"user":"%s","tenant":"%s","allowed_tenants":%s,"endpoint":"%s"}',
        getattr(user, "sub", "unknown"),
        tenant_ctx.tenant_id,
        list(allowed_tenants)[:5],  # Log first 5 for debugging
        endpoint,
    )
    
    raise TenantAccessDeniedError(
        user_id=getattr(user, "sub", "unknown"),
        tenant_id=tenant_ctx.tenant_id,
        reason="Tenant not in user's allowed tenant list",
    )
