"""
Stable UUID representation of validated tenant identifiers for PostgreSQL UUID columns.

IngestedFileV2.business_id is typed as UUID. Raw tenant slugs (e.g. acme_corp) are not
UUIDs — we derive a deterministic UUID v5 per tenant so admins can scope queries without
cross-tenant leakage. When the client submits a literal UUID business_id, that value is preserved.
"""

from __future__ import annotations

import uuid
from typing import Optional

# Fixed namespace UUID for Marketing Advantage tenant derivation (RFC 4128 name-based uuid5).
MAI_TENANT_NAMESPACE = uuid.UUID("321e4567-e89b-12d3-a456-426614174001")


def _parse_literal_uuid(raw: Optional[str]) -> Optional[uuid.UUID]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return uuid.UUID(text)
    except (ValueError, AttributeError):
        return None


def storage_business_uuid_for_tenant(
    validated_tenant_id: str,
    raw_request_business_id: Optional[str] = None,
) -> uuid.UUID:
    """
    Persistable business_id for ingestion file rows.

    - If ``raw_request_business_id`` is already a UUID string/object, returns that UUID (canonicalized).
    - Otherwise derives ``uuid.uuid5(MAI_TENANT_NAMESPACE, "mai:tenant:<tenant>")``.
    """
    literal = _parse_literal_uuid(raw_request_business_id)
    if literal is not None:
        return literal

    slug = (validated_tenant_id or "default").strip().lower()
    if not slug:
        slug = "default"

    return uuid.uuid5(MAI_TENANT_NAMESPACE, f"mai:tenant:{slug}")
