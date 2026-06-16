"""
Public tenant configuration read API (Phase 6).

GET /api/v2/config/{client_id} — sanitized, secret-free tenant config for browsers.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from app.core.config.client_config_resolver import (
    ConfigValidationError,
    get_client_config,
)
from app.core.config.public_tenant_config import PublicTenantConfig
from app.utils.tenant_validator import TenantValidationError, validate_tenant_id

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v2/config",
    tags=["Public Tenant Config"],
)


def _validated_tenant_path(client_id: str, endpoint: str) -> str:
    try:
        ctx = validate_tenant_id(
            client_id,
            source="path",
            endpoint=endpoint,
            allow_default=True,
        )
        return ctx.tenant_id
    except TenantValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{client_id}", response_model=PublicTenantConfig)
async def get_public_tenant_config(client_id: str) -> PublicTenantConfig:
    """
    Return a frontend-safe projection of the tenant ClientConfig.

    Secrets backends, secret refs, internal paths, and credential material
    are stripped via ``ClientConfig.to_public()`` / ``PublicTenantConfig``.
    """
    cid = _validated_tenant_path(client_id, "get_public_tenant_config")

    try:
        config = get_client_config(cid, apply_env=False)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Default configuration not found: {exc}",
        ) from exc
    except ConfigValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Tenant configuration invalid: {exc}",
        ) from exc
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Schema validation failed: {exc}",
        ) from exc

    public = config.to_public()
    # Belt-and-suspenders: assert serialized payload contains no secret keys.
    public.model_dump_public()
    logger.debug(
        "[public_tenant_config_api] Served public config client_id=%r version=%s",
        cid,
        public.version,
    )
    return public
