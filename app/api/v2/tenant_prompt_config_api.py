"""
Thin tenant prompt-config wrapper over Client JSON (same persistence as rag-config).

GET  /api/v2/tenants/{client_id}/prompt-config  → read prompt_template_id + rewrite_enabled
PATCH /api/v2/tenants/{client_id}/prompt-config → partial update (admin)
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.v2.rag_config_api import (
    _CONFIGS_DIR,
    _atomic_write_json,
    _merged_effective_client_dict,
    _validate_config_dict_raises,
    _validated_tenant_path,
)
from app.auth.guards import require_role
from app.core.config.client_config_resolver import get_client_config
from app.core.prompts.library_loader import PROMPTS_DIR
from app.core.prompts.ssot import enforce_library_first_prompt_persist
from app.utils.path_sanitizer import sanitize_client_id

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v2/tenants",
    tags=["Tenant Prompt Config"],
)


class TenantPromptConfigResponse(BaseModel):
    client_id: str
    prompt_template_id: Optional[str] = None
    rewrite_enabled: bool = False


class TenantPromptConfigPatch(BaseModel):
    prompt_template_id: Optional[str] = None
    rewrite_enabled: Optional[bool] = None


def _validate_template_id_exists(template_id: str) -> None:
    path = PROMPTS_DIR / f"{template_id}.json"
    if not path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown prompt template_id '{template_id}'.",
        )


def _rewrite_enabled_from_config(client_id: str) -> bool:
    try:
        cfg = get_client_config(client_id)
        if cfg.retrieval is not None:
            return bool(cfg.retrieval.rewrite_enabled)
    except Exception:
        pass
    return False


@router.get("/{client_id}/prompt-config", response_model=TenantPromptConfigResponse)
async def get_tenant_prompt_config(client_id: str):
    """Read tenant prompt settings from merged Client JSON."""
    cid = _validated_tenant_path(client_id, "get_tenant_prompt_config")
    cfg = get_client_config(cid)
    template_id = cfg.retrieval.prompt_template_id if cfg.retrieval else None
    return TenantPromptConfigResponse(
        client_id=cid,
        prompt_template_id=template_id,
        rewrite_enabled=_rewrite_enabled_from_config(cid),
    )


@router.patch(
    "/{client_id}/prompt-config",
    response_model=TenantPromptConfigResponse,
    dependencies=[Depends(require_role("admin"))],
)
async def patch_tenant_prompt_config(client_id: str, patch: TenantPromptConfigPatch):
    """Partial update of tenant prompt settings; persists Client JSON."""
    cid = _validated_tenant_path(client_id, "patch_tenant_prompt_config")
    merged = _merged_effective_client_dict(cid)
    retrieval = merged.setdefault("retrieval", {})

    if "prompt_template_id" in patch.model_fields_set:
        tid = patch.prompt_template_id
        if tid is not None:
            tid = str(tid).strip()
            if tid:
                _validate_template_id_exists(tid)
                retrieval["prompt_template_id"] = tid
            else:
                retrieval["prompt_template_id"] = None
        else:
            retrieval["prompt_template_id"] = None

    if "rewrite_enabled" in patch.model_fields_set:
        retrieval["rewrite_enabled"] = bool(patch.rewrite_enabled)

    merged = enforce_library_first_prompt_persist(merged, client_id=cid)
    _validate_config_dict_raises(merged)

    safe_id = sanitize_client_id(cid)
    out_path = (_CONFIGS_DIR / f"{safe_id}.json").resolve()
    if not str(out_path).startswith(str(_CONFIGS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid client_id.")
    _atomic_write_json(out_path, merged)

    from app.core.pipeline_factory import pipeline_factory
    from app.core.config.effective_tenant_runtime import (
        invalidate_effective_tenant_runtime_cache,
    )

    pipeline_factory.invalidate(cid)
    invalidate_effective_tenant_runtime_cache(cid)
    try:
        from app.services.ingestion.ingestion_service_v2 import clear_ingestion_pipeline_cache

        clear_ingestion_pipeline_cache(cid)
    except Exception as e:
        logger.warning(
            "[TenantPromptConfig] Failed to clear ingestion pipeline cache: %s", e
        )

    cfg = get_client_config(cid)
    template_id = cfg.retrieval.prompt_template_id if cfg.retrieval else None
    return TenantPromptConfigResponse(
        client_id=cid,
        prompt_template_id=template_id,
        rewrite_enabled=_rewrite_enabled_from_config(cid),
    )
