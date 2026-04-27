"""
Pipeline Template Gallery API
Endpoints for listing, loading, and managing pipeline configuration templates.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v2/pipeline-templates",
    tags=["Pipeline Templates"],
)

_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "core" / "configs" / "pipeline_templates"
_SLUG_RE = __import__("re").compile(r"^[a-z0-9_-]{1,64}$")


def _load_all_templates() -> List[Dict[str, Any]]:
    templates = []
    if not _TEMPLATES_DIR.exists():
        return templates
    for p in sorted(_TEMPLATES_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            templates.append(data)
        except Exception as e:
            logger.warning("[pipeline_template_api] Failed to load %s: %s", p.name, e)
    return templates


def _load_template(template_id: str) -> Optional[Dict[str, Any]]:
    if not _SLUG_RE.match(template_id):
        return None
    p = _TEMPLATES_DIR / f"{template_id}.json"
    if not p.exists():
        return None
    resolved = p.resolve()
    if not str(resolved).startswith(str(_TEMPLATES_DIR.resolve())):
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


@router.get("/")
async def list_templates(
    tag: Optional[str] = Query(None, description="Filter by tag"),
):
    """List all available pipeline templates."""
    templates = _load_all_templates()
    if tag:
        templates = [t for t in templates if tag in t.get("tags", [])]
    summaries = []
    for t in templates:
        summaries.append({
            "template_id": t.get("template_id"),
            "name": t.get("name"),
            "description": t.get("description"),
            "tags": t.get("tags", []),
            "pii_enabled": t.get("pii_enabled", False),
            "recommended_default": t.get("recommended_default", False),
            "version": t.get("version"),
            "created_by": t.get("created_by"),
        })
    return {"templates": summaries, "count": len(summaries)}


@router.get("/{template_id}")
async def get_template(template_id: str):
    """Get a specific pipeline template with full config patch."""
    template = _load_template(template_id)
    if template is None:
        raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found.")
    return template


class SaveTemplateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field("", max_length=500)
    tags: List[str] = Field(default_factory=list)
    pii_enabled: bool = False
    config_patch: Dict[str, Any] = Field(default_factory=dict)


@router.post("/")
async def save_template(req: SaveTemplateRequest):
    """Save current pipeline configuration as a reusable template."""
    import re
    slug = re.sub(r"[^a-z0-9_-]", "_", req.name.lower().strip())[:64]
    if not slug:
        raise HTTPException(status_code=400, detail="Invalid template name.")

    _TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

    template = {
        "template_id": slug,
        "name": req.name,
        "description": req.description,
        "tags": req.tags,
        "pii_enabled": req.pii_enabled,
        "recommended_default": False,
        "version": "1.0",
        "created_by": "user",
        "config_patch": req.config_patch,
    }

    p = _TEMPLATES_DIR / f"{slug}.json"
    p.write_text(json.dumps(template, indent=2), encoding="utf-8")

    return {"status": "saved", "template_id": slug}
