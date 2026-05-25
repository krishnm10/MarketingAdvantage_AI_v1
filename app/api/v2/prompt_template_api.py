"""
================================================================================
Marketing Advantage AI — Prompt Template Library API
File: app/api/v2/prompt_template_api.py

Endpoints:
  GET    /api/v2/prompt-templates              → List all templates
  POST   /api/v2/prompt-templates              → Create a template
  GET    /api/v2/prompt-templates/{id}         → Get a template by ID
  PUT    /api/v2/prompt-templates/{id}         → Update a template
  DELETE /api/v2/prompt-templates/{id}         → Delete a template
  POST   /api/v2/prompt-templates/{id}/preview → Preview rendered output

Design:
  - Templates are persisted as JSON files in app/core/configs/prompts/.
  - All template IDs are URL-safe slugs (alphanumeric + hyphens).
  - Previews render against sample context chunks so engineers can validate
    templates before associating them with a client pipeline.
  - All mutation endpoints require admin role.
================================================================================
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field, field_validator

from app.ai.contracts.generator_contract import CitationStyle, PromptTemplate
from app.auth.generate_token import verify_access_token
from app.core.prompts.library_loader import (
    PROMPTS_DIR,
    strip_and_verify,
    system_instructions_from_record,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v2/prompt-templates",
    tags=["Prompt Templates"],
)

_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "core" / "configs" / "prompts"
_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9\-]{0,62}[a-z0-9]$")


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def _admin_dep():
    try:
        from app.auth.guards import require_role
        return Depends(require_role("admin"))
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PromptTemplateCreate(BaseModel):
    template_id:          str   = Field(
        ...,
        description="URL-safe slug (lowercase alphanumeric + hyphens, 2–64 chars).",
        examples=["marketing-rag-v1"],
    )
    name:                 str   = Field(..., min_length=2, max_length=128)
    system_instructions:  str   = Field(
        ...,
        min_length=20,
        description="System prompt text defining assistant persona and grounding policy.",
    )
    context_format:       str   = Field(
        "[{index}] {text}\n  [Source: {source}]",
        description="Format string for each context chunk. Supports {index}, {text}, {source}, {page}.",
    )
    question_prefix:      str   = Field("Question:", max_length=64)
    answer_prefix:        str   = Field("Answer:", max_length=64)
    citation_style:       str   = Field(
        "inline_numeric",
        description="'inline_numeric', 'inline_source', or 'none'.",
    )
    max_context_chars:    Optional[int] = Field(
        None,
        ge=500,
        description="Hard cap on total context characters. None = governed by token budget.",
    )
    version:              str   = Field("1.0.0")
    tags:                 List[str] = Field(default_factory=list)
    metadata:             Dict[str, Any] = Field(default_factory=dict)
    examples:             List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Few-shot authoring reference; never injected into LLM paths.",
    )

    @field_validator("template_id")
    @classmethod
    def validate_slug(cls, v: str) -> str:
        if not _SLUG_RE.match(v):
            raise ValueError(
                "template_id must be a lowercase URL-safe slug "
                "(alphanumeric + hyphens, 2–64 chars, no leading/trailing hyphens)."
            )
        return v

    @field_validator("citation_style")
    @classmethod
    def validate_citation(cls, v: str) -> str:
        valid = {cs.value for cs in CitationStyle}
        if v not in valid:
            raise ValueError(f"citation_style must be one of {sorted(valid)}.")
        return v


class PromptTemplateUpdate(BaseModel):
    name:                 Optional[str]  = None
    system_instructions:  Optional[str]  = None
    context_format:       Optional[str]  = None
    question_prefix:      Optional[str]  = None
    answer_prefix:        Optional[str]  = None
    citation_style:       Optional[str]  = None
    max_context_chars:    Optional[int]  = None
    version:              Optional[str]  = None
    tags:                 Optional[List[str]] = None
    metadata:             Optional[Dict[str, Any]] = None
    examples:             Optional[List[Dict[str, Any]]] = None


class PreviewRequest(BaseModel):
    query:   str = Field(..., description="Sample question to render.")
    chunks:  List[Dict[str, Any]] = Field(
        default_factory=lambda: [
            {
                "text": "Sample retrieved passage about marketing ROI metrics.",
                "metadata": {"source": "marketing_report_q3.pdf", "page_number": 12},
            }
        ],
        description="Sample context chunks to render against the template.",
    )


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _template_path(template_id: str) -> Path:
    return _TEMPLATES_DIR / f"{template_id}.json"


def _load_template(template_id: str) -> dict:
    path = _template_path(template_id)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Prompt template '{template_id}' not found.",
        )
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Prompt template '{template_id}' on disk is invalid JSON "
                f"(line {e.lineno}, column {e.colno}). "
                "Repair the file under app/core/configs/prompts/ or restore from backup."
            ),
        ) from e


def _save_template(data: dict) -> None:
    path = _template_path(data["template_id"])
    tmp = path.with_suffix(path.suffix + ".tmp")
    body = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)


def _has_examples(data: dict) -> bool:
    examples = data.get("examples")
    return isinstance(examples, list) and len(examples) > 0


def _template_description(data: dict) -> str:
    meta = data.get("metadata")
    if isinstance(meta, dict) and meta.get("description"):
        return str(meta["description"])
    desc = data.get("description")
    return str(desc) if desc else ""


_oauth2_optional = OAuth2PasswordBearer(tokenUrl="/api/v2/auth/token", auto_error=False)


def _require_admin_token(token: Optional[str]) -> dict:
    """Same role gate as PATCH tenant prompt-config (admin only)."""
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = verify_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if payload.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Access forbidden: insufficient role privileges",
        )
    return payload


def _examples_list(data: dict) -> List[Dict[str, Any]]:
    examples = data.get("examples")
    return examples if isinstance(examples, list) else []


def _build_raw_template_response(data: dict, template_id: str) -> dict:
    """Full on-disk record for Prompt Library authoring (no strip_and_verify)."""
    instructions = system_instructions_from_record(data) or ""
    return {
        "template_id": data.get("template_id", template_id),
        "name": data.get("name"),
        "system_instructions": instructions,
        "context_format": data.get(
            "context_format", "[{index}] {text}\n  [Source: {source}]"
        ),
        "question_prefix": data.get("question_prefix", "Question:"),
        "answer_prefix": data.get("answer_prefix", "Answer:"),
        "citation_style": data.get("citation_style", "inline_numeric"),
        "max_context_chars": data.get("max_context_chars"),
        "version": data.get("version", "1.0.0"),
        "tags": data.get("tags", []),
        "metadata": data.get("metadata", {}),
        "examples": _examples_list(data),
        "examples_note": data.get(
            "examples_note",
            "Authoring reference only. Never injected into LLM.",
        ),
        "has_examples": _has_examples(data),
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
    }


def _build_prompt_template(data: dict) -> PromptTemplate:
    return PromptTemplate(
        template_id=data["template_id"],
        name=data["name"],
        system_instructions=data["system_instructions"],
        context_format=data.get("context_format", "[{index}] {text}\n  [Source: {source}]"),
        question_prefix=data.get("question_prefix", "Question:"),
        answer_prefix=data.get("answer_prefix", "Answer:"),
        citation_style=CitationStyle(data.get("citation_style", "inline_numeric")),
        max_context_chars=data.get("max_context_chars"),
        version=data.get("version", "1.0.0"),
        tags=data.get("tags", []),
        metadata=data.get("metadata", {}),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("", response_model=List[dict])
async def list_templates():
    """List all prompt templates in the library."""
    results = []
    for path in sorted(PROMPTS_DIR.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            results.append({
                "template_id": data.get("template_id"),
                "name": data.get("name"),
                "description": _template_description(data),
                "has_examples": _has_examples(data),
            })
        except Exception as e:
            logger.warning("Skipping malformed template %s: %s", path.name, e)
    return results


@router.post("", response_model=dict, status_code=201)
async def create_template(req: PromptTemplateCreate):
    """Create a new prompt template."""
    path = _template_path(req.template_id)
    if path.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Template '{req.template_id}' already exists. Use PUT to update.",
        )

    now = int(time.time())
    data = req.model_dump()
    data["created_at"] = now
    data["updated_at"] = now
    _save_template(data)

    logger.info(
        "[PromptTemplateAPI] Created template='%s' version=%s",
        req.template_id, req.version,
    )
    return {"status": "created", **data}


@router.get("/{template_id}", response_model=dict)
async def get_template(
    template_id: str,
    raw: bool = Query(
        False,
        description="Admin-only: return full on-disk template including examples.",
    ),
    token: Optional[str] = Depends(_oauth2_optional),
):
    """Get a prompt template by ID.

    Default (``raw=false``): stripped ``system_instructions`` for LLM preview; no examples.
    ``raw=true``: admin auth required; full authoring record including ``examples``.
    """
    data = _load_template(template_id)
    if raw:
        _require_admin_token(token)
        return _build_raw_template_response(data, template_id)

    instructions = system_instructions_from_record(data) or ""
    return {
        "template_id": data.get("template_id", template_id),
        "name": data.get("name"),
        "system_instructions": strip_and_verify(instructions),
        "has_examples": _has_examples(data),
        "examples_note": data.get(
            "examples_note",
            "Authoring reference only. Never injected into LLM.",
        ),
    }


@router.put("/{template_id}", response_model=dict)
async def update_template(template_id: str, req: PromptTemplateUpdate):
    """Update fields of an existing prompt template."""
    data = _load_template(template_id)

    updates = req.model_dump(exclude_none=True)
    if "citation_style" in updates:
        valid = {cs.value for cs in CitationStyle}
        if updates["citation_style"] not in valid:
            raise HTTPException(
                status_code=422,
                detail=f"citation_style must be one of {sorted(valid)}.",
            )
    data.update(updates)
    data["updated_at"] = int(time.time())
    _save_template(data)

    logger.info("[PromptTemplateAPI] Updated template='%s'", template_id)
    return {"status": "updated", **data}


@router.delete("/{template_id}", response_model=dict)
async def delete_template(template_id: str):
    """Delete a prompt template."""
    path = _template_path(template_id)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Template '{template_id}' not found.",
        )
    path.unlink()
    logger.info("[PromptTemplateAPI] Deleted template='%s'", template_id)
    return {"status": "deleted", "template_id": template_id}


@router.post("/{template_id}/preview", response_model=dict)
async def preview_template(template_id: str, req: PreviewRequest):
    """
    Preview how a template renders with sample query and context chunks.
    Useful for validating templates before associating with a pipeline.
    """
    data = _load_template(template_id)
    tmpl = _build_prompt_template(data)

    from app.ai.contracts.reranker_contract import ScoredCandidate
    import dataclasses

    mock_chunks = [
        ScoredCandidate(
            chunk_id=str(i),
            text=c.get("text", ""),
            vector_score=0.9,
            rerank_score=0.9,
            metadata=c.get("metadata", {}),
        )
        for i, c in enumerate(req.chunks)
    ]

    system_prompt  = tmpl.render_system_prompt()
    context_text   = tmpl.render_context(mock_chunks)
    full_prompt    = tmpl.render_full_prompt(req.query, mock_chunks)
    estimated_chars = len(system_prompt) + len(full_prompt)

    return {
        "template_id":      template_id,
        "system_prompt":    system_prompt,
        "rendered_context": context_text,
        "full_user_turn":   full_prompt,
        "estimated_chars":  estimated_chars,
        "estimated_tokens": estimated_chars // 4,
    }
