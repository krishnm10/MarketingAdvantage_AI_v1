"""
Single source of truth for reading Prompt Library JSON files.

Templates live under ``app/core/configs/prompts/{template_id}.json`` — same
directory as [`app/api/v2/prompt_template_api`](../api/v2/prompt_template_api.py).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "configs" / "prompts"


def load_prompt_library_raw(template_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a prompt template JSON by id.

    Returns None for invalid slug, missing file, or unreadable JSON.
    """
    if not template_id or not str(template_id).strip():
        return None
    try:
        from app.utils.path_sanitizer import validate_slug

        safe = validate_slug(str(template_id).strip(), max_length=64)
    except ValueError:
        logger.debug("[PromptLibrary] rejected template_id slug %r", template_id[:48])
        return None

    path = PROMPTS_DIR / f"{safe}.json"
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("[PromptLibrary] failed to read %s: %s", path, e)
        return None


def system_instructions_from_record(data: Dict[str, Any]) -> Optional[str]:
    """Extract primary instruction text (matches rag_api / PromptTemplateCreate)."""
    if not data:
        return None
    text = data.get("system_instructions")
    if text is None:
        text = data.get("content")
    if text is None:
        return None
    s = str(text).strip()
    return s or None


def get_system_instructions(template_id: str) -> Optional[str]:
    """Convenience for callers that only need the system-instructions string."""
    raw = load_prompt_library_raw(template_id)
    return system_instructions_from_record(raw) if raw else None
