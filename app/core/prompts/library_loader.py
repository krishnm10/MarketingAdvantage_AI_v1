"""
Single source of truth for reading Prompt Library JSON files.

Templates live under ``app/core/configs/prompts/{template_id}.json`` — same
directory as [`app/api/v2/prompt_template_api`](../api/v2/prompt_template_api.py).

``system_instructions`` is the only field injected into LLM paths. Few-shot
``examples`` are authoring reference only (see ``examples_note`` in each JSON).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "configs" / "prompts"

assert PROMPTS_DIR.is_dir(), (
    f"Templates directory not found: {PROMPTS_DIR}. Check path in library_loader.py."
)

_DOLLAR_AMOUNT_RE = re.compile(r"\$[\d,]+(?:\.\d{2})?")


def strip_examples_from_instructions(text: str) -> str:
    """
    Last-line safety net for legacy inline examples.

    Migrated templates store examples in a separate JSON field; this should be a
    no-op on current templates.
    """
    if not text:
        return text
    return text.strip()


def strip_and_verify(text: str) -> str:
    """Strip (no-op on migrated templates); revert if new dollar amounts appear."""
    original = text or ""
    stripped = strip_examples_from_instructions(original)
    orig_amounts = set(_DOLLAR_AMOUNT_RE.findall(original))
    stripped_amounts = set(_DOLLAR_AMOUNT_RE.findall(stripped))
    new_amounts = stripped_amounts - orig_amounts
    if new_amounts:
        logger.warning(
            "[PromptLibrary] strip introduced dollar amounts %s; reverting",
            sorted(new_amounts),
        )
        return original
    return stripped


def load_prompt_library_raw(template_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a prompt template JSON by id (full record, including ``examples``).

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
    """Extract primary instruction text; never reads ``examples``."""
    if not data:
        return None
    text = data.get("system_instructions")
    if text is None:
        text = data.get("content")
    if text is None:
        return None
    s = str(text).strip()
    return s or None


def load_template(template_id: str) -> Optional[str]:
    """Return ``system_instructions`` only — never reads ``examples``."""
    raw = load_prompt_library_raw(template_id)
    return system_instructions_from_record(raw) if raw else None


def get_system_instructions(template_id: str) -> Optional[str]:
    """Alias for :func:`load_template`."""
    return load_template(template_id)


def validate_all_templates() -> None:
    """Fail fast if any template ``system_instructions`` contains ``$`` amounts."""
    for path in sorted(PROMPTS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(
                f"Prompt template {path.name} is unreadable: {e}"
            ) from e
        instructions = system_instructions_from_record(data)
        if instructions is None:
            raise RuntimeError(
                f"Prompt template {path.name} missing system_instructions"
            )
        hits = _DOLLAR_AMOUNT_RE.findall(instructions)
        if hits:
            raise RuntimeError(
                f"Prompt template {path.name} system_instructions contains "
                f"dollar amounts: {hits[:5]}"
            )


validate_all_templates()
