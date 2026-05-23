"""
Prompt SSOT — single resolution path for generation instructions.

Canonical field: ``ClientConfig.retrieval.prompt_template_id`` → Prompt Library JSON.
Legacy fallbacks (preset map, inline template) emit warnings via ``resolve_prompt_ssot``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

from app.core.config.client_config_schema import (
    PROMPT_PREVIEW_MAX_CHARS,
    PromptSSOTSource,
)
from app.core.prompts.library_loader import (
    load_prompt_library_raw,
    system_instructions_from_record,
)

if TYPE_CHECKING:
    from app.core.config.client_config_schema import ClientConfig

logger = logging.getLogger(__name__)

# Immutable last-resort instructions — generation must never hard-crash on missing files.
_EMERGENCY_FALLBACK_PROMPT: str = (
    "You are a helpful assistant. Answer the user's question using ONLY the retrieved "
    "context provided below. If the context does not contain enough information, say so "
    "clearly. Do not invent facts. Cite source numbers [1], [2], etc. when referencing "
    "specific passages."
)

PROMPT_PRESET_LIBRARY: dict[str, str] = {
    "rag_context": "preset-rag-context",
    "cot": "preset-chain-of-thought",
    "few_shot": "preset-few-shot",
    "instruction_tuned": "preset-instruction-tuned",
    "system": "preset-system",
}


@dataclass(frozen=True, slots=True)
class PromptSSOTResolution:
    effective_template_id: Optional[str]
    source: PromptSSOTSource
    configured_prompt_type: Optional[str]
    library_found: bool
    preview: Optional[str]
    legacy_inline_detected: bool
    instructions: str


def emergency_fallback_instructions() -> str:
    """Return the immutable emergency prompt (public for tests)."""
    return _EMERGENCY_FALLBACK_PROMPT


def _truncate_preview(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    s = text.strip()
    if len(s) <= PROMPT_PREVIEW_MAX_CHARS:
        return s
    return s[: PROMPT_PREVIEW_MAX_CHARS - 3] + "..."


def _emergency_resolution(
    *,
    template_id: Optional[str],
    configured_prompt_type: Optional[str],
    reason: str,
) -> PromptSSOTResolution:
    logger.critical(
        "[PromptSSOT] Library load failed for template_id=%r (%s); "
        "using immutable emergency fallback prompt.",
        template_id,
        reason,
    )
    fb = _EMERGENCY_FALLBACK_PROMPT
    return PromptSSOTResolution(
        effective_template_id=template_id,
        source="emergency_fallback",
        configured_prompt_type=configured_prompt_type,
        library_found=False,
        preview=_truncate_preview(fb),
        legacy_inline_detected=False,
        instructions=fb,
    )


def _load_library(template_id: str) -> tuple[Optional[str], bool]:
    """
    Load library JSON by id.

    Returns (instructions, found). On disk/read failure callers should use
    ``_emergency_resolution`` — this function does not log CRITICAL itself so
    the caller can attach tenant context.
    """
    raw = load_prompt_library_raw(template_id)
    if not raw:
        return None, False
    instructions = system_instructions_from_record(raw)
    if not instructions:
        return None, False
    return instructions, True


def resolve_prompt_ssot(config: ClientConfig) -> PromptSSOTResolution:
    """
    Resolve effective prompt template and generation instructions for a tenant.

    Priority:
      1. retrieval.prompt_template_id (library)
      2. prompt.template_id (library)
      3. prompt.prompt_type → preset library id
      4. prompt.template (legacy inline)
      5. default_builtin / emergency_fallback
    """
    prompt_cfg = getattr(config, "prompt", None)
    configured_prompt_type: Optional[str] = None
    configured_tid: Optional[str] = None

    if config.retrieval and config.retrieval.prompt_template_id:
        configured_tid = str(config.retrieval.prompt_template_id).strip() or None

    if prompt_cfg:
        configured_prompt_type = (prompt_cfg.prompt_type or None)
        if configured_prompt_type:
            configured_prompt_type = str(configured_prompt_type).strip() or None

    # 1 — retrieval.prompt_template_id (SSOT)
    if configured_tid:
        instructions, found = _load_library(configured_tid)
        if found and instructions:
            return PromptSSOTResolution(
                effective_template_id=configured_tid,
                source="library",
                configured_prompt_type=configured_prompt_type,
                library_found=True,
                preview=_truncate_preview(instructions),
                legacy_inline_detected=_has_inline_template(prompt_cfg) and bool(
                    (prompt_cfg.template or "").strip()
                ),
                instructions=instructions,
            )
        return _emergency_resolution(
            template_id=configured_tid,
            configured_prompt_type=configured_prompt_type,
            reason="missing_or_empty_library_file",
        )

    # 2 — prompt.template_id
    if prompt_cfg and getattr(prompt_cfg, "template_id", None):
        tid = str(prompt_cfg.template_id).strip()
        if tid:
            instructions, found = _load_library(tid)
            if found and instructions:
                return PromptSSOTResolution(
                    effective_template_id=tid,
                    source="library",
                    configured_prompt_type=configured_prompt_type,
                    library_found=True,
                    preview=_truncate_preview(instructions),
                    legacy_inline_detected=False,
                    instructions=instructions,
                )
            return _emergency_resolution(
                template_id=tid,
                configured_prompt_type=configured_prompt_type,
                reason="prompt_node_template_id_load_failed",
            )

    # 3 — preset map from prompt_type
    if (
        prompt_cfg
        and getattr(prompt_cfg, "enabled", False)
        and configured_prompt_type
        and configured_prompt_type != "custom"
    ):
        preset_id = PROMPT_PRESET_LIBRARY.get(configured_prompt_type)
        if preset_id:
            instructions, found = _load_library(preset_id)
            if found and instructions:
                return PromptSSOTResolution(
                    effective_template_id=preset_id,
                    source="preset_mapped",
                    configured_prompt_type=configured_prompt_type,
                    library_found=True,
                    preview=_truncate_preview(instructions),
                    legacy_inline_detected=False,
                    instructions=instructions,
                )
            logger.critical(
                "[PromptSSOT] Preset library file missing for prompt_type=%r preset_id=%r; "
                "trying built-in template.",
                configured_prompt_type,
                preset_id,
            )
        from app.core.pipeline_nodes.prompt_node import _BUILTIN_TEMPLATES

        builtin = _BUILTIN_TEMPLATES.get(
            configured_prompt_type, _BUILTIN_TEMPLATES["rag_context"]
        )
        return PromptSSOTResolution(
            effective_template_id=None,
            source="default_builtin",
            configured_prompt_type=configured_prompt_type,
            library_found=False,
            preview=_truncate_preview(builtin),
            legacy_inline_detected=False,
            instructions=builtin,
        )

    # 4 — legacy inline template
    if prompt_cfg and _has_inline_template(prompt_cfg):
        inline = (prompt_cfg.template or "").strip()
        return PromptSSOTResolution(
            effective_template_id=None,
            source="legacy_inline",
            configured_prompt_type=configured_prompt_type,
            library_found=False,
            preview=_truncate_preview(inline),
            legacy_inline_detected=True,
            instructions=inline,
        )

    # 5 — default built-in
    from app.core.pipeline_nodes.prompt_node import _BUILTIN_TEMPLATES

    builtin = _BUILTIN_TEMPLATES["rag_context"]
    return PromptSSOTResolution(
        effective_template_id=None,
        source="default_builtin",
        configured_prompt_type=configured_prompt_type,
        library_found=False,
        preview=_truncate_preview(builtin),
        legacy_inline_detected=False,
        instructions=builtin,
    )


def _has_inline_template(prompt_cfg: object) -> bool:
    if not prompt_cfg:
        return False
    t = getattr(prompt_cfg, "template", None)
    return bool(t and str(t).strip())


def get_generation_instructions(config: ClientConfig) -> str:
    """Full system-instruction text for LLM generation (never None)."""
    return resolve_prompt_ssot(config).instructions


def prompt_ssot_to_state(resolution: PromptSSOTResolution):
    """Map resolution dataclass to Pydantic ``PromptSSOTState``."""
    from app.core.config.client_config_schema import PromptSSOTState

    return PromptSSOTState(
        effective_template_id=resolution.effective_template_id,
        source=resolution.source,
        configured_prompt_type=resolution.configured_prompt_type,
        library_found=resolution.library_found,
        preview=resolution.preview,
        legacy_inline_detected=resolution.legacy_inline_detected,
    )


def _clear_inline_prompt_template(config: ClientConfig) -> ClientConfig:
    """Remove legacy ``prompt.template`` from a ClientConfig (library-first persist)."""
    prompt_cfg = getattr(config, "prompt", None)
    if not prompt_cfg:
        return config
    if getattr(prompt_cfg, "template", None) in (None, ""):
        return config
    return config.model_copy(
        update={"prompt": prompt_cfg.model_copy(update={"template": None})}
    )


def sync_prompt_ssot_to_config(config: ClientConfig) -> ClientConfig:
    """
    When ``retrieval.prompt_template_id`` is empty but ``prompt.prompt_type`` is a preset,
    set the library id so persisted JSON matches generation SSOT.

    Always clears ``prompt.template`` when a library id is (or becomes) canonical.
    """
    out = config
    if config.retrieval.prompt_template_id:
        return _clear_inline_prompt_template(out)

    prompt_cfg = getattr(config, "prompt", None)
    if not prompt_cfg or not getattr(prompt_cfg, "enabled", False):
        return out
    ptype = (prompt_cfg.prompt_type or "").strip()
    if not ptype or ptype == "custom":
        return out
    preset_id = PROMPT_PRESET_LIBRARY.get(ptype)
    if not preset_id:
        return out
    raw = load_prompt_library_raw(preset_id)
    if not raw:
        return out
    updates = {"prompt_template_id": preset_id}
    new_retrieval = out.retrieval.model_copy(update=updates)
    out = out.model_copy(update={"retrieval": new_retrieval})
    return _clear_inline_prompt_template(out)


def enforce_library_first_prompt_persist(
    config_data: Dict[str, Any],
    *,
    client_id: str,
) -> Dict[str, Any]:
    """
    Normalize tenant JSON before disk write: sync preset → library id, strip inline template.

    New tenants must not accumulate ``prompt.template``; generation uses Prompt Library only.
    """
    from app.core.config.client_config_schema import ClientConfig
    from app.utils.path_sanitizer import sanitize_client_id

    data = dict(config_data)
    safe_id = sanitize_client_id(client_id)
    data["client_id"] = safe_id

    retrieval = data.get("retrieval") if isinstance(data.get("retrieval"), dict) else {}
    library_id = (retrieval.get("prompt_template_id") or "").strip() or None
    prompt = data.get("prompt")
    if isinstance(prompt, dict):
        prompt.pop("custom_template", None)
        if library_id:
            prompt["template"] = None

    try:
        cfg = ClientConfig.from_dict(data)
        cfg = sync_prompt_ssot_to_config(cfg)
        if cfg.retrieval.prompt_template_id:
            data.setdefault("retrieval", {})[
                "prompt_template_id"
            ] = cfg.retrieval.prompt_template_id
            library_id = cfg.retrieval.prompt_template_id
        if isinstance(data.get("prompt"), dict) and library_id:
            data["prompt"]["template"] = None
    except Exception as exc:
        logger.warning(
            "[PromptSSOT] enforce_library_first_prompt_persist partial for %r: %s",
            client_id,
            exc,
        )
    return data
