"""
Prompt Node — Phase 1 Runtime Implementation
Configurable prompt template rendering for the RAG pipeline.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Placeholder pattern: {variable_name}
_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")

# Standard prompt type templates
_BUILTIN_TEMPLATES = {
    "rag_context": (
        "You are an intelligent assistant. Answer the question using ONLY the context provided.\n"
        "If the context does not contain enough information, say so clearly.\n"
        "Cite the source number [1], [2], etc. when referencing specific facts.\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer:"
    ),
    "system": "{system_prompt}",
    "few_shot": (
        "{system_prompt}\n\n"
        "Examples:\n{examples}\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer:"
    ),
    "cot": (
        "You are an intelligent assistant. Think step by step.\n"
        "Use ONLY the context provided to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Let me think through this step by step:\n"
    ),
    "instruction_tuned": (
        "### Instruction\n{system_prompt}\n\n"
        "### Input\nContext:\n{context}\n\n"
        "Question: {query}\n\n"
        "### Response\n"
    ),
}


class PromptNode:
    """
    Configurable prompt template rendering node.

    Supports built-in prompt types (rag_context, system, few_shot, cot,
    instruction_tuned) and custom templates with {placeholder} variables.
    """

    def __init__(
        self,
        *,
        prompt_type: str = "rag_context",
        template: Optional[str] = None,
        template_id: Optional[str] = None,
        variable_map: Optional[Dict[str, str]] = None,
        max_tokens_warning: int = 3000,
    ):
        self._prompt_type = prompt_type
        self._template = template
        self._template_id = template_id
        self._variable_map = variable_map or {
            "context": "retriever.output",
            "query": "user.input",
        }
        self._max_tokens_warning = max_tokens_warning

    def node_type(self) -> str:
        return "prompt_node"

    def positions(self) -> list:
        from app.ai.contracts.pipeline_node_contract import NodePosition
        return [NodePosition.PRE_LLM]

    def render(
        self,
        query: str,
        context: str,
        *,
        history: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        examples: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Render the configured prompt template with provided variables.
        Returns dict with rendered_prompt, variables_used, token_count, warnings.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        # Resolve template
        if self._template:
            template = self._template
        elif self._template_id:
            template = self._load_template(self._template_id)
            if template is None:
                warnings.append(f"Template '{self._template_id}' not found, using default")
                template = _BUILTIN_TEMPLATES.get(self._prompt_type, _BUILTIN_TEMPLATES["rag_context"])
        else:
            template = _BUILTIN_TEMPLATES.get(self._prompt_type, _BUILTIN_TEMPLATES["rag_context"])

        # Build variable context
        variables = {
            "query": query,
            "context": context,
            "history": history or "",
            "metadata": str(metadata or {}),
            "system_prompt": system_prompt or "You are a helpful assistant.",
            "examples": examples or "",
        }

        # Find placeholders and render
        placeholders = _PLACEHOLDER_RE.findall(template)
        variables_used = {}
        rendered = template
        for placeholder in placeholders:
            value = variables.get(placeholder, "")
            if not value and placeholder not in variables:
                warnings.append(f"Unresolved placeholder: {{{placeholder}}}")
            rendered = rendered.replace(f"{{{placeholder}}}", str(value))
            variables_used[placeholder] = f"<{len(str(value))} chars>"

        # Token count estimation (rough: 1 token ~ 4 chars for English)
        estimated_tokens = len(rendered) // 4
        if estimated_tokens > self._max_tokens_warning:
            warnings.append(
                f"Estimated token count ({estimated_tokens}) exceeds warning threshold ({self._max_tokens_warning})"
            )

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "rendered_prompt": rendered,
            "variables_used": variables_used,
            "token_count": estimated_tokens,
            "warnings": warnings,
            "latency_ms": latency_ms,
        }

    def validate_template(
        self,
        template: str,
        variable_map: Dict[str, str],
    ) -> Dict[str, Any]:
        """Validate a template string and variable map."""
        errors: List[str] = []
        warnings: List[str] = []
        placeholders = _PLACEHOLDER_RE.findall(template)

        if not template.strip():
            errors.append("Template cannot be empty")

        # Check for unresolvable placeholders
        known_vars = {"query", "context", "history", "metadata", "system_prompt", "examples"}
        known_vars.update(variable_map.keys())
        for p in placeholders:
            if p not in known_vars:
                warnings.append(f"Placeholder '{{{p}}}' has no mapping in variable_map")

        # PII pre-save check using security middleware
        try:
            from app.middleware.security_middleware import redact_pii
            _, pii_types = redact_pii(template)
            if pii_types:
                warnings.append(f"Template contains literal PII patterns: {pii_types}")
        except ImportError:
            pass

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "placeholders_found": placeholders,
        }

    def validate_config(self) -> List[str]:
        """Return validation errors."""
        errors = []
        if self._prompt_type not in _BUILTIN_TEMPLATES and self._prompt_type != "custom":
            errors.append(f"Unknown prompt_type: {self._prompt_type}")
        if self._prompt_type == "custom" and not self._template:
            errors.append("Custom prompt_type requires a template")
        return errors

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute as pipeline node."""
        return self.render(
            query=context.get("query", ""),
            context=context.get("context", ""),
            history=context.get("history"),
            metadata=context.get("metadata"),
            system_prompt=context.get("system_prompt"),
            examples=context.get("examples"),
        )

    @staticmethod
    def _load_template(template_id: str) -> Optional[str]:
        """Load a saved template from the prompts directory."""
        import json
        from pathlib import Path

        # Validate slug safety
        try:
            from app.utils.path_sanitizer import validate_slug
            safe_id = validate_slug(template_id)
        except (ImportError, ValueError):
            logger.warning("[PromptNode] Invalid template_id: %s", template_id[:20])
            return None

        prompts_dir = Path(__file__).resolve().parents[1] / "configs" / "prompts"
        template_path = prompts_dir / f"{safe_id}.json"

        if not template_path.exists():
            return None

        # Containment check
        try:
            resolved = template_path.resolve()
            if not str(resolved).startswith(str(prompts_dir.resolve())):
                logger.warning("[PromptNode] Path containment violation for template: %s", safe_id)
                return None
        except Exception:
            return None

        try:
            with template_path.open() as f:
                data = json.load(f)
            return data.get("system_instructions") or data.get("content") or data.get("template")
        except Exception as e:
            logger.warning("[PromptNode] Failed to load template '%s': %s", safe_id, e)
            return None
