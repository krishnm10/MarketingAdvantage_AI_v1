"""
Output Formatter and Security Gate — Phase 2/3 Runtime
Formats RAG output and enforces trust/PII guardrails.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TOXIC_PATTERNS = [
    re.compile(r"\b(kill|murder|attack|bomb|weapon)\b", re.IGNORECASE),
    re.compile(r"\b(hack|exploit|breach|steal)\b", re.IGNORECASE),
]


class OutputFormatter:
    """
    Output formatting and security gate node.
    Runs after Post-LLM PII middleware and before final API response.
    """

    VALID_FORMATS = {"plain_text", "markdown", "json", "structured_fields"}

    def __init__(
        self,
        *,
        response_format: str = "plain_text",
        json_schema: Optional[Dict[str, Any]] = None,
        strip_boilerplate: bool = False,
        min_trust_score: float = 0.0,
        block_on_low_trust: bool = False,
        toxicity_filter: str = "disabled",
    ):
        self._format = response_format
        self._json_schema = json_schema
        self._strip_boilerplate = strip_boilerplate
        self._min_trust_score = min_trust_score
        self._block_on_low_trust = block_on_low_trust
        self._toxicity_filter = toxicity_filter

    def node_type(self) -> str:
        return "output_formatter"

    def format_output(
        self,
        raw_answer: str,
        *,
        pii_redacted: bool = False,
        pii_entities_found: Optional[List[str]] = None,
        trust_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        blocked = False
        block_reason: Optional[str] = None

        if self._block_on_low_trust and trust_score is not None:
            if trust_score < self._min_trust_score:
                blocked = True
                block_reason = (
                    f"Trust score ({trust_score:.2f}) below threshold "
                    f"({self._min_trust_score:.2f})"
                )

        if not blocked and self._toxicity_filter == "rule_based":
            toxicity_result = self._check_toxicity_rules(raw_answer)
            if toxicity_result:
                blocked = True
                block_reason = f"Toxicity detected: {toxicity_result}"

        if blocked:
            formatted = f"[BLOCKED] {block_reason}"
        else:
            formatted = self._apply_format(raw_answer)

        if self._strip_boilerplate and not blocked:
            formatted = self._strip(formatted)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "formatted_output": formatted,
            "format_type": self._format,
            "blocked": blocked,
            "block_reason": block_reason,
            "trust_gate_passed": not blocked,
            "latency_ms": latency_ms,
        }

    def _apply_format(self, text: str) -> str:
        if self._format == "json" and self._json_schema:
            try:
                return json.dumps({"answer": text}, indent=2)
            except Exception:
                return text
        return text

    @staticmethod
    def _strip(text: str) -> str:
        prefixes = ("Sure,", "Of course,", "Here's", "I'd be happy")
        lines = text.strip().split("\n")
        stripped = [l for l in lines if not l.strip().startswith(prefixes)]
        return "\n".join(stripped) if stripped else text

    @staticmethod
    def _check_toxicity_rules(text: str) -> Optional[str]:
        """Basic rule-based toxicity check."""
        for pattern in _TOXIC_PATTERNS:
            match = pattern.search(text)
            if match:
                return f"matched pattern near: ...{match.group(0)}..."
        return None

    def validate_config(self) -> List[str]:
        errors: List[str] = []
        if self._format not in OutputFormatter.VALID_FORMATS:
            errors.append(f"Invalid response_format: {self._format}")
        if not 0.0 <= self._min_trust_score <= 1.0:
            errors.append("min_trust_score must be 0.0-1.0")
        return errors
