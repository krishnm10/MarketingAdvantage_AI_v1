"""
RegexPIIMiddleware — Phase 1 regex-based PII detection and redaction node.

Delegates to the existing security_middleware patterns for PII and prompt
injection detection. Adds Luhn validation for credit card matches.

Never logs or stores raw PII — only entity type names.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from app.ai.contracts.pii_middleware_contract import (
    PIIAction,
    PIIMiddlewareContract,
    PIIScanResult,
    PIISeverity,
)
from app.middleware.security_middleware import (
    _PII_PATTERNS,
    _VALIDATED_PII,
    detect_prompt_injection,
    redact_pii,
)

logger = logging.getLogger(__name__)

_VALID_POSITIONS = {"pre_embedding", "pre_llm", "post_llm"}

_SEVERITY_MAP: Dict[str, PIISeverity] = {
    # CRITICAL — unique national identifiers
    "aadhaar":     PIISeverity.CRITICAL,
    "ssn_us":      PIISeverity.CRITICAL,
    "credit_card": PIISeverity.CRITICAL,
    # HIGH — semi-unique identifiers
    "pan":         PIISeverity.HIGH,
    "iban":        PIISeverity.HIGH,
    "passport_in": PIISeverity.HIGH,
    # MEDIUM — contact / routing info
    "ifsc":        PIISeverity.MEDIUM,
    "phone_in":    PIISeverity.MEDIUM,
    "email":       PIISeverity.MEDIUM,
    # LOW — infrastructure identifiers
    "ipv4":        PIISeverity.LOW,
}

_SEVERITY_RANK = {
    PIISeverity.LOW: 0,
    PIISeverity.MEDIUM: 1,
    PIISeverity.HIGH: 2,
    PIISeverity.CRITICAL: 3,
}


def _redact_pii_with_luhn(text: str) -> Tuple[str, List[str]]:
    """
    Enhanced PII redaction with validation for high-ambiguity patterns.

    Delegates to security_middleware.redact_pii() which handles both
    simple regex patterns and validated patterns (credit_card → Luhn,
    iban → ISO 13616 checksum) in a single pass.
    """
    return redact_pii(text)


class RegexPIIMiddleware(PIIMiddlewareContract):
    """Phase 1 regex-based PII middleware with optional audit logging."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._positions: List[str] = cfg.get("position", ["pre_llm"])
        self._action: PIIAction = PIIAction(cfg.get("action", "REDACT"))
        self._block_on_severity: Optional[PIISeverity] = (
            PIISeverity(cfg["block_on_severity"])
            if cfg.get("block_on_severity")
            else None
        )
        self._trust_score_penalty: float = cfg.get("trust_score_penalty", 0.0)
        self._custom_patterns: Dict[str, re.Pattern] = {}
        for cp in cfg.get("custom_patterns", []):
            if "name" in cp and "pattern" in cp:
                self._custom_patterns[cp["name"]] = re.compile(cp["pattern"])
        self._audit_log_enabled: bool = cfg.get("audit_log_enabled", False)
        self._audit_service = None

    def _get_audit_service(self):
        if self._audit_service is None:
            from app.services.security.security_audit_service import (
                SecurityAuditService,
            )
            self._audit_service = SecurityAuditService()
        return self._audit_service

    def _max_severity(self, entities: List[str]) -> Optional[PIISeverity]:
        if not entities:
            return None
        best: Optional[PIISeverity] = None
        for e in entities:
            sev = _SEVERITY_MAP.get(e)
            if sev and (best is None or _SEVERITY_RANK[sev] > _SEVERITY_RANK[best]):
                best = sev
        return best

    def scan_text(
        self, text: str, *, position: str = "pre_llm",
    ) -> PIIScanResult:
        start = time.perf_counter()

        redacted, entities = _redact_pii_with_luhn(text)

        for name, pattern in self._custom_patterns.items():
            if pattern.search(redacted):
                entities.append(name)
                redacted = pattern.sub(f"[{name.upper()}_REDACTED]", redacted)

        injection_detected, _ = detect_prompt_injection(text)
        if injection_detected:
            entities.append("prompt_injection")

        severity = self._max_severity(entities)
        blocked = False
        if (
            self._block_on_severity
            and severity
            and _SEVERITY_RANK.get(severity, 0)
            >= _SEVERITY_RANK.get(self._block_on_severity, 0)
        ):
            blocked = True

        latency = (time.perf_counter() - start) * 1000

        result = PIIScanResult(
            redacted_text=redacted,
            entities_found=entities,
            action_taken=self._action if entities else None,
            severity_max=severity,
            blocked=blocked,
            latency_ms=round(latency, 2),
        )

        if entities and self._audit_log_enabled:
            self._fire_audit_log(
                position=position,
                entities=entities,
                severity=severity,
                latency=latency,
            )

        return result

    def scan_chunks(
        self, chunks: list, *, position: str = "pre_llm",
    ) -> Tuple[list, PIIScanResult]:
        start = time.perf_counter()
        redacted_chunks: list = []
        all_entities: List[str] = []
        affected_ids: List[str] = []

        for chunk in chunks:
            if isinstance(chunk, dict):
                text = chunk.get("text", chunk.get("content", ""))
                chunk_id = chunk.get("id", chunk.get("chunk_id"))
            elif isinstance(chunk, str):
                text = chunk
                chunk_id = None
            else:
                text = getattr(chunk, "text", getattr(chunk, "content", str(chunk)))
                chunk_id = getattr(chunk, "id", getattr(chunk, "chunk_id", None))

            result = self.scan_text(text, position=position)

            if isinstance(chunk, dict):
                new_chunk = {**chunk}
                new_chunk["text"] = result.redacted_text
                if "content" in new_chunk:
                    new_chunk["content"] = result.redacted_text
                redacted_chunks.append(new_chunk)
            elif isinstance(chunk, str):
                redacted_chunks.append(result.redacted_text)
            else:
                redacted_chunks.append(result.redacted_text)

            for e in result.entities_found:
                if e not in all_entities:
                    all_entities.append(e)

            if result.entities_found and chunk_id:
                affected_ids.append(str(chunk_id))

        total_latency = (time.perf_counter() - start) * 1000
        severity = self._max_severity(all_entities)
        blocked = False
        if (
            self._block_on_severity
            and severity
            and _SEVERITY_RANK.get(severity, 0)
            >= _SEVERITY_RANK.get(self._block_on_severity, 0)
        ):
            blocked = True

        aggregated = PIIScanResult(
            redacted_text=f"[{len(chunks)} chunks scanned]",
            entities_found=all_entities,
            action_taken=self._action if all_entities else None,
            severity_max=severity,
            blocked=blocked,
            latency_ms=round(total_latency, 2),
        )

        return redacted_chunks, aggregated

    def alignment_status(self) -> str:
        if not self._positions:
            return "disabled"
        configured = set(self._positions)
        if configured >= _VALID_POSITIONS:
            return "aligned"
        if configured & _VALID_POSITIONS:
            return "partial"
        return "error"

    def _fire_audit_log(
        self,
        *,
        position: str,
        entities: List[str],
        severity: Optional[PIISeverity],
        latency: float,
        business_id: str = "system",
        pipeline_id: Optional[str] = None,
        node_id: Optional[str] = None,
        chunk_ids: Optional[List[str]] = None,
    ) -> None:
        """Best-effort async audit log write; never blocks the scan pipeline."""
        try:
            svc = self._get_audit_service()
            coro = svc.log_pii_event(
                business_id=business_id,
                pipeline_id=pipeline_id,
                node_id=node_id,
                position=position,
                entities_detected=entities,
                action_taken=self._action.value,
                severity_max=severity.value if severity else None,
                chunk_ids_affected=chunk_ids,
                latency_ms=round(latency, 2),
            )
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(coro)
            except RuntimeError:
                asyncio.run(coro)
        except Exception:
            logger.warning(
                "Failed to fire security audit log for entities=%s", entities,
            )
