"""
PII Middleware contract — abstract interface for PII detection and redaction.

Implementations must never expose raw PII in logs, return values, or audit
records. Only entity TYPE names (e.g. "email", "credit_card") may be surfaced.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class PIIAction(str, Enum):
    REDACT = "REDACT"
    MASK = "MASK"
    TOKENIZE = "TOKENIZE"
    HASH = "HASH"
    BLOCK = "BLOCK"


class PIISeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class PIIScanResult:
    redacted_text: str
    entities_found: List[str] = field(default_factory=list)
    action_taken: Optional[PIIAction] = None
    severity_max: Optional[PIISeverity] = None
    blocked: bool = False
    latency_ms: float = 0.0


class PIIMiddlewareContract(abc.ABC):
    @abc.abstractmethod
    def scan_text(
        self, text: str, *, position: str = "pre_llm",
    ) -> PIIScanResult:
        ...

    @abc.abstractmethod
    def scan_chunks(
        self, chunks: list, *, position: str = "pre_llm",
    ) -> Tuple[list, PIIScanResult]:
        """Returns (redacted_chunks, aggregated PIIScanResult)."""
        ...

    @abc.abstractmethod
    def alignment_status(self) -> str:
        """Returns 'aligned' | 'partial' | 'disabled' | 'error'."""
        ...
