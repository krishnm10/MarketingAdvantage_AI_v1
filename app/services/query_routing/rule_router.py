"""
Deterministic L0 rule router — zero external calls.
"""

from __future__ import annotations

import re
from typing import Optional, Set

from app.services.query_routing.types import (
    RouteDecision,
    RouterLayer,
    blocked_decision,
    chitchat_decision,
    clarification_decision,
    meta_help_decision,
    structured_decision,
)

_GREETING_EXACT: Set[str] = {
    "hi",
    "hello",
    "hey",
    "hiya",
    "howdy",
    "greetings",
    "good morning",
    "good afternoon",
    "good evening",
    "good day",
    "morning",
    "afternoon",
    "evening",
    "yo",
    "sup",
    "hola",
}

_FAREWELL_EXACT: Set[str] = {
    "bye",
    "goodbye",
    "good bye",
    "see you",
    "see ya",
    "later",
    "take care",
    "farewell",
    "ciao",
    "good night",
    "gn",
}

_THANKS_EXACT: Set[str] = {
    "thanks",
    "thank you",
    "thx",
    "ty",
    "much appreciated",
    "appreciate it",
    "cheers",
}

_GREETING_RE = re.compile(
    r"^(?:hi|hello|hey|hiya|howdy|greetings|good\s+(?:morning|afternoon|evening|day))[\s!.,?]*$",
    re.IGNORECASE,
)
_FAREWELL_RE = re.compile(
    r"^(?:bye|goodbye|good\s*bye|see\s+you|see\s+ya|later|take\s+care|farewell|ciao|good\s+night)[\s!.,?]*$",
    re.IGNORECASE,
)
_THANKS_RE = re.compile(
    r"^(?:thanks|thank\s+you|thx|ty|much\s+appreciated|appreciate\s+it|cheers)[\s!.,?]*$",
    re.IGNORECASE,
)
_META_RE = re.compile(
    r"(?i)\b(?:what\s+can\s+you\s+do|what\s+do\s+you\s+do|how\s+can\s+you\s+help|"
    r"help\s+me|capabilities|who\s+are\s+you|what\s+are\s+you)\b"
)
_STRUCTURED_ID_RE = re.compile(r"\b[A-Z]{2,}-[A-Z0-9]{3,}\b")
_INJECTION_RE = re.compile(
    r"(?i)(?:ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions|"
    r"disregard\s+(?:the\s+)?(?:system|above)\s+prompt|"
    r"you\s+are\s+now\s+(?:a\s+)?(?:dan|jailbreak)|"
    r"<\s*/?\s*system\s*>|"
    r"\[\s*INST\s*\])"
)
_INTERROGATIVE_RE = re.compile(
    r"(?i)\b(?:what|who|where|when|why|how|which|show|list|find|give|tell)\b|\?"
)
_DOMAIN_TOKEN_RE = re.compile(
    r"(?i)\b(?:invoice|invoices|order|orders|vendor|total|due\s+date|payment|"
    r"line\s+item|tax|subtotal|amount|billing|po|purchase)\b"
)

_BARE_DOMAIN_TOKENS: Set[str] = {
    "invoice",
    "invoices",
    "order",
    "orders",
    "vendor",
    "total",
    "tax",
    "payment",
    "billing",
}


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


class RuleRouter:
    """Synchronous rule-based query classifier."""

    def route(
        self,
        raw_query: str,
        has_chat_history: bool = False,
    ) -> Optional[RouteDecision]:
        q = (raw_query or "").strip()
        if not q:
            return clarification_decision(
                reason_code="empty_query",
                layer=RouterLayer.HEURISTIC,
            )

        norm = _normalize(q)

        if _INJECTION_RE.search(q):
            m = _INJECTION_RE.search(q)
            return blocked_decision(
                matched_pattern=m.group(0) if m else None,
                reason_code="injection_pattern",
            )

        if norm in _GREETING_EXACT or _GREETING_RE.match(q):
            return chitchat_decision(
                reason_code="greeting",
                matched_pattern=norm if norm in _GREETING_EXACT else "greeting_re",
            )

        if norm in _FAREWELL_EXACT or _FAREWELL_RE.match(q):
            return chitchat_decision(
                reason_code="farewell",
                matched_pattern=norm if norm in _FAREWELL_EXACT else "farewell_re",
            )

        if norm in _THANKS_EXACT or _THANKS_RE.match(q):
            return chitchat_decision(
                reason_code="thanks",
                matched_pattern=norm if norm in _THANKS_EXACT else "thanks_re",
            )

        meta_m = _META_RE.search(q)
        if meta_m:
            return meta_help_decision(
                matched_pattern=meta_m.group(0),
                reason_code="meta_help",
            )

        id_m = _STRUCTURED_ID_RE.search(q.upper())
        if id_m:
            return structured_decision(
                matched_pattern=id_m.group(0),
                reason_code="structured_id",
            )

        words = norm.split()
        word_count = len(words)
        has_domain = bool(_DOMAIN_TOKEN_RE.search(q))
        has_question = "?" in q or bool(_INTERROGATIVE_RE.search(q))

        if (
            word_count <= 3
            and not has_domain
            and not has_question
            and not has_chat_history
        ):
            return clarification_decision(
                reason_code="ambiguous_short",
                layer=RouterLayer.HEURISTIC,
                matched_pattern=f"words={word_count}",
            )

        if word_count == 1 and norm in _BARE_DOMAIN_TOKENS and not has_chat_history:
            return clarification_decision(
                reason_code="bare_domain_token",
                layer=RouterLayer.HEURISTIC,
                matched_pattern=norm,
            )

        return None
