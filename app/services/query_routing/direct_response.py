"""
Direct (non-RAG) responses for L0-routed chat turns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.services.query_routing.types import QueryRoute, RouteDecision


@dataclass
class DirectResponse:
    answer: str
    route: str
    retrieval_skipped: bool = True
    sources_count: int = 0
    confidence: float = 1.0

    def to_api_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "route": self.route,
            "retrieval_skipped": self.retrieval_skipped,
            "sources_count": self.sources_count,
            "confidence": self.confidence,
        }


def _tenant_display_name(tenant_config: Any) -> str:
    if tenant_config is None:
        return "your invoice assistant"
    name = getattr(tenant_config, "client_id", None) or getattr(
        tenant_config, "name", None
    )
    if name and str(name).strip():
        return f"{str(name).strip()} assistant"
    return "your invoice assistant"


def build_direct_response(
    decision: RouteDecision,
    raw_query: str,
    tenant_config: Any = None,
) -> DirectResponse:
    del raw_query
    assistant = _tenant_display_name(tenant_config)
    route = decision.route

    if route == QueryRoute.CHITCHAT:
        if decision.reason_code == "farewell":
            answer = (
                "Goodbye! Feel free to return when you need invoice details, "
                "totals, due dates, or a full extraction."
            )
        elif decision.reason_code == "thanks":
            answer = (
                "You're welcome! Let me know if you need anything else from your invoices."
            )
        else:
            answer = (
                f"Hello! I'm {assistant}. I can help with invoice numbers, due dates, "
                "line items, totals, and full document extraction. What would you like to look up?"
            )
    elif route == QueryRoute.META_HELP:
        answer = (
            f"I'm {assistant}. I can help you:\n"
            "- Look up invoice or order numbers (e.g. INV-1234)\n"
            "- Find due dates, totals, tax, and payment details\n"
            "- Extract a full invoice breakdown\n"
            "- Answer targeted questions about fields in your documents\n\n"
            "Ask a specific question or paste an identifier to get started."
        )
    elif route == QueryRoute.CLARIFICATION:
        answer = (
            "I can search your ingested invoices and documents. "
            "Please ask a specific question — for example: \"What is the due date?\", "
            "\"Show total for INV-6640\", or \"Extract everything from this invoice.\""
        )
    elif route == QueryRoute.BLOCKED:
        answer = (
            "I can't process that request. Please ask a question about your invoices "
            "or documents without override instructions."
        )
    else:
        answer = (
            f"Hello! I'm {assistant}. How can I help with your invoices today?"
        )

    return DirectResponse(
        answer=answer,
        route=route.value,
        retrieval_skipped=True,
        sources_count=0,
        confidence=decision.confidence,
    )
