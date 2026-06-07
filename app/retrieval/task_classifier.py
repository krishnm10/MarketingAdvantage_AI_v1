from __future__ import annotations

"""
L1 task classification for retrieval/chat.

Phase 1 scope:
- Deterministic, read-only classification.
- No behavior change when flags are off.
- Only used for QueryRoute.KNOWLEDGE in /api/v2/retrieve/chat.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from app.services.query_routing.types import QueryRoute, RouteDecision

from app.retrieval.predicate_parser import ParsedPredicates, parse_predicates
from app.retrieval.types_retrieve import DomainType


class TaskType(str, Enum):
    """High-level task categories for retrieval/chat."""

    SINGLE_QA = "single_qa"
    DOCSET_FILTER = "docset_filter"
    AGGREGATE = "aggregate"
    DISCREPANCY = "discrepancy"
    AUDIT = "audit"
    POLICY_LOOKUP = "policy_lookup"
    OPEN_QA = "open_qa"
    MIXED = "mixed"


@dataclass(frozen=True)
class TaskPlan:
    """
    Immutable L1 task plan attached to debug_info and traces.

    Phase 1 deliberately keeps this compact and read-only.
    """

    route: QueryRoute
    task_type: TaskType
    domain: DomainType
    predicates: Optional[ParsedPredicates]
    confidence: float = 0.5

    def to_debug_dict(self) -> Dict[str, Any]:
        """Compact, JSON-serializable representation for debug_info and tracing."""
        return {
            "route": self.route.value,
            "task_type": self.task_type.value,
            "domain": self.domain.value,
            "confidence": float(self.confidence),
            "predicates": self.predicates.to_debug_dict() if self.predicates else None,
        }


def _detect_domain(text: str) -> DomainType:
    """
    Very coarse domain detector.

    Phase 1: invoice is the only production-targeted domain for task-type heuristics;
    generic aggregate/discrepancy parsing is advisory only. Core classification stays
    domain-agnostic so future adapters (contracts, tickets, policies, etc.) can plug in.
    """
    q = (text or "").lower()
    if any(tok in q for tok in ("invoice", "invoices", "vendor", "tax amount", "late fee")):
        return DomainType.INVOICE
    return DomainType.GENERIC


def classify_task(
    *,
    raw_query: str,
    route_decision: RouteDecision,
    client_id: Optional[str] = None,
    rewritten_query: Optional[str] = None,
) -> TaskPlan:
    """
    Deterministic L1 task classifier.

    - Only inspects strings and RouteDecision.
    - Never calls external services or LLMs.
    - Safe to call in hot paths; behavior is read-only.
    """
    del client_id  # reserved for future per-tenant overrides

    route = route_decision.route
    # Default: treat as open-ended Q&A unless clearly docset/aggregate/audit.
    base_text = (rewritten_query or raw_query or "").strip()
    domain = _detect_domain(base_text)

    # Default task type depending on route.
    if route == QueryRoute.STRUCTURED:
        default_task = TaskType.SINGLE_QA
    elif route == QueryRoute.KNOWLEDGE:
        default_task = TaskType.OPEN_QA
    else:
        # CHITCHAT, META_HELP, CLARIFICATION, BLOCKED are not analyzed in Phase 1.
        return TaskPlan(
            route=route,
            task_type=TaskType.OPEN_QA,
            domain=domain,
            predicates=None,
            confidence=0.0,
        )

    predicates = parse_predicates(text=base_text, domain=domain)

    # Phase 1: invoice-specific task-type mapping; other domains keep default_task.
    # Does not touch retrieval/ranking — metadata for debug/trace only.
    task_type = default_task
    confidence = 0.5

    if domain == DomainType.INVOICE and route == QueryRoute.KNOWLEDGE:
        kinds = predicates.kinds if predicates else []
        if "docset_filter" in kinds:
            task_type = TaskType.DOCSET_FILTER
            confidence = 0.8
        elif "aggregate" in kinds:
            task_type = TaskType.AGGREGATE
            confidence = 0.8
        elif "discrepancy" in kinds:
            task_type = TaskType.DISCREPANCY
            confidence = 0.8
        elif "audit" in kinds:
            task_type = TaskType.AUDIT
            confidence = 0.7
        elif "policy_lookup" in kinds:
            task_type = TaskType.POLICY_LOOKUP
            confidence = 0.7

        # Mixed indicators: fall back to MIXED with lower confidence.
        if len({k for k in kinds if k in {"docset_filter", "aggregate", "discrepancy", "audit"}}) > 1:
            task_type = TaskType.MIXED
            confidence = 0.6

    return TaskPlan(
        route=route,
        task_type=task_type,
        domain=domain,
        predicates=predicates,
        confidence=confidence,
    )

