from __future__ import annotations

"""
Deterministic predicate parser for L1 task classification.

Phase 1 scope:
- Lightweight regex/keyword detection for invoice-style queries.
- No side effects; safe to call in hot paths.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.retrieval.types_retrieve import DomainType


@dataclass(frozen=True)
class ParsedPredicates:
    """
    Normalized predicate hints extracted from a natural-language query.

    Phase 1 keeps this intentionally simple and domain-agnostic; future
    phases can extend fields without breaking callers.
    """

    kinds: List[str]
    raw_text: str
    numeric_thresholds: Dict[str, float]

    def to_debug_dict(self) -> Dict[str, Any]:
        return {
            "kinds": list(self.kinds),
            "raw_text": self.raw_text,
            "numeric_thresholds": dict(self.numeric_thresholds),
        }


def parse_predicates(*, text: str, domain: DomainType) -> Optional[ParsedPredicates]:
    """
    Best-effort, deterministic predicate parsing.

    This function MUST NEVER raise; callers treat None as "no predicates inferred".
    """
    q = (text or "").lower()
    if not q:
        return None

    kinds: List[str] = []
    thresholds: Dict[str, float] = {}

    try:
        if domain == DomainType.INVOICE:
            _parse_invoice_predicates(q, kinds, thresholds)
        else:
            _parse_generic_predicates(q, kinds, thresholds)
    except Exception:
        # Parsing is advisory only — never affect primary behavior.
        return None

    if not kinds and not thresholds:
        return None

    return ParsedPredicates(kinds=kinds, raw_text=text, numeric_thresholds=thresholds)


def _parse_invoice_predicates(q: str, kinds: List[str], thresholds: Dict[str, float]) -> None:
    # Docset-style language.
    if any(phrase in q for phrase in ("show me all", "show all", "list all", "find all", "find every", "list invoices", "find invoices")):
        kinds.append("docset_filter")

    # Overdue / late payment language.
    if any(tok in q for tok in ("overdue", "past due", "late payment", "late fee", "late payment penalty", "late payment penalties")):
        if "docset_filter" not in kinds:
            kinds.append("docset_filter")

    # Tax thresholds, e.g. "tax above 10%".
    if "tax" in q and any(term in q for term in ("above", "over", "greater than", ">", "at least")):
        import re

        m = re.search(r"tax[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*%", q)
        if m:
            try:
                val = float(m.group(1))
                thresholds["tax_gt_percent"] = val
            except ValueError:
                pass
        kinds.append("docset_filter")

    # Aggregation language.
    if any(tok in q for tok in ("how many", "count of", "total number of", "sum of", "total amount", "aggregate")):
        kinds.append("aggregate")

    # Discrepancy / audit language.
    if any(tok in q for tok in ("discrepancies", "discrepancy", "mismatch", "differences", "reconcile", "reconciliation")):
        kinds.append("discrepancy")

    if any(tok in q for tok in ("audit", "review", "compliance", "violations", "violations of")):
        kinds.append("audit")


def _parse_generic_predicates(q: str, kinds: List[str], thresholds: Dict[str, float]) -> None:
    # Phase 1: generic aggregate/discrepancy/audit hints are advisory only (debug/trace).
    # Invoice is the first production-targeted domain; the platform stays domain-agnostic.
    # Very small generic heuristics reused across domains.
    if any(phrase in q for phrase in ("show me all", "list all", "find all", "find every")):
        kinds.append("docset_filter")

    if any(tok in q for tok in ("how many", "count of", "total number of", "sum of", "aggregate")):
        kinds.append("aggregate")

    if any(tok in q for tok in ("discrepancies", "discrepancy", "mismatch", "differences", "compare")):
        kinds.append("discrepancy")

    if any(tok in q for tok in ("audit", "review", "violations", "policy", "compliance")):
        kinds.append("audit")

