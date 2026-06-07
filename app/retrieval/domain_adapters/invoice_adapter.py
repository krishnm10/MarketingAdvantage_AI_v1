from __future__ import annotations

"""
Invoice domain adapter for document-set analysis (Phase 2, debug-only).

Constraints:
- Operates ONLY on already-retrieved RankedResult content.
- Deterministic, regex/keyword-based; no DB/LLM/external calls.
- Never mutates ranked_results or affects retrieval/ranking/answers.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

from app.retrieval.document_match import DocumentMatch
from app.retrieval.types_retrieve import DomainType, RankedResult
from app.retrieval.analysis_adapter_registry import register_adapter


@dataclass(frozen=True)
class InvoiceDocumentMatch:
    file_id: str
    overdue: bool
    has_late_fee: bool
    tax_gt_threshold: bool
    sample_chunk_ids: List[str]

    def to_debug_dict(self) -> Dict:
        return {
            "file_id": self.file_id,
            "overdue": self.overdue,
            "has_late_fee": self.has_late_fee,
            "tax_gt_threshold": self.tax_gt_threshold,
            "sample_chunk_ids": list(self.sample_chunk_ids),
        }


class InvoiceDocumentSetAdapter:
    """
    Deterministic invoice adapter for docset analysis.

    Phase 2: uses small internal heuristics and a fixed tax threshold
    (e.g. 10%) instead of tenant-configured values to keep scope tight.
    """

    # Internal Phase 2 constant; can be made configurable later if needed.
    _TAX_THRESHOLD_PERCENT: float = 10.0

    def __init__(self, *, max_chunks_per_doc: int = 3) -> None:
        self._max_chunks_per_doc = max(1, int(max_chunks_per_doc))

    def analyze(
        self,
        *,
        grouped_chunks: Mapping[Optional[str], Iterable[RankedResult]],
        raw_query: str,
        task_plan,
        tenant_runtime,
    ) -> List[DocumentMatch]:
        """
        Return one DocumentMatch per file_id with any positive invoice signal.

        Inclusion is disjunctive: overdue OR has_late_fee OR tax_gt_threshold.
        This is not conjunctive query satisfaction; docs_matched in AnalysisResult
        counts documents included here, not documents matching every query clause.
        """
        invoice_matches: List[InvoiceDocumentMatch] = []

        for file_id, chunks in grouped_chunks.items():
            if not file_id:
                continue
            chunk_list = list(chunks)
            if not chunk_list:
                continue

            overdue = any(_is_overdue_text(c.text or "") for c in chunk_list)
            has_late_fee = any(_has_late_fee_text(c.text or "") for c in chunk_list)
            tax_gt_threshold = any(
                _has_tax_above_threshold_text(c.text or "", self._TAX_THRESHOLD_PERCENT)
                for c in chunk_list
            )

            if not (overdue or has_late_fee or tax_gt_threshold):
                continue

            sample_ids = [c.chunk_id for c in chunk_list[: self._max_chunks_per_doc]]
            invoice_matches.append(
                InvoiceDocumentMatch(
                    file_id=file_id,
                    overdue=overdue,
                    has_late_fee=has_late_fee,
                    tax_gt_threshold=tax_gt_threshold,
                    sample_chunk_ids=sample_ids,
                )
            )

        # Normalize to the shared DocumentMatch contract before returning to core.
        normalized: List[DocumentMatch] = []
        for m in invoice_matches:
            attrs: Dict[str, bool] = {
                "overdue": m.overdue,
                "has_late_fee": m.has_late_fee,
                "tax_gt_threshold": m.tax_gt_threshold,
            }
            normalized.append(
                DocumentMatch(
                    file_id=m.file_id,
                    attributes=attrs,
                    sample_chunk_ids=m.sample_chunk_ids,
                )
            )

        return normalized


def _is_overdue_text(text: str) -> bool:
    t = text.lower()
    return any(
        phrase in t
        for phrase in (
            "overdue",
            "past due",
            "payment overdue",
            "overdue amount",
        )
    )


def _has_late_fee_text(text: str) -> bool:
    t = text.lower()
    return any(
        phrase in t
        for phrase in (
            "late fee",
            "late payment fee",
            "late payment penalty",
            "late payment penalties",
        )
    )


def _has_tax_above_threshold_text(text: str, threshold_percent: float) -> bool:
    import re

    t = text.lower()
    if "tax" not in t:
        return False

    # Look for patterns like "tax 12%" or "tax rate: 15%" within a bounded window.
    pattern = r"tax[^.]{0,60}?(\d{1,3}(?:\.\d{1,2})?)\s*%"
    for match in re.finditer(pattern, t):
        try:
            val = float(match.group(1))
        except ValueError:
            continue
        # Strictly greater than threshold; e.g. > 10.0, not >=.
        if val > threshold_percent:
            return True
    return False


INVOICE_ADAPTER_DOMAIN = DomainType.INVOICE
INVOICE_ADAPTER = InvoiceDocumentSetAdapter()
register_adapter(INVOICE_ADAPTER_DOMAIN, INVOICE_ADAPTER)

