from __future__ import annotations

"""
Phase 4 — Active result shaping (presentation + grounding layer only).

Constraints (enforced by design, not by this module alone):
- Never changes retrieval, ranking, reranking, fusion, or routing.
- Uses DocumentMatch as a binary document-eligibility filter only.
- Preserves retrieval document order; no new scores or comparators.
- Applies deterministic document and chunk truncation only.
"""

from typing import Dict, List, Optional, Sequence, Set

from app.retrieval.document_set_analysis import AnalysisResult, DocumentMatch
from app.retrieval.types_retrieve import RankedResult
from app.retrieval.components import RuntimeComponents


def _ordered_unique_file_ids(ranked_results: Sequence[RankedResult]) -> List[str]:
    """
    Stable projection of distinct file_ids from ranked results.

    Order is defined solely by first appearance in the incoming ranked_results
    sequence, which itself is already a finalized retrieval/reranking output.
    """
    seen: Set[str] = set()
    ordered: List[str] = []
    for r in ranked_results:
        fid = getattr(r, "file_id", None)
        if not fid:
            continue
        if fid in seen:
            continue
        seen.add(fid)
        ordered.append(fid)
    return ordered


def shape_docset_results(
    *,
    ranked_results: List[RankedResult],
    analysis: AnalysisResult,
    matches: List[DocumentMatch],
    runtime: RuntimeComponents,
) -> Optional[List[RankedResult]]:
    """
    Deterministically filter and truncate RankedResult list based on DocumentMatch.

    Behavior:
    - Eligible documents are the strict subset of file_ids present in matches.
    - Document ordering is a stable projection of the incoming ranked_results
      order; this function never sorts or recomputes scores.
    - If docset_max_docs_returned > 0, we keep only the first N eligible
      documents encountered in retrieval order.
    - If docset_max_chunks_per_doc_view > 0, we cap chunks per eligible
      document, again without reordering.

    Returns a new list of RankedResult when shaping is applied, or None to
    indicate "no-op / fail-open" so callers can preserve Phase 3 behavior.
    """
    if not getattr(runtime, "enable_docset_result_shaping", False):
        return None

    if not ranked_results:
        return None

    # Analysis must be present and non-degraded; otherwise we fail open.
    if analysis is None or getattr(analysis, "degraded", False):
        return None

    if not matches:
        return None

    # Canonical eligibility set: file_ids from DocumentMatch.
    eligible_ids: Set[str] = set()
    for m in matches:
        fid = getattr(m, "file_id", None)
        if fid:
            eligible_ids.add(str(fid))

    if not eligible_ids:
        return None

    # Stable document ordering derived solely from retrieval output.
    ordered_file_ids = _ordered_unique_file_ids(ranked_results)
    eligible_ordered_docs = [fid for fid in ordered_file_ids if fid in eligible_ids]
    if not eligible_ordered_docs:
        # Analysis reported matches that do not correspond to retrieved docs;
        # fail open and preserve Phase 3 behavior.
        return None

    max_docs = getattr(runtime, "docset_max_docs_returned", 0) or 0
    if max_docs > 0:
        active_doc_ids = eligible_ordered_docs[: max_docs]
    else:
        active_doc_ids = eligible_ordered_docs

    if not active_doc_ids:
        return None

    active_doc_set: Set[str] = set(active_doc_ids)
    max_chunks_per_doc = getattr(runtime, "docset_max_chunks_per_doc_view", 0) or 0
    chunks_per_doc: Dict[str, int] = {}

    shaped: List[RankedResult] = []
    for r in ranked_results:
        fid = getattr(r, "file_id", None)
        if not fid or fid not in active_doc_set:
            continue

        if max_chunks_per_doc > 0:
            used = chunks_per_doc.get(fid, 0)
            if used >= max_chunks_per_doc:
                continue
            chunks_per_doc[fid] = used + 1

        shaped.append(r)

    # If filtering produced an empty list, treat this as a no-op / fail-open.
    if not shaped:
        return None

    return shaped

