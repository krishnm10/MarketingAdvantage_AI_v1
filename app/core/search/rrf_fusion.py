"""
Reciprocal Rank Fusion (RRF) — merges two ranked result lists into one.

Reference:
  Cormack, Clarke & Büttcher (2009). "Reciprocal Rank Fusion outperforms
  Condorcet and individual Rank Learning methods."

Used by RAGPipeline.query() when SearchMode.HYBRID is active.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class FusedHit:
    """One fused result after RRF merging."""
    id: str
    text: str
    rrf_score: float
    vector_score: float
    keyword_score: float
    metadata: Dict[str, Any]


def reciprocal_rank_fusion(
    vector_hits: List[Dict[str, Any]],
    keyword_hits: List[Dict[str, Any]],
    *,
    alpha: float = 0.7,
    k_constant: int = 60,
    top_k: int = 20,
) -> List[FusedHit]:
    """
    Merge vector and keyword results using weighted RRF.

    Args:
        vector_hits:  List of dicts with at least ``id``, ``text``, ``score``, ``metadata``.
        keyword_hits: List of dicts with at least ``id``, ``text``, ``score``, ``metadata``.
        alpha:        Weight for vector results (1-alpha for keyword). Default 0.7.
        k_constant:   RRF constant (higher → less steep rank decay). Default 60.
        top_k:        Number of fused results to return.

    Returns:
        List of FusedHit sorted by descending rrf_score.
    """
    scores: Dict[str, float] = {}
    docs: Dict[str, Dict[str, Any]] = {}
    vec_scores: Dict[str, float] = {}
    kw_scores: Dict[str, float] = {}

    # Accumulate vector hits
    for rank, hit in enumerate(vector_hits):
        doc_id = hit["id"]
        rrf = alpha / (k_constant + rank + 1)
        scores[doc_id] = scores.get(doc_id, 0.0) + rrf
        vec_scores[doc_id] = hit.get("score", 0.0)
        if doc_id not in docs:
            docs[doc_id] = hit

    # Accumulate keyword hits
    for rank, hit in enumerate(keyword_hits):
        doc_id = hit["id"]
        rrf = (1.0 - alpha) / (k_constant + rank + 1)
        scores[doc_id] = scores.get(doc_id, 0.0) + rrf
        kw_scores[doc_id] = hit.get("score", 0.0)
        if doc_id not in docs:
            docs[doc_id] = hit

    # Sort by combined RRF score
    ranked_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]

    return [
        FusedHit(
            id=doc_id,
            text=docs[doc_id].get("text", ""),
            rrf_score=scores[doc_id],
            vector_score=vec_scores.get(doc_id, 0.0),
            keyword_score=kw_scores.get(doc_id, 0.0),
            metadata=docs[doc_id].get("metadata", {}),
        )
        for doc_id in ranked_ids
    ]
