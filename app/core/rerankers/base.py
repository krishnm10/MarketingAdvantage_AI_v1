"""
================================================================================
Marketing Advantage AI — Reranker Base Contract
File: app/core/rerankers/base.py

PURPOSE:
  Defines the strict interface every reranker backend must implement.

WHY RERANKING MATTERS IN ENTERPRISE RAG:
  Vector similarity search is fast but imprecise — cosine distance between
  embeddings is a proxy for relevance, not a guarantee.

  Rerankers are cross-attention models that DIRECTLY score (query, passage)
  pairs. They are much slower than vector search but dramatically more
  accurate. The correct pattern is always:

    VectorDB search → top-20 candidates (fast, high recall)
         ↓
    Reranker        → top-5 results    (slow, high precision)
         ↓
    LLM             → answer from top-5 (best quality context)

  Without a reranker, your LLM receives noisy context.
  With a reranker, it receives the most relevant passages.

INTERFACE DESIGN:
  rerank() accepts a query + list of candidates (from VectorDB search),
  returns the same list trimmed to top_k and sorted by rerank_score.

  Candidates in  → [{"id", "text", "score", "metadata"}, ...]
  Candidates out → same dicts + "rerank_score" added, sorted desc, top_k only
================================================================================
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RerankerInfo:
    """Identity metadata for a loaded reranker."""
    provider: str
    model:    str


@dataclass
class RerankCandidate:
    """
    Normalized candidate document for reranking.

    Attributes:
        id:            Document ID (from VectorDB hit).
        text:          Raw document text chunk.
        vector_score:  Original cosine similarity score from VectorDB.
        rerank_score:  Score assigned by the reranker (higher = more relevant).
                       None before reranking.
        metadata:      Original document metadata from VectorDB.
    """
    id:           str
    text:         str
    vector_score: float
    metadata:     Dict[str, Any]
    rerank_score: Optional[float] = None

    @classmethod
    def from_vector_hit(cls, hit: Dict[str, Any]) -> "RerankCandidate":
        """
        Build a RerankCandidate from a raw VectorDB hit dict.
        Compatible with VectorHit objects from app/core/vectordb/base.py.
        """
        return cls(
            id=str(hit.get("id", "")),
            text=str(hit.get("text", "")),
            vector_score=float(hit.get("score", 0.0)),
            metadata=dict(hit.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize back to dict for downstream consumers."""
        return {
            "id":           self.id,
            "text":         self.text,
            "score":        self.rerank_score   # rerank_score replaces score
                            if self.rerank_score is not None
                            else self.vector_score,
            "vector_score": self.vector_score,
            "rerank_score": self.rerank_score,
            "metadata":     self.metadata,
        }


class BaseReranker(abc.ABC):
    """All reranker connectors implement this interface."""

    @property
    @abc.abstractmethod
    def info(self) -> RerankerInfo:
        raise NotImplementedError

    @abc.abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        *,
        top_k: int = 5,
    ) -> List[RerankCandidate]:
        """
        Score and sort candidates by relevance to the query.

        Args:
            query:      User search query string.
            candidates: List of RerankCandidate from VectorDB search.
            top_k:      Number of results to return (trim after sorting).

        Returns:
            Top-k RerankCandidates sorted descending by rerank_score.
            Each candidate has rerank_score populated.
        """
        raise NotImplementedError

    def health_check(self) -> bool:
        """
        Optional quick sanity check (model loaded, API reachable).
        Override in connectors that make network calls.
        """
        return True

    # ── Shared utility ────────────────────────────────────────────────────

    @staticmethod
    def from_vector_hits(hits: List[Any]) -> List[RerankCandidate]:
        """
        Convert a list of VectorHit objects OR raw dicts to RerankCandidates.
        Accepts both so callers don't need to convert manually.
        """
        candidates = []
        for h in hits:
            if isinstance(h, RerankCandidate):
                candidates.append(h)
            elif hasattr(h, "id"):
                # VectorHit dataclass from vectordb/base.py
                candidates.append(
                    RerankCandidate(
                        id=str(h.id),
                        text=str(h.text),
                        vector_score=float(h.score),
                        metadata=dict(h.metadata),
                    )
                )
            elif isinstance(h, dict):
                candidates.append(RerankCandidate.from_vector_hit(h))
            else:
                raise TypeError(
                    f"[Reranker] Cannot convert type '{type(h).__name__}' "
                    "to RerankCandidate."
                )
        return candidates
