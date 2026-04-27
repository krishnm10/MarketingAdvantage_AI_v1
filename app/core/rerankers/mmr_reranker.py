"""
MMR (Maximal Marginal Relevance) Reranker
Diversifies results by penalizing similarity to already-selected documents.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.core.rerankers.base import BaseReranker, RerankCandidate, RerankerInfo

logger = logging.getLogger(__name__)


class MMRReranker(BaseReranker):
    """Maximal Marginal Relevance reranker for diversity."""

    def __init__(self, *, mmr_lambda: float = 0.7, **kwargs):
        self._lambda = max(0.0, min(1.0, mmr_lambda))

    @property
    def info(self) -> RerankerInfo:
        return RerankerInfo(
            provider="mmr",
            model=f"mmr(lambda={self._lambda})",
        )

    def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        *,
        top_k: int = 5,
    ) -> List[RerankCandidate]:
        if not candidates:
            return []

        selected: List[RerankCandidate] = []
        remaining = list(candidates)

        for _ in range(min(top_k, len(remaining))):
            best_idx = 0
            best_score = float("-inf")

            for i, cand in enumerate(remaining):
                relevance = cand.vector_score or 0.0
                max_sim = 0.0
                if selected:
                    max_sim = max(
                        self._text_similarity(cand.text, s.text)
                        for s in selected
                    )
                mmr_score = self._lambda * relevance - (1 - self._lambda) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            chosen = remaining.pop(best_idx)
            chosen.rerank_score = best_score
            selected.append(chosen)

        return selected

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """Simple Jaccard similarity as a lightweight proxy."""
        if not a or not b:
            return 0.0
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)
