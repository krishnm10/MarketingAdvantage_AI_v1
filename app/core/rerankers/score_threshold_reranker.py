"""
Score Threshold Reranker
Filters candidates below a configurable similarity/rerank score threshold.
"""
from __future__ import annotations

import logging
from typing import List

from app.core.rerankers.base import BaseReranker, RerankCandidate, RerankerInfo

logger = logging.getLogger(__name__)


class ScoreThresholdReranker(BaseReranker):
    """Filters candidates below a score threshold."""

    def __init__(self, *, score_threshold: float = 0.6, min_results: int = 1, **kwargs):
        self._threshold = score_threshold
        self._min_results = max(1, min_results)

    @property
    def info(self) -> RerankerInfo:
        return RerankerInfo(
            provider="score_threshold",
            model=f"score_threshold({self._threshold})",
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

        sorted_cands = sorted(
            candidates, key=lambda c: c.vector_score or 0.0, reverse=True
        )

        above_threshold = [
            c for c in sorted_cands if (c.vector_score or 0.0) >= self._threshold
        ]

        if len(above_threshold) >= self._min_results:
            result = above_threshold[:top_k]
        else:
            result = sorted_cands[: max(self._min_results, top_k)]

        for c in result:
            if c.rerank_score is None:
                c.rerank_score = c.vector_score

        logger.info(
            "[ScoreThresholdReranker] %d/%d candidates above threshold %.2f",
            len(above_threshold),
            len(candidates),
            self._threshold,
        )
        return result
