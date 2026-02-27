"""
================================================================================
Marketing Advantage AI — Cohere Reranker
File: app/core/rerankers/cohere_v1.py

Cohere Rerank is the highest-quality cloud reranker API available.
It consistently tops enterprise retrieval benchmarks.

MODELS:
  - rerank-english-v3.0      → English only, highest English quality ✅
  - rerank-multilingual-v3.0 → 100+ languages (Hindi, Telugu, Tamil etc.)
  - rerank-english-v2.0      → older, cheaper fallback

FREE TIER:
  1,000 rerank calls/month free.
  Each call can rerank up to 1000 documents.

WHEN TO USE COHERE RERANKER:
  - Client needs maximum retrieval quality and can afford API cost
  - Multilingual Indian business documents (use multilingual-v3.0)
  - Regulatory/compliance use cases needing highest accuracy

Install:
  pip install cohere
================================================================================
"""

from __future__ import annotations

import logging
from typing import List, Optional

from app.core.rerankers.base import BaseReranker, RerankerInfo, RerankCandidate

logger = logging.getLogger(__name__)


class CohereReranker(BaseReranker):
    """
    Cohere Rerank API connector.

    Args:
        api_key:    Cohere API key (from env via factory — never hardcode).
        model:      Cohere reranker model name.
        max_chunks_per_doc: Max text segments per document
                            (split long docs if >512 tokens).
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "rerank-english-v3.0",
        max_chunks_per_doc: int = 1,
    ):
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "cohere not installed. Run: pip install cohere"
            )

        import cohere as _co

        self._client = _co.Client(api_key)
        self._model  = model
        self._max_chunks = int(max_chunks_per_doc)

        logger.info("[CohereReranker] Initialized | model=%s", model)

    @property
    def info(self) -> RerankerInfo:
        return RerankerInfo(provider="cohere", model=self._model)

    def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        *,
        top_k: int = 5,
    ) -> List[RerankCandidate]:
        """
        Call Cohere Rerank API to score all (query, passage) pairs.

        Cohere returns results indexed into the original list,
        so we map scores back by index (not by text matching).
        """
        if not candidates:
            logger.debug("[CohereReranker] No candidates to rerank.")
            return []

        top_k = min(int(top_k), len(candidates))
        docs  = [c.text for c in candidates]

        logger.debug(
            "[CohereReranker] Calling Cohere Rerank | model=%s | "
            "docs=%d | top_n=%d | query: %r",
            self._model, len(docs), top_k, query[:80],
        )

        res = self._client.rerank(
            query=query,
            documents=docs,
            top_n=top_k,
            model=self._model,
            max_chunks_per_doc=self._max_chunks,
        )

        # Cohere returns only top_n results (not all candidates)
        # Map scores back to original candidates by Cohere's index
        scored_indices = set()
        for r in res.results:
            idx = r.index
            candidates[idx].rerank_score = float(r.relevance_score)
            scored_indices.add(idx)

        # Assign 0.0 to any candidate Cohere didn't include in top_n
        for i, c in enumerate(candidates):
            if i not in scored_indices:
                c.rerank_score = 0.0

        # Sort all candidates desc and return top_k
        reranked = sorted(
            candidates,
            key=lambda c: c.rerank_score or 0.0,
            reverse=True,
        )

        logger.info(
            "[CohereReranker] Reranked %d → top %d | "
            "top_score=%.4f | bottom_score=%.4f",
            len(candidates), top_k,
            reranked[0].rerank_score or 0.0,
            reranked[top_k - 1].rerank_score or 0.0,
        )

        return reranked[:top_k]

    def health_check(self) -> bool:
        """Verify Cohere API is reachable with a minimal rerank call."""
        try:
            self._client.rerank(
                query="test",
                documents=["test document"],
                top_n=1,
                model=self._model,
            )
            return True
        except Exception as exc:
            logger.warning("[CohereReranker] health_check failed: %s", exc)
            return False
