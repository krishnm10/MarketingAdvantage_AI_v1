"""
================================================================================
Marketing Advantage AI — CrossEncoder Reranker
File: app/core/rerankers/crossencoder_v1.py

WHAT IS A CROSSENCODER?
  A CrossEncoder jointly encodes (query, passage) together using full
  cross-attention — meaning every query token attends to every passage
  token. This is much more accurate than comparing separately-encoded
  embeddings (like a BiEncoder / SentenceTransformer).

  The trade-off: CrossEncoders are ~100x slower than vector similarity,
  which is why they are only used on a small candidate set (top-20).

RECOMMENDED MODELS (all free, local):
  - cross-encoder/ms-marco-MiniLM-L-6-v2   → fastest, great for prod
  - cross-encoder/ms-marco-MiniLM-L-12-v2  → best quality/speed balance ✅
  - cross-encoder/ms-marco-electra-base     → strong, larger

Install:
  pip install sentence-transformers
================================================================================
"""

from __future__ import annotations

import logging
from typing import List, Optional

from app.core.rerankers.base import BaseReranker, RerankerInfo, RerankCandidate

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """
    HuggingFace CrossEncoder reranker via sentence-transformers.

    Args:
        model_name: CrossEncoder model ID from HuggingFace hub.
        device:     'cpu' or 'cuda' or 'mps'.
        max_length: Max token length for (query + passage) pair.
                    Truncates longer passages rather than erroring.
        batch_size: How many pairs to score in one forward pass.
                    Increase on GPU, decrease on CPU if memory is tight.
    """

    def __init__(
        self,
        *,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 16,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

        self._model_name = model_name
        self._batch_size = int(batch_size)
        self._model = CrossEncoder(
            model_name,
            device=device,
            max_length=int(max_length),
        )

        logger.info(
            "[CrossEncoderReranker] Loaded | model=%s | device=%s | "
            "batch_size=%d | max_length=%d",
            model_name, device, batch_size, max_length,
        )

    @property
    def info(self) -> RerankerInfo:
        return RerankerInfo(provider="crossencoder", model=self._model_name)

    def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        *,
        top_k: int = 5,
    ) -> List[RerankCandidate]:
        """
        Score each (query, passage) pair and return top_k sorted desc.

        The CrossEncoder outputs a single relevance score per pair.
        Higher = more relevant. No fixed scale (model-dependent).
        """
        if not candidates:
            logger.debug("[CrossEncoderReranker] No candidates to rerank.")
            return []

        # Clamp top_k to available candidates
        top_k = min(int(top_k), len(candidates))

        # Build (query, passage) pairs — one per candidate
        pairs = [(query, c.text) for c in candidates]

        logger.debug(
            "[CrossEncoderReranker] Scoring %d pairs for query: %r",
            len(pairs), query[:80],
        )

        # Batch score all pairs in one forward pass (efficient on CPU)
        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

        # Attach scores to candidates
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)

        # Sort descending by rerank_score and return top_k
        reranked = sorted(
            candidates,
            key=lambda c: c.rerank_score or 0.0,
            reverse=True,
        )

        logger.info(
            "[CrossEncoderReranker] Reranked %d → top %d | "
            "top_score=%.4f | bottom_score=%.4f",
            len(candidates), top_k,
            reranked[0].rerank_score or 0.0,
            reranked[top_k - 1].rerank_score or 0.0,
        )

        return reranked[:top_k]
