"""
================================================================================
Marketing Advantage AI — FlashRank Reranker
File: app/core/rerankers/flashrank_v1.py

FlashRank is an ultra-lightweight, ultra-fast reranker library designed
specifically for production latency requirements.

WHY FLASHRANK:
  - 4x faster than standard CrossEncoder on CPU
  - Tiny memory footprint (ONNX quantized models)
  - Sub-20ms reranking of 20 candidates on CPU
  - Perfect for real-time user-facing APIs

RECOMMENDED MODELS:
  - ms-marco-MiniLM-L-12-v2    → best quality in FlashRank ✅
  - ms-marco-MultiBERT-L-12    → multilingual
  - rank-T5-flan               → T5-based, different strengths
  - ms-marco-TinyBERT-L-2-v2  → absolute fastest, lowest quality

Install:
  pip install flashrank
================================================================================
"""

from __future__ import annotations

import logging
from typing import List

from app.core.rerankers.base import BaseReranker, RerankerInfo, RerankCandidate

logger = logging.getLogger(__name__)


class FlashRankReranker(BaseReranker):
    """
    FlashRank reranker — optimized for production latency on CPU.

    Args:
        model_name: FlashRank model name (NOT a HuggingFace path —
                    FlashRank has its own model registry).
        max_length: Token limit per passage.
        cache_dir:  Directory to cache FlashRank models
                    (default: ~/.cache/flashrank).
    """

    def __init__(
        self,
        *,
        model_name: str = "ms-marco-MiniLM-L-12-v2",
        max_length: int = 512,
        cache_dir: str = None,
    ):
        try:
            from flashrank import Ranker
        except ImportError:
            raise ImportError(
                "flashrank not installed. Run: pip install flashrank"
            )

        from flashrank import Ranker

        init_kwargs = {
            "model_name": model_name,
            "max_length": int(max_length),
        }
        if cache_dir:
            init_kwargs["cache_dir"] = cache_dir

        self._ranker = Ranker(**init_kwargs)
        self._model_name = model_name

        logger.info(
            "[FlashRankReranker] Loaded | model=%s | max_length=%d",
            model_name, max_length,
        )

    @property
    def info(self) -> RerankerInfo:
        return RerankerInfo(provider="flashrank", model=self._model_name)

    def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        *,
        top_k: int = 5,
    ) -> List[RerankCandidate]:
        """
        Rerank using FlashRank's optimized ONNX inference.

        FlashRank expects passages as list of dicts:
          [{"id": ..., "text": ..., "meta": {...}}, ...]

        Returns candidates sorted by FlashRank score descending.
        """
        if not candidates:
            logger.debug("[FlashRankReranker] No candidates to rerank.")
            return []

        from flashrank import RerankRequest

        top_k = min(int(top_k), len(candidates))

        # Build FlashRank passage list — store original index in meta
        # so we can map scores back to our candidates
        passages = [
            {
                "id": str(i),           # FlashRank uses positional id
                "text": c.text,
                "meta": {"original_index": i},
            }
            for i, c in enumerate(candidates)
        ]

        request = RerankRequest(query=query, passages=passages)

        logger.debug(
            "[FlashRankReranker] Reranking %d passages | query: %r",
            len(passages), query[:80],
        )

        results = self._ranker.rerank(request)

        # Map FlashRank scores back to original candidates by positional ID
        for result in results:
            original_idx = int(result["meta"]["original_index"])
            candidates[original_idx].rerank_score = float(result["score"])

        # Set a small negative score for any candidates FlashRank didn't score
        for c in candidates:
            if c.rerank_score is None:
                c.rerank_score = -1.0

        reranked = sorted(
            candidates,
            key=lambda c: c.rerank_score or 0.0,
            reverse=True,
        )

        logger.info(
            "[FlashRankReranker] Reranked %d → top %d | "
            "top_score=%.4f | bottom_score=%.4f",
            len(candidates), top_k,
            reranked[0].rerank_score or 0.0,
            reranked[top_k - 1].rerank_score or 0.0,
        )

        return reranked[:top_k]
