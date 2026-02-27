"""
================================================================================
Marketing Advantage AI — BGE Reranker Connector
File: app/core/rerankers/bge_reranker_v1.py

BAAI BGE Reranker is currently the highest-quality FREE reranker
on the MTEB benchmark. It significantly outperforms standard CrossEncoders
on retrieval tasks while remaining completely free and local.

RECOMMENDED MODELS:
  - BAAI/bge-reranker-base      → balanced quality/speed (768M params) ✅
  - BAAI/bge-reranker-large     → best quality, slower
  - BAAI/bge-reranker-v2-m3     → multilingual (great for Hindi/Telugu/Tamil)
  - BAAI/bge-reranker-v2-gemma  → highest quality, needs more VRAM

WHY bge-reranker-v2-m3 IS IDEAL FOR INDIAN ENTERPRISE:
  It supports 100+ languages including Hindi, Telugu, Tamil, Kannada,
  Bengali — making it perfect for multilingual Indian business content.

Install:
  pip install sentence-transformers FlagEmbedding
================================================================================
"""

from __future__ import annotations

import logging
from typing import List

from app.core.rerankers.base import BaseReranker, RerankerInfo, RerankCandidate

logger = logging.getLogger(__name__)


class BGEReranker(BaseReranker):
    """
    BAAI BGE Reranker — state-of-the-art free local reranker.

    Internally uses FlagEmbedding for bge-reranker-v2 models
    and falls back to sentence-transformers CrossEncoder for older models,
    so both model families are supported transparently.

    Args:
        model_name:  BGE reranker model ID from HuggingFace hub.
        device:      'cpu' or 'cuda' or 'mps'.
        max_length:  Token limit per (query, passage) pair.
        batch_size:  Pairs per forward pass.
        use_fp16:    Use FP16 for faster GPU inference (ignored on CPU).
    """

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 16,
        use_fp16: bool = False,
    ):
        self._model_name = model_name
        self._batch_size = int(batch_size)
        self._device = device

        # Try FlagEmbedding first (supports newer bge-reranker-v2 models)
        # Fall back to sentence-transformers CrossEncoder for older models
        self._model = None
        self._use_flag = False

        try:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(
                model_name,
                use_fp16=use_fp16 and device != "cpu",
            )
            self._use_flag = True
            logger.info(
                "[BGEReranker] Loaded via FlagEmbedding | model=%s | device=%s",
                model_name, device,
            )
        except ImportError:
            # FlagEmbedding not installed — fall back to CrossEncoder
            logger.info(
                "[BGEReranker] FlagEmbedding not found. "
                "Falling back to sentence-transformers CrossEncoder."
            )
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(
                    model_name,
                    device=device,
                    max_length=max_length,
                )
                self._use_flag = False
                logger.info(
                    "[BGEReranker] Loaded via CrossEncoder | model=%s | device=%s",
                    model_name, device,
                )
            except ImportError:
                raise ImportError(
                    "Neither FlagEmbedding nor sentence-transformers is installed.\n"
                    "Run one of:\n"
                    "  pip install FlagEmbedding\n"
                    "  pip install sentence-transformers"
                )

    @property
    def info(self) -> RerankerInfo:
        return RerankerInfo(provider="bge_reranker", model=self._model_name)

    def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        *,
        top_k: int = 5,
    ) -> List[RerankCandidate]:
        if not candidates:
            logger.debug("[BGEReranker] No candidates to rerank.")
            return []

        top_k = min(int(top_k), len(candidates))
        pairs = [(query, c.text) for c in candidates]

        logger.debug(
            "[BGEReranker] Scoring %d pairs via %s | query: %r",
            len(pairs),
            "FlagEmbedding" if self._use_flag else "CrossEncoder",
            query[:80],
        )

        if self._use_flag:
            # FlagEmbedding returns raw float scores
            scores = self._model.compute_score(
                pairs,
                batch_size=self._batch_size,
                normalize=True,          # normalize to [0, 1] for interpretability
            )
        else:
            # CrossEncoder returns numpy floats
            scores = self._model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
            )

        # Ensure scores is iterable list of floats
        if not isinstance(scores, list):
            try:
                scores = scores.tolist()
            except AttributeError:
                scores = [float(scores)]

        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)

        reranked = sorted(
            candidates,
            key=lambda c: c.rerank_score or 0.0,
            reverse=True,
        )

        logger.info(
            "[BGEReranker] Reranked %d → top %d | "
            "top_score=%.4f | bottom_score=%.4f",
            len(candidates), top_k,
            reranked[0].rerank_score or 0.0,
            reranked[top_k - 1].rerank_score or 0.0,
        )

        return reranked[:top_k]
