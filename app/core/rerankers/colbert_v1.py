"""
================================================================================
Marketing Advantage AI — ColBERT Reranker
File: app/core/rerankers/colbert_v1.py

WHAT IS COLBERT?
  ColBERT (Contextualized Late Interaction over BERT) is fundamentally
  different from CrossEncoders:

  CrossEncoder:  encodes (query + passage) TOGETHER → 1 score
  ColBERT:       encodes query and passage SEPARATELY → token-level
                 late interaction → MaxSim score

  This makes ColBERT:
  - Much faster than CrossEncoder at inference time
  - More accurate than BiEncoders (SentenceTransformers)
  - Excellent for long documents (token-level matching)

SUPPORTED BACKENDS (we support both):
  1. RAGatouille  → simplest, wraps ColBERT cleanly  (pip install ragatouille)
  2. stanford-oval/ColBERT → original implementation (pip install colbert-ai)

RECOMMENDED MODELS:
  - colbert-ir/colbertv2.0          → best quality ✅
  - jinaai/jina-colbert-v2          → multilingual ColBERT (great for India)
================================================================================
"""

from __future__ import annotations

import logging
from typing import List, Optional

from app.core.rerankers.base import BaseReranker, RerankerInfo, RerankCandidate

logger = logging.getLogger(__name__)


class ColBERTReranker(BaseReranker):
    """
    ColBERT late-interaction reranker.

    Uses RAGatouille as the primary backend (easiest to install and use).
    Falls back to raw ColBERT library if RAGatouille is not available.

    Args:
        model_name:  ColBERT model from HuggingFace hub.
        index_root:  Optional path for ColBERT index cache.
        device:      'cpu' or 'cuda'.
        batch_size:  Passages per scoring batch.
    """

    def __init__(
        self,
        *,
        model_name: str = "colbert-ir/colbertv2.0",
        index_root: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 16,
    ):
        self._model_name = model_name
        self._batch_size = int(batch_size)
        self._device     = device
        self._model      = None
        self._backend    = None

        # ── Try RAGatouille first (recommended) ──────────────────────
        try:
            from ragatouille import RAGPretrainedModel
            self._model   = RAGPretrainedModel.from_pretrained(model_name)
            self._backend = "ragatouille"
            logger.info(
                "[ColBERTReranker] Loaded via RAGatouille | model=%s",
                model_name,
            )
            return
        except ImportError:
            logger.info(
                "[ColBERTReranker] RAGatouille not found. "
                "Trying colbert-ai backend..."
            )

        # ── Fallback: raw ColBERT library ────────────────────────────
        try:
            from colbert.infra import Run, RunConfig, ColBERTConfig
            from colbert.modeling.checkpoint import Checkpoint
            self._colbert_config = ColBERTConfig(
                doc_maxlen=220,
                nbits=2,
                root=index_root or ".colbert_cache",
            )
            self._checkpoint = Checkpoint(
                model_name,
                colbert_config=self._colbert_config,
            )
            self._backend = "colbert-ai"
            logger.info(
                "[ColBERTReranker] Loaded via colbert-ai | model=%s",
                model_name,
            )
            return
        except ImportError:
            pass

        # ── Neither backend available ─────────────────────────────────
        raise ImportError(
            "ColBERT backend not found. Install one of:\n"
            "  pip install ragatouille          (recommended)\n"
            "  pip install colbert-ai           (original implementation)"
        )

    @property
    def info(self) -> RerankerInfo:
        return RerankerInfo(
            provider=f"colbert[{self._backend}]",
            model=self._model_name,
        )

    def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        *,
        top_k: int = 5,
    ) -> List[RerankCandidate]:
        if not candidates:
            logger.debug("[ColBERTReranker] No candidates to rerank.")
            return []

        top_k = min(int(top_k), len(candidates))

        logger.debug(
            "[ColBERTReranker] Reranking %d candidates via %s | query: %r",
            len(candidates), self._backend, query[:80],
        )

        if self._backend == "ragatouille":
            return self._rerank_ragatouille(query, candidates, top_k)
        elif self._backend == "colbert-ai":
            return self._rerank_colbert_ai(query, candidates, top_k)

        raise RuntimeError(
            f"[ColBERTReranker] Unknown backend: {self._backend}"
        )

    def _rerank_ragatouille(
        self,
        query: str,
        candidates: List[RerankCandidate],
        top_k: int,
    ) -> List[RerankCandidate]:
        """Rerank using RAGatouille .rerank() interface."""

        # Build docs list for RAGatouille
        # RAGatouille expects: [{"content": str, "doc_id": str, ...}, ...]
        docs = [
            {"content": c.text, "doc_id": str(i)}
            for i, c in enumerate(candidates)
        ]

        results = self._model.rerank(
            query=query,
            documents=docs,
            k=top_k,
        )

        # RAGatouille returns [{content, score, rank, doc_id}, ...]
        # Map scores back to our candidates by doc_id (positional index)
        scored = {}
        for r in results:
            idx = int(r["doc_id"])
            scored[idx] = float(r.get("score", r.get("result_score", 0.0)))

        for i, c in enumerate(candidates):
            c.rerank_score = scored.get(i, 0.0)

        reranked = sorted(
            candidates,
            key=lambda c: c.rerank_score or 0.0,
            reverse=True,
        )

        logger.info(
            "[ColBERTReranker] Reranked %d → top %d | "
            "top_score=%.4f",
            len(candidates), top_k,
            reranked[0].rerank_score or 0.0,
        )

        return reranked[:top_k]

    def _rerank_colbert_ai(
        self,
        query: str,
        candidates: List[RerankCandidate],
        top_k: int,
    ) -> List[RerankCandidate]:
        """Rerank using raw colbert-ai Checkpoint.score()."""
        import torch

        # Encode query once
        Q = self._checkpoint.queryFromText([query])

        passages = [c.text for c in candidates]
        D = self._checkpoint.docFromText(
            passages,
            bsize=self._batch_size,
        )

        # MaxSim scoring: each query token finds best matching doc token
        scores = self._checkpoint.score(Q, D).tolist()
        if not isinstance(scores, list):
            scores = [scores]

        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)

        reranked = sorted(
            candidates,
            key=lambda c: c.rerank_score or 0.0,
            reverse=True,
        )

        logger.info(
            "[ColBERTReranker] Reranked %d → top %d | "
            "top_score=%.4f",
            len(candidates), top_k,
            reranked[0].rerank_score or 0.0,
        )

        return reranked[:top_k]
