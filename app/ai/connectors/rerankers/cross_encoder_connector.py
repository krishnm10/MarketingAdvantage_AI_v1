# =============================================================================
# app/ai/connectors/rerankers/cross_encoder_connector.py
#
# CrossEncoderReranker — Phase 2 connector
#
# Wraps sentence-transformers CrossEncoder under the Phase 1 RerankerContract.
# Plugs into the RAG pipeline AFTER VectorDB ANN retrieval, not instead of it.
#
# Supported models (from reranker_catalog.yaml):
#   cross-encoder/ms-marco-MiniLM-L-6-v2   (fast, English, CPU-friendly)
#   cross-encoder/ms-marco-MiniLM-L-12-v2  (higher quality, CPU-friendly)
#   BAAI/bge-reranker-large                 (BGE, en+zh, GPU recommended)
#   BAAI/bge-reranker-v2-m3                 (multilingual, large context)
#
# Install:
#   pip install sentence-transformers
# =============================================================================

from __future__ import annotations

import logging
import time
from typing import List, Optional

from app.ai.contracts.reranker_contract import (
    RerankerCapabilities,
    RerankerCandidate,
    RerankerContract,
    RerankerInputTooLongError,
    RerankerProvider,
    RerankerScoreSpace,
    ScoredCandidate,
)

logger = logging.getLogger(__name__)

_DEFAULT_CAPS: dict = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": RerankerCapabilities(
        max_input_tokens_per_pair=512,
        score_space=RerankerScoreSpace.LOGIT,
        supports_batch_scoring=True,
        lang_support=["en"],
        tokenizer_family="wordpiece",
        notes="6-layer MiniLM; fast CPU cross-encoder.",
    ),
    "cross-encoder/ms-marco-MiniLM-L-12-v2": RerankerCapabilities(
        max_input_tokens_per_pair=512,
        score_space=RerankerScoreSpace.LOGIT,
        supports_batch_scoring=True,
        lang_support=["en"],
        tokenizer_family="wordpiece",
        notes="12-layer MiniLM; higher accuracy at moderate cost.",
    ),
    "BAAI/bge-reranker-large": RerankerCapabilities(
        max_input_tokens_per_pair=512,
        score_space=RerankerScoreSpace.LOGIT,
        supports_batch_scoring=True,
        lang_support=["en", "zh"],
        tokenizer_family="wordpiece",
        notes="BGE-large; strong en/zh performance.",
    ),
    "BAAI/bge-reranker-v2-m3": RerankerCapabilities(
        max_input_tokens_per_pair=8192,
        score_space=RerankerScoreSpace.LOGIT,
        supports_batch_scoring=True,
        lang_support=["*"],
        tokenizer_family="sentencepiece",
        notes="Multilingual BGE; 8K context; GPU recommended.",
    ),
}


class CrossEncoderReranker(RerankerContract):
    """
    Phase 1 cross-encoder reranker backed by sentence-transformers.

    Re-scores the top-K ANN candidates returned by the VectorDB; it does NOT
    perform its own retrieval. Cross-encoders read query and passage together,
    so they capture interaction signals that bi-encoder embeddings miss.

    Args:
        model_id   : HuggingFace model path.
        device     : "cpu", "cuda", or "mps".
        batch_size : Passages per forward pass (tune for GPU memory).
        max_length : Token truncation limit (default: model's catalog limit).
    """

    def __init__(
        self,
        *,
        model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        max_length: Optional[int] = None,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder as _CE
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

        self._model_id  = model_id
        self._device    = device
        self._batch_size = batch_size

        caps = _DEFAULT_CAPS.get(model_id)
        if caps is None:
            logger.warning(
                "[CrossEncoderReranker] model_id=%r not in capability table; "
                "using fallback caps (512 tokens, logit space).",
                model_id,
            )
            caps = RerankerCapabilities(
                max_input_tokens_per_pair=512,
                score_space=RerankerScoreSpace.LOGIT,
                supports_batch_scoring=True,
                lang_support=["en"],
                tokenizer_family="wordpiece",
            )
        self._capabilities = caps
        self._max_length = max_length or caps.max_input_tokens_per_pair

        logger.info(
            "[CrossEncoderReranker] Loading model=%r device=%s batch_size=%d",
            model_id, device, batch_size,
        )
        self._model = _CE(
            model_id,
            device=device,
            max_length=self._max_length,
        )
        logger.info("[CrossEncoderReranker] Model ready.")

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def capabilities(self) -> RerankerCapabilities:
        return self._capabilities

    def rerank(
        self,
        query: str,
        candidates: List[RerankerCandidate],
        *,
        top_k: int,
    ) -> List[ScoredCandidate]:
        if not candidates:
            return []

        t0 = time.perf_counter()

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, c.text) for c in candidates]

        scores: list = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

        # Pair original candidates with their scores; sort descending
        scored = [
            c.to_scored(float(s))
            for c, s in zip(candidates, scores)
        ]
        scored.sort(key=lambda x: x.rerank_score, reverse=True)
        result = scored[:top_k]

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "[CrossEncoderReranker] model=%r | in=%d | out=%d | top_score=%.4f | %.1fms",
            self._model_id,
            len(candidates),
            len(result),
            result[0].rerank_score if result else 0.0,
            elapsed,
        )
        return result

    def health_check(self) -> bool:
        try:
            test_result = self._model.predict([("health check", "test passage")])
            return test_result is not None
        except Exception as e:
            logger.warning("[CrossEncoderReranker] health_check failed: %s", e)
            return False
