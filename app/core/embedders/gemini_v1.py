"""
================================================================================
Marketing Advantage AI — Google Gemini Embedder
File: app/core/embedders/gemini_v1.py

Implements BaseEmbedder using the Google Generative AI text-embedding models.

Supported models (as of 2026):
  - gemini-embedding-001     (3072 dims, multilingual, stable — replaces text-embedding-004)
  - gemini-embedding-2       (3072 dims, latest stable generation)
  - gemini-embedding-2-preview (3072 dims, preview channel)

Migration note:
  text-embedding-004 was retired from the Gemini API in 2026 and is no longer
  available via v1 or v1beta. Use gemini-embedding-001 as the drop-in replacement.

Design:
  - Uses the new google-genai SDK (google.genai) which routes to the v1 API
    by default (the deprecated google.generativeai package defaulted to v1beta
    where gemini-embedding-001 is not available).
  - Uses task_type differentiation: RETRIEVAL_QUERY for queries,
    RETRIEVAL_DOCUMENT for chunks. Critical for asymmetric retrieval quality.
  - Batches document embeddings to stay within API limits (100 texts per call).
  - Normalises output vectors (L2) to match cosine similarity expectations.
  - Reads the API key exclusively from env vars — never hard-coded.

Install:
  pip install google-genai
================================================================================
"""

from __future__ import annotations

import logging
from typing import List

from app.core.embedders.base import BaseEmbedder, EmbedderInfo, _l2_normalize

logger = logging.getLogger(__name__)

# Known dimensions per model (all current Gemini embedding models are 3072-dim)
_DIM_MAP: dict = {
    "gemini-embedding-001":       3072,
    "gemini-embedding-2":         3072,
    "gemini-embedding-2-preview": 3072,
    # Legacy — kept for catalog backward-compatibility but will fail at runtime
    "text-embedding-004":         768,
    "embedding-001":              768,
}

# Google API hard-limit per batch embed call
class GeminiEmbedder(BaseEmbedder):
    """
    Google Gemini embedding connector using the google-genai SDK (v1 API).

    Args:
        api_key   : Google AI API key (resolved from env via factory).
        model     : Gemini embedding model ID.
        normalize : Whether to L2-normalise output vectors.
    """

    def __init__(
        self,
        *,
        api_key:   str,
        model:     str = "gemini-embedding-001",
        normalize: bool = True,
    ) -> None:
        try:
            from google import genai as _genai_pkg
            from google.genai import types as _types_pkg
        except ImportError:
            raise ImportError(
                "google-genai is not installed. "
                "Run: pip install google-genai"
            )

        self._client   = _genai_pkg.Client(api_key=api_key)
        self._types    = _types_pkg
        self._model    = model
        self._normalize = bool(normalize)
        self._dim      = _DIM_MAP.get(model, 0)

        logger.info(
            "[GeminiEmbedder] Initialized | model=%s | dim=%s | sdk=google-genai",
            model, self._dim or "auto",
        )

    # ── BaseEmbedder interface ──────────────────────────────────────────────

    @property
    def info(self) -> EmbedderInfo:
        return EmbedderInfo(provider="gemini", model=self._model, dim=self._dim)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a retrieval query.
        Uses task_type=RETRIEVAL_QUERY for asymmetric retrieval.
        """
        result = self._client.models.embed_content(
            model=self._model,
            contents=text,
            config=self._types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        vec = [float(v) for v in result.embeddings[0].values]
        if self._dim <= 0:
            self._dim = len(vec)
        return _l2_normalize(vec) if self._normalize else vec

    def _embed_one_document(self, text: str) -> List[float]:
        """Single-chunk RETRIEVAL_DOCUMENT embedding (reliable shape vs batched calls)."""
        result = self._client.models.embed_content(
            model=self._model,
            contents=text,
            config=self._types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        vec = [float(v) for v in result.embeddings[0].values]
        if self._dim <= 0:
            self._dim = len(vec)
        return _l2_normalize(vec) if self._normalize else vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document chunks for indexing.
        Uses task_type=RETRIEVAL_DOCUMENT for passage-side embedding.
        Batches calls to stay within the 100-text API limit.
        """
        if not texts:
            return []

        all_vecs: List[List[float]] = []
        for i in range(0, len(texts), _MAX_BATCH_SIZE):
            batch = texts[i : i + _MAX_BATCH_SIZE]
            result = self._client.models.embed_content(
                model=self._model,
                contents=batch,
                config=self._types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            batch_vecs = [[float(v) for v in emb.values] for emb in result.embeddings]

            # google-genai may return fewer embeddings than inputs for multi-item
            # `contents`; Chroma upsert requires 1:1 lengths — fall back per text.
            if len(batch_vecs) != len(batch):
                logger.warning(
                    "[GeminiEmbedder] embed_content returned %d embeddings for %d texts "
                    "(model=%s); embedding sequentially.",
                    len(batch_vecs),
                    len(batch),
                    self._model,
                )
                batch_vecs = [self._embed_one_document(t) for t in batch]
            else:
                if self._dim <= 0 and batch_vecs:
                    self._dim = len(batch_vecs[0])
                if self._normalize:
                    batch_vecs = [_l2_normalize(v) for v in batch_vecs]
            all_vecs.extend(batch_vecs)

        logger.debug(
            "[GeminiEmbedder] Embedded %d documents | model=%s | dim=%d",
            len(texts), self._model, self._dim,
        )
        return all_vecs
