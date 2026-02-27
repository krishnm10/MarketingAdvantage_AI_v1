"""
================================================================================
Marketing Advantage AI — Cohere Embedder
File: app/core/embedders/cohere_v1.py

Install:
  pip install cohere

Cohere embedding API supports specifying input_type:
  - "search_query"
  - "search_document"
This is ideal for enterprise-grade retrieval accuracy.
================================================================================
"""

from __future__ import annotations

from typing import List, Optional

from app.core.embedders.base import BaseEmbedder, EmbedderInfo, _l2_normalize


class CohereEmbedder(BaseEmbedder):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        normalize: bool = True,
    ):
        try:
            import cohere
        except ImportError:
            raise ImportError("cohere not installed. Run: pip install cohere")

        import cohere as _co

        self._client = _co.Client(api_key)
        self._model = model
        self._normalize = bool(normalize)
        self._dim = 0  # discovered after first call

    @property
    def info(self) -> EmbedderInfo:
        return EmbedderInfo(provider="cohere", model=self._model, dim=int(self._dim))

    def _ensure_dim(self, vec: List[float]) -> None:
        if self._dim <= 0:
            self._dim = len(vec)

    def embed_query(self, text: str) -> List[float]:
        res = self._client.embed(
            texts=[text],
            model=self._model,
            input_type="search_query",
        )
        vec = [float(x) for x in res.embeddings[0]]
        self._ensure_dim(vec)
        return _l2_normalize(vec) if self._normalize else vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        res = self._client.embed(
            texts=texts,
            model=self._model,
            input_type="search_document",
        )
        out = []
        for emb in res.embeddings:
            v = [float(x) for x in emb]
            self._ensure_dim(v)
            out.append(_l2_normalize(v) if self._normalize else v)
        return out
