"""
================================================================================
Marketing Advantage AI — OpenAI Embedder
File: app/core/embedders/openai_v1.py

Install:
  pip install openai

Supports:
  - text-embedding-3-small
  - text-embedding-3-large

Notes:
- OpenAI supports batch embedding (recommended for ingestion throughput).
================================================================================
"""

from __future__ import annotations

from typing import List, Optional

from app.core.embedders.base import BaseEmbedder, EmbedderInfo, _l2_normalize


_DIM_MAP = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(BaseEmbedder):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        organization: Optional[str] = None,
        normalize: bool = True,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")

        self._client = OpenAI(api_key=api_key, organization=organization)
        self._model = model
        self._normalize = bool(normalize)

        self._dim = int(_DIM_MAP.get(model, 0))
        if self._dim <= 0:
            # We can still run; dim will be discovered on first call.
            self._dim = 0

    @property
    def info(self) -> EmbedderInfo:
        dim = self._dim if self._dim > 0 else 0
        return EmbedderInfo(provider="openai", model=self._model, dim=dim)

    def _ensure_dim(self, vec: List[float]) -> None:
        if self._dim <= 0:
            self._dim = len(vec)

    def embed_query(self, text: str) -> List[float]:
        res = self._client.embeddings.create(model=self._model, input=[text])
        vec = [float(x) for x in res.data[0].embedding]
        self._ensure_dim(vec)
        return _l2_normalize(vec) if self._normalize else vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        res = self._client.embeddings.create(model=self._model, input=texts)

        # Ensure original order (some SDKs return ordered, but keep it safe)
        items = sorted(res.data, key=lambda d: d.index)
        vecs = []
        for it in items:
            v = [float(x) for x in it.embedding]
            self._ensure_dim(v)
            vecs.append(_l2_normalize(v) if self._normalize else v)
        return vecs
