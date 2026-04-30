"""
================================================================================
Marketing Advantage AI — HuggingFace SentenceTransformers Embedder
File: app/core/embedders/huggingface_st_v1.py

Best default-free open-source choice for enterprise deployments.

Install:
  pip install sentence-transformers

Supports:
  - all-MiniLM-L6-v2 (fast)
  - all-mpnet-base-v2 (strong)
  - BAAI/bge-* (very strong)
  - intfloat/e5-* (use PromptedEmbedder wrapper for query/passage prefixes)
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from app.core.embedders.base import BaseEmbedder, EmbedderInfo, _l2_normalize


class HuggingFaceSTEmbedder(BaseEmbedder):
    def __init__(
        self,
        *,
        model: str,
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

        self._model_name = model
        self._device = device
        self._batch_size = int(batch_size)
        self._normalize = bool(normalize)

        self._m = SentenceTransformer(model, device=device)
        self._dim = int(self._m.get_sentence_embedding_dimension())

    @property
    def info(self) -> EmbedderInfo:
        return EmbedderInfo(provider="huggingface-st", model=self._model_name, dim=self._dim)

    def embed_query(self, text: str) -> List[float]:
        vec = self._m.encode(text, normalize_embeddings=self._normalize).tolist()
        # extra guard: some models/versions may not normalize even if asked
        return _l2_normalize(vec) if self._normalize else [float(x) for x in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = self._m.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        ).tolist()
        if not self._normalize:
            return [[float(x) for x in v] for v in vecs]
        return [_l2_normalize([float(x) for x in v]) for v in vecs]
