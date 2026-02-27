"""
================================================================================
Marketing Advantage AI — Embedder Contracts
File: app/core/embedders/base.py

This is the standard interface for ALL embedding backends:
- HuggingFace / sentence-transformers
- Ollama embeddings
- OpenAI embeddings
- Cohere embeddings
- Any future enterprise embedding service

Important:
- NO DEFAULT embedder is selected here.
- Client config must pick embedder type + model explicitly.
================================================================================
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class EmbedderInfo:
    """Small metadata bundle to expose model identity and vector dimension."""
    provider: str
    model: str
    dim: int


class BaseEmbedder(abc.ABC):
    """
    All embedding providers implement this.

    We separate query vs document embedding because many modern embedders
    work better with different prompts/prefixes (e.g., e5: 'query:'/'passage:').
    """

    @property
    @abc.abstractmethod
    def info(self) -> EmbedderInfo:
        raise NotImplementedError

    @abc.abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a user query for retrieval."""
        raise NotImplementedError

    @abc.abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed document chunks for indexing."""
        raise NotImplementedError


def _l2_normalize(vec: List[float]) -> List[float]:
    """Pure-python L2 normalization (no numpy dependency)."""
    s = 0.0
    for v in vec:
        s += float(v) * float(v)
    if s <= 0.0:
        return vec
    inv = (s ** 0.5)
    return [float(v) / inv for v in vec]
