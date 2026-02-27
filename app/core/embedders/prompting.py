"""
================================================================================
Marketing Advantage AI — Prompted Embedder Wrapper
File: app/core/embedders/prompting.py

Wrap any BaseEmbedder to apply query/document prefixes or instructions.

Examples:
- e5 models: query_prefix="query: ", document_prefix="passage: "
- BGE: query_prefix="Represent this sentence for searching relevant passages: "
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.core.embedders.base import BaseEmbedder, EmbedderInfo


@dataclass(frozen=True)
class EmbeddingPrompts:
    query_prefix: str = ""
    document_prefix: str = ""


class PromptedEmbedder(BaseEmbedder):
    def __init__(self, base: BaseEmbedder, prompts: EmbeddingPrompts):
        self._base = base
        self._prompts = prompts

    @property
    def info(self) -> EmbedderInfo:
        # keep original provider/model but mark wrapper in provider for debugging
        i = self._base.info
        return EmbedderInfo(provider=f"{i.provider}+prompted", model=i.model, dim=i.dim)

    def embed_query(self, text: str) -> List[float]:
        return self._base.embed_query(f"{self._prompts.query_prefix}{text}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [f"{self._prompts.document_prefix}{t}" for t in texts]
        return self._base.embed_documents(prefixed)
