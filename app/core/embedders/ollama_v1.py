"""
================================================================================
Marketing Advantage AI — Ollama Embedder
File: app/core/embedders/ollama_v1.py

Local embedding provider via Ollama.
Install:
  pip install ollama

Notes:
- Ollama embeddings do not always have a true batch endpoint, so we implement
  safe parallel batching via a small threadpool.
================================================================================
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from app.core.embedders.base import BaseEmbedder, EmbedderInfo, _l2_normalize


class OllamaEmbedder(BaseEmbedder):
    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        max_workers: int = 4,
        normalize: bool = True,
    ):
        try:
            import ollama
        except ImportError:
            raise ImportError("ollama not installed. Run: pip install ollama")

        import ollama as _ol

        self._client = _ol.Client(host=base_url)
        self._model = model
        self._max_workers = int(max_workers)
        self._normalize = bool(normalize)

        # Determine dim once (cheap + avoids config mismatch later)
        test = self._client.embeddings(model=self._model, prompt="dim_probe")["embedding"]
        self._dim = len(test)

    @property
    def info(self) -> EmbedderInfo:
        return EmbedderInfo(provider="ollama", model=self._model, dim=int(self._dim))
        
    @property
    def kind(self) -> str:
        return "ollama"


    def _embed_one(self, text: str) -> List[float]:
        vec = self._client.embeddings(model=self._model, prompt=text)["embedding"]
        vec = [float(x) for x in vec]
        return _l2_normalize(vec) if self._normalize else vec

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            return list(pool.map(self._embed_one, texts))
