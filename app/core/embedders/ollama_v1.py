"""
================================================================================
Marketing Advantage AI — Ollama Embedder
File: app/core/embedders/ollama_v1.py

Local embedding provider via Ollama.
Install:
  pip install ollama

Notes:
- Prefer Ollama's batch embed API when available.
- Fall back to the legacy single-prompt endpoint only for older clients.
================================================================================
"""

from __future__ import annotations

import logging
import time
from typing import List

from app.core.embedders.base import BaseEmbedder, EmbedderInfo, _l2_normalize

logger = logging.getLogger(__name__)


def _is_non_retryable_ollama_error(exc: Exception) -> bool:
    """Model missing / bad request — retrying will not help."""
    msg = str(exc).lower()
    return (
        "not found" in msg
        or "status code: 404" in msg
        or "status code: 400" in msg
        or "unknown model" in msg
    )


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
        self._supports_batch_embed = hasattr(self._client, "embed")

        # Determine dim once — retry with exponential backoff so a
        # momentarily-unavailable Ollama doesn't crash the whole pipeline.
        _MAX_RETRIES = 3
        _BACKOFF = [1, 2, 4]  # seconds
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                test = self.embed_query("dim_probe")
                self._dim = len(test)
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                if _is_non_retryable_ollama_error(exc):
                    logger.warning(
                        "[OllamaEmbedder] dim_probe failed (non-retryable): %s",
                        exc,
                    )
                    break
                wait = _BACKOFF[attempt] if attempt < len(_BACKOFF) else _BACKOFF[-1]
                logger.warning(
                    "[OllamaEmbedder] dim_probe attempt %d/%d failed (%s). "
                    "Retrying in %ds …",
                    attempt + 1, _MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)

        if last_err is not None:
            raise RuntimeError(
                f"[OllamaEmbedder] Failed to probe embedding dimension after "
                f"{_MAX_RETRIES} attempts. Is Ollama running at {base_url}? "
                f"Last error: {last_err}"
            ) from last_err

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
        if self._supports_batch_embed:
            response = self._client.embed(model=self._model, input=texts)
            return [
                _l2_normalize([float(x) for x in vec]) if self._normalize else [float(x) for x in vec]
                for vec in response["embeddings"]
            ]

        return [self._embed_one(text) for text in texts]
