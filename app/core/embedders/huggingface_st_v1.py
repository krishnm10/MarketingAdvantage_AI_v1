"""
================================================================================
Marketing Advantage AI — HuggingFace SentenceTransformers Embedder
File: app/core/embedders/huggingface_st_v1.py

FIXES APPLIED (vs original):
  FIX-1: Removed unused imports — dataclass, Optional
  FIX-2: Added kind = "huggingface-st" class attr
         (pipeline_factory AssembledPipeline.__repr__ + _EmbedderAdapter.kind)
  FIX-3: Added embedding_dim property
         (ingestion_service_v2 STEP 6 skips redundant dim probe)
  FIX-4: Lazy model load — SentenceTransformer() moved out of __init__
         (pipeline_factory.build() no longer blocks on import)
  FIX-5: self._dim = None at __init__, set in _load_model()
  FIX-6: Added _resolve_device() — "auto" maps to CUDA → MPS → CPU
  FIX-7: device="cpu" default → "auto"

Install:
pip install sentence-transformers
================================================================================
"""

from __future__ import annotations

import logging
from typing import List

from app.core.embedders.base import BaseEmbedder, EmbedderInfo, _l2_normalize

logger = logging.getLogger(__name__)


class HuggingFaceSTEmbedder(BaseEmbedder):

    # FIX-2: Required by _EmbedderAdapter.kind in ingestion_service_v2.py
    # and by AssembledPipeline log line: f"... via {vectordb.kind}"
    kind = "huggingface-st"

    def __init__(
        self,
        *,
        model: str,
        device: str = "auto",      # FIX-7: was "cpu", now "auto"
        batch_size: int = 32,
        normalize: bool = True,
    ):
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

        self._model_name = model
        self._device = self._resolve_device(device)   # FIX-6
        self._batch_size = int(batch_size)
        self._normalize = bool(normalize)

        # FIX-4 + FIX-5: do NOT load model here.
        # _load_model() is called on first embed call.
        # This keeps pipeline_factory.build() fast and non-blocking.
        self._m = None        # populated by _load_model()
        self._dim = None      # populated by _load_model()

        logger.info(
            "[HFSTEmbedder] Configured | model=%s | device=%s | "
            "batch=%d | normalize=%s",
            model, self._device, batch_size, normalize,
        )

    # ------------------------------------------------------------------
    # FIX-6: Device resolver — SentenceTransformer does NOT accept "auto"
    # in all versions. Resolve here before passing to ST constructor.
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(device: str) -> str:
        if device and device.lower() != "auto":
            return device  # explicit: "cpu", "cuda", "cuda:1", "mps"
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    # ------------------------------------------------------------------
    # FIX-4: Lazy loader — called on first embed, not on __init__
    # Thread-safe: no mutable state shared across calls during inference.
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        if self._m is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info(
            "[HFSTEmbedder] Loading '%s' on %s ...", self._model_name, self._device
        )
        self._m = SentenceTransformer(self._model_name, device=self._device)
        self._dim = int(self._m.get_sentence_embedding_dimension())   # FIX-5
        logger.info("[HFSTEmbedder] ✅ Loaded | dim=%d", self._dim)

    # ------------------------------------------------------------------
    # BaseEmbedder interface
    # ------------------------------------------------------------------

    @property
    def info(self) -> EmbedderInfo:
        return EmbedderInfo(
            provider="huggingface-st",
            model=self._model_name,
            dim=self._dim or 0,     # 0 before first load — pipeline_factory probes
        )

    # FIX-3: Direct property consumed by ingestion_service_v2 STEP 6:
    #   embedding_dim = getattr(embedder._emb, "embedding_dim", None)
    # When non-zero, the dim probe (embed_query "dimension probe") is SKIPPED.
    # After pipeline_factory.build() calls embed_query() once, self._dim is set
    # and all subsequent ingestion calls skip the redundant probe.
    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            self._load_model()
        return self._dim

    def embed_query(self, text: str) -> List[float]:
        self._load_model()   # FIX-4: no-op after first call
        vec = self._m.encode(
            text,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        ).tolist()
        return _l2_normalize(vec) if self._normalize else [float(x) for x in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self._load_model()   # FIX-4: no-op after first call
        vecs = self._m.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        ).tolist()
        if not self._normalize:
            return [[float(x) for x in v] for v in vecs]
        return [_l2_normalize([float(x) for x in v]) for v in vecs]

    def __repr__(self) -> str:
        status = "loaded" if self._m is not None else "lazy/not-loaded"
        return (
            f"HuggingFaceSTEmbedder(model={self._model_name!r}, "
            f"device={self._device!r}, batch={self._batch_size}, "
            f"normalize={self._normalize}, status={status})"
        )
