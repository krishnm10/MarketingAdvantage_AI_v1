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
from typing import List, Optional

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
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        hf_token: Optional[str] = None,
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
        self._trust_remote_code = bool(trust_remote_code)
        self._revision = revision.strip() if isinstance(revision, str) and revision.strip() else None
        self._hf_token = hf_token.strip() if isinstance(hf_token, str) and hf_token.strip() else None

        # FIX-4 + FIX-5: do NOT load model here.
        # _load_model() is called on first embed call.
        # This keeps pipeline_factory.build() fast and non-blocking.
        self._m = None        # populated by _load_model()
        self._dim = None      # populated by _load_model()

        logger.info(
            "[HFSTEmbedder] Configured | model=%s | device=%s | "
            "batch=%d | normalize=%s | trust_remote_code=%s",
            model, self._device, batch_size, normalize, self._trust_remote_code,
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

    def _sentence_transformer_kwargs(self) -> dict:
        kwargs = {"device": self._device}
        if self._revision:
            kwargs["revision"] = self._revision
        if self._trust_remote_code:
            kwargs["trust_remote_code"] = True
        if self._hf_token:
            kwargs["token"] = self._hf_token
        return kwargs

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
        st_kwargs = self._sentence_transformer_kwargs()
        try:
            self._m = SentenceTransformer(self._model_name, **st_kwargs)
        except TypeError:
            # Older sentence-transformers may not accept top-level trust_remote_code.
            revision = st_kwargs.pop("revision", None)
            trust_remote_code = st_kwargs.pop("trust_remote_code", False)
            token = st_kwargs.pop("token", None)
            model_kwargs = {}
            tokenizer_kwargs = {}
            if trust_remote_code:
                model_kwargs["trust_remote_code"] = True
                tokenizer_kwargs["trust_remote_code"] = True
            if token:
                model_kwargs["token"] = token
                tokenizer_kwargs["token"] = token
            if revision:
                model_kwargs["revision"] = revision
                tokenizer_kwargs["revision"] = revision
            self._m = SentenceTransformer(
                self._model_name,
                device=self._device,
                model_kwargs=model_kwargs or None,
                tokenizer_kwargs=tokenizer_kwargs or None,
            )
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
        """
        PERF-FIX: Process in sub-batches of _batch_size to avoid blocking
        the executor thread indefinitely for very large inputs (1000+ texts).
        sentence_transformers handles internal batching per sub-call.
        GPU/CPU is selected by _resolve_device() at __init__ time.
        Returns: [[float, ...], [float, ...], ...]
        """
        if not texts:
            return []
        self._load_model()  # FIX-4: no-op after first call

        all_vecs: list = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            batch_vecs = self._m.encode(
                batch,
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
            ).tolist()
            all_vecs.extend(batch_vecs)

        if self._normalize:
            return [_l2_normalize(v) for v in all_vecs]
        return [[float(x) for x in v] for v in all_vecs]

    def __repr__(self) -> str:
        status = "loaded" if self._m is not None else "lazy/not-loaded"
        return (
            f"HuggingFaceSTEmbedder(model={self._model_name!r}, "
            f"device={self._device!r}, batch={self._batch_size}, "
            f"normalize={self._normalize}, trust_remote_code={self._trust_remote_code}, "
            f"status={status})"
        )
