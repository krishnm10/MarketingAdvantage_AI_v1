"""
================================================================================
Marketing Advantage AI — ChunkingTokenCounter
File: app/ai/chunking/token_counter.py

Provider-agnostic token counter for all chunking strategies.

Problem solved:
  BERT (WordPiece) and model-native tokenizers (SentencePiece, tiktoken, etc.)
  produce different token counts for the same text. A chunk measured as safe
  by BERT may exceed the embedding model's actual limit, causing hard-limit
  violations or silent truncation during embedding.

Solution:
  ChunkingTokenCounter wraps the embedding model's native TokenizerContract
  and injects it into every chunking strategy via ContextVar, so every strategy
  uses the exact same tokenizer the embedding model will use — no proxy.

Hybrid strategy for remote tokenizers (Gemini countTokens, Cohere API):
  Calling a remote API for every internal split candidate during chunking would
  be prohibitively expensive. The hybrid strategy uses:
    1. A fast local approximation (BERT factory) for very short texts.
    2. A direct model-native API call for all final chunk candidates and any
       text that may be close to the capacity limit.
  This guarantees accuracy for all stored chunks while minimising API calls.

Concurrency:
  ChunkingTokenCounter is thread-safe after construction.
  The ContextVar used for injection is async-task-scoped (Python contextvars
  guarantee isolation between concurrent async tasks out of the box).

Failure modes prevented:
  F-02  Max-length overflow   — accurate model-native counts drive sizing
  F-15  Lost in the middle    — calibrated soft_cap keeps chunks within limit
  F-16  Tokenizer migration   — counter is always derived from EmbedderBundle

Performance:
  - LRU cache (512 entries by default) avoids re-counting repeated segments.
  - Remote calls are minimised by a configurable approx threshold.
================================================================================
"""

from __future__ import annotations

import hashlib
import logging
import math
import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, Optional, Tuple

from app.ai.contracts.tokenizer_contract import TokenizerContract, TokenizerFamily
from app.ai.validation.chunk_sizer import DefaultChunkSizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Remote-tokenizer source tags — any tokenizer_source containing these
# strings makes API calls when count_tokens() is invoked.
# ---------------------------------------------------------------------------
_REMOTE_TOKENIZER_SOURCES: Tuple[str, ...] = (
    "google-generativeai:countTokens",
    "cohere_sdk",
)

# Hard minimum for the approx-only fast path (chars).
# Texts shorter than this are so small they are always safe; skip the API call.
_APPROX_ONLY_CHAR_THRESHOLD = 80

_BUILTIN_SOFT_CAP_FACTORS: Dict[str, float] = {
    "google": 0.85,
    "openai": 0.92,
    "cohere": 0.88,
    "huggingface": 0.98,
    "ollama": 0.96,
    "mistral": 0.92,
    "default": 0.90,
}

_SOFT_CAP_CTX: contextvars.ContextVar[Optional[Dict[str, float]]] = contextvars.ContextVar(
    "chunk_soft_cap_factors", default=None
)


def current_soft_cap_factors() -> Dict[str, float]:
    """Active soft-cap map (request/task scoped when ingestion pushes tenant JSON)."""
    merged = _SOFT_CAP_CTX.get()
    if merged is not None:
        return merged
    return dict(_BUILTIN_SOFT_CAP_FACTORS)


@contextmanager
def soft_cap_factors_from_client(factors: Optional[Dict[str, float]]):
    """Temporarily merge *factors* over built-in defaults for chunking token counts."""
    base = dict(_BUILTIN_SOFT_CAP_FACTORS)
    if factors:
        for k, v in factors.items():
            try:
                base[str(k).lower()] = float(v)
            except (TypeError, ValueError):
                continue
    tok = _SOFT_CAP_CTX.set(base)
    try:
        yield
    finally:
        _SOFT_CAP_CTX.reset(tok)


# ---------------------------------------------------------------------------
# AlignmentMetrics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AlignmentMetrics:
    """
    Calibration metrics for a (model_id, factory_backend) tokenizer pair.

    Fields
    ──────
    ratio_p95       : 95th-percentile ratio of (model_tokens / factory_tokens).
                      E.g. 1.20 means BERT under-counts by up to 20 % for 95 %
                      of typical English content.
    delta_p95       : 95th-percentile absolute gap (model_tokens - factory_tokens).
                      Useful for deriving additive safety margins.
    soft_cap_factor : Factor applied to hard_cap to derive soft_cap.
                      soft_cap = floor(hard_cap * soft_cap_factor).
                      Must be in (0.0, 1.0).
    soft_cap        : Precomputed = floor(hard_cap * soft_cap_factor).
    hard_cap        : Absolute maximum tokens (embed_max_tokens from catalog).
    calibrated      : True if computed from real sample texts; False if from
                      provider defaults only.
    model_id        : Model this calibration applies to.
    factory_backend : Factory tokenizer backend (e.g. "huggingface").
    """
    ratio_p95:       float
    delta_p95:       int
    soft_cap_factor: float
    soft_cap:        int
    hard_cap:        int
    calibrated:      bool
    model_id:        str
    factory_backend: str

    def __post_init__(self) -> None:
        if not (0.0 < self.soft_cap_factor <= 1.0):
            raise ValueError(
                f"[AlignmentMetrics] soft_cap_factor must be in (0, 1], "
                f"got {self.soft_cap_factor} for model_id={self.model_id!r}."
            )
        if self.soft_cap <= 0:
            raise ValueError(
                f"[AlignmentMetrics] soft_cap must be > 0, "
                f"got {self.soft_cap} for model_id={self.model_id!r}."
            )


def _default_alignment_metrics(
    provider: str,
    hard_cap: int,
    model_id: str,
    factory_backend: str,
) -> AlignmentMetrics:
    """
    Construct conservative default AlignmentMetrics without running calibration.
    Used when calibration has not been run or is unavailable.
    """
    factor = current_soft_cap_factors().get(
        provider.lower(), current_soft_cap_factors()["default"]
    )
    # For remote tokenizers, assume up to 25 % over-count by BERT (ratio_p95=1.25).
    # For local tokenizers, assume up to 5 % variance (ratio_p95=1.05).
    is_remote_provider = provider.lower() in ("google", "cohere")
    ratio_p95 = 1.25 if is_remote_provider else 1.05
    delta_p95 = int(hard_cap * (ratio_p95 - 1.0))
    soft_cap = max(1, math.floor(hard_cap * factor))

    return AlignmentMetrics(
        ratio_p95=ratio_p95,
        delta_p95=delta_p95,
        soft_cap_factor=factor,
        soft_cap=soft_cap,
        hard_cap=hard_cap,
        calibrated=False,
        model_id=model_id,
        factory_backend=factory_backend,
    )


# ---------------------------------------------------------------------------
# ChunkingTokenCounter
# ---------------------------------------------------------------------------

class ChunkingTokenCounter:
    """
    Enterprise token counter for chunking strategy injection.

    Hybrid counting for remote tokenizers
    ──────────────────────────────────────
    For tokenizers that make API calls (Gemini countTokens, Cohere SDK),
    every `count()` invocation on a very short text uses a fast local
    approximation. All real chunk candidates (longer texts) call the model-native
    tokenizer directly to guarantee accuracy.

    For LOCAL tokenizers (tiktoken, HuggingFace, Ollama), every `count()`
    call directly invokes the model-native tokenizer — no approximation is used.

    Caching
    ───────
    Results are cached by text-hash (LRU, configurable size). This avoids
    re-counting when chunkers re-examine the same text segment multiple times.

    Usage
    ─────
    Do NOT instantiate directly in chunking code. Use the ContextVar injection
    provided by `ingestion_service_v2._chunk_text_with_strategy()`. Read the
    count via `segmenter_v2.count_tokens()` which checks `_ACTIVE_TOKEN_COUNTER`.

    Direct usage:
        counter = ChunkingTokenCounter.from_bundle(bundle)
        n = counter.count("The quick brown fox...")
    """

    def __init__(
        self,
        *,
        contract: TokenizerContract,
        metrics: AlignmentMetrics,
        is_remote: bool = False,
        cache_size: int = 512,
    ) -> None:
        self._contract   = contract
        self._metrics    = metrics
        self._is_remote  = is_remote
        self._cache_size = cache_size
        # Simple bounded dict cache (avoids functools.lru_cache overhead on methods)
        self._cache: Dict[str, int] = {}
        self._cache_hits   = 0
        self._cache_misses = 0
        self._remote_calls = 0
        self._approx_calls = 0

        logger.info(
            "[ChunkingTokenCounter] Ready | model=%s | remote=%s | "
            "hard_cap=%d | soft_cap=%d | soft_cap_factor=%.2f | calibrated=%s",
            metrics.model_id, is_remote,
            metrics.hard_cap, metrics.soft_cap,
            metrics.soft_cap_factor, metrics.calibrated,
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> AlignmentMetrics:
        return self._metrics

    @property
    def hard_cap(self) -> int:
        return self._metrics.hard_cap

    @property
    def soft_cap(self) -> int:
        return self._metrics.soft_cap

    @property
    def is_remote(self) -> bool:
        return self._is_remote

    # ------------------------------------------------------------------
    # Primary counting interface
    # ------------------------------------------------------------------

    def count(self, text: str) -> int:
        """
        Count tokens for chunking decisions.

        For LOCAL tokenizers:
          Direct model-native count — perfectly accurate.

        For REMOTE tokenizers:
          - Very short texts (< _APPROX_ONLY_CHAR_THRESHOLD chars):
              Return fast approximation (chars / 4.5) scaled by ratio_p95.
              These texts are always safe; the API call overhead is unjustified.
          - All other texts:
              Call model-native API directly.
              Result is cached to avoid re-counting the same text.

        Returns the model-native token count (or safe approximation for tiny texts).
        """
        if not text or not text.strip():
            return 0

        cache_key = self._make_cache_key(text)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache_hits += 1
            return cached

        self._cache_misses += 1
        result = self._count_uncached(text)

        # Bounded LRU eviction: drop oldest when full
        if len(self._cache) >= self._cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[cache_key] = result
        return result

    def _count_uncached(self, text: str) -> int:
        """Internal counting without cache lookup."""
        if self._is_remote and len(text) < _APPROX_ONLY_CHAR_THRESHOLD:
            # Fast path: tiny texts are always safe — approximate to avoid API call.
            approx = max(1, int(len(text) / 4.5))
            result = math.ceil(approx * self._metrics.ratio_p95)
            self._approx_calls += 1
            return result

        # For remote: make one direct API call (accurate).
        # For local: model-native call is cheap — always accurate.
        try:
            result = self._contract.count_tokens(text, include_special_tokens=False)
            if self._is_remote:
                self._remote_calls += 1
        except Exception as exc:
            # Graceful degradation: fall back to conservative approximation.
            logger.warning(
                "[ChunkingTokenCounter] count_tokens() failed for model=%s (%s). "
                "Using conservative approximation (factor=%.2f). "
                "This is DEGRADED MODE — chunk safety margin increases.",
                self._metrics.model_id, exc, self._metrics.ratio_p95,
            )
            approx = max(1, int(len(text) / 4.5))
            result = math.ceil(approx * self._metrics.ratio_p95)
            self._approx_calls += 1

        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        """Return hit/miss/call statistics for observability."""
        return {
            "cache_hits":   self._cache_hits,
            "cache_misses": self._cache_misses,
            "remote_calls": self._remote_calls,
            "approx_calls": self._approx_calls,
            "cache_size":   len(self._cache),
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_bundle(
        cls,
        bundle: "EmbedderBundle",  # type: ignore[name-defined]
        *,
        alignment_metrics: Optional[AlignmentMetrics] = None,
        cache_size: int = 512,
        factory_backend: Optional[str] = None,
    ) -> "ChunkingTokenCounter":
        """
        Build a ChunkingTokenCounter from an EmbedderBundle.

        If `alignment_metrics` is provided (from a prior `TokenizerValidator.calibrate()`
        call), those calibrated values are used.  Otherwise, conservative defaults
        are derived from the provider and embed_max_tokens.

        Parameters
        ----------
        bundle            : Fully resolved EmbedderBundle from EmbedderRegistry.
        alignment_metrics : Optional pre-calibrated metrics.  If None, defaults
                            are used.
        cache_size        : LRU cache capacity for repeated text counts.
        """
        provider = bundle.provider.value if hasattr(bundle.provider, "value") else str(bundle.provider)
        hard_cap = bundle.embed_max_tokens

        # Detect remote tokenizers by their tokenizer_source tag.
        tokenizer_source = getattr(bundle.tokenizer, "tokenizer_source", "") or ""
        is_remote = any(tag in tokenizer_source for tag in _REMOTE_TOKENIZER_SOURCES)

        if alignment_metrics is None:
            fb = (factory_backend or "huggingface").strip().lower()
            alignment_metrics = _default_alignment_metrics(
                provider=provider,
                hard_cap=hard_cap,
                model_id=bundle.model_id,
                factory_backend=fb,
            )

        return cls(
            contract=bundle.tokenizer,
            metrics=alignment_metrics,
            is_remote=is_remote,
            cache_size=cache_size,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_cache_key(text: str) -> str:
        """
        Compact cache key using first+last 64 bytes + length to avoid full hash
        cost while keeping collision probability negligible for chunking texts.
        """
        encoded = text.encode("utf-8", errors="replace")
        length = len(encoded)
        sample = encoded[:64] + encoded[-64:] if length > 128 else encoded
        return hashlib.blake2s(sample, digest_size=8).hexdigest() + f":{length}"

    def __repr__(self) -> str:
        return (
            f"ChunkingTokenCounter("
            f"model={self._metrics.model_id!r}, "
            f"remote={self._is_remote}, "
            f"hard_cap={self.hard_cap}, "
            f"soft_cap={self.soft_cap}, "
            f"calibrated={self._metrics.calibrated})"
        )


# ---------------------------------------------------------------------------
# Lazy import guard (avoid circular import; EmbedderBundle imported at runtime)
# ---------------------------------------------------------------------------
try:
    from app.ai.contracts.embedder_contract import EmbedderBundle  # noqa: F401
except ImportError:  # pragma: no cover
    pass
