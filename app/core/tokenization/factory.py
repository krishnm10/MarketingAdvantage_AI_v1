# =============================================================================
# app/core/tokenization/factory.py — Tokenizer Factory (Strategy Pattern)
# =============================================================================
#
# Central factory for resolving and caching tokenizer backends.
#
# Configuration (env vars):
#   DEFAULT_TOKENIZER_BACKEND = huggingface | spacy | nltk | whitespace
#   HF_TOKENIZER_MODEL        = bert-base-multilingual-cased  (HuggingFace)
#   SPACY_MODEL                = en_core_web_sm                (spaCy)
#
# Usage:
#   from app.core.tokenization import get_tokenizer, count_tokens
#
#   tokenizer = get_tokenizer()                  # uses DEFAULT_TOKENIZER_BACKEND
#   tokenizer = get_tokenizer("huggingface")     # explicit backend
#   n = count_tokens("Hello world")              # convenience function
#   tokens = tokenize("Hello world")             # convenience function
#
# Design:
#   - Factory caches one instance per backend (lazy singleton)
#   - Graceful fallback: if requested backend fails to load → whitespace
#   - Thread-safe: all backend construction is idempotent
# =============================================================================

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from app.core.tokenization.base import BaseTokenizer, TokenizerBackend
from app.utils.logger import log_info, log_warning


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY — cached backend resolution
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=8)
def _build_backend(backend: str) -> BaseTokenizer:
    """
    Construct and cache a tokenizer backend instance.

    Lazy imports: heavy dependencies (transformers, spacy, nltk) are only
    imported when the corresponding backend is first requested.

    Falls back to WhitespaceTokenizer on any construction error.
    """
    from app.core.tokenization.backends import (
        HuggingFaceTokenizer,
        NLTKTokenizer,
        SpacyTokenizer,
        WhitespaceTokenizer,
    )

    _BACKEND_MAP: Dict[str, type] = {
        TokenizerBackend.HUGGINGFACE: HuggingFaceTokenizer,
        TokenizerBackend.SPACY: SpacyTokenizer,
        TokenizerBackend.NLTK: NLTKTokenizer,
        TokenizerBackend.WHITESPACE: WhitespaceTokenizer,
    }

    backend_lower = backend.strip().lower()
    cls = _BACKEND_MAP.get(backend_lower)

    if cls is None:
        valid = ", ".join(sorted(_BACKEND_MAP.keys()))
        log_warning(
            f"[Tokenization] Unknown backend '{backend}'. "
            f"Valid: {valid}. Falling back to whitespace."
        )
        return WhitespaceTokenizer()

    try:
        instance = cls()
        log_info(f"[Tokenization] Factory built backend: {instance.name}")
        return instance
    except Exception as e:
        log_warning(
            f"[Tokenization] Failed to build '{backend}' backend: {e}. "
            f"Falling back to whitespace tokenizer."
        )
        return WhitespaceTokenizer()


def _default_backend() -> str:
    """Resolve default backend from env var.

    Default is 'huggingface' for accurate subword tokenization.
    Falls back to 'whitespace' automatically if HuggingFace is unavailable
    (handled by _build_backend's graceful fallback).
    """
    return os.getenv("DEFAULT_TOKENIZER_BACKEND", "huggingface").strip().lower()


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def get_tokenizer(backend: Optional[str] = None) -> BaseTokenizer:
    """
    Get a tokenizer instance for the specified backend.

    Args:
        backend: One of "huggingface", "spacy", "nltk", "whitespace".
                 If None, reads DEFAULT_TOKENIZER_BACKEND from env.

    Returns:
        Cached BaseTokenizer instance. Falls back to whitespace on error.
    """
    resolved = (backend or _default_backend()).strip().lower()
    return _build_backend(resolved)


def count_tokens(text: str, backend: Optional[str] = None) -> int:
    """
    Count tokens using the specified (or default) tokenizer backend.

    Drop-in replacement for segmenter_v2.count_tokens() but with
    accurate subword tokenization instead of whitespace approximation.
    """
    return get_tokenizer(backend).count_tokens(text)


def tokenize(text: str, backend: Optional[str] = None) -> List[str]:
    """
    Tokenize text using the specified (or default) tokenizer backend.
    """
    return get_tokenizer(backend).tokenize(text)


def list_backends() -> List[str]:
    """Return sorted list of all supported backend names."""
    return sorted(e.value for e in TokenizerBackend)


def clear_tokenizer_cache() -> None:
    """Invalidate the backend LRU cache. Useful for testing."""
    _build_backend.cache_clear()


def get_bridge_for_contract(contract: "Any") -> BaseTokenizer:
    """
    Wrap a ``TokenizerContract`` in a ``ChunkingTokenizerBridge`` so that it
    can be passed to any caller that expects a ``BaseTokenizer``.

    This is the single entry-point for the Phase A migration bridge.
    Callers that receive the returned object are transparently using the
    embedding model's native tokenizer for token counting, replacing the
    generic ``bert-base-multilingual-cased`` approximation.

    Parameters
    ----------
    contract : TokenizerContract
        A fully-initialised ``TokenizerContract`` instance, typically
        sourced from ``EmbedderBundle.tokenizer`` via ``EmbedderRegistry``.

    Returns
    -------
    BaseTokenizer
        A ``ChunkingTokenizerBridge`` instance (subclass of ``BaseTokenizer``)
        that delegates all calls to ``contract``.

    Raises
    ------
    TypeError
        If ``contract`` is not a ``TokenizerContract`` instance.

    Example
    -------
    >>> from app.ai.registry import embedder_registry
    >>> from app.core.tokenization.factory import get_bridge_for_contract
    >>>
    >>> bundle = embedder_registry.get("BAAI/bge-large-en-v1.5")
    >>> tokenizer = get_bridge_for_contract(bundle.tokenizer)
    >>> n = tokenizer.count_tokens("Hello world")  # uses BGE's native tokenizer
    """
    from app.ai.contracts.chunking_bridge import ChunkingTokenizerBridge  # lazy — prevents circular import at module load
    return ChunkingTokenizerBridge(contract)
