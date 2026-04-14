# =============================================================================
# app/core/tokenization/base.py — Abstract Base Tokenizer
# =============================================================================
#
# Defines the contract that all tokenizer backends must implement.
# Keeps the interface minimal: tokenize() and count_tokens().
# =============================================================================

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class TokenizerBackend(str, Enum):
    """Supported tokenizer backends. Matches DEFAULT_TOKENIZER_BACKEND env var."""
    HUGGINGFACE = "huggingface"
    SPACY = "spacy"
    NLTK = "nltk"
    WHITESPACE = "whitespace"  # fallback — zero dependencies


class BaseTokenizer(ABC):
    """
    Abstract base class for all tokenizer backends.

    Every backend must implement:
      - tokenize(text) → List[str]  (split text into tokens)
      - count_tokens(text) → int    (return token count)
      - name → str                  (human-readable backend name)

    Implementations should be:
      - Thread-safe (no mutable shared state after __init__)
      - Deterministic (same input → same output)
      - CPU-optimized (no GPU requirements)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier."""
        ...

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Split text into a list of token strings.

        Args:
            text: Input text (any language, including Indic scripts).

        Returns:
            List of token strings. Empty list for empty input.
        """
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the text.

        For most backends this is equivalent to len(self.tokenize(text)),
        but some backends (e.g. HuggingFace) can count without materializing
        the full token list, which is faster for large texts.

        Args:
            text: Input text.

        Returns:
            Non-negative integer token count. 0 for empty input.
        """
        ...
