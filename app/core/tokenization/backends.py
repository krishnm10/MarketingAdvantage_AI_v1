# =============================================================================
# app/core/tokenization/backends.py — Tokenizer Backend Implementations
# =============================================================================
#
# Three production backends + one zero-dependency fallback:
#
#   1. HuggingFaceTokenizer  — Rust-based Fast Tokenizers (RECOMMENDED)
#      Best for: raw speed on CPU, Indic scripts via ai4bharat/indic-bert
#      Deps: tokenizers (already installed), transformers
#
#   2. SpacyTokenizer        — spaCy linguistic tokenizer
#      Best for: linguistically-aware tokenization, sentence boundary detection
#      Deps: spacy + language model (e.g. en_core_web_sm)
#
#   3. NLTKTokenizer         — NLTK word tokenizer + Indic NLP fallback
#      Best for: research-grade tokenization, Indic NLP Library integration
#      Deps: nltk (already installed)
#
#   4. WhitespaceTokenizer   — Zero-dependency fallback
#      Best for: ultra-fast approximate counting when no library is available
#
# All backends are:
#   - Lazy-loaded (heavy imports only happen when the backend is first used)
#   - Thread-safe after construction
#   - CPU-only (no GPU required)
# =============================================================================

from __future__ import annotations

import os
import re
import unicodedata
from typing import List, Optional

from app.core.tokenization.base import BaseTokenizer
from app.utils.logger import log_info, log_warning


# ─────────────────────────────────────────────────────────────────────────────
# INDIC SCRIPT DETECTION — shared across backends
# ─────────────────────────────────────────────────────────────────────────────

# Unicode block ranges for major Indic scripts
_INDIC_RANGES = (
    ("\u0900", "\u097F"),  # Devanagari (Hindi, Marathi, Sanskrit)
    ("\u0980", "\u09FF"),  # Bengali
    ("\u0A00", "\u0A7F"),  # Gurmukhi (Punjabi)
    ("\u0A80", "\u0AFF"),  # Gujarati
    ("\u0B00", "\u0B7F"),  # Odia
    ("\u0B80", "\u0BFF"),  # Tamil
    ("\u0C00", "\u0C7F"),  # Telugu
    ("\u0C80", "\u0CFF"),  # Kannada
    ("\u0D00", "\u0D7F"),  # Malayalam
)

_INDIC_PATTERN = re.compile(
    "[" + "".join(f"{lo}-{hi}" for lo, hi in _INDIC_RANGES) + "]"
)


def contains_indic(text: str) -> bool:
    """Fast check: does text contain any Indic script characters?"""
    return bool(_INDIC_PATTERN.search(text))


def _normalize_unicode(text: str) -> str:
    """NFKC normalize — collapses compatibility characters."""
    return unicodedata.normalize("NFKC", text)


# ─────────────────────────────────────────────────────────────────────────────
# 1. HUGGING FACE FAST TOKENIZER (RECOMMENDED — Rust-based, CPU-optimized)
# ─────────────────────────────────────────────────────────────────────────────

class HuggingFaceTokenizer(BaseTokenizer):
    """
    Production tokenizer using HuggingFace's Rust-based Fast Tokenizers.

    Model selection (env: HF_TOKENIZER_MODEL):
      - Default: "bert-base-multilingual-cased" (104 languages, Indic included)
      - Indic-optimized: "ai4bharat/indic-bert" (lower fertility on Indic scripts)
      - English-only: "bert-base-uncased"

    Why this is fast:
      - tokenizers library is written in Rust, called from Python via FFI
      - Encodes 1M tokens/sec on a single CPU core
      - No GPU required, no torch dependency for tokenization alone

    Fertility control for Indic scripts:
      - Standard BERT wordpiece can split a single Hindi word into 5-8 subwords
      - ai4bharat/indic-bert has a vocabulary trained on Indic text → 1-3 subwords
      - The factory auto-selects indic-bert when Indic text is detected (optional)
    """

    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or os.getenv(
            "HF_TOKENIZER_MODEL", "bert-base-multilingual-cased"
        )
        self._tokenizer = None  # lazy load

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None:
            return
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                use_fast=True,  # Force Rust-based Fast Tokenizer
            )
            log_info(
                f"[Tokenization] HuggingFace Fast Tokenizer loaded: "
                f"{self._model_name} (use_fast=True)"
            )
        except Exception as e:
            log_warning(
                f"[Tokenization] Failed to load HuggingFace tokenizer "
                f"'{self._model_name}': {e}. Falling back to whitespace."
            )
            raise

    @property
    def name(self) -> str:
        return f"huggingface:{self._model_name}"

    def tokenize(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        self._ensure_loaded()
        text = _normalize_unicode(text)
        encoding = self._tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return self._tokenizer.convert_ids_to_tokens(encoding["input_ids"])

    def count_tokens(self, text: str) -> int:
        if not text or not text.strip():
            return 0
        self._ensure_loaded()
        text = _normalize_unicode(text)
        # encode() returns just the IDs — faster than full __call__
        return len(self._tokenizer.encode(text, add_special_tokens=False))


# ─────────────────────────────────────────────────────────────────────────────
# 2. SPACY TOKENIZER (Linguistic tokenization)
# ─────────────────────────────────────────────────────────────────────────────

class SpacyTokenizer(BaseTokenizer):
    """
    spaCy-based linguistic tokenizer.

    Uses spaCy's rule-based tokenizer which is language-aware and handles
    contractions, punctuation attachment, and script boundaries properly.

    Model selection (env: SPACY_MODEL):
      - Default: "xx_sent_ud_sm" (multi-language) or "en_core_web_sm"
      - Falls back to blank "xx" model if no model is installed

    Note: spaCy tokenization is ~10x slower than HuggingFace Fast Tokenizers
    but produces linguistically cleaner token boundaries.
    """

    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or os.getenv("SPACY_MODEL", "en_core_web_sm")
        self._nlp = None  # lazy load

    def _ensure_loaded(self) -> None:
        if self._nlp is not None:
            return
        try:
            import spacy
            try:
                self._nlp = spacy.load(
                    self._model_name,
                    disable=["ner", "parser", "lemmatizer", "textcat"],
                )
            except OSError:
                log_warning(
                    f"[Tokenization] spaCy model '{self._model_name}' not found. "
                    f"Falling back to blank 'xx' (multi-language) model."
                )
                self._nlp = spacy.blank("xx")
            # Increase max length for large documents
            self._nlp.max_length = 5_000_000
            log_info(f"[Tokenization] spaCy tokenizer loaded: {self._model_name}")
        except ImportError:
            raise ImportError(
                "spaCy is not installed. Install with: pip install spacy"
            )

    @property
    def name(self) -> str:
        return f"spacy:{self._model_name}"

    def tokenize(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        self._ensure_loaded()
        text = _normalize_unicode(text)
        doc = self._nlp(text)
        return [token.text for token in doc if not token.is_space]

    def count_tokens(self, text: str) -> int:
        if not text or not text.strip():
            return 0
        self._ensure_loaded()
        text = _normalize_unicode(text)
        doc = self._nlp(text)
        return sum(1 for token in doc if not token.is_space)


# ─────────────────────────────────────────────────────────────────────────────
# 3. NLTK TOKENIZER (with Indic NLP Library fallback)
# ─────────────────────────────────────────────────────────────────────────────

class NLTKTokenizer(BaseTokenizer):
    """
    NLTK-based word tokenizer with optional Indic NLP Library integration.

    Strategy:
      1. If text contains Indic script AND indic_nlp_library is installed → use it
      2. Otherwise → use nltk.word_tokenize (Penn Treebank style)

    Indic NLP Library handles:
      - Devanagari conjuncts (Hindi, Marathi, Sanskrit)
      - Telugu/Kannada script-specific tokenization rules
      - Proper compound word splitting for Indic morphology
    """

    def __init__(self):
        self._nltk_ready = False
        self._indic_available = None  # None = not yet checked

    def _ensure_loaded(self) -> None:
        if self._nltk_ready:
            return
        try:
            import nltk
            # Ensure punkt_tab tokenizer data is available
            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
            self._nltk_ready = True
            log_info("[Tokenization] NLTK tokenizer ready (punkt_tab)")
        except ImportError:
            raise ImportError(
                "NLTK is not installed. Install with: pip install nltk"
            )

    def _check_indic_available(self) -> bool:
        if self._indic_available is None:
            try:
                from indicnlp.tokenize import indic_tokenize  # noqa: F401
                self._indic_available = True
                log_info("[Tokenization] Indic NLP Library available for Indic scripts")
            except ImportError:
                self._indic_available = False
        return self._indic_available

    def _tokenize_indic(self, text: str) -> List[str]:
        """Tokenize Indic-script text using indic_nlp_library."""
        from indicnlp.tokenize import indic_tokenize
        tokens = indic_tokenize.trivial_tokenize(text)
        return [t for t in tokens if t.strip()]

    @property
    def name(self) -> str:
        return "nltk"

    def tokenize(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        self._ensure_loaded()
        text = _normalize_unicode(text)

        # Check if text has Indic script and Indic NLP is available
        if contains_indic(text) and self._check_indic_available():
            return self._tokenize_indic(text)

        from nltk.tokenize import word_tokenize
        return word_tokenize(text)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))


# ─────────────────────────────────────────────────────────────────────────────
# 4. WHITESPACE TOKENIZER (Zero-dependency fallback)
# ─────────────────────────────────────────────────────────────────────────────

class WhitespaceTokenizer(BaseTokenizer):
    """
    Zero-dependency fallback tokenizer using whitespace splitting.

    This is the same approach as the existing count_tokens() in segmenter_v2.py.
    Kept for backward compatibility and as an ultra-fast fallback when no
    ML libraries are available.

    Performance: ~10M tokens/sec (pure Python string split).
    Accuracy: approximate — does not handle subword tokenization, punctuation
    attachment, or script-specific boundaries.
    """

    @property
    def name(self) -> str:
        return "whitespace"

    def tokenize(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        return text.split()

    def count_tokens(self, text: str) -> int:
        if not text or not text.strip():
            return 0
        return len(text.split())
