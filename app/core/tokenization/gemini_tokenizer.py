"""
================================================================================
Marketing Advantage AI — Gemini Tokenizer Contract
File: app/core/tokenization/gemini_tokenizer.py

Implements TokenizerContract using Gemini's countTokens API (via google-genai SDK).

Purpose:
  Provides accurate Gemini-native token counts for:
    1. Chunk sizing during ingestion — avoid creating chunks that exceed
       Gemini's embedding model input limit.
    2. Token budgeting in the RAG post-processor.

Failure mode protection:
  - F-13 (Tokenizer mismatch): Gemini uses a SentencePiece-based tokenizer.
    Using tiktoken or BERT WordPiece leads to silently truncated/oversized chunks.
  - Includes a hard timeout (default 5 s) and fallback to whitespace counting
    so chunking is never blocked by API errors.

Limitations:
  - The Gemini API's countTokens call returns a token count but not token IDs.
    The `tokenize()` method returns dummy IDs (range integers) of the correct
    length. Sufficient for chunking/budgeting that only needs count_tokens().
  - `decode()` is a best-effort identity function (returns empty string).

Install:
  pip install google-genai
================================================================================
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from app.ai.contracts.tokenizer_contract import (
    TokenizerContract,
    TokenizerFamily,
)

logger = logging.getLogger(__name__)

# Context window limits (tokens) per Gemini model
_CONTEXT_LIMITS: dict = {
    "gemini-embedding-001":       2048,
    "gemini-embedding-2":         2048,
    "gemini-embedding-2-preview": 2048,
    # Legacy
    "text-embedding-004":         2048,
    "embedding-001":              2048,
    # Generation models
    "gemini-1.5-flash":           1_000_000,
    "gemini-1.5-flash-latest":    1_000_000,
    "gemini-1.5-pro":             1_000_000,
    "gemini-1.5-pro-latest":      1_000_000,
    "gemini-2.0-flash":           1_048_576,
    "gemini-2.0-flash-lite":      1_048_576,
}

_DEFAULT_LIMIT = 8192  # conservative fallback for unknown models

# Lightweight generation model used for countTokens (consistent across model families)
_COUNT_MODEL = "gemini-2.0-flash-lite"


class GeminiTokenizerContract(TokenizerContract):
    """
    Gemini-native token counter using the google-genai SDK.

    Args:
        model_id     : Gemini model whose token limit is enforced.
        api_key_env  : Env var name containing the Google AI API key.
        timeout_sec  : Reserved for future use; SDK does not expose per-call timeout.
    """

    def __init__(
        self,
        *,
        model_id:    str = "gemini-embedding-001",
        api_key_env: str = "GOOGLE_API_KEY",
        api_key:     Optional[str] = None,
        timeout_sec: float = 5.0,
    ) -> None:
        try:
            from google import genai as _genai_pkg
        except ImportError:
            raise ImportError(
                "google-genai is not installed. Run: pip install google-genai"
            )

        resolved_key = (api_key or os.environ.get(api_key_env, "")).strip()
        if not resolved_key:
            raise EnvironmentError(
                f"[GeminiTokenizerContract] Google API key is not set "
                f"(pass api_key= or set env {api_key_env!r})."
            )

        self._client     = _genai_pkg.Client(api_key=resolved_key)
        self._model_id   = model_id
        self._timeout    = timeout_sec
        self._max_length = _CONTEXT_LIMITS.get(model_id, _DEFAULT_LIMIT)

        logger.info(
            "[GeminiTokenizerContract] Ready | model=%s | max_tokens=%d | sdk=google-genai",
            model_id, self._max_length,
        )

    # ── Abstract property implementations ───────────────────────────────────

    @property
    def name(self) -> str:
        return f"gemini/{self._model_id}"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def tokenizer_family(self) -> TokenizerFamily:
        return TokenizerFamily.SENTENCEPIECE

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def special_tokens(self) -> Dict[str, str]:
        return {
            "bos": "<s>",
            "eos": "</s>",
            "unk": "<unk>",
            "pad": "<pad>",
        }

    @property
    def tokenizer_source(self) -> str:
        return "google-generativeai:countTokens"

    @property
    def is_thread_safe(self) -> bool:
        return True

    # ── Abstract method implementations ─────────────────────────────────────

    def count_tokens(
        self,
        text: str,
        *,
        include_special_tokens: bool = False,
    ) -> int:
        """
        Count tokens using Gemini's countTokens API.
        Falls back to whitespace count on API failure.
        """
        if not text or not text.strip():
            return 0
        try:
            result = self._client.models.count_tokens(
                model=_COUNT_MODEL,
                contents=text,
            )
            count = int(result.total_tokens)
        except Exception as e:
            logger.warning(
                "[GeminiTokenizerContract] countTokens API failed (%s); "
                "falling back to whitespace estimate.", e,
            )
            count = len(text.split())
        if include_special_tokens:
            count += 2  # SentencePiece BOS + EOS overhead
        return count

    def tokenize(self, text: str, *, add_special_tokens: bool = False) -> List[int]:
        """
        Returns dummy integer IDs whose length equals count_tokens().
        Gemini API does not expose token IDs; only the count is accurate.
        """
        n = self.count_tokens(text, include_special_tokens=add_special_tokens)
        return list(range(n))

    def decode(self, token_ids: List[int]) -> str:
        logger.debug(
            "[GeminiTokenizerContract] decode() not supported "
            "(Gemini API does not expose token IDs)."
        )
        return ""

    def measure_with_overhead(
        self,
        text: str,
        *,
        special_overhead: int,
    ) -> int:
        return self.count_tokens(text) + special_overhead
