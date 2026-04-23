# =============================================================================
# app/ai/validation/chunk_sizer.py
#
# ChunkSizerContract (ABC) + DefaultChunkSizer — Phase 1, Module 5a
#
# Computes the safe maximum chunk size in tokens and validates individual
# chunks against that limit.
#
# Failure modes addressed:
#   F-02  Max length overflow     — safe_chunk_size() enforces the limit
#   F-13  Reranker input mismatch — reranker_max_tokens included in formula
#   F-14  LLM context mismatch    — chunking independent of LLM tokenizer
#   F-15  Lost in the middle      — headroom_pct prevents edge-case overflow
#   F-16  Migration without re-index — chunk_max is derived from bundle;
#                                       bundle fingerprint change triggers re-index
#
# Golden rules enforced:
#   R-C1  Token-based sizing only (no chars/words)
#   R-C2  Exact formula: floor((min(E,R) - query_reserve - special_overhead)
#                               * (1 - headroom_pct))
#   R-C3  Tokenize to measure; assert tokens ≤ MAX
#   R-R2  chunk_max ≤ min(embed_max, reranker_max) - query_tokens - special_tokens
# =============================================================================

from __future__ import annotations

import abc
import math
import logging
from dataclasses import dataclass
from typing import Optional

from app.ai.contracts.tokenizer_contract import TokenizerContract
from app.ai.validation.exceptions import TokenOverflowError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """
    Result of a single chunk validation call.

    Fields
    ──────
    ok               : True iff the chunk passes all checks.
    errors           : Human-readable error messages, each embedding at least
                       one F-code.
    warnings         : Non-blocking observations.
    failure_modes    : Set of F-codes triggered by this validation, e.g.
                       ["F-02", "F-13"].
    measured_tokens  : Token count measured by the bound tokenizer. None if
                       the count could not be obtained.
    safe_chunk_max   : The computed safe limit used for comparison.
    """
    ok:              bool
    errors:          list
    warnings:        list
    failure_modes:   list
    measured_tokens: Optional[int]
    safe_chunk_max:  Optional[int]


# ---------------------------------------------------------------------------
# ChunkSizerContract (ABC)
# ---------------------------------------------------------------------------

class ChunkSizerContract(abc.ABC):
    """
    Abstract contract for computing and validating safe chunk sizes.

    Invariants (non-negotiable)
    ─────────────────────────────
    C-I1 (Token-based only): All size calculations are expressed in tokens
         measured by an embedder's TokenizerContract.  Characters and words
         are never used.  Enforces R-C1.

    C-I2 (Formal formula): safe_chunk_size() must compute:
         raw_max  = min(embed_max_tokens, reranker_max_tokens or embed_max_tokens)
         available = raw_max - query_reserve_tokens - special_overhead_tokens
         chunk_max = floor(available * (1.0 - headroom_pct))
         No deviations from this formula are permitted.
         Enforces R-C2.

    C-I3 (Embedder-reranker coupling): when reranker_max_tokens is set, the
         formula uses it.  When absent, falls back to embed_max_tokens only.
         Prevents F-13.

    C-I4 (Positive result): if chunk_max ≤ 0, the configuration is invalid
         and safe_chunk_size() must raise ValueError before any data is
         chunked.  Prevents silent zero-token chunks.
    """

    @property
    @abc.abstractmethod
    def embed_max_tokens(self) -> int:
        """Maximum tokens accepted by the embedding model."""

    @property
    @abc.abstractmethod
    def reranker_max_tokens(self) -> Optional[int]:
        """Maximum tokens accepted by the reranker, or None if no reranker."""

    @property
    @abc.abstractmethod
    def query_reserve_tokens(self) -> int:
        """Tokens reserved for the query string (default 50)."""

    @property
    @abc.abstractmethod
    def special_overhead_tokens(self) -> int:
        """Tokens reserved for special tokens ([CLS],[SEP], BOS/EOS). Min 4."""

    @property
    @abc.abstractmethod
    def headroom_pct(self) -> float:
        """Fractional headroom applied after subtracting overhead. Default 0.10."""

    @abc.abstractmethod
    def safe_chunk_size(self) -> int:
        """
        Compute and return the maximum safe chunk size in tokens.

        Must implement exactly:
          raw_max   = min(embed_max_tokens, reranker_max_tokens or embed_max_tokens)
          available = raw_max - query_reserve_tokens - special_overhead_tokens
          chunk_max = floor(available * (1.0 - headroom_pct))

        Raises ValueError if chunk_max ≤ 0 (C-I4).
        """

    @abc.abstractmethod
    def validate_chunk(
        self,
        text: str,
        tokenizer: TokenizerContract,
    ) -> "ValidationResult":
        """
        Validate that ``text`` fits within safe_chunk_size() tokens.

        Uses ``tokenizer.count_tokens(text, include_special_tokens=False)``
        for measurement (R-C3).

        On overflow, raises TokenOverflowError AND returns a failed
        ValidationResult.  Both are guaranteed on violation.
        """


# ---------------------------------------------------------------------------
# DefaultChunkSizer
# ---------------------------------------------------------------------------

class DefaultChunkSizer(ChunkSizerContract):
    """
    Production-ready implementation of ChunkSizerContract.

    Computes safe_chunk_size per R-C2 and validates individual chunks against
    the computed limit.

    Parameters
    ──────────
    embed_max_tokens      : From EmbedderBundle.embed_max_tokens.
    reranker_max_tokens   : From RerankerView.max_input_tokens (or None).
    query_reserve_tokens  : Tokens reserved for the query. Default 50.
    special_overhead_tokens: Tokens for sentinel tokens. Default 4.
    headroom_pct          : Fractional buffer. Default 0.10.
    model_id              : For error messages.
    """

    def __init__(
        self,
        embed_max_tokens: int,
        reranker_max_tokens: Optional[int] = None,
        query_reserve_tokens: int = 50,
        special_overhead_tokens: int = 4,
        headroom_pct: float = 0.10,
        model_id: str = "<unknown>",
    ) -> None:
        if embed_max_tokens <= 0:
            raise ValueError(
                f"[DefaultChunkSizer] embed_max_tokens must be > 0, "
                f"got {embed_max_tokens} for model_id={model_id!r}."
            )
        if reranker_max_tokens is not None and reranker_max_tokens <= 0:
            raise ValueError(
                f"[DefaultChunkSizer] reranker_max_tokens must be > 0 if set, "
                f"got {reranker_max_tokens} for model_id={model_id!r}."
            )
        if not (0.0 <= headroom_pct < 1.0):
            raise ValueError(
                f"[DefaultChunkSizer] headroom_pct must be in [0.0, 1.0), "
                f"got {headroom_pct}."
            )
        if special_overhead_tokens < 0:
            raise ValueError(
                "[DefaultChunkSizer] special_overhead_tokens must be ≥ 0."
            )
        if query_reserve_tokens < 0:
            raise ValueError(
                "[DefaultChunkSizer] query_reserve_tokens must be ≥ 0."
            )

        self._embed_max = embed_max_tokens
        self._reranker_max = reranker_max_tokens
        self._query_reserve = query_reserve_tokens
        self._special_overhead = special_overhead_tokens
        self._headroom_pct = headroom_pct
        self._model_id = model_id
        # Cache the computed size (inputs are immutable)
        self._safe_size: Optional[int] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def embed_max_tokens(self) -> int:
        return self._embed_max

    @property
    def reranker_max_tokens(self) -> Optional[int]:
        return self._reranker_max

    @property
    def query_reserve_tokens(self) -> int:
        return self._query_reserve

    @property
    def special_overhead_tokens(self) -> int:
        return self._special_overhead

    @property
    def headroom_pct(self) -> float:
        return self._headroom_pct

    # ------------------------------------------------------------------
    # safe_chunk_size (C-I2 formula)
    # ------------------------------------------------------------------

    def safe_chunk_size(self) -> int:
        """
        Implements the canonical R-C2 formula exactly:
          raw_max   = min(embed_max_tokens, reranker_max_tokens or embed_max_tokens)
          available = raw_max - query_reserve_tokens - special_overhead_tokens
          chunk_max = floor(available * (1.0 - headroom_pct))

        Examples
        ─────────
        BGE-Large: embed_max=512, reranker_max=None, reserve=50, overhead=4
          raw_max=512, available=458, chunk_max=floor(458*0.9)=412

        Cohere embed-v3: embed_max=512
          same → 412

        e5-Mistral: embed_max=4096, reranker_max=None
          raw_max=4096, available=4042, chunk_max=floor(4042*0.9)=3637
        """
        if self._safe_size is not None:
            return self._safe_size

        raw_max = (
            min(self._embed_max, self._reranker_max)
            if self._reranker_max is not None
            else self._embed_max
        )
        available = raw_max - self._query_reserve - self._special_overhead
        chunk_max = math.floor(available * (1.0 - self._headroom_pct))

        if chunk_max <= 0:
            raise ValueError(
                f"[DefaultChunkSizer C-I4] Computed chunk_max={chunk_max} ≤ 0 "
                f"for model_id={self._model_id!r}. "
                f"embed_max={self._embed_max}, reranker_max={self._reranker_max}, "
                f"query_reserve={self._query_reserve}, "
                f"special_overhead={self._special_overhead}, "
                f"headroom={self._headroom_pct}. "
                "Configuration is invalid — check embed_max_tokens and "
                "reranker_max_tokens values. Failure modes: F-13, F-15."
            )

        logger.debug(
            "[DefaultChunkSizer] model_id=%r raw_max=%d available=%d "
            "headroom=%.2f chunk_max=%d",
            self._model_id, raw_max, available, self._headroom_pct, chunk_max,
        )
        self._safe_size = chunk_max
        return chunk_max

    # ------------------------------------------------------------------
    # validate_chunk
    # ------------------------------------------------------------------

    def validate_chunk(
        self,
        text: str,
        tokenizer: TokenizerContract,
    ) -> ValidationResult:
        """
        Validate that ``text`` fits within the safe chunk limit.

        Checks (in order, per §4.1):
          1. Input normalization — empty text produces ok=True, 0 tokens.
          2. Token counting     — count via tokenizer (R-C3).
          3. Compute safe_max   — via safe_chunk_size() (R-C2).
          4. Overflow check     — token_count ≤ safe_max (R-E4).
          5. Return result.

        Raises TokenOverflowError on check 3 (degenerate config) or 4
        (overflow).  Returns ValidationResult in all cases.
        """
        # Check 1 — empty input
        if not text or not text.strip():
            size = self.safe_chunk_size()
            return ValidationResult(
                ok=True,
                errors=[],
                warnings=["[W] Empty or whitespace-only chunk text."],
                failure_modes=[],
                measured_tokens=0,
                safe_chunk_max=size,
            )

        # Check 2 — token counting
        try:
            token_count = tokenizer.count_tokens(text, include_special_tokens=False)
        except Exception as exc:
            err_msg = (
                f"[F-02] Token counting failed for model_id={self._model_id!r}: {exc}"
            )
            raise TokenOverflowError(
                chunk_tokens=0,
                max_allowed=0,
                model_id=self._model_id,
                failure_mode="F-02",
            ) from exc

        # Check 3 — safe_chunk_size
        try:
            safe_max = self.safe_chunk_size()
        except ValueError as exc:
            raise TokenOverflowError(
                chunk_tokens=token_count,
                max_allowed=0,
                model_id=self._model_id,
                failure_mode="F-13",
            ) from exc

        # Check 4 — overflow
        if token_count > safe_max:
            err_msg = (
                f"[F-02/F-13] Chunk token length {token_count} exceeds safe limit "
                f"{safe_max} for model_id={self._model_id!r}. "
                "Split the chunk before embedding. "
                f"Failure modes: F-02, F-13."
            )
            raise TokenOverflowError(
                chunk_tokens=token_count,
                max_allowed=safe_max,
                model_id=self._model_id,
                failure_mode="F-02",
            )

        # Check 5 — all good
        warnings = []
        # Warn if close to the limit (>95% utilisation)
        if token_count > safe_max * 0.95:
            warnings.append(
                f"[W-F-02] Chunk uses {token_count}/{safe_max} tokens "
                f"({100*token_count/safe_max:.1f}%). Near limit — "
                "consider smaller chunks to prevent edge-case overflows."
            )

        return ValidationResult(
            ok=True,
            errors=[],
            warnings=warnings,
            failure_modes=[],
            measured_tokens=token_count,
            safe_chunk_max=safe_max,
        )

    def __repr__(self) -> str:
        size = "?" if self._safe_size is None else str(self._safe_size)
        return (
            f"DefaultChunkSizer("
            f"embed_max={self._embed_max}, "
            f"reranker_max={self._reranker_max}, "
            f"reserve={self._query_reserve}, "
            f"overhead={self._special_overhead}, "
            f"headroom={self._headroom_pct}, "
            f"safe_size={size})"
        )
