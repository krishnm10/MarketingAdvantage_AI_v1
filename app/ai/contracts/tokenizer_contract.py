# =============================================================================
# app/ai/contracts/tokenizer_contract.py
#
# TokenizerContract — Phase 1, Module 1
#
# The single canonical tokenizer contract for the Phase 1 embedder system.
# Every tokenizer used by the embedding layer must implement this ABC.
#
# MASTER RULE: Each component uses ONLY its own native tokenizer for its own
# purpose.  No cross-component tokenizer sharing is permitted.
#
# Failure modes addressed:
#   F-01  Vocabulary mismatch  — model_id + family enforce native binding
#   F-02  Max-length overflow  — max_length + count_tokens invariant
#   F-03  Special token injection — special_tokens fidelity invariant
#   F-13  Reranker input mismatch — accurate counts via measure_with_overhead
#   F-14  LLM context mismatch  — family isolation separates tokenizer scopes
#   F-15  Lost in the middle    — correct max_length drives safe chunk sizing
#   F-16  Migration without re-index — model_id + family are versioned fields
#
# Golden rules enforced:
#   R-E1  Never use a third-party tokenizer with an embedding model
#   R-E2  Always use the model's bundled tokenizer
#   R-E3  Special tokens must match exactly per family
#   R-E4  Respect model max sequence length — never rely on auto-truncation
#   R-C3  Tokenize to measure; assert tokens ≤ MAX
# =============================================================================

from __future__ import annotations

import abc
from enum import Enum
from typing import Dict, List


# ---------------------------------------------------------------------------
# TokenizerResolutionError
# Raised when a tokenizer cannot be loaded, identified, or bound to a model.
# ---------------------------------------------------------------------------

class TokenizerResolutionError(RuntimeError):
    """
    Raised when a tokenizer cannot be resolved for a given model.

    Fields
    ------
    model_id  : canonical model identifier (e.g. "BAAI/bge-large-en-v1.5")
    provider  : provider string (e.g. "huggingface", "ollama")
    reason    : human-readable explanation of the failure
    """

    def __init__(self, model_id: str, provider: str, reason: str) -> None:
        self.model_id = model_id
        self.provider = provider
        self.reason = reason
        super().__init__(
            f"[TokenizerResolutionError] model_id={model_id!r} "
            f"provider={provider!r} — {reason}"
        )


# ---------------------------------------------------------------------------
# TokenizerFamily
# Enum of all supported tokenizer families. A family determines vocabulary,
# special-token conventions, and subword splitting rules.
# Mixing families between components is a hard failure (F-01).
# ---------------------------------------------------------------------------

class TokenizerFamily(str, Enum):
    """
    Canonical tokenizer family identifiers.

    Values must match the `tokenizer_family` field in embedder_catalog.yaml.
    """
    TIKTOKEN      = "tiktoken"       # OpenAI / Qwen BPE variant; cl100k_base, o200k_base
    WORDPIECE     = "wordpiece"      # BERT, DistilBERT, Cohere embed, nomic-embed-text
    SENTENCEPIECE = "sentencepiece"  # Mistral, LLaMA, E5-Mistral, mT5
    BPE           = "bpe"            # GPT-2-style, RoBERTa, mxbai-embed
    WHITESPACE    = "whitespace"     # Zero-dependency fallback — NEVER used in production
    UNKNOWN       = "unknown"        # Family could not be determined; blocks production


# ---------------------------------------------------------------------------
# TokenizerContract (ABC)
# ---------------------------------------------------------------------------

class TokenizerContract(abc.ABC):
    """
    Abstract contract that ALL tokenizers used inside EmbedderBundle must satisfy.

    Design rules
    ─────────────
    • Implementations must be deterministic: same input → same output.
    • After construction, implementations must be thread-safe (no mutable
      shared state between calls) OR set is_thread_safe = False so that callers
      know to use per-thread instances.
    • max_length is read once at construction time from the underlying
      tokenizer metadata (config.json / tokenizer.json / vendor docs).
      It must never be hard-coded to a generic default.

    Invariants (non-negotiable)
    ─────────────────────────────
    I-T1 (Native binding): model_id refers only to the model family the
         tokenizer was trained for. Cross-model tokenizers are forbidden.
         Prevents F-01 / F-16.

    I-T2 (Family consistency): tokenizer_family must match the tokenizer
         type discovered from metadata at load time. Mismatch triggers
         TokenizerResolutionError, not a silent fallback.
         Prevents F-01 / F-16.

    I-T3 (Max-length accuracy): max_length must equal the model's
         max_position_embeddings or vendor-documented limit. No auto-
         detection defaults are permitted. Failure to read this value
         must raise TokenizerResolutionError.
         Prevents F-02 / F-13 / F-14 / F-15. Enforces R-E4, R-C2, R-L3.

    I-T4 (Special token fidelity): special_tokens must contain all
         sentinel tokens required by the family (BERT → [CLS]/[SEP],
         RoBERTa → <s>/</s>, GPT → <|endoftext|>). Values must match
         the underlying tokenizer's definitions exactly.
         Prevents F-03 / F-07. Enforces R-E3.

    I-T5 (Count equivalence): count_tokens(text, include_special_tokens=x)
         must equal len(tokenize(text, add_special_tokens=x)) for all
         inputs. Both methods must be pure functions of text + flag.
         Prevents F-02. Enforces R-C3.

    I-T6 (Thread safety declaration): is_thread_safe reflects actual
         implementation safety. Callers rely on this to decide whether
         serialization is needed under concurrency.
    """

    # ------------------------------------------------------------------
    # Abstract properties — every implementation must declare these.
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Human-readable identifier, e.g. ``"openai/cl100k_base"``
        or ``"hf/BAAI/bge-large-en-v1.5"``.
        Used in log messages and AlignmentReport fields.
        """

    @property
    @abc.abstractmethod
    def model_id(self) -> str:
        """
        Canonical model identifier this tokenizer is native to.
        e.g. ``"openai/text-embedding-3-large"``, ``"BAAI/bge-large-en-v1.5"``.
        Must be identical to the ``model_id`` in EmbedderBundle.
        Enforces I-T1.
        """

    @property
    @abc.abstractmethod
    def tokenizer_family(self) -> TokenizerFamily:
        """
        Family classification.  Determines vocabulary and special-token rules.
        Must match the value in embedder_catalog.yaml.  Enforces I-T2.
        """

    @property
    @abc.abstractmethod
    def max_length(self) -> int:
        """
        Maximum number of tokens (inclusive) the embedding model accepts.
        Read from config.json ``max_position_embeddings`` or vendor docs.
        Must be > 0.  Enforces I-T3 / R-E4.
        """

    @property
    @abc.abstractmethod
    def special_tokens(self) -> Dict[str, str]:
        """
        Mapping from semantic role to concrete token string.

        Required keys per family:
          wordpiece   : "cls" → "[CLS]",  "sep" → "[SEP]",
                        "pad" → "[PAD]",  "unk" → "[UNK]"
          sentencepiece: "bos" → "<s>",   "eos" → "</s>",
                         "unk" → "<unk>", "pad" → "<pad>"  (if present)
          tiktoken    : "endoftext" → "<|endoftext|>"
          bpe         : "bos" → "<s>",    "eos" → "</s>",
                        "unk" → "<unk>"   (model-dependent)
          whitespace  : {}  (no special tokens)

        Values must match the underlying tokenizer's definitions exactly.
        Enforces I-T4 / R-E3.
        """

    @property
    @abc.abstractmethod
    def tokenizer_source(self) -> str:
        """
        Source of truth string describing where tokenizer metadata was loaded.
        Examples:
          ``"tiktoken:cl100k_base"``
          ``"hf:config.json+tokenizer.json"``
          ``"ollama:modelfile"``
          ``"openai_docs"``
        """

    @property
    @abc.abstractmethod
    def is_thread_safe(self) -> bool:
        """
        True if the implementation is safe for concurrent use after
        construction with no external locking.
        Enforces I-T6.
        """

    # ------------------------------------------------------------------
    # Abstract methods — core tokenizer operations.
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def tokenize(self, text: str, *, add_special_tokens: bool = False) -> List[int]:
        """
        Tokenize ``text`` using the model's native tokenizer.

        Parameters
        ----------
        text               : Input text to tokenize.
        add_special_tokens : When True, prepend/append the family's required
                             sentinel tokens (e.g. [CLS], [SEP] for BERT).

        Returns
        -------
        List of integer token IDs in vocabulary order.

        Raises
        ------
        TokenizerResolutionError
            If the underlying tokenizer is not loaded or cannot process input.

        Notes
        -----
        The length of the returned list must satisfy I-T5:
          len(tokenize(text, add_special_tokens=x))
          == count_tokens(text, include_special_tokens=x)
        for all text and flag combinations.
        """

    @abc.abstractmethod
    def count_tokens(
        self,
        text: str,
        *,
        include_special_tokens: bool = False,
    ) -> int:
        """
        Return the number of tokens that ``text`` produces.

        This must be a pure function of ``(text, include_special_tokens)``.
        Identical to ``len(tokenize(text, add_special_tokens=include_special_tokens))``,
        but may be faster for some backends (e.g. tiktoken encodes IDs without
        materializing a Python list).

        Parameters
        ----------
        text                    : Input text to count tokens for.
        include_special_tokens  : When True, include the overhead of the
                                  family's sentinel tokens in the count.

        Returns
        -------
        Non-negative integer.  0 for empty or whitespace-only input.

        Invariant (I-T5)
        ─────────────────
        count_tokens(text, include_special_tokens=x)
        == len(tokenize(text, add_special_tokens=x))
        for all text, x.
        """

    @abc.abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to a unicode string.

        This is the inverse of ``tokenize(text, add_special_tokens=False)``.
        Special-token IDs (e.g. [CLS], [SEP]) must be stripped or decoded
        transparently depending on the underlying tokenizer's convention.

        Parameters
        ----------
        token_ids : List of integer token IDs as returned by ``tokenize``.

        Returns
        -------
        Decoded unicode string.
        """

    @abc.abstractmethod
    def measure_with_overhead(
        self,
        text: str,
        *,
        special_overhead: int,
    ) -> int:
        """
        Return the token count for ``text`` plus a caller-supplied overhead.

        Used by ChunkSizerContract and TokenizerValidator to account for
        the model's sentinel tokens without calling the full tokenizer with
        add_special_tokens=True (which is slower for some backends).

        Parameters
        ----------
        text             : Input text to measure.
        special_overhead : Additional tokens to add (e.g. 2 for [CLS]/[SEP],
                           4 for BOS/EOS + padding). Must be ≥ 0.

        Returns
        -------
        count_tokens(text, include_special_tokens=False) + special_overhead.

        The caller is responsible for passing a correct overhead value that
        matches the model's actual sentinel token budget.  Enforces R-C2.
        """

    # ------------------------------------------------------------------
    # Concrete helpers — not overrideable; derived from abstract methods.
    # ------------------------------------------------------------------

    def fits_in_model(self, text: str, *, special_overhead: int = 0) -> bool:
        """
        Return True if ``text`` (plus overhead) fits within ``max_length``.

        Uses ``measure_with_overhead`` so the overhead contract is respected.
        Convenience method for fast path checks before raising errors.
        """
        return self.measure_with_overhead(text, special_overhead=special_overhead) <= self.max_length

    def assert_fits_in_model(self, text: str, *, special_overhead: int = 0) -> None:
        """
        Raise ``TokenizerResolutionError`` if ``text`` (plus overhead) exceeds
        ``max_length``.

        This is the canonical overflow guard — every ingestion stage must call
        this (via ChunkSizerContract) before embedding.  Enforces R-E4 / R-C3.
        """
        measured = self.measure_with_overhead(text, special_overhead=special_overhead)
        if measured > self.max_length:
            raise TokenizerResolutionError(
                model_id=self.model_id,
                provider="",
                reason=(
                    f"Token count {measured} exceeds max_length {self.max_length} "
                    f"(special_overhead={special_overhead}). "
                    "Enforce chunking before embedding. Failure mode: F-02."
                ),
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_id={self.model_id!r}, "
            f"family={self.tokenizer_family.value!r}, "
            f"max_length={self.max_length}, "
            f"thread_safe={self.is_thread_safe})"
        )
