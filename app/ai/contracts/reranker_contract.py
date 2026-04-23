# =============================================================================
# app/ai/contracts/reranker_contract.py
#
# RerankerContract — Phase 2, Module 1
#
# Defines the canonical Phase 1-aligned contract for reranking in the RAG
# pipeline. Mirrors the EmbedderContract design: ABC + dataclasses + strict
# invariants, no concrete imports, no side effects.
#
# Reranking pipeline position:
#   VectorDB top-K ANN results → RerankerContract.rerank() → pruned top-M
#
# Design rules:
#   - Rerankers operate on (query, passage) pairs; they do NOT replace ANN retrieval.
#   - Score space must be declared so downstream components normalise correctly.
#   - Max-input-tokens must be enforced per-pair, not just per document.
#   - Every implementation must declare its tokenizer family to prevent
#     cross-wiring with the embedder's tokenizer (F-13 protection).
#
# Failure modes addressed:
#   F-13  Reranker input mismatch   — max_input_tokens_per_pair + own tokenizer
#   F-11  Score space confusion     — RerankerScoreSpace enum enforced
#   F-08  Pipeline component mismatch — lang_support + tokenizer_family declared
# =============================================================================

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RerankerScoreSpace(str, Enum):
    """
    Score space that the reranker outputs.

    PROBABILITY  — sigmoid-normalised [0.0, 1.0], higher = more relevant.
    LOGIT        — raw cross-encoder logit, unbounded; requires normalisation
                   before threshold gating.
    COSINE       — cosine similarity in [-1.0, 1.0]; used by bi-encoder
                   rerankers like ColBERT.
    """
    PROBABILITY = "probability"
    LOGIT       = "logit"
    COSINE      = "cosine"


class RerankerProvider(str, Enum):
    """Canonical reranker provider identifiers."""
    HUGGINGFACE   = "huggingface"
    COHERE        = "cohere"
    COLBERT       = "colbert"
    FLASHRANK     = "flashrank"
    OPENAI_LLM    = "openai_llm"    # LLM-as-judge (OpenAI)
    ANTHROPIC_LLM = "anthropic_llm" # LLM-as-judge (Anthropic)
    CUSTOM        = "custom"


# ---------------------------------------------------------------------------
# Capability & bundle dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RerankerCapabilities:
    """
    Immutable descriptor for a reranker's operating envelope.

    Attributes
    ----------
    max_input_tokens_per_pair
        Hard token limit for the concatenated (query + passage) input fed to
        the reranker.  Content exceeding this must be truncated by the caller
        — never silently by the model.
    score_space
        The numeric scale of scores returned by ``rerank()``.
    supports_batch_scoring
        True when the model can score all (query, passage) pairs in one
        forward pass, enabling efficient batch calls.
    lang_support
        ISO-639-1 language codes this model reliably supports.
        ``["*"]`` means all languages (with quality caveats).
    tokenizer_family
        Family of the reranker's own tokenizer.  Must NOT be shared with the
        embedder tokenizer (F-13).  ``None`` when the reranker is an API
        service that manages its own tokenisation internally.
    notes
        Short human-readable note on model lineage or limitations.
    """
    max_input_tokens_per_pair: int
    score_space:               RerankerScoreSpace
    supports_batch_scoring:    bool
    lang_support:              List[str]
    tokenizer_family:          Optional[str] = None
    notes:                     str           = ""


@dataclass(frozen=True)
class RerankerCandidate:
    """
    Immutable (query, passage) pair ready for scoring.

    Produced by converting VectorDB search results before reranking.
    """
    chunk_id:     str
    text:         str
    vector_score: float
    metadata:     Dict[str, Any] = field(default_factory=dict)

    def to_scored(self, rerank_score: float) -> "ScoredCandidate":
        return ScoredCandidate(
            chunk_id=self.chunk_id,
            text=self.text,
            vector_score=self.vector_score,
            rerank_score=rerank_score,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class ScoredCandidate:
    """
    Immutable output from the reranker — a candidate with its rerank score.

    The ``rerank_score`` is the authoritative relevance signal; ``vector_score``
    is preserved for diagnostics and RRF fusion.
    """
    chunk_id:     str
    text:         str
    vector_score: float
    rerank_score: float
    metadata:     Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":           self.chunk_id,
            "text":         self.text,
            "score":        self.rerank_score,
            "vector_score": self.vector_score,
            "rerank_score": self.rerank_score,
            "metadata":     self.metadata,
        }


# ---------------------------------------------------------------------------
# RerankerContract ABC
# ---------------------------------------------------------------------------

class RerankerContract(abc.ABC):
    """
    Canonical interface every Phase 1 reranker implementation must satisfy.

    Contract invariants
    -------------------
    I-R1  ``rerank()`` returns candidates sorted descending by rerank_score.
    I-R2  Returned count ≤ top_k.  If fewer candidates are supplied than
          top_k, all are returned (no padding).
    I-R3  ``score_space`` matches the declared
          ``capabilities.score_space`` — never silently remapped.
    I-R4  Scores for the same (query, passage) pair are deterministic at
          temperature=0 (or the equivalent for the backend).
    I-R5  The reranker does NOT embed the query — it re-scores ANN candidates
          that already passed through VectorDB retrieval.
    I-R6  Truncation of inputs that exceed max_input_tokens_per_pair is the
          caller's responsibility; the implementation should raise
          ``RerankerInputTooLongError`` rather than silently truncating.
    """

    @property
    @abc.abstractmethod
    def model_id(self) -> str:
        """Canonical model identifier matching the reranker catalog."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def capabilities(self) -> RerankerCapabilities:
        """Immutable capabilities descriptor for this reranker."""
        raise NotImplementedError

    @abc.abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[RerankerCandidate],
        *,
        top_k: int,
    ) -> List[ScoredCandidate]:
        """
        Score and rank (query, passage) pairs.

        Parameters
        ----------
        query : str
            Raw user query — unprefixed (prefixes are for embedders, not
            rerankers).
        candidates : list[RerankerCandidate]
            ANN top-K candidates from VectorDB retrieval.  This list is the
            *input* to the reranker, not retrieved independently.
        top_k : int
            Maximum number of results to return after sorting.

        Returns
        -------
        list[ScoredCandidate]
            Sorted descending by rerank_score, length ≤ top_k.
        """
        raise NotImplementedError

    def health_check(self) -> bool:
        """
        Verifies that the underlying model or API is reachable.
        Implementations calling remote APIs should override this.
        """
        return True


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class RerankerInputTooLongError(ValueError):
    """
    Raised when a (query, passage) pair exceeds max_input_tokens_per_pair.

    Corresponds to failure mode F-13.
    """
    def __init__(
        self,
        model_id: str,
        actual_tokens: int,
        max_allowed: int,
    ) -> None:
        self.model_id = model_id
        self.actual_tokens = actual_tokens
        self.max_allowed = max_allowed
        super().__init__(
            f"[RerankerInputTooLongError F-13] model_id={model_id!r}: "
            f"pair_tokens={actual_tokens} exceeds max_input_tokens_per_pair={max_allowed}. "
            "Truncate the passage before passing to the reranker."
        )


class RerankerResolutionError(RuntimeError):
    """
    Raised when a reranker cannot be resolved from the catalog or registry.
    """
    def __init__(self, model_id: str, reason: str) -> None:
        self.model_id = model_id
        self.reason = reason
        super().__init__(
            f"[RerankerResolutionError] model_id={model_id!r} — {reason}"
        )
