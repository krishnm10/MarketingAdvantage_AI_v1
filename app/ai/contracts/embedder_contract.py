# =============================================================================
# app/ai/contracts/embedder_contract.py
#
# EmbedderContract (ABC) + EmbedderBundle (dataclass) — Phase 1, Module 2
#
# MASTER RULE: EmbedderBundle is the ONLY object passed between pipeline
# stages.  No stage may resolve a tokenizer independently.
#
# Store format is always {vector, raw_text, metadata} — NEVER {vector, token_ids}.
#
# Failure modes addressed:
#   F-01  Vocabulary mismatch      — bound_tokenizer injected by registry only
#   F-04  Dimension mismatch       — dimension field is immutable, validated
#   F-05  Embedding space drift    — model_id + fingerprint ties index to bundle
#   F-06  Normalization mismatch   — is_normalized invariant
#   F-07  Asymmetric encoding      — query_prefix / passage_prefix enforced
#   F-08  Distance metric mismatch — distance_metric propagated to VectorDB
#   F-09  ANN algorithm mismatch   — fingerprint detects index/bundle drift
#   F-10  Metadata schema conflict — out-of-scope for this module (VectorDB layer)
#   F-11  Score space incompatibility — is_normalized + distance_metric pairing
#   F-16  Migration without re-index — embedding_fingerprint() enforces re-index
#
# Golden rules enforced:
#   R-E1  Never use third-party tokenizer with an embedding model
#   R-E2  Always use the model's bundled tokenizer via AutoTokenizer
#   R-R1  Store {vector, raw_text, metadata}
#   R-R3  Query pre-processing consistent at retrieval AND reranking
# =============================================================================

from __future__ import annotations

import abc
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Optional, Any

from app.ai.contracts.tokenizer_contract import TokenizerContract, TokenizerFamily


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EmbedderProvider(str, Enum):
    """
    Supported embedding providers.
    Allowed values in embedder_catalog.yaml ``provider`` field.
    """
    OPENAI      = "openai"
    COHERE      = "cohere"
    HUGGINGFACE = "huggingface"
    OLLAMA      = "ollama"
    ANTHROPIC   = "anthropic"
    GOOGLE      = "google"
    MISTRAL     = "mistral"


class DistanceMetric(str, Enum):
    """
    Vector distance metrics.
    The value stored in ``EmbedderBundle.distance_metric`` MUST match
    the metric used when creating the VectorDB collection.
    Mismatch → F-08.
    """
    COSINE      = "cosine"
    DOTPRODUCT  = "dotproduct"
    EUCLIDEAN   = "euclidean"


class VerificationStatus(str, Enum):
    """
    Confidence level of tokenizer resolution for an EmbedderBundle.

    VERIFIED          — tokenizer family, max_length, and special_tokens
                        all match metadata discovered at load time.
                        Required for production ingestion.

    CATALOG_MISMATCH  — catalog overrides discovered metadata; ingestion
                        may proceed only with explicit operator approval.

    UNVERIFIED        — tokenizer family or max_length could not be
                        conclusively determined.  MUST be blocked from
                        production ingestion unless a feature flag grants
                        explicit override.
    """
    VERIFIED         = "verified"
    CATALOG_MISMATCH = "catalog_mismatch"
    UNVERIFIED       = "unverified"


# ---------------------------------------------------------------------------
# RerankerView
# Minimal read-only view of a reranker bundle consumed by the validator.
# Using a structural protocol avoids circular imports between the reranker
# and embedder contract modules.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RerankerView:
    """
    Lightweight view of a reranker configuration used by TokenizerValidator.

    This is NOT a full reranker contract — it carries only the fields required
    to perform alignment checks (F-11, F-12, F-13, R-R2).

    Fields
    ------
    model_id            : Reranker model identifier, e.g. "cohere/rerank-v3.0".
    max_input_tokens    : Maximum passage length the reranker can process.
                          chunk_max must be ≤ this value (F-13, R-R2).
    tokenizer_family    : Tokenizer family used by the reranker.
                          Must be validated independently — never the same
                          instance as the embedder's tokenizer (MASTER RULE).
    score_space         : Score normalization convention.
                          Use ``"0_1"`` for probability outputs (Cohere Rerank).
                          Use ``"logits"`` for raw cross-encoder logits
                          (e.g. BGE-Reranker, range roughly -10 to +12).
                          Mixing these scales without renormalization → F-11.
    lang_support        : Language coverage, e.g. ``"en"`` or ``"multilingual"``.
                          Checked against corpus language for F-12 warnings.
    """
    model_id:         str
    max_input_tokens: int
    tokenizer_family: TokenizerFamily
    score_space:      Literal["0_1", "logits"]
    lang_support:     str = "en"


# ---------------------------------------------------------------------------
# EmbedderContract (ABC)
# ---------------------------------------------------------------------------

class EmbedderContract(abc.ABC):
    """
    Abstract contract for all embedding backend implementations used in
    the Phase 1 tokenizer-safe pipeline.

    Relationship to existing BaseEmbedder
    ──────────────────────────────────────
    This contract is a strict superset of ``app/core/embedders/base.BaseEmbedder``.
    New provider implementations for Phase 1 should implement both; existing
    implementations are wrapped at registry resolution time by an adapter that
    satisfies this contract without touching the original files.

    Invariants (non-negotiable)
    ─────────────────────────────
    I-E1 (Tokenizer injection): ``bound_tokenizer`` is set exclusively by
         EmbedderRegistry during EmbedderBundle construction.  Concrete
         implementations MUST NOT call AutoTokenizer.from_pretrained,
         tiktoken.get_encoding, or any tokenizer factory internally.
         Violation breaks F-01 / F-02 / F-16 prevention.

    I-E2 (Prefix discipline): ``embed_documents`` must apply ``passage_prefix``
         and ``embed_query`` must apply ``query_prefix`` when non-empty,
         deterministically, before every call to the underlying API.
         Same policy at both index time and query time.  Prevents F-07 / F-05.
         Enforces R-R3.

    I-E3 (Dimension correctness): every vector returned by ``embed_documents``
         and ``embed_query`` must have length exactly equal to ``dimension``.
         Any mismatch raises AlignmentError.  Prevents F-04 / F-05 / F-09.

    I-E4 (Distance metric): ``distance_metric`` must match the VectorDB
         collection metric.  Alignment is enforced by TokenizerValidator at
         pipeline construction time.  Prevents F-08.

    I-E5 (Normalization honesty): if ``is_normalized`` is True all output
         vectors must be unit-norm in L2 space.  If False no implicit
         normalization may be applied.  Prevents F-06 / F-11.
    """

    # ------------------------------------------------------------------
    # Abstract properties
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def provider(self) -> EmbedderProvider:
        """Provider enum value; must match catalog entry."""

    @property
    @abc.abstractmethod
    def model_id(self) -> str:
        """
        Exact canonical model identifier.
        e.g. ``"openai/text-embedding-3-large"``, ``"BAAI/bge-large-en-v1.5"``.
        Must match ``EmbedderBundle.model_id``.
        """

    @property
    @abc.abstractmethod
    def bound_tokenizer(self) -> TokenizerContract:
        """
        Read-only.  Set once at registry resolution time.
        Must be the same instance as ``EmbedderBundle.tokenizer``.
        Enforces I-E1 / MASTER CONTRACT.
        """

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """
        Output vector dimensionality.
        Must equal ``len(embed_query("probe"))`` and the catalog ``dimension``
        field.  Enforces I-E3.
        """

    @property
    @abc.abstractmethod
    def distance_metric(self) -> DistanceMetric:
        """
        Intended distance metric for VectorDB indexing and querying.
        Must be consistent with the VectorDB collection metric.
        Enforces I-E4.
        """

    @property
    @abc.abstractmethod
    def is_normalized(self) -> bool:
        """
        True iff output vectors are L2-normalized (unit norm).
        Enforces I-E5.
        """

    @property
    @abc.abstractmethod
    def query_prefix(self) -> str:
        """
        Prefix prepended to query text before embedding.
        Empty string if the model does not use asymmetric prefixes.
        """

    @property
    @abc.abstractmethod
    def passage_prefix(self) -> str:
        """
        Prefix prepended to document/passage text before embedding.
        Empty string if the model does not use asymmetric prefixes.
        """

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document/passage texts for indexing.

        Contract
        ────────
        1. If ``passage_prefix`` is non-empty, prepend it to every element of
           ``texts`` before passing to the underlying embedding API.
        2. The caller guarantees that every element has been validated to fit
           within ``bound_tokenizer.max_length`` (via ChunkSizerContract /
           TokenizerValidator).  The embedder must NOT silently truncate.
        3. Every returned vector must have length == ``dimension``.

        Returns
        -------
        List of float vectors, one per input text, each of length ``dimension``.

        Raises
        ------
        AlignmentError (from app.ai.validation)
            If any returned vector length ≠ ``dimension``.
        """

    @abc.abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string for retrieval.

        Contract
        ────────
        1. If ``query_prefix`` is non-empty, prepend it to ``text`` before
           passing to the underlying embedding API.
        2. Uses ``bound_tokenizer`` for measurement/validation only — never for
           generating token IDs sent to an external API (R-E1).
        3. The returned vector must have length == ``dimension``.

        Returns
        -------
        Single float vector of length ``dimension``.
        """

    # ------------------------------------------------------------------
    # Concrete helper — strips model-specific prefix before reranking.
    # ------------------------------------------------------------------

    def strip_query_prefix(self, text: str) -> str:
        """
        Return ``text`` with ``query_prefix`` stripped from the start.

        Rerankers must receive raw text without the embedding model's prefix.
        E.g. E5-Mistral: ``"query: what is RAG?"`` → ``"what is RAG?"``.
        Enforces R-R3.
        """
        pfx = self.query_prefix
        if pfx and text.startswith(pfx):
            return text[len(pfx):]
        return text

    def strip_passage_prefix(self, text: str) -> str:
        """
        Return ``text`` with ``passage_prefix`` stripped from the start.

        Rerankers receive raw passage text without the embedding model's prefix.
        Enforces R-R3.
        """
        pfx = self.passage_prefix
        if pfx and text.startswith(pfx):
            return text[len(pfx):]
        return text

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_id={self.model_id!r}, "
            f"dim={self.dimension}, "
            f"metric={self.distance_metric.value!r}, "
            f"normalized={self.is_normalized})"
        )


# ---------------------------------------------------------------------------
# EmbedderBundle (dataclass)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbedderBundle:
    """
    The single, immutable object passed between all pipeline stages.

    EmbedderBundle is the only artifact that may carry tokenizer, embedder,
    and configuration state between ingestion, retrieval, reranking, and LLM
    context-building modules.  No stage may reconstruct or re-resolve any of
    these fields independently.

    Fields
    ──────
    model_id            : Canonical model identifier.
    provider            : Provider enum value.
    tokenizer           : Resolved TokenizerContract bound to this model.
    embedder            : Resolved EmbedderContract with ``bound_tokenizer``
                          set to exactly the same instance as ``tokenizer``.
    dimension           : Output vector dimensionality.
    distance_metric     : Metric for VectorDB indexing and querying.
    is_normalized       : Whether output vectors are L2-normalized.
    embed_max_tokens    : Maximum tokens for this embedder/tokenizer pair.
    default_reranker_id : Optional reference to the recommended reranker.
    tokenizer_family    : Mirrors ``tokenizer.tokenizer_family`` for fast
                          alignment checks without unpacking the tokenizer.
    verification_status : Quality of tokenizer resolution.

    Invariants (non-negotiable)
    ─────────────────────────────
    B-I1 (Single object): This is the only container of embedder+tokenizer
         state in the pipeline.  No stage may instantiate a second one.

    B-I2 (Tokenizer identity): ``embedder.bound_tokenizer is tokenizer``
         must be True.  Checked eagerly in ``__post_init__``.

    B-I3 (Dimension parity): ``embedder.dimension == dimension``
         must be True.  Checked eagerly in ``__post_init__``.

    B-I4 (Metric parity): ``embedder.distance_metric == distance_metric``
         must be True.  Checked eagerly in ``__post_init__``.

    B-I5 (Family parity): ``tokenizer.tokenizer_family == tokenizer_family``
         must be True.  Checked eagerly in ``__post_init__``.

    B-I6 (Verification semantics):
         ``"unverified"`` blocks production ingestion unless a feature flag
         grants explicit override.  Only ``"verified"`` is production-safe.

    Failure modes prevented
    ────────────────────────
    F-01  via tokenizer_family field and B-I2
    F-04  via dimension field and B-I3
    F-05  via embedding_fingerprint()
    F-06  via is_normalized + distance_metric pairing check
    F-08  via distance_metric and B-I4
    F-16  via embedding_fingerprint() stored in VectorDB collection metadata
    """

    model_id:            str
    provider:            EmbedderProvider
    tokenizer:           TokenizerContract
    embedder:            EmbedderContract
    dimension:           int
    distance_metric:     DistanceMetric
    is_normalized:       bool
    embed_max_tokens:    int
    default_reranker_id: Optional[str]
    tokenizer_family:    TokenizerFamily
    verification_status: VerificationStatus

    def __post_init__(self) -> None:
        """
        Eager invariant checks.
        Runs immediately at construction — never deferred to embed/query time.
        Enforces B-I2 through B-I5.
        """
        # B-I2 — tokenizer identity
        if self.embedder.bound_tokenizer is not self.tokenizer:
            raise ValueError(
                f"[EmbedderBundle B-I2] embedder.bound_tokenizer is not the same "
                f"instance as bundle.tokenizer for model_id={self.model_id!r}. "
                "EmbedderRegistry must inject the tokenizer before constructing "
                "the bundle.  Failure modes: F-01, F-02, F-16."
            )

        # B-I3 — dimension parity
        if self.embedder.dimension != self.dimension:
            raise ValueError(
                f"[EmbedderBundle B-I3] embedder.dimension={self.embedder.dimension} "
                f"≠ bundle.dimension={self.dimension} for model_id={self.model_id!r}. "
                "Failure modes: F-04, F-05, F-09."
            )

        # B-I4 — metric parity
        if self.embedder.distance_metric != self.distance_metric:
            raise ValueError(
                f"[EmbedderBundle B-I4] embedder.distance_metric="
                f"{self.embedder.distance_metric.value!r} "
                f"≠ bundle.distance_metric={self.distance_metric.value!r} "
                f"for model_id={self.model_id!r}.  Failure mode: F-08."
            )

        # B-I5 — family parity
        if self.tokenizer.tokenizer_family != self.tokenizer_family:
            raise ValueError(
                f"[EmbedderBundle B-I5] tokenizer.tokenizer_family="
                f"{self.tokenizer.tokenizer_family.value!r} "
                f"≠ bundle.tokenizer_family={self.tokenizer_family.value!r} "
                f"for model_id={self.model_id!r}.  Failure modes: F-01, F-16."
            )

        # Additional guard: embed_max_tokens must be positive
        if self.embed_max_tokens <= 0:
            raise ValueError(
                f"[EmbedderBundle] embed_max_tokens must be > 0, "
                f"got {self.embed_max_tokens} for model_id={self.model_id!r}."
            )

        # Additional guard: dimension must be positive
        if self.dimension <= 0:
            raise ValueError(
                f"[EmbedderBundle] dimension must be > 0, "
                f"got {self.dimension} for model_id={self.model_id!r}."
            )

    # ------------------------------------------------------------------
    # Pipeline helpers
    # ------------------------------------------------------------------

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed document texts using the bound embedder.
        Delegates to ``embedder.embed_documents`` which applies passage_prefix.
        Store the raw text alongside vectors: {vector, raw_text, metadata}.
        """
        return self.embedder.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query string using the bound embedder.
        Delegates to ``embedder.embed_query`` which applies query_prefix.
        """
        return self.embedder.embed_query(text)

    def reranker_query_text(self, query: str) -> str:
        """
        Return the query text formatted for a reranker (no embedding prefix).

        The reranker receives the plain query, never the embedding model's
        ``query_prefix`` prepended version.  Enforces R-R3:
          "query: what is RAG?" for embedding → "what is RAG?" for reranker
          (strip model-specific prefix before reranker).

        Usage
        ─────
        ::

            plain_query = bundle.reranker_query_text("query: what is RAG?")
            # → "what is RAG?"  (prefix stripped)
            reranker.rerank(plain_query, candidates)

        Returns
        -------
        Query string with embedding ``query_prefix`` stripped, if present.
        """
        return self.embedder.strip_query_prefix(query)

    def reranker_passage_text(self, passage: str) -> str:
        """
        Return the passage text formatted for a reranker (no embedding prefix).

        Mirrors ``reranker_query_text`` for the passage side.  The reranker
        receives the raw passage, never the embedding model's passage_prefix.
        Enforces R-R3.
        """
        return self.embedder.strip_passage_prefix(passage)

    # ------------------------------------------------------------------
    # Token counting helpers (delegation to bound tokenizer)
    # ------------------------------------------------------------------

    def count_query_tokens(self, query: str) -> int:
        """
        Count tokens in a query using the bound tokenizer.

        The query_prefix is included in the count to accurately measure the
        full token budget consumed by the embedder.
        """
        prefixed = (self.embedder.query_prefix + query) if self.embedder.query_prefix else query
        return self.tokenizer.count_tokens(prefixed, include_special_tokens=False)

    def count_passage_tokens(self, passage: str) -> int:
        """
        Count tokens in a passage/document using the bound tokenizer.

        The passage_prefix is included to reflect the full token budget.
        """
        prefixed = (
            (self.embedder.passage_prefix + passage)
            if self.embedder.passage_prefix
            else passage
        )
        return self.tokenizer.count_tokens(prefixed, include_special_tokens=False)

    # ------------------------------------------------------------------
    # Embedding-space fingerprint (migration guard — F-16)
    # ------------------------------------------------------------------

    def embedding_fingerprint(self) -> str:
        """
        Derive a stable hex fingerprint from embedding-space identity fields.

        The fingerprint is stored in VectorDB collection metadata at index
        creation time.  On subsequent pipeline builds, this fingerprint is
        re-derived and compared.  A difference triggers an AlignmentError
        tagged [F-05, F-16], preventing the old index from being queried
        with the new embedding space.

        Fields included in the fingerprint:
          provider, model_id, tokenizer_family, dimension,
          distance_metric, is_normalized

        Fields intentionally excluded:
          default_reranker_id  — reranker can change without re-indexing
          verification_status  — operational metadata, not space identity
          embed_max_tokens     — chunking change ≠ embedding space change

        Returns
        -------
        Lowercase hex SHA-256 of the identity JSON (first 16 chars for brevity,
        full 64 chars available from the raw digest for auditing).
        """
        identity: Dict[str, Any] = {
            "provider":         self.provider.value,
            "model_id":         self.model_id,
            "tokenizer_family": self.tokenizer_family.value,
            "dimension":        self.dimension,
            "distance_metric":  self.distance_metric.value,
            "is_normalized":    self.is_normalized,
        }
        raw = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return digest

    def embedding_fingerprint_short(self) -> str:
        """First 16 hex characters of the fingerprint for display purposes."""
        return self.embedding_fingerprint()[:16]

    # ------------------------------------------------------------------
    # Rich repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"EmbedderBundle("
            f"model_id={self.model_id!r}, "
            f"provider={self.provider.value!r}, "
            f"dim={self.dimension}, "
            f"metric={self.distance_metric.value!r}, "
            f"normalized={self.is_normalized}, "
            f"max_tokens={self.embed_max_tokens}, "
            f"family={self.tokenizer_family.value!r}, "
            f"status={self.verification_status.value!r}, "
            f"fp={self.embedding_fingerprint_short()!r})"
        )
