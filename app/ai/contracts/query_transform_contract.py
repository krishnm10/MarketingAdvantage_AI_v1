# =============================================================================
# app/ai/contracts/query_transform_contract.py
#
# QueryTransformContract — Phase 2, Module 2
#
# Defines the canonical interface for query-time transformations that run
# BEFORE retrieval to improve recall and semantic coverage.
#
# Supported strategies
# ────────────────────
# HYDE          — LLM generates a hypothetical document that would answer
#                 the query; the embedding of that document is used instead
#                 of the raw query embedding, bridging the vocabulary gap
#                 between short queries and long passages.
#
# MULTI_QUERY   — LLM generates N diverse query variants; each variant is
#                 retrieved independently and results are fused via RRF.
#                 Addresses recall gaps from single-phrasing queries.
#
# STEP_BACK     — LLM generates a higher-level "step-back" question before
#                 the specific query; retrieval runs on both; improves
#                 multi-hop reasoning coverage.
#
# DECOMPOSITION — LLM decomposes a complex question into sub-questions for
#                 sequential or parallel retrieval.
#
# Design rules:
#   - Transforms are applied before ANN retrieval (not as post-processing).
#   - Transforms never mutate the original query; they produce new texts.
#   - Transforms that require an LLM declare it via ``requires_llm = True``.
#   - Transforms are stateless; all state lives in the returned dataclass.
# =============================================================================

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class QueryTransformStrategy(str, Enum):
    """
    Canonical strategy identifiers for query transformation.
    Values correspond to the ``strategy`` field in the pipeline config.
    """
    HYDE          = "hyde"
    MULTI_QUERY   = "multi_query"
    STEP_BACK     = "step_back"
    DECOMPOSITION = "decomposition"
    PASSTHROUGH   = "passthrough"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransformedQuery:
    """
    Immutable result of a query transformation.

    Attributes
    ----------
    original_query
        The raw user input, preserved verbatim.
    expanded_queries
        One or more texts that should be used for retrieval.
        For PASSTHROUGH: ``[original_query]``.
        For HYDE: a single hypothetical document text.
        For MULTI_QUERY: N diverse query phrasings.
        For STEP_BACK: ``[step_back_question, original_query]``.
        For DECOMPOSITION: the ordered sub-questions.
    strategy
        The transform strategy that produced this result.
    llm_model_used
        Model name used for generation, if any.  ``None`` for PASSTHROUGH.
    metadata
        Any debug data (raw LLM output, token counts, latency_ms, etc.).
    """
    original_query:  str
    expanded_queries: List[str]
    strategy:        QueryTransformStrategy
    llm_model_used:  Optional[str]             = None
    metadata:        dict                       = field(default_factory=dict)

    @property
    def primary_query(self) -> str:
        """The main query text for single-query retrieval paths."""
        return self.expanded_queries[0] if self.expanded_queries else self.original_query

    @property
    def is_expanded(self) -> bool:
        """True when more than one query was produced."""
        return len(self.expanded_queries) > 1


# ---------------------------------------------------------------------------
# QueryTransformContract ABC
# ---------------------------------------------------------------------------

class QueryTransformContract(abc.ABC):
    """
    Canonical interface every query transformer must implement.

    Contract invariants
    -------------------
    I-QT1  ``transform()`` must always include the original query in the
           returned ``expanded_queries`` unless the strategy semantically
           replaces it (e.g., HYDE replaces with the hypothetical text).
    I-QT2  ``transform()`` must be stateless — repeated calls with the same
           input must return semantically equivalent results.
    I-QT3  LLM failures are surfaced as logged warnings and the implementation
           falls back to a PASSTHROUGH result, never raising uncaught errors.
    I-QT4  ``strategy`` on the returned ``TransformedQuery`` must match
           ``self.strategy``.
    """

    @property
    @abc.abstractmethod
    def strategy(self) -> QueryTransformStrategy:
        """The strategy this transformer implements."""
        raise NotImplementedError

    @property
    def requires_llm(self) -> bool:
        """
        Whether this transformer requires an LLM backend.
        Override to True for HYDE, MULTI_QUERY, etc.
        """
        return False

    @abc.abstractmethod
    def transform(self, query: str) -> TransformedQuery:
        """
        Transform the user query for retrieval.

        Parameters
        ----------
        query : str
            Raw, sanitised user query.

        Returns
        -------
        TransformedQuery
            Contains one or more expanded queries for retrieval.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Passthrough implementation (zero-dependency, always available)
# ---------------------------------------------------------------------------

class PassthroughTransform(QueryTransformContract):
    """
    Identity transform — returns the original query unchanged.
    Used as the default when no transform is configured.
    """

    @property
    def strategy(self) -> QueryTransformStrategy:
        return QueryTransformStrategy.PASSTHROUGH

    def transform(self, query: str) -> TransformedQuery:
        return TransformedQuery(
            original_query=query,
            expanded_queries=[query],
            strategy=QueryTransformStrategy.PASSTHROUGH,
            llm_model_used=None,
        )
