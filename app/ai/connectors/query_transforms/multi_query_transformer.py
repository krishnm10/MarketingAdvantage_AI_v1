# =============================================================================
# app/ai/connectors/query_transforms/multi_query_transformer.py
#
# MultiQueryTransformer — Phase 2 connector
#
# Generates N diverse phrasings of the user query, retrieves documents for
# each, then merges results via Reciprocal Rank Fusion (RRF) to maximise
# recall.
#
# Motivation:
#   A single query phrasing may miss relevant documents due to vocabulary
#   mismatch between the user's language and the indexed content.
#   Multi-query expansion addresses this by sampling the embedding space
#   from multiple angles.
#
# Pipeline position: BEFORE ANN retrieval.
# The caller (RAGPipeline) is responsible for running separate VectorDB
# searches for each expanded query and fusing the results.
#
# Contract: returns all generated queries in ``TransformedQuery.expanded_queries``
# including the original query as the last element (unless deduplicated).
# =============================================================================

from __future__ import annotations

import logging
import os
from typing import List, Optional

from app.ai.contracts.query_transform_contract import (
    PassthroughTransform,
    QueryTransformContract,
    QueryTransformStrategy,
    TransformedQuery,
)

logger = logging.getLogger(__name__)

_MULTI_QUERY_SYSTEM = (
    "You are a query diversification assistant. Generate {n} distinct "
    "alternative phrasings of the user's question that would help retrieve "
    "relevant information from a document database. Each phrasing should use "
    "different vocabulary or framing while preserving the original intent. "
    "Output ONLY a numbered list, one per line, no extra text."
)

_MULTI_QUERY_USER = "Original question: {query}\n\nGenerate {n} alternative phrasings:"


class MultiQueryTransformer(QueryTransformContract):
    """
    Generates N alternative query phrasings for multi-retrieval RRF fusion.

    Args:
        n_variants     : Number of alternative phrasings to generate.
        include_original: Append the original query to the variant list.
        model          : LLM model for generation.
        api_key_env    : Env var name for the API key.
        base_url       : OpenAI-compatible base URL.
        temperature    : Sampling temperature (>0 encourages diversity).
    """

    @property
    def strategy(self) -> QueryTransformStrategy:
        return QueryTransformStrategy.MULTI_QUERY

    @property
    def requires_llm(self) -> bool:
        return True

    def __init__(
        self,
        *,
        n_variants: int = 3,
        include_original: bool = True,
        model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
        temperature: float = 0.7,
    ) -> None:
        if n_variants < 1:
            raise ValueError("n_variants must be ≥ 1.")

        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise EnvironmentError(
                f"[MultiQueryTransformer] API key env var {api_key_env!r} is not set."
            )

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")

        self._n_variants      = n_variants
        self._include_original = include_original
        self._model           = model
        self._temperature     = temperature
        self._fallback        = PassthroughTransform()

        init_kwargs = {"api_key": api_key}
        if base_url:
            init_kwargs["base_url"] = base_url
        self._client = OpenAI(**init_kwargs)

        logger.info(
            "[MultiQueryTransformer] Ready | model=%s | n=%d | include_original=%s",
            model, n_variants, include_original,
        )

    def transform(self, query: str) -> TransformedQuery:
        """
        Generate n_variants alternative phrasings.
        Falls back to passthrough on any LLM failure.
        """
        try:
            system_msg = _MULTI_QUERY_SYSTEM.format(n=self._n_variants)
            user_msg   = _MULTI_QUERY_USER.format(query=query, n=self._n_variants)

            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=self._temperature,
                max_tokens=300,
            )

            raw   = (resp.choices[0].message.content or "").strip()
            lines = [
                line.lstrip("0123456789.-) ").strip()
                for line in raw.splitlines()
                if line.strip()
            ]
            variants: List[str] = [l for l in lines if len(l) > 10][: self._n_variants]

            if not variants:
                logger.warning(
                    "[MultiQueryTransformer] No valid variants parsed; using passthrough."
                )
                return self._fallback.transform(query)

            expanded = list(variants)
            if self._include_original and query not in expanded:
                expanded.append(query)

            logger.info(
                "[MultiQueryTransformer] Generated %d variants for query=%r",
                len(expanded), query[:60],
            )
            return TransformedQuery(
                original_query=query,
                expanded_queries=expanded,
                strategy=QueryTransformStrategy.MULTI_QUERY,
                llm_model_used=self._model,
                metadata={
                    "n_variants_requested": self._n_variants,
                    "n_variants_generated": len(expanded),
                },
            )

        except Exception as e:
            logger.warning(
                "[MultiQueryTransformer] LLM call failed (%s); using passthrough.", e
            )
            return TransformedQuery(
                original_query=query,
                expanded_queries=[query],
                strategy=QueryTransformStrategy.PASSTHROUGH,
                llm_model_used=None,
                metadata={"multi_query_error": str(e)},
            )
