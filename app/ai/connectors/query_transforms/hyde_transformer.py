# =============================================================================
# app/ai/connectors/query_transforms/hyde_transformer.py
#
# HyDETransformer — Phase 2 connector
#
# Hypothetical Document Embeddings (HyDE):
#   1. The LLM generates a short hypothetical document that WOULD answer
#      the user query (without seeing any real context).
#   2. The embedding of this hypothetical document is used as the query
#      vector instead of the raw query embedding.
#   3. This bridges the vocabulary gap between short queries and long passages
#      because the hypothetical answer lives closer to real answers in the
#      embedding space.
#
# Reference: Gao et al. (2022) — "Precise Zero-Shot Dense Retrieval without
# Relevance Labels" (https://arxiv.org/abs/2212.10496)
#
# Failure safety:
#   If the LLM call fails or returns an empty/trivially short response,
#   the transformer falls back to PASSTHROUGH (original query).
#   This prevents pipeline failures while preserving best-effort quality.
#
# Pipeline position: before ANN retrieval, not after.
# =============================================================================

from __future__ import annotations

import logging
import os
from typing import Optional

from app.ai.contracts.query_transform_contract import (
    PassthroughTransform,
    QueryTransformContract,
    QueryTransformStrategy,
    TransformedQuery,
)

logger = logging.getLogger(__name__)

_HYDE_SYSTEM_PROMPT = (
    "You are a knowledge assistant. Your task is to write a short factual "
    "passage (2-4 sentences) that would directly answer the user's question. "
    "Write the passage as if it were an excerpt from a relevant document. "
    "Do not say you don't know. Do not reference the question itself."
)

_HYDE_USER_TEMPLATE = "Question: {query}\n\nHypothetical passage:"


class HyDETransformer(QueryTransformContract):
    """
    Generates a hypothetical document to use as the retrieval query embedding.

    Args:
        model          : LLM model identifier (e.g., "gpt-4o-mini").
        api_key_env    : Name of the env var holding the API key.
        base_url       : Optional OpenAI-compatible base URL.
        max_tokens     : Max tokens for the hypothetical document.
        min_useful_len : Minimum character count for a useful hypothesis;
                         shorter responses fall back to the original query.
        temperature    : Generation temperature (0.0 for most deterministic).
    """

    @property
    def strategy(self) -> QueryTransformStrategy:
        return QueryTransformStrategy.HYDE

    @property
    def requires_llm(self) -> bool:
        return True

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
        max_tokens: int = 200,
        min_useful_len: int = 40,
        temperature: float = 0.0,
    ) -> None:
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise EnvironmentError(
                f"[HyDETransformer] API key env var {api_key_env!r} is not set."
            )

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")

        self._model         = model
        self._max_tokens    = max_tokens
        self._min_useful    = min_useful_len
        self._temperature   = temperature
        self._fallback      = PassthroughTransform()

        init_kwargs = {"api_key": api_key}
        if base_url:
            init_kwargs["base_url"] = base_url
        self._client = OpenAI(**init_kwargs)

        logger.info("[HyDETransformer] Ready | model=%s", model)

    def transform(self, query: str) -> TransformedQuery:
        """
        Generate a hypothetical answer and return it as the single query for
        embedding.  Falls back to the original query on any LLM failure.
        """
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _HYDE_SYSTEM_PROMPT},
                    {"role": "user",   "content": _HYDE_USER_TEMPLATE.format(query=query)},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            hypothesis = (resp.choices[0].message.content or "").strip()

            if len(hypothesis) < self._min_useful:
                logger.info(
                    "[HyDETransformer] Hypothesis too short (%d chars); "
                    "falling back to original query.", len(hypothesis),
                )
                return self._fallback.transform(query)

            logger.info(
                "[HyDETransformer] Generated %d-char hypothesis for query=%r",
                len(hypothesis), query[:60],
            )
            return TransformedQuery(
                original_query=query,
                expanded_queries=[hypothesis],
                strategy=QueryTransformStrategy.HYDE,
                llm_model_used=self._model,
                metadata={
                    "hypothesis_chars": len(hypothesis),
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
                    "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
                },
            )

        except Exception as e:
            logger.warning(
                "[HyDETransformer] LLM call failed (%s); using passthrough.", e
            )
            return TransformedQuery(
                original_query=query,
                expanded_queries=[query],
                strategy=QueryTransformStrategy.PASSTHROUGH,
                llm_model_used=None,
                metadata={"hyde_error": str(e)},
            )
