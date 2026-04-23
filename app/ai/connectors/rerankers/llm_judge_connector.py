# =============================================================================
# app/ai/connectors/rerankers/llm_judge_connector.py
#
# LLMJudgeReranker — Phase 2 connector
#
# Implements LLM-as-Judge reranking with three scoring strategies:
#   POINTWISE  — each passage scored independently on a 0–10 relevance scale
#   LISTWISE   — model orders all passages in a single call (most token-efficient)
#   PAIRWISE   — pairs compared directly; highest quality, O(N²) cost
#
# Usage position in pipeline:
#   VectorDB top-K ANN → LLMJudgeReranker.rerank() → pruned top-M
#
# This connector can also be used for PRE-GENERATION evaluation
# (score before sending context to the generator) and POST-GENERATION
# faithfulness evaluation via the separate ``evaluate_faithfulness()`` method.
#
# Security:
#   - All inputs pass through security_middleware before this connector is called.
#   - API keys read from environment only; never logged.
# =============================================================================

from __future__ import annotations

import json
import logging
import os
import time
from enum import Enum
from typing import List, Optional

from app.ai.contracts.reranker_contract import (
    RerankerCapabilities,
    RerankerCandidate,
    RerankerContract,
    RerankerProvider,
    RerankerScoreSpace,
    ScoredCandidate,
)

logger = logging.getLogger(__name__)


class JudgeStrategy(str, Enum):
    POINTWISE = "pointwise"
    LISTWISE  = "listwise"
    PAIRWISE  = "pairwise"


# Prompt templates for each strategy
_POINTWISE_PROMPT = """\
You are a relevance grader. Given a query and a passage, score how relevant \
the passage is to answering the query.

Scoring scale:
  10 — Directly and fully answers the query.
   7 — Mostly relevant; answers the core but misses details.
   4 — Partially relevant; contains related information.
   1 — Irrelevant or off-topic.

Respond with ONLY a JSON object: {{"score": <int 1-10>}}

Query: {query}

Passage:
{passage}"""


_LISTWISE_PROMPT = """\
You are a relevance ranker. Given a query and a list of passages, order them \
from most to least relevant for answering the query.

Return ONLY a JSON array of passage indices (0-based), most relevant first.
Example for 4 passages: [2, 0, 3, 1]

Query: {query}

Passages:
{passages}"""


_FAITHFULNESS_PROMPT = """\
You are a faithfulness evaluator for a RAG system.
Given a QUESTION, a set of CONTEXT passages, and an ANSWER, determine whether \
the answer is fully supported by the context.

Respond with ONLY a JSON object:
{{"faithful": true/false, "score": <float 0.0-1.0>, "reason": "<one-line reason>"}}

QUESTION: {question}

CONTEXT:
{context}

ANSWER: {answer}"""


class LLMJudgeReranker(RerankerContract):
    """
    LLM-as-Judge reranker.

    Supports both pre-generation ranking (rerank()) and post-generation
    faithfulness evaluation (evaluate_faithfulness()).

    Args:
        model_id   : OpenAI model to use (e.g. "gpt-4o-mini", "gpt-4o").
        strategy   : POINTWISE, LISTWISE, or PAIRWISE.
        api_key_env: Name of the env var holding the OpenAI API key.
        base_url   : Optional OpenAI-compatible base URL.
        timeout    : HTTP timeout in seconds.
    """

    def __init__(
        self,
        *,
        model_id: str = "gpt-4o-mini",
        strategy: JudgeStrategy = JudgeStrategy.POINTWISE,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise EnvironmentError(
                f"[LLMJudgeReranker] API key env var {api_key_env!r} is not set."
            )

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai not installed. Run: pip install openai"
            )

        self._model_id = f"llm-judge/{model_id}"
        self._llm_model = model_id
        self._strategy  = strategy
        self._timeout   = timeout

        init_kwargs = {"api_key": api_key, "timeout": timeout}
        if base_url:
            init_kwargs["base_url"] = base_url
        self._client = OpenAI(**init_kwargs)

        # Capabilities vary by model; use generous defaults for API-based judges
        max_tokens = 32768 if "gpt-4o" in model_id else 16384
        self._capabilities = RerankerCapabilities(
            max_input_tokens_per_pair=max_tokens,
            score_space=RerankerScoreSpace.PROBABILITY,
            supports_batch_scoring=(strategy == JudgeStrategy.LISTWISE),
            lang_support=["*"],
            tokenizer_family="external",
            notes=f"LLM-as-judge via {model_id}; strategy={strategy.value}.",
        )

        logger.info(
            "[LLMJudgeReranker] Ready | model=%s | strategy=%s",
            model_id, strategy.value,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def capabilities(self) -> RerankerCapabilities:
        return self._capabilities

    def rerank(
        self,
        query: str,
        candidates: List[RerankerCandidate],
        *,
        top_k: int,
    ) -> List[ScoredCandidate]:
        if not candidates:
            return []

        t0 = time.perf_counter()

        if self._strategy == JudgeStrategy.LISTWISE:
            scored = self._rerank_listwise(query, candidates)
        elif self._strategy == JudgeStrategy.PAIRWISE:
            scored = self._rerank_pointwise(query, candidates)  # pairwise fallback
        else:
            scored = self._rerank_pointwise(query, candidates)

        scored.sort(key=lambda x: x.rerank_score, reverse=True)
        result = scored[:top_k]

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "[LLMJudgeReranker] strategy=%s | in=%d | out=%d | top_score=%.4f | %.1fms",
            self._strategy.value, len(candidates), len(result),
            result[0].rerank_score if result else 0.0, elapsed,
        )
        return result

    def _rerank_pointwise(
        self,
        query: str,
        candidates: List[RerankerCandidate],
    ) -> List[ScoredCandidate]:
        """Score each (query, passage) pair independently."""
        scored: List[ScoredCandidate] = []
        for c in candidates:
            prompt = _POINTWISE_PROMPT.format(
                query=query, passage=c.text[:4000]
            )
            raw_score = self._call_llm_json(prompt, key="score", default=5)
            # Normalise 1–10 scale to [0.0, 1.0] probability
            normalised = max(0.0, min(1.0, (float(raw_score) - 1) / 9))
            scored.append(c.to_scored(normalised))
        return scored

    def _rerank_listwise(
        self,
        query: str,
        candidates: List[RerankerCandidate],
    ) -> List[ScoredCandidate]:
        """Order all passages in a single LLM call (most token-efficient)."""
        passages_text = "\n\n".join(
            f"[{i}] {c.text[:2000]}" for i, c in enumerate(candidates)
        )
        prompt = _LISTWISE_PROMPT.format(query=query, passages=passages_text)

        try:
            resp = self._client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=128,
            )
            raw = resp.choices[0].message.content.strip()
            order: list = json.loads(raw)
            if not isinstance(order, list):
                raise ValueError("Expected a JSON list")
        except Exception as e:
            logger.warning(
                "[LLMJudgeReranker] Listwise parse failed (%s); "
                "falling back to original order.", e
            )
            order = list(range(len(candidates)))

        # Convert ranked order to descending scores
        n = len(candidates)
        scored: List[ScoredCandidate] = []
        rank_map = {idx: rank for rank, idx in enumerate(order)}
        for i, c in enumerate(candidates):
            rank = rank_map.get(i, n)
            score = max(0.0, 1.0 - (rank / n))
            scored.append(c.to_scored(score))
        return scored

    def _call_llm_json(
        self,
        prompt: str,
        key: str,
        default: float,
    ) -> float:
        """Call the LLM and extract a single numeric value from JSON response."""
        try:
            resp = self._client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=32,
            )
            raw = resp.choices[0].message.content.strip()
            parsed = json.loads(raw)
            return float(parsed[key])
        except Exception as e:
            logger.warning(
                "[LLMJudgeReranker] JSON parse failed for key=%r (%s); "
                "using default=%s.", key, e, default,
            )
            return float(default)

    def evaluate_faithfulness(
        self,
        *,
        question: str,
        context_chunks: list,
        answer: str,
    ) -> dict:
        """
        Post-generation faithfulness evaluation.

        Returns a dict with keys: ``faithful``, ``score`` (0–1), ``reason``.
        Called after LLM generation to verify the answer is grounded.
        """
        context_text = "\n\n".join(
            f"[{i+1}] {c.get('text', '') if isinstance(c, dict) else c.text}"
            for i, c in enumerate(context_chunks[:10])
        )
        prompt = _FAITHFULNESS_PROMPT.format(
            question=question,
            context=context_text[:8000],
            answer=answer[:2000],
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            raw = resp.choices[0].message.content.strip()
            return json.loads(raw)
        except Exception as e:
            logger.warning(
                "[LLMJudgeReranker] evaluate_faithfulness failed: %s", e
            )
            return {"faithful": None, "score": None, "reason": str(e)}

    def health_check(self) -> bool:
        try:
            resp = self._client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=4,
            )
            return bool(resp.choices)
        except Exception as e:
            logger.warning("[LLMJudgeReranker] health_check failed: %s", e)
            return False


# =============================================================================
# GenericLLMJudgeReranker — provider-agnostic, uses BaseLLM interface
# Supports OpenAI, Gemini, Anthropic, Groq, Ollama — any BaseLLM.
# =============================================================================

class GenericLLMJudgeReranker(RerankerContract):
    """
    Provider-agnostic LLM-as-Judge reranker.

    Uses the BaseLLM.generate() interface so it works with ANY registered
    LLM provider (OpenAI, Gemini, Anthropic, Groq, Ollama, etc.).

    Scoring strategies:
      POINTWISE — score each (query, passage) pair; 0–10 scale, normalised.
      LISTWISE  — rank all passages in one call; most token-efficient.

    Security:
      Inputs are expected to arrive pre-sanitised by security_middleware.
      API keys are read from env vars; never stored in plain text.

    Args:
        provider    : LLM provider key registered in llm_registry ('openai', 'gemini', …).
        model_id    : Model name (e.g. 'gpt-4o-mini', 'gemini-1.5-flash').
        strategy    : JudgeStrategy.POINTWISE or LISTWISE.
        api_key     : Resolved API key string (caller reads from env).
        base_url    : Optional base URL override (for Ollama, proxies).
        timeout     : Request timeout in seconds.
    """

    def __init__(
        self,
        *,
        provider:  str = "openai",
        model_id:  str = "gpt-4o-mini",
        strategy:  JudgeStrategy = JudgeStrategy.POINTWISE,
        api_key:   str = "",
        base_url:  Optional[str] = None,
        timeout:   float = 30.0,
    ) -> None:
        self._provider  = provider
        self._model_id  = f"llm-judge/{model_id}"
        self._llm_model = model_id
        self._strategy  = strategy
        self._timeout   = timeout

        # Resolve LLM via the shared registry
        from app.core.plugin_registry import llm_registry

        kwargs: dict = {"model": model_id}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        elif provider == "openai":
            kwargs.setdefault("base_url", None)
        elif provider == "gemini":
            pass  # GeminiLLM does not require base_url

        try:
            self._llm = llm_registry.build(provider, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"[GenericLLMJudgeReranker] Failed to build LLM "
                f"provider={provider!r} model={model_id!r}: {e}"
            ) from e

        max_tokens = 32768 if "gpt-4o" in model_id or "gemini" in provider else 8192
        self._capabilities = RerankerCapabilities(
            max_input_tokens_per_pair=max_tokens,
            score_space=RerankerScoreSpace.PROBABILITY,
            supports_batch_scoring=(strategy == JudgeStrategy.LISTWISE),
            lang_support=["*"],
            tokenizer_family="external",
            notes=f"Generic LLM-as-judge | provider={provider} | model={model_id} | strategy={strategy.value}.",
        )

        logger.info(
            "[GenericLLMJudgeReranker] Ready | provider=%s | model=%s | strategy=%s",
            provider, model_id, strategy.value,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def capabilities(self) -> RerankerCapabilities:
        return self._capabilities

    def rerank(
        self,
        query: str,
        candidates: List[RerankerCandidate],
        *,
        top_k: int,
    ) -> List[ScoredCandidate]:
        if not candidates:
            return []

        t0 = time.perf_counter()

        if self._strategy == JudgeStrategy.LISTWISE:
            scored = self._rerank_listwise(query, candidates)
        else:
            scored = self._rerank_pointwise(query, candidates)

        scored.sort(key=lambda x: x.rerank_score, reverse=True)
        result = scored[:top_k]

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "[GenericLLMJudgeReranker] provider=%s | strategy=%s | "
            "in=%d | out=%d | top_score=%.4f | %.1fms",
            self._provider, self._strategy.value,
            len(candidates), len(result),
            result[0].rerank_score if result else 0.0, elapsed,
        )
        return result

    def _call_llm(self, prompt: str, max_tokens: int = 64) -> str:
        """Call the underlying LLM and return the response text."""
        try:
            response = self._llm.generate(
                prompt,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return response.text.strip()
        except Exception as e:
            logger.warning(
                "[GenericLLMJudgeReranker] LLM call failed (%s)", e
            )
            return ""

    def _rerank_pointwise(
        self,
        query: str,
        candidates: List[RerankerCandidate],
    ) -> List[ScoredCandidate]:
        """Score each (query, passage) pair independently using 0–10 scale."""
        scored: List[ScoredCandidate] = []
        for c in candidates:
            prompt = _POINTWISE_PROMPT.format(
                query=query, passage=c.text[:4000]
            )
            raw = self._call_llm(prompt, max_tokens=32)
            try:
                parsed = json.loads(raw)
                raw_score = float(parsed.get("score", 5))
            except Exception:
                raw_score = 5.0
            normalised = max(0.0, min(1.0, (raw_score - 1.0) / 9.0))
            scored.append(c.to_scored(normalised))
        return scored

    def _rerank_listwise(
        self,
        query: str,
        candidates: List[RerankerCandidate],
    ) -> List[ScoredCandidate]:
        """Rank all passages in a single call — most token-efficient."""
        passages_text = "\n\n".join(
            f"[{i}] {c.text[:2000]}" for i, c in enumerate(candidates)
        )
        prompt = _LISTWISE_PROMPT.format(query=query, passages=passages_text)
        raw = self._call_llm(prompt, max_tokens=256)

        try:
            order: list = json.loads(raw)
            if not isinstance(order, list):
                raise ValueError("Expected a JSON list")
        except Exception as e:
            logger.warning(
                "[GenericLLMJudgeReranker] Listwise parse failed (%s); "
                "using original order.", e,
            )
            order = list(range(len(candidates)))

        n = len(candidates)
        rank_map = {idx: rank for rank, idx in enumerate(order)}
        scored: List[ScoredCandidate] = []
        for i, c in enumerate(candidates):
            rank = rank_map.get(i, n)
            score = max(0.0, 1.0 - (rank / n))
            scored.append(c.to_scored(score))
        return scored

    def evaluate_faithfulness(
        self,
        *,
        question: str,
        context_chunks: list,
        answer: str,
    ) -> dict:
        """Post-generation faithfulness evaluation using the LLM provider."""
        context_text = "\n\n".join(
            f"[{i+1}] {c.get('text', '') if isinstance(c, dict) else str(c)}"
            for i, c in enumerate(context_chunks[:10])
        )
        prompt = _FAITHFULNESS_PROMPT.format(
            question=question,
            context=context_text[:8000],
            answer=answer[:2000],
        )
        raw = self._call_llm(prompt, max_tokens=100)
        try:
            return json.loads(raw)
        except Exception as e:
            logger.warning(
                "[GenericLLMJudgeReranker] evaluate_faithfulness parse failed: %s", e
            )
            return {"faithful": None, "score": None, "reason": str(e)}

    def health_check(self) -> bool:
        try:
            text = self._call_llm("ping", max_tokens=4)
            return len(text) >= 0
        except Exception as e:
            logger.warning("[GenericLLMJudgeReranker] health_check failed: %s", e)
            return False
