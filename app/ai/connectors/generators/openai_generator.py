# =============================================================================
# app/ai/connectors/generators/openai_generator.py
#
# OpenAIGenerator — Phase 2 connector
#
# Implements GeneratorContract backed by the OpenAI Chat Completions API.
# Wraps the existing app/core/llms/openai_v1.py at the Phase 2 contract level,
# adding:
#   - Structured PromptTemplate rendering
#   - Adaptive token budgeting (computes max_new_tokens from context window)
#   - Citation style enforcement
#   - Structured GenerationResult with token usage
#
# Supported models: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo,
# and any OpenAI-compatible API (Azure, LM Studio, vLLM, LocalAI).
#
# Install: pip install openai
# =============================================================================

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from app.ai.contracts.generator_contract import (
    CitationStyle,
    GenerationRequest,
    GenerationResult,
    GeneratorContract,
    GeneratorProvider,
    PromptTemplate,
)
from app.ai.contracts.reranker_contract import ScoredCandidate

logger = logging.getLogger(__name__)

# Known context window sizes for common models
_CONTEXT_WINDOWS: dict = {
    "gpt-4o":           128_000,
    "gpt-4o-mini":      128_000,
    "gpt-4-turbo":      128_000,
    "gpt-4":              8_192,
    "gpt-3.5-turbo":   16_385,
    "gpt-3.5-turbo-16k": 16_385,
}

# Default RAG system prompt — used when no PromptTemplate is provided
_DEFAULT_SYSTEM_PROMPT = """\
You are a Marketing Intelligence Assistant for enterprise businesses.
Answer the question using ONLY the context passages provided below.
If the context does not contain sufficient information to answer, state clearly: \
"I don't have enough information in the provided context to answer this."
Cite the source number (e.g. [1], [2]) when referencing a specific passage.
Be concise, factual, and precise. Do not speculate beyond the context."""


class OpenAIGenerator(GeneratorContract):
    """
    Phase 2 GeneratorContract backed by OpenAI Chat Completions.

    Adaptive token budgeting:
      max_new_tokens = context_window - prompt_tokens_estimate - safety_margin
      This ensures the model never silently truncates its output due to token
      exhaustion.

    Args:
        model          : OpenAI model name (e.g. "gpt-4o-mini").
        api_key_env    : Env var name holding the API key.
        organization_env: Env var name for org ID (optional).
        base_url       : OpenAI-compatible base URL.
        default_temperature : Default temperature (overridden per request).
        default_max_new_tokens : Default token limit (overridden per request).
        safety_margin  : Tokens reserved to prevent context overflow.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        organization_env: Optional[str] = None,
        base_url: Optional[str] = None,
        default_temperature: float = 0.3,
        default_max_new_tokens: int = 1024,
        safety_margin: int = 256,
    ) -> None:
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise EnvironmentError(
                f"[OpenAIGenerator] API key env var {api_key_env!r} is not set."
            )

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")

        self._model                 = model
        self._default_temperature   = default_temperature
        self._default_max_new_tokens = default_max_new_tokens
        self._safety_margin         = safety_margin
        self._context_window        = _CONTEXT_WINDOWS.get(model, 8192)

        init_kwargs = {"api_key": api_key}
        if organization_env:
            org = os.environ.get(organization_env, "").strip()
            if org:
                init_kwargs["organization"] = org
        if base_url:
            init_kwargs["base_url"] = base_url

        self._client = OpenAI(**init_kwargs)

        logger.info(
            "[OpenAIGenerator] Ready | model=%s | context_window=%d",
            model, self._context_window,
        )

    @property
    def provider(self) -> GeneratorProvider:
        return GeneratorProvider.OPENAI

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def max_context_tokens(self) -> int:
        return self._context_window

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate a grounded answer.

        1. Render system prompt from request.prompt_template (or default).
        2. Render context + question as the user turn.
        3. Compute adaptive max_new_tokens from context window.
        4. Call OpenAI and return structured GenerationResult.
        """
        t0 = time.perf_counter()

        template = request.prompt_template
        system_prompt = (
            template.render_system_prompt()
            if template
            else _DEFAULT_SYSTEM_PROMPT
        )

        if template:
            user_content = template.render_full_prompt(
                query=request.query,
                chunks=request.context_chunks,
            )
        else:
            context = self._default_render_context(request.context_chunks)
            user_content = (
                f"Context:\n{context}\n\n"
                f"Question: {request.query}\n\nAnswer:"
            )

        # Adaptive token budget
        temperature   = request.temperature if request.temperature is not None else self._default_temperature
        max_new_tokens = request.max_new_tokens

        if max_new_tokens is None:
            # Conservative estimate: ~4 chars per token for system + user turn
            estimated_prompt_tokens = (
                len(system_prompt) + len(user_content)
            ) // 4
            remaining = self._context_window - estimated_prompt_tokens - self._safety_margin
            max_new_tokens = max(64, min(self._default_max_new_tokens, remaining))

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ]

        try:
            kwargs = {
                "model":       self._model,
                "messages":    messages,
                "temperature": temperature,
                "max_tokens":  max_new_tokens,
            }
            if request.stop_sequences:
                kwargs["stop"] = request.stop_sequences

            resp = self._client.chat.completions.create(**kwargs)

            answer       = resp.choices[0].message.content or ""
            finish_reason = resp.choices[0].finish_reason or "stop"
            prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
            completion_tokens = resp.usage.completion_tokens if resp.usage else 0

        except Exception as e:
            logger.error(
                "[OpenAIGenerator] Generation failed | model=%s | error=%s",
                self._model, e,
            )
            answer        = f"[Generation error: {e}]"
            finish_reason = "error"
            prompt_tokens = completion_tokens = 0

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        logger.info(
            "[OpenAIGenerator] model=%s | prompt_tokens=%d | "
            "completion_tokens=%d | finish=%s | %.1fms",
            self._model, prompt_tokens, completion_tokens,
            finish_reason, latency_ms,
        )

        return GenerationResult(
            answer=answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            model=self._model,
            latency_ms=latency_ms,
            metadata={
                "correlation_id": request.correlation_id,
                "max_new_tokens_used": max_new_tokens,
            },
        )

    @staticmethod
    def _default_render_context(chunks: list) -> str:
        """Fallback context renderer when no PromptTemplate is provided."""
        if not chunks:
            return "No relevant context found."
        parts = []
        for i, chunk in enumerate(chunks, 1):
            if isinstance(chunk, ScoredCandidate):
                text   = chunk.text.strip()
                source = (
                    chunk.metadata.get("source")
                    or chunk.metadata.get("file_name")
                    or "unknown"
                )
            else:
                text   = str(chunk.get("text", "")).strip()
                source = (
                    chunk.get("metadata", {}).get("source")
                    or chunk.get("metadata", {}).get("file_name")
                    or "unknown"
                )
            parts.append(f"[{i}] {text}\n  [Source: {source}]")
        return "\n\n".join(parts)

    def health_check(self) -> bool:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=4,
            )
            return bool(resp.choices)
        except Exception as e:
            logger.warning("[OpenAIGenerator] health_check failed: %s", e)
            return False
