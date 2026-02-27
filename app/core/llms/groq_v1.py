"""
================================================================================
Marketing Advantage AI — Groq LLM Connector
File: app/core/llms/groq_v1.py

Groq provides hardware-accelerated inference — the fastest LLM API available.
Free tier is very generous (ideal for Indian startups).

Best models on Groq:
  - llama-3.3-70b-versatile   (best quality on Groq)
  - llama3-8b-8192            (fastest, great for chain step 1)
  - mixtral-8x7b-32768        (large context, good reasoning)
  - gemma2-9b-it              (lightweight, fast)

Install:
  pip install groq
================================================================================
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator, Dict, List, Optional

from app.core.llms.base import BaseLLM, LLMInfo, LLMResponse

logger = logging.getLogger(__name__)


class GroqLLM(BaseLLM):
    """
    Groq LPU-accelerated LLM connector.

    Ideal for:
      - Step 1 of LLMChain (fast entity extraction from query)
      - Low-latency RAG for real-time user-facing features
      - Cost-effective high-throughput processing

    Args:
        model:   Groq model ID e.g. 'llama3-8b-8192'
        api_key: Groq API key (from env via factory)
        timeout: HTTP timeout in seconds
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        timeout: float = 30.0,
    ):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("groq not installed. Run: pip install groq")

        from groq import Groq as _Groq

        self._client = _Groq(api_key=api_key, timeout=timeout)
        self._model  = model

        logger.info("[GroqLLM] Initialized | model=%s", model)

    @property
    def info(self) -> LLMInfo:
        return LLMInfo(
            provider="groq",
            model=self._model,
            supports_streaming=True,
            supports_system_prompt=True,
        )

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
    ) -> LLMResponse:
        messages = self._build_messages(prompt, system_prompt)
        return self.chat(messages, temperature=temperature,
                         max_tokens=max_tokens, stop=stop)

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
    ) -> LLMResponse:
        kwargs: Dict = {
            "model":       self._model,
            "messages":    messages,
            "temperature": float(temperature),
            "max_tokens":  int(max_tokens),
        }
        if stop:
            kwargs["stop"] = stop

        res = self._client.chat.completions.create(**kwargs)
        choice = res.choices[0]
        usage  = res.usage

        return LLMResponse(
            text=choice.message.content or "",
            model=res.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            finish_reason=choice.finish_reason or "stop",
            raw=res,
        )

    async def stream(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        messages = self._build_messages(prompt, system_prompt)
        with self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
