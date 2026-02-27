"""
================================================================================
Marketing Advantage AI — OpenAI LLM Connector
File: app/core/llms/openai_v1.py

Supports:
  - gpt-4o, gpt-4o-mini
  - gpt-4-turbo, gpt-4
  - gpt-3.5-turbo
  - Any OpenAI-compatible API (Azure OpenAI, LM Studio, vLLM, LocalAI)
    via custom base_url

Install:
  pip install openai
================================================================================
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator, Dict, List, Optional

from app.core.llms.base import BaseLLM, LLMInfo, LLMResponse

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """
    OpenAI GPT connector.

    Also works as a drop-in for any OpenAI-compatible API by setting
    base_url to the target server (e.g., Azure, LM Studio, vLLM).

    Args:
        model:          OpenAI model name e.g. 'gpt-4o-mini'
        api_key:        OpenAI API key (read from env via factory — never hardcode)
        organization:   Optional OpenAI organization ID
        base_url:       Override API base (for OpenAI-compatible servers)
        timeout:        HTTP timeout in seconds
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")

        from openai import OpenAI as _OAI

        init_kwargs: Dict = {
            "api_key": api_key,
            "timeout": timeout,
        }
        if organization:
            init_kwargs["organization"] = organization
        if base_url:
            init_kwargs["base_url"] = base_url

        self._client = _OAI(**init_kwargs)
        self._model  = model

        logger.info(
            "[OpenAILLM] Initialized | model=%s | base_url=%s",
            model, base_url or "default"
        )

    @property
    def info(self) -> LLMInfo:
        return LLMInfo(
            provider="openai",
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
        """Yield SSE text chunks from OpenAI streaming endpoint."""
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
