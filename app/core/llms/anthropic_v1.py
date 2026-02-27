"""
================================================================================
Marketing Advantage AI — Anthropic Claude LLM Connector
File: app/core/llms/anthropic_v1.py

Supports:
  - claude-3-5-sonnet-20241022  (best reasoning, long context 200K)
  - claude-3-5-haiku-20241022   (fastest Claude, low cost)
  - claude-3-opus-20240229      (most powerful, highest cost)

Claude excels at:
  - Long document analysis (200K context window)
  - Multi-step reasoning chains
  - Structured output generation
  - Safety-critical enterprise use cases

Install:
  pip install anthropic
================================================================================
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator, Dict, List, Optional

from app.core.llms.base import BaseLLM, LLMInfo, LLMResponse

logger = logging.getLogger(__name__)


class AnthropicLLM(BaseLLM):
    """
    Anthropic Claude connector.

    Important Anthropic API difference:
      Claude uses a separate 'system' parameter (not inside messages list).
      We handle this transparently in chat() so callers never need to know.

    Args:
        model:   Claude model ID e.g. 'claude-3-5-sonnet-20241022'
        api_key: Anthropic API key (from env via factory)
        timeout: HTTP timeout in seconds
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        timeout: float = 120.0,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic not installed. Run: pip install anthropic")

        import anthropic as _ant

        self._client = _ant.Anthropic(api_key=api_key, timeout=timeout)
        self._model  = model

        logger.info("[AnthropicLLM] Initialized | model=%s", model)

    @property
    def info(self) -> LLMInfo:
        return LLMInfo(
            provider="anthropic",
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
        # Claude: extract system prompt from messages list (its API is different)
        system_text: Optional[str] = None
        user_messages: List[Dict[str, str]] = []

        for msg in messages:
            if msg["role"] == "system":
                # Claude takes system as a top-level parameter
                system_text = msg["content"]
            else:
                user_messages.append(msg)

        kwargs: Dict = {
            "model":       self._model,
            "max_tokens":  int(max_tokens),
            "temperature": float(temperature),
            "messages":    user_messages,
        }
        if system_text:
            kwargs["system"] = system_text
        if stop:
            kwargs["stop_sequences"] = stop

        res = self._client.messages.create(**kwargs)

        # Anthropic content is a list of blocks — join text blocks
        text = "".join(
            block.text
            for block in res.content
            if hasattr(block, "text")
        )

        usage = res.usage
        return LLMResponse(
            text=text,
            model=res.model,
            prompt_tokens=usage.input_tokens if usage else 0,
            completion_tokens=usage.output_tokens if usage else 0,
            total_tokens=(usage.input_tokens + usage.output_tokens) if usage else 0,
            finish_reason=res.stop_reason or "stop",
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
        """Yield streaming text from Claude."""
        messages = self._build_messages(prompt, system_prompt)
        # Extract system from messages
        system_text = None
        user_msgs = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                user_msgs.append(msg)

        kwargs: Dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": user_msgs,
        }
        if system_text:
            kwargs["system"] = system_text

        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                if text:
                    yield text
