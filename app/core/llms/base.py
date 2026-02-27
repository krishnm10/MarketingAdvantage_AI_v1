"""
================================================================================
Marketing Advantage AI — LLM Base Contract
File: app/core/llms/base.py

PURPOSE:
  Defines the strict interface that every LLM backend must implement.
  This ensures the RAG pipeline and LLMChain work identically regardless
  of whether the underlying model is Ollama, OpenAI, Groq, Anthropic or Gemini.

DESIGN:
  - generate()  → single prompt → single response (simplest case)
  - chat()      → multi-turn messages → response (full conversation support)
  - stream()    → async generator for streaming responses (UI use cases)
  - info        → model identity metadata
================================================================================
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Optional


@dataclass(frozen=True)
class LLMInfo:
    """Identity metadata for a loaded LLM."""
    provider:    str
    model:       str
    supports_streaming: bool = False
    supports_system_prompt: bool = True


@dataclass
class LLMResponse:
    """
    Structured response from any LLM call.
    Wraps raw text with usage stats (token counts) for observability.
    """
    text:              str
    model:             str
    prompt_tokens:     int = 0
    completion_tokens: int = 0
    total_tokens:      int = 0
    finish_reason:     str = "stop"
    raw:               Optional[object] = None   # original SDK response object


class BaseLLM(abc.ABC):
    """All LLM connectors implement this interface."""

    @property
    @abc.abstractmethod
    def info(self) -> LLMInfo:
        raise NotImplementedError

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
    ) -> LLMResponse:
        """
        Single-turn generation.
        Convenience wrapper — internally calls chat() for most providers.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
    ) -> LLMResponse:
        """
        Multi-turn conversation.

        Args:
            messages: List of {"role": "system"/"user"/"assistant", "content": "..."}
        """
        raise NotImplementedError

    async def stream(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """
        Async streaming response — yields text chunks as they arrive.
        Default implementation falls back to a single non-streaming generate().
        Override in providers that support native streaming.
        """
        response = self.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        yield response.text

    # ── Shared helper ────────────────────────────────────────────────────

    @staticmethod
    def _build_messages(
        prompt: str,
        system_prompt: Optional[str],
    ) -> List[Dict[str, str]]:
        """Builds standard message list from prompt + optional system prompt."""
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages
