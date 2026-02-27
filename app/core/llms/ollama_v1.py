"""
================================================================================
Marketing Advantage AI — Ollama LLM Connector
File: app/core/llms/ollama_v1.py

Supports:
  - Any model pulled via Ollama (llama3.2, mistral, gemma2, phi3, etc.)
  - Local CPU and GPU inference
  - Native streaming via Ollama Python SDK

Your existing app/services/ingestion/llm_rewriter.py is NOT touched.
This connector provides a new clean interface for the pluggable pipeline.

Install:
  pip install ollama
================================================================================
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator, Dict, List, Optional

from app.core.llms.base import BaseLLM, LLMInfo, LLMResponse

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """
    Ollama local LLM connector.

    Args:
        model:       Ollama model name e.g. 'llama3.2', 'mistral', 'gemma2:9b'
        base_url:    Ollama server URL (default: http://localhost:11434)
        timeout:     Request timeout in seconds
        keep_alive:  Duration to keep model loaded in memory
                     e.g. "5m", "1h", "0" (unload immediately)
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        keep_alive: str = "5m",
    ):
        try:
            import ollama
        except ImportError:
            raise ImportError("ollama not installed. Run: pip install ollama")

        import ollama as _ol

        self._model     = model
        self._keep_alive = keep_alive
        self._client    = _ol.Client(host=base_url, timeout=timeout)

        logger.info(
            "[OllamaLLM] Initialized | model=%s | url=%s", model, base_url
        )

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def info(self) -> LLMInfo:
        return LLMInfo(
            provider="ollama",
            model=self._model,
            supports_streaming=True,
            supports_system_prompt=True,
        )

    # ── Core generation ─────────────────────────────────────────────────

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
        options: Dict = {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        }
        if stop:
            options["stop"] = stop

        response = self._client.chat(
            model=self._model,
            messages=messages,
            options=options,
            keep_alive=self._keep_alive,
        )

        content = response["message"]["content"]
        usage   = response.get("usage") or {}

        return LLMResponse(
            text=content,
            model=self._model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            finish_reason=response.get("done_reason", "stop"),
            raw=response,
        )

    # ── Streaming ────────────────────────────────────────────────────────

    async def stream(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """Yield text chunks from Ollama streaming response."""
        messages = self._build_messages(prompt, system_prompt)
        for chunk in self._client.chat(
            model=self._model,
            messages=messages,
            stream=True,
            options={"temperature": temperature, "num_predict": max_tokens},
            keep_alive=self._keep_alive,
        ):
            delta = chunk.get("message", {}).get("content", "")
            if delta:
                yield delta
