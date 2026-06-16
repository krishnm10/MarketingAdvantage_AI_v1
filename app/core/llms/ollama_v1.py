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
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from app.core.llms.base import BaseLLM, LLMInfo, LLMResponse

logger = logging.getLogger(__name__)


def _ollama_get_field(response: Any, key: str, default: Any = None) -> Any:
    """Read a field from Ollama ChatResponse (Pydantic) or dict."""
    if hasattr(response, key):
        val = getattr(response, key, default)
        return default if val is None else val
    if isinstance(response, dict):
        return response.get(key, default)
    return default


def _extract_ollama_token_usage(response: Any) -> Tuple[int, int, int]:
    """
    Map Ollama native counts to OpenAI-style prompt/completion/total.

    Ollama reports prompt_eval_count and eval_count per API response.
    """
    usage = _ollama_get_field(response, "usage") or {}
    if isinstance(usage, dict) and usage:
        pt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        ct = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        tt = int(usage.get("total_tokens") or 0)
        if tt <= 0 and (pt > 0 or ct > 0):
            tt = pt + ct
        if pt > 0 or ct > 0 or tt > 0:
            return pt, ct, tt

    pt = int(_ollama_get_field(response, "prompt_eval_count", 0) or 0)
    ct = int(_ollama_get_field(response, "eval_count", 0) or 0)
    tt = pt + ct if (pt > 0 or ct > 0) else 0
    return pt, ct, tt


def _ollama_message_content(response: Any) -> str:
    msg = _ollama_get_field(response, "message")
    if msg is None:
        return ""
    if hasattr(msg, "content"):
        return str(getattr(msg, "content", "") or "")
    if isinstance(msg, dict):
        return str(msg.get("content") or "")
    return ""


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
        timeout: int = 250,  # seconds; retrieve/chat overrides via instantiate_llm + OLLAMA_LLM_TIMEOUT_SECONDS
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
            "[OllamaLLM] Initialized | model=%s | url=%s | timeout=%ss",
            model,
            base_url,
            timeout,
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

        content = _ollama_message_content(response)
        pt, ct, tt = _extract_ollama_token_usage(response)
        finish_reason = _ollama_get_field(response, "done_reason", "stop")

        return LLMResponse(
            text=content,
            model=self._model,
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=tt,
            finish_reason=str(finish_reason) if finish_reason is not None else "stop",
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
        """Yield text chunks from Ollama streaming response.

        The Ollama Python SDK is synchronous, so we offload the blocking
        iterator to a thread via asyncio.to_thread to avoid starving the
        event loop under concurrent requests.
        """
        import asyncio
        import queue as _queue

        messages = self._build_messages(prompt, system_prompt)
        q: _queue.Queue[Optional[str]] = _queue.Queue()

        def _sync_stream() -> None:
            try:
                for chunk in self._client.chat(
                    model=self._model,
                    messages=messages,
                    stream=True,
                    options={"temperature": temperature, "num_predict": max_tokens},
                    keep_alive=self._keep_alive,
                ):
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        q.put(delta)
            finally:
                q.put(None)  # sentinel

        task = asyncio.get_running_loop().run_in_executor(None, _sync_stream)
        while True:
            try:
                item = await asyncio.to_thread(q.get, timeout=0.5)
            except Exception:
                if task.done():
                    break
                continue
            if item is None:
                break
            yield item
        await task  # propagate any exception
