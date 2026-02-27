"""
================================================================================
Marketing Advantage AI — Google Gemini LLM Connector
File: app/core/llms/gemini_v1.py

Supports:
  - gemini-1.5-pro-latest     (best quality, 1M context)
  - gemini-1.5-flash-latest   (fastest Gemini, low cost)
  - gemini-2.0-flash          (latest 2026 model, fast + capable)

Gemini excels at:
  - Multi-modal input (text + image in same request)
  - Very long context windows (up to 1M tokens)
  - Strong multilingual support (excellent for Indian languages)

Install:
  pip install google-generativeai
================================================================================
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator, Dict, List, Optional

from app.core.llms.base import BaseLLM, LLMInfo, LLMResponse

logger = logging.getLogger(__name__)


class GeminiLLM(BaseLLM):
    """
    Google Gemini connector via google-generativeai SDK.

    Args:
        model:       Gemini model ID e.g. 'gemini-1.5-flash-latest'
        api_key:     Google AI API key (from env via factory)
        timeout:     Request timeout in seconds
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        timeout: float = 60.0,
    ):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. "
                "Run: pip install google-generativeai"
            )

        import google.generativeai as _genai

        _genai.configure(api_key=api_key)
        self._model_name = model
        self._timeout    = timeout

        # Safety settings: permissive for enterprise marketing content
        # Adjust as required for your content policies
        self._safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        self._model = _genai.GenerativeModel(
            model_name=model,
            safety_settings=self._safety_settings,
        )

        logger.info("[GeminiLLM] Initialized | model=%s", model)

    @property
    def info(self) -> LLMInfo:
        return LLMInfo(
            provider="gemini",
            model=self._model_name,
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
        import google.generativeai as _genai

        # Gemini API uses "parts" format — convert from OpenAI-style messages
        gemini_history = []
        system_instruction = None
        final_prompt = ""

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                # Gemini handles system as model instruction
                system_instruction = content

            elif role == "user":
                final_prompt = content        # last user message = prompt
                if gemini_history or system_instruction:
                    gemini_history.append({"role": "user", "parts": [content]})

            elif role == "assistant":
                gemini_history.append({"role": "model", "parts": [content]})

        # Rebuild model with system instruction if present
        if system_instruction:
            model = _genai.GenerativeModel(
                model_name=self._model_name,
                safety_settings=self._safety_settings,
                system_instruction=system_instruction,
            )
        else:
            model = self._model

        gen_config = _genai.types.GenerationConfig(
            temperature=float(temperature),
            max_output_tokens=int(max_tokens),
            stop_sequences=stop or [],
        )

        if gemini_history:
            # Multi-turn chat
            chat_session = model.start_chat(history=gemini_history[:-1])
            res = chat_session.send_message(
                final_prompt, generation_config=gen_config
            )
        else:
            # Single-turn generation
            res = model.generate_content(final_prompt, generation_config=gen_config)

        text = res.text if hasattr(res, "text") else ""
        usage = getattr(res, "usage_metadata", None)

        return LLMResponse(
            text=text,
            model=self._model_name,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
            total_tokens=getattr(usage, "total_token_count", 0) if usage else 0,
            finish_reason="stop",
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
        """Yield streaming text chunks from Gemini."""
        import google.generativeai as _genai

        gen_config = _genai.types.GenerationConfig(
            temperature=float(temperature),
            max_output_tokens=int(max_tokens),
        )

        full_prompt = (
            f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        )

        res = self._model.generate_content(
            full_prompt,
            generation_config=gen_config,
            stream=True,
        )
        for chunk in res:
            text = chunk.text if hasattr(chunk, "text") else ""
            if text:
                yield text
