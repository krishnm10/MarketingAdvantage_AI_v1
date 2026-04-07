# =============================================
# vision_encoder_api_v1.py
#
# Cloud API Multimodal Vision Encoder Connectors
#
# Connectors:
#   OpenAIVisionEncoder     → GPT-4o / GPT-4-vision
#   AnthropicVisionEncoder  → Claude 3.5 Sonnet / Claude 3.7
#   GeminiVisionEncoder     → Gemini 2.0 Flash / 1.5 Pro
#
# Use when AI_PROFILE="api".
# API keys are read from environment variables.
#
# Security:
#   - API keys never logged
#   - File contents base64-encoded in-memory only
#   - Timeouts enforced per call
# =============================================

import asyncio
import base64
import os
import mimetypes
import logging
from typing import Dict, Any, List, Optional

import httpx

from app.ai.contracts import MultimodalVisionEncoder
from app.utils.logger import log_info, log_warning

logger = logging.getLogger(__name__)


# -------------------------------------------------
# Shared utility
# -------------------------------------------------
def _image_to_base64(image_path: str) -> tuple:
    """Returns (base64_string, media_type)."""
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "image/jpeg"
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8"), mime_type


_DEFAULT_PROMPTS = {
    "caption": "Describe this image in comprehensive detail, including all text, data, and visual elements.",
    "ocr": "Extract ALL text from this image exactly as it appears.",
    "chart": (
        "Analyze this chart thoroughly:\n"
        "TITLE: [chart title]\nTYPE: [bar/line/pie/scatter]\n"
        "X_AXIS: [label and range]\nY_AXIS: [label and range]\n"
        "KEY_VALUES: [important numbers]\nTREND: [main trend]\nINSIGHT: [business insight]"
    ),
    "document": "Extract the complete text content of this document, preserving structure.",
    "vqa": "Answer the following question about this image:",
    "embed": "Provide a detailed description of this image for semantic indexing.",
}

_API_TIMEOUT = 120.0  # seconds per API call
_MAX_RETRIES = 2


# =============================================
# Connector 1: OpenAI GPT-4o Vision
# =============================================
class OpenAIVisionEncoder(MultimodalVisionEncoder):
    """GPT-4o / GPT-4-vision via OpenAI API."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        detail: str = "high",
    ):
        self._model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._max_tokens = max_tokens
        self._detail = detail

        if not self._api_key:
            raise EnvironmentError(
                "OpenAI vision encoder requires OPENAI_API_KEY in environment."
            )

    @property
    def model_name(self) -> str:
        return f"openai/{self._model}"

    @property
    def supports_video_frames(self) -> bool:
        return True

    @property
    def supports_batch(self) -> bool:
        return True

    @property
    def max_image_size_mb(self) -> float:
        return 20.0

    async def _call_api(self, image_path: str, prompt: str) -> str:
        b64, mime = _image_to_base64(image_path)
        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64}",
                                "detail": self._detail,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        async with httpx.AsyncClient(timeout=_API_TIMEOUT) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    async def encode(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        mode: str = "caption",
    ) -> Dict[str, Any]:
        effective_prompt = prompt or _DEFAULT_PROMPTS.get(mode, _DEFAULT_PROMPTS["caption"])
        result_text = await self._call_api(image_path, effective_prompt)

        is_chart = any(k in result_text.lower() for k in ["chart", "graph", "axis", "trend", "percent"])
        return {
            "caption":     result_text,
            "ocr_text":    result_text if mode == "ocr" else None,
            "chart_data":  result_text if (mode == "chart" or is_chart) else None,
            "embedding":   None,
            "is_chart":    is_chart,
            "is_document": mode == "document",
            "confidence":  0.95,
            "model_used":  self.model_name,
            "tokens_used": None,
        }

    async def encode_batch(
        self,
        image_paths: List[str],
        prompt: Optional[str] = None,
        mode: str = "caption",
    ) -> List[Dict[str, Any]]:
        tasks = [self.encode(p, prompt=prompt, mode=mode) for p in image_paths]
        return list(await asyncio.gather(*tasks))


# =============================================
# Connector 2: Anthropic Claude Vision
# =============================================
class AnthropicVisionEncoder(MultimodalVisionEncoder):
    """Claude 3.5 Sonnet / Claude 3.7 via Anthropic API."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
    ):
        self._model = model
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._max_tokens = max_tokens

        if not self._api_key:
            raise EnvironmentError("Anthropic vision encoder requires ANTHROPIC_API_KEY.")

    @property
    def model_name(self) -> str:
        return f"anthropic/{self._model}"

    @property
    def supports_video_frames(self) -> bool:
        return True

    @property
    def supports_batch(self) -> bool:
        return True

    @property
    def max_image_size_mb(self) -> float:
        return 20.0

    async def _call_api(self, image_path: str, prompt: str) -> str:
        b64, mime = _image_to_base64(image_path)
        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime,
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        async with httpx.AsyncClient(timeout=_API_TIMEOUT) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return "".join(
                block["text"] for block in data["content"]
                if block.get("type") == "text"
            ).strip()

    async def encode(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        mode: str = "caption",
    ) -> Dict[str, Any]:
        effective_prompt = prompt or _DEFAULT_PROMPTS.get(mode, _DEFAULT_PROMPTS["caption"])
        result_text = await self._call_api(image_path, effective_prompt)

        is_chart = any(k in result_text.lower() for k in ["chart", "graph", "axis", "trend"])
        return {
            "caption":     result_text,
            "ocr_text":    result_text if mode == "ocr" else None,
            "chart_data":  result_text if (mode == "chart" or is_chart) else None,
            "embedding":   None,
            "is_chart":    is_chart,
            "is_document": mode == "document",
            "confidence":  0.96,
            "model_used":  self.model_name,
            "tokens_used": None,
        }

    async def encode_batch(
        self,
        image_paths: List[str],
        prompt: Optional[str] = None,
        mode: str = "caption",
    ) -> List[Dict[str, Any]]:
        tasks = [self.encode(p, prompt=prompt, mode=mode) for p in image_paths]
        return list(await asyncio.gather(*tasks))


# =============================================
# Connector 3: Google Gemini Vision
# =============================================
class GeminiVisionEncoder(MultimodalVisionEncoder):
    """Gemini 2.0 Flash / 1.5 Pro via Google AI Studio API."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
    ):
        self._model = model
        self._api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self._max_tokens = max_tokens

        if not self._api_key:
            raise EnvironmentError(
                "Gemini vision encoder requires GOOGLE_API_KEY in environment."
            )

    @property
    def model_name(self) -> str:
        return f"google/{self._model}"

    @property
    def supports_video_frames(self) -> bool:
        return True

    @property
    def supports_batch(self) -> bool:
        return True

    @property
    def max_image_size_mb(self) -> float:
        return 20.0

    async def _call_api(self, image_path: str, prompt: str) -> str:
        b64, mime = _image_to_base64(image_path)
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": mime,
                                "data": b64,
                            }
                        },
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": self._max_tokens,
                "temperature": 0.0,
            },
        }
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent?key={self._api_key}"
        )
        async with httpx.AsyncClient(timeout=_API_TIMEOUT) as client:
            resp = await client.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts).strip()

    async def encode(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        mode: str = "caption",
    ) -> Dict[str, Any]:
        effective_prompt = prompt or _DEFAULT_PROMPTS.get(mode, _DEFAULT_PROMPTS["caption"])
        result_text = await self._call_api(image_path, effective_prompt)

        is_chart = any(k in result_text.lower() for k in ["chart", "graph", "axis", "trend"])
        return {
            "caption":     result_text,
            "ocr_text":    result_text if mode == "ocr" else None,
            "chart_data":  result_text if (mode == "chart" or is_chart) else None,
            "embedding":   None,
            "is_chart":    is_chart,
            "is_document": mode == "document",
            "confidence":  0.93,
            "model_used":  self.model_name,
            "tokens_used": None,
        }

    async def encode_batch(
        self,
        image_paths: List[str],
        prompt: Optional[str] = None,
        mode: str = "caption",
    ) -> List[Dict[str, Any]]:
        tasks = [self.encode(p, prompt=prompt, mode=mode) for p in image_paths]
        return list(await asyncio.gather(*tasks))
