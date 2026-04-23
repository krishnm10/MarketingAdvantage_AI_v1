# =============================================
# app/ai/registry/ai_providers.py
#
# AI Provider Registry — CPU / GPU / API resolution with singleton caching
#
# SINGLETON PATTERN:
#   GPU models load multi-GB weights once per process.
#   Without singletons, each call to get_*() creates a NEW instance,
#   triggering a full model load on every request — unusable in production.
#
#   Fix: module-level dict + threading.Lock for each provider type.
#   Thread-safe double-checked locking ensures exactly one instance per key.
#
# GPU FALLBACK PATTERN:
#   Missing GPU provider files cause ImportError at runtime, not startup.
#   Fix: each GPU path has a try/except that falls back to CPU with a warning.
#
# Rules:
#   - This is the ONLY place that knows CPU vs GPU vs API
#   - Lazy imports only (avoid heavy startup cost)
#   - No ingestion logic
# =============================================

import logging
import threading
from typing import Any, Dict

from app.config.ai_config import (
    AI_PROFILE,
    AUDIO_MODEL_CPU,
    AUDIO_MODEL_GPU,
    LOG_AI_PROVIDER_SELECTION,
    VISION_MODEL_CPU,
    VISION_MODEL_GPU,
    VISION_QUANTIZE,
    VISION_FLASH_ATTENTION,
    VISION_MAX_PIXELS,
    VISION_API_PROVIDER,
    VISION_API_MODEL,
)

logger = logging.getLogger(__name__)

# ── Module-level singleton registry ──────────────────────────────────────────
# Keys are strings like "stt_gpu", "captioner_cpu", "vision_api_openai".
# Values are fully initialized provider instances (model weights loaded).
_PROVIDER_SINGLETONS: Dict[str, Any] = {}
_PROVIDER_LOCK = threading.Lock()


def _get_or_create(key: str, factory):
    """
    Thread-safe singleton factory.

    Double-checked locking: first check without lock (fast path), then
    re-check inside lock (safe path). Ensures one model load per process.
    """
    instance = _PROVIDER_SINGLETONS.get(key)
    if instance is not None:
        return instance
    with _PROVIDER_LOCK:
        instance = _PROVIDER_SINGLETONS.get(key)
        if instance is None:
            instance = factory()
            _PROVIDER_SINGLETONS[key] = instance
    return instance


def reset_singletons() -> None:
    """
    Clear all cached provider instances.
    Call after AI_PROFILE or model config changes (e.g., in tests or hot-reload).
    """
    with _PROVIDER_LOCK:
        _PROVIDER_SINGLETONS.clear()
    logger.info("[AI Registry] All provider singletons cleared")


# -------------------------------------------------
# Audio: Speech → Text
# -------------------------------------------------
def get_speech_to_text():
    """
    Resolve SpeechToText provider based on AI_PROFILE.
    Singleton: GPU model loaded at most once per process.
    """
    if AI_PROFILE == "gpu":
        def _factory():
            if LOG_AI_PROVIDER_SELECTION:
                logger.info("[AI Registry] Loading GPU SpeechToText: %s", AUDIO_MODEL_GPU)
            try:
                from app.ai.providers.gpu.whisper_gpu_v1 import WhisperGPUSpeechToText
                return WhisperGPUSpeechToText(model_size=AUDIO_MODEL_GPU)
            except ImportError:
                logger.warning(
                    "[AI Registry] GPU WhisperSpeechToText unavailable — falling back to CPU"
                )
                from app.ai.providers.local.whisper_cpu_v1 import WhisperCPUSpeechToText
                return WhisperCPUSpeechToText(model_size=AUDIO_MODEL_CPU)
        return _get_or_create(f"stt_{AI_PROFILE}_{AUDIO_MODEL_GPU}", _factory)

    # Default: CPU
    def _cpu_factory():
        if LOG_AI_PROVIDER_SELECTION:
            logger.info("[AI Registry] Loading CPU SpeechToText: %s", AUDIO_MODEL_CPU)
        from app.ai.providers.local.whisper_cpu_v1 import WhisperCPUSpeechToText
        return WhisperCPUSpeechToText(model_size=AUDIO_MODEL_CPU)

    return _get_or_create(f"stt_cpu_{AUDIO_MODEL_CPU}", _cpu_factory)


# -------------------------------------------------
# Image: Image → Caption
# -------------------------------------------------
def get_image_captioner():
    """
    Resolve ImageCaptioner provider.
    Singleton: GPU model loaded at most once per process.
    Falls back to CPU if GPU provider file is missing.
    """
    if AI_PROFILE == "gpu":
        def _factory():
            if LOG_AI_PROVIDER_SELECTION:
                logger.info("[AI Registry] Loading GPU ImageCaptioner")
            try:
                from app.ai.providers.gpu.image_caption_gpu_v1 import ImageCaptionerGPU
                return ImageCaptionerGPU()
            except ImportError:
                logger.warning(
                    "[AI Registry] GPU ImageCaptioner unavailable — falling back to CPU"
                )
                from app.ai.providers.local.image_caption_cpu_v1 import ImageCaptionerCPU
                return ImageCaptionerCPU()
        return _get_or_create("captioner_gpu", _factory)

    def _cpu_factory():
        if LOG_AI_PROVIDER_SELECTION:
            logger.info("[AI Registry] Loading CPU ImageCaptioner")
        from app.ai.providers.local.image_caption_cpu_v1 import ImageCaptionerCPU
        return ImageCaptionerCPU()

    return _get_or_create("captioner_cpu", _cpu_factory)


# -------------------------------------------------
# Video: Video → Text
# -------------------------------------------------
def get_video_to_text():
    """
    Resolve VideoToText provider.
    Singleton: GPU model loaded at most once per process.
    Falls back to CPU if GPU provider file is missing.
    """
    if AI_PROFILE == "gpu":
        def _factory():
            if LOG_AI_PROVIDER_SELECTION:
                logger.info("[AI Registry] Loading GPU VideoToText")
            try:
                from app.ai.providers.gpu.video_to_text_gpu_v1 import VideoToTextGPU
                return VideoToTextGPU()
            except ImportError:
                logger.warning(
                    "[AI Registry] GPU VideoToText unavailable — falling back to CPU"
                )
                from app.ai.providers.local.video_to_text_cpu_v1 import VideoToTextCPU
                return VideoToTextCPU()
        return _get_or_create("video_to_text_gpu", _factory)

    def _cpu_factory():
        if LOG_AI_PROVIDER_SELECTION:
            logger.info("[AI Registry] Loading CPU VideoToText")
        from app.ai.providers.local.video_to_text_cpu_v1 import VideoToTextCPU
        return VideoToTextCPU()

    return _get_or_create("video_to_text_cpu", _cpu_factory)


# -------------------------------------------------
# Visual Explanation (charts / tables)
# -------------------------------------------------
def get_visual_explainer():
    """
    Resolve VisualExplainer provider.
    Singleton: GPU model loaded at most once per process.
    Falls back to CPU if GPU provider file is missing.
    """
    if AI_PROFILE == "gpu":
        def _factory():
            if LOG_AI_PROVIDER_SELECTION:
                logger.info("[AI Registry] Loading GPU VisualExplainer")
            try:
                from app.ai.providers.gpu.visual_explainer_gpu_v1 import VisualExplainerGPU
                return VisualExplainerGPU()
            except ImportError:
                logger.warning(
                    "[AI Registry] GPU VisualExplainer unavailable — falling back to CPU"
                )
                from app.ai.providers.local.visual_explainer_cpu_v1 import VisualExplainerCPU
                return VisualExplainerCPU()
        return _get_or_create("visual_explainer_gpu", _factory)

    def _cpu_factory():
        if LOG_AI_PROVIDER_SELECTION:
            logger.info("[AI Registry] Loading CPU VisualExplainer")
        from app.ai.providers.local.visual_explainer_cpu_v1 import VisualExplainerCPU
        return VisualExplainerCPU()

    return _get_or_create("visual_explainer_cpu", _cpu_factory)


# -------------------------------------------------
# Vision: Multimodal Vision Encoder
# -------------------------------------------------
def get_vision_encoder():
    """
    Resolve MultimodalVisionEncoder based on AI_PROFILE.

    AI_PROFILE routing:
        "cpu"  → VisionEncoderCPUV1 (Qwen2.5-VL-2B → moondream2 fallback)
        "gpu"  → QwenVLGPUEncoder   (Qwen2.5-VL-7B with BF16 + Flash Attn)
        "api"  → OpenAI / Anthropic / Google based on VISION_API_PROVIDER

    All providers implement MultimodalVisionEncoder — callers
    never import a concrete class directly.

    Singleton: GPU and CPU models loaded at most once per process.
    API providers are stateless (no weights); still cached for connection reuse.
    """
    if AI_PROFILE == "api":
        provider = (VISION_API_PROVIDER or "openai").lower()
        cache_key = f"vision_api_{provider}_{VISION_API_MODEL or 'default'}"

        def _api_factory():
            if LOG_AI_PROVIDER_SELECTION:
                logger.info(
                    "[AI Registry] Initializing API VisionEncoder: %s/%s",
                    VISION_API_PROVIDER, VISION_API_MODEL,
                )
            if provider == "openai":
                from app.ai.providers.api.vision_encoder_api_v1 import OpenAIVisionEncoder
                return OpenAIVisionEncoder(model=VISION_API_MODEL)
            if provider == "anthropic":
                from app.ai.providers.api.vision_encoder_api_v1 import AnthropicVisionEncoder
                return AnthropicVisionEncoder(model=VISION_API_MODEL)
            if provider == "google":
                from app.ai.providers.api.vision_encoder_api_v1 import GeminiVisionEncoder
                return GeminiVisionEncoder(model=VISION_API_MODEL)
            raise ValueError(f"Unknown VISION_API_PROVIDER: '{provider}'")

        return _get_or_create(cache_key, _api_factory)

    if AI_PROFILE == "gpu":
        cache_key = f"vision_gpu_{VISION_MODEL_GPU}"

        def _gpu_factory():
            if LOG_AI_PROVIDER_SELECTION:
                logger.info("[AI Registry] Loading GPU VisionEncoder: %s", VISION_MODEL_GPU)
            try:
                from app.ai.providers.gpu.vision_encoder_gpu_v1 import QwenVLGPUEncoder
                return QwenVLGPUEncoder(
                    model_name=VISION_MODEL_GPU,
                    quantize=VISION_QUANTIZE,
                    use_flash_attention=VISION_FLASH_ATTENTION,
                    max_pixels=VISION_MAX_PIXELS,
                )
            except ImportError:
                logger.warning(
                    "[AI Registry] GPU VisionEncoder unavailable — falling back to CPU"
                )
                from app.ai.providers.local.vision_encoder_cpu_v1 import VisionEncoderCPUV1
                return VisionEncoderCPUV1(model_name=VISION_MODEL_CPU)

        return _get_or_create(cache_key, _gpu_factory)

    # Default: CPU
    cache_key = f"vision_cpu_{VISION_MODEL_CPU}"

    def _cpu_factory():
        if LOG_AI_PROVIDER_SELECTION:
            logger.info(
                "[AI Registry] Configuring CPU VisionEncoder (auto-fallback chain): %s",
                VISION_MODEL_CPU,
            )
        from app.ai.providers.local.vision_encoder_cpu_v1 import VisionEncoderCPUV1
        return VisionEncoderCPUV1(model_name=VISION_MODEL_CPU)

    return _get_or_create(cache_key, _cpu_factory)
