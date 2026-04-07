# =============================================
# registry.py
#
# AI Provider Registry
#
# This file resolves AI capability implementations
# based on runtime configuration.
#
# Rules:
# - This is the ONLY place that knows CPU vs GPU vs API
# - Lazy imports only (avoid heavy startup cost)
# - No ingestion logic
# =============================================

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


# -------------------------------------------------
# Audio: Speech → Text
# -------------------------------------------------
def get_speech_to_text():
    """
    Resolve SpeechToText provider based on AI_PROFILE.
    """
    if AI_PROFILE == "gpu":
        if LOG_AI_PROVIDER_SELECTION:
            print("[AI Registry] Using GPU SpeechToText provider")

        from app.ai.providers.gpu.whisper_gpu_v1 import WhisperGPUSpeechToText
        return WhisperGPUSpeechToText(model_size=AUDIO_MODEL_GPU)

    # Default: CPU
    if LOG_AI_PROVIDER_SELECTION:
        print("[AI Registry] Using CPU SpeechToText provider")

    from app.ai.providers.local.whisper_cpu_v1 import WhisperCPUSpeechToText
    return WhisperCPUSpeechToText(model_size=AUDIO_MODEL_CPU)


# -------------------------------------------------
# Image: Image → Caption
# -------------------------------------------------
def get_image_captioner():
    """
    Resolve ImageCaptioner provider.
    """
    if AI_PROFILE == "gpu":
        if LOG_AI_PROVIDER_SELECTION:
            print("[AI Registry] Using GPU ImageCaptioner provider")

        from app.ai.providers.gpu.image_caption_gpu_v1 import ImageCaptionerGPU
        return ImageCaptionerGPU()

    if LOG_AI_PROVIDER_SELECTION:
        print("[AI Registry] Using CPU ImageCaptioner provider")

    from app.ai.providers.local.image_caption_cpu_v1 import ImageCaptionerCPU
    return ImageCaptionerCPU()


# -------------------------------------------------
# Video: Video → Text
# -------------------------------------------------
def get_video_to_text():
    """
    Resolve VideoToText provider.
    """
    if AI_PROFILE == "gpu":
        if LOG_AI_PROVIDER_SELECTION:
            print("[AI Registry] Using GPU VideoToText provider")

        from app.ai.providers.gpu.video_to_text_gpu_v1 import VideoToTextGPU
        return VideoToTextGPU()

    if LOG_AI_PROVIDER_SELECTION:
        print("[AI Registry] Using CPU VideoToText provider")

    from app.ai.providers.local.video_to_text_cpu_v1 import VideoToTextCPU
    return VideoToTextCPU()


# -------------------------------------------------
# Visual Explanation (charts / tables)
# -------------------------------------------------
def get_visual_explainer():
    """
    Resolve VisualExplainer provider.
    """
    if AI_PROFILE == "gpu":
        if LOG_AI_PROVIDER_SELECTION:
            print("[AI Registry] Using GPU VisualExplainer provider")

        from app.ai.providers.gpu.visual_explainer_gpu_v1 import VisualExplainerGPU
        return VisualExplainerGPU()

    if LOG_AI_PROVIDER_SELECTION:
        print("[AI Registry] Using CPU VisualExplainer provider")

    from app.ai.providers.local.visual_explainer_cpu_v1 import VisualExplainerCPU
    return VisualExplainerCPU()


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
    """
    if AI_PROFILE == "api":
        if LOG_AI_PROVIDER_SELECTION:
            print(f"[AI Registry] Using API VisionEncoder: {VISION_API_PROVIDER}/{VISION_API_MODEL}")

        provider = (VISION_API_PROVIDER or "openai").lower()

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

    if AI_PROFILE == "gpu":
        if LOG_AI_PROVIDER_SELECTION:
            print(f"[AI Registry] Using GPU VisionEncoder: {VISION_MODEL_GPU}")
        from app.ai.providers.gpu.vision_encoder_gpu_v1 import QwenVLGPUEncoder
        return QwenVLGPUEncoder(
            model_name=VISION_MODEL_GPU,
            quantize=VISION_QUANTIZE,
            use_flash_attention=VISION_FLASH_ATTENTION,
            max_pixels=VISION_MAX_PIXELS,
        )

    # Default: CPU (lazy — actual model resolved at first encode call)
    if LOG_AI_PROVIDER_SELECTION:
        print(f"[AI Registry] Configured CPU VisionEncoder (auto-fallback chain)")
    from app.ai.providers.local.vision_encoder_cpu_v1 import VisionEncoderCPUV1
    return VisionEncoderCPUV1(model_name=VISION_MODEL_CPU)
