# app/ai/registry/__init__.py
#
# Unified export surface for app.ai.registry.
#
# Two concerns live here:
#   1. EmbedderRegistry  — Phase 1 embedder / tokenizer bundle resolution
#   2. AI Provider fns   — CPU/GPU/API routing for vision, audio, video
#      (migrated from the former app/ai/registry.py module file)

from app.ai.registry.embedder_registry import (
    EmbedderRegistry,
    UnknownEmbedderModelError,
    embedder_registry,
)
from app.ai.registry.ai_providers import (
    get_speech_to_text,
    get_image_captioner,
    get_video_to_text,
    get_visual_explainer,
    get_vision_encoder,
)

__all__ = [
    # Phase 1 — Embedder registry
    "EmbedderRegistry",
    "UnknownEmbedderModelError",
    "embedder_registry",
    # AI provider resolution (CPU / GPU / API)
    "get_speech_to_text",
    "get_image_captioner",
    "get_video_to_text",
    "get_visual_explainer",
    "get_vision_encoder",
]
