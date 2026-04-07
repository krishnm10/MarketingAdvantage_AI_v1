# =============================================
# ai_config.py
#
# Global runtime configuration for AI execution.
# This file MUST remain:
# - lightweight
# - import-safe
# - free of model logic
#
# Any change here should switch behavior
# system-wide without code refactors.
# =============================================

import os

# -------------------------------------------------
# AI Execution Profile
#
# cpu  → local CPU-only models (current setup)
# gpu  → local GPU-backed models
# api  → cloud API providers (OpenAI, Anthropic, Google)
# dist → distributed / remote inference (future)
# -------------------------------------------------
AI_PROFILE = os.getenv("AI_PROFILE", "cpu")   # allowed: "cpu", "gpu", "api", "dist"


# -------------------------------------------------
# Audio configuration
# -------------------------------------------------
AUDIO_MODEL_CPU = "base"     # whisper: tiny | base | small
AUDIO_MODEL_GPU = "large"    # future use


# -------------------------------------------------
# Vision Encoder Configuration
#
# CPU models:
#   Qwen/Qwen2.5-VL-3B-Instruct  → best CPU quality, ~30s load
#   vikhyatk/moondream2           → ultra-fast 1.8B fallback
#
# GPU models:
#   Qwen/Qwen2.5-VL-7B-Instruct  → sweet spot (24GB GPU)
#   Qwen/Qwen2.5-VL-72B-Instruct → max quality (80GB, 4-bit: 40GB)
#   lmms-lab/llava-onevision-qwen2-7b-ov-hf → good for video
#   OpenGVLab/InternVL3-8B        → best for dense documents
#
# API models:
#   openai/gpt-4o                 → best API quality
#   anthropic/claude-3-5-sonnet-20241022 → best for tables/charts
# -------------------------------------------------
VISION_MODEL_CPU = os.getenv("VISION_MODEL_CPU", "Qwen/Qwen2.5-VL-3B-Instruct")
VISION_MODEL_GPU = os.getenv("VISION_MODEL_GPU", "Qwen/Qwen2.5-VL-7B-Instruct")

# API vision provider (only when AI_PROFILE="api")
VISION_API_PROVIDER = os.getenv("VISION_API_PROVIDER", "openai")    # openai | anthropic | google
VISION_API_MODEL = os.getenv("VISION_API_MODEL", "gpt-4o")

# GPU quantization
VISION_QUANTIZE = os.getenv("VISION_QUANTIZE", "4bit")              # none | 4bit | 8bit
VISION_FLASH_ATTENTION = os.getenv("VISION_FLASH_ATTENTION", "true").lower() == "true"

# Pixel budget (Qwen2.5-VL supports up to 16384 tokens)
VISION_MAX_PIXELS = int(os.getenv("VISION_MAX_PIXELS", str(1280 * 28 * 28)))
VISION_MIN_PIXELS = int(os.getenv("VISION_MIN_PIXELS", str(256 * 28 * 28)))

# Batch processing
VISION_BATCH_SIZE_CPU = int(os.getenv("VISION_BATCH_SIZE_CPU", "1"))
VISION_BATCH_SIZE_GPU = int(os.getenv("VISION_BATCH_SIZE_GPU", "4"))


# -------------------------------------------------
# Image / Legacy configuration
# -------------------------------------------------
IMAGE_CAPTION_MODEL_CPU = "light"
IMAGE_CAPTION_MODEL_GPU = "large"


# -------------------------------------------------
# Video configuration
# -------------------------------------------------
VIDEO_FRAME_SAMPLE_RATE = 1.0   # frames per second
VIDEO_MAX_DURATION_SEC = 1800   # safety cap (30 mins)
VIDEO_VISION_FRAMES = int(os.getenv("VIDEO_VISION_FRAMES", "8"))


# -------------------------------------------------
# LLM behavior flags
# -------------------------------------------------
ENABLE_VISUAL_EXPLANATION = True
ENABLE_AUDIO_LANGUAGE_DETECTION = True


# -------------------------------------------------
# Safety / performance controls
# -------------------------------------------------
MAX_CONCURRENT_MEDIA_TASKS = 2      # keep CPU safe
MEDIA_TIMEOUT_SECONDS = 900         # hard timeout


# -------------------------------------------------
# Debug / observability
# -------------------------------------------------
LOG_AI_PROVIDER_SELECTION = True
