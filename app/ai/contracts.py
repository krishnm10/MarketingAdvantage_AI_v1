# =============================================
# contracts.py
#
# AI Capability Interfaces
#
# This file defines WHAT the system needs,
# not HOW it is implemented.
#
# Rules:
# - No model imports
# - No config imports
# - No side effects
# - Async-first interfaces
# =============================================

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


# -------------------------------------------------
# Audio → Text
# -------------------------------------------------
class SpeechToText(ABC):
    """
    Converts audio into raw textual transcript.
    """

    @abstractmethod
    async def transcribe(self, audio_path: str) -> str:
        """
        Args:
            audio_path: Absolute or relative path to audio file

        Returns:
            Plain text transcript (no timestamps, no formatting)
        """
        raise NotImplementedError


# -------------------------------------------------
# Image → Caption / Text
# -------------------------------------------------
class ImageCaptioner(ABC):
    """
    Converts images into descriptive text.
    Used for photos, charts, diagrams.
    """

    @abstractmethod
    async def caption(self, image_path: str) -> Dict[str, Any]:
        """
        Args:
            image_path: Path to image file

        Returns:
            {
              "caption": str,
              "ocr_text": Optional[str],
              "objects": Optional[List[str]],
              "is_chart": Optional[bool]
            }
        """
        raise NotImplementedError


# -------------------------------------------------
# Video → Text
# -------------------------------------------------
class VideoToText(ABC):
    """
    Converts video into semantic text using
    audio + frame sampling.
    """

    @abstractmethod
    async def extract(self, video_path: str) -> Dict[str, Any]:
        """
        Args:
            video_path: Path to video file

        Returns:
            {
              "transcript": Optional[str],
              "frame_captions": List[str],
              "duration_sec": Optional[float]
            }
        """
        raise NotImplementedError


# -------------------------------------------------
# Visual / Chart Explanation
# -------------------------------------------------
class VisualExplainer(ABC):
    """
    Converts numeric / chart-like text into
    semantic explanation.
    """

    @abstractmethod
    async def explain(self, text: str) -> str:
        """
        Args:
            text: Raw chart / table / visual text

        Returns:
            Human-readable semantic explanation
        """
        raise NotImplementedError


# -------------------------------------------------
# Optional: Language Detection
# -------------------------------------------------
class LanguageDetector(ABC):
    """
    Detects language of text/audio.
    """

    @abstractmethod
    async def detect(self, text: str) -> str:
        """
        Returns:
            ISO language code (e.g. 'en', 'hi', 'fr')
        """
        raise NotImplementedError


# -------------------------------------------------
# Multimodal Vision Encoder (enterprise-grade)
#
# Neural multimodal vision encoder contract.
# Replaces OCR-only ImageCaptioner for all new paths.
# Supports images, video frames, charts, tables, docs.
#
# Implementations:
#   CPU : Qwen2.5-VL-2B, moondream2
#   GPU : Qwen2.5-VL-7B/72B, LLaVA-OneVision, InternVL3
#   API : GPT-4o Vision, Claude Vision, Gemini Vision
# -------------------------------------------------
class MultimodalVisionEncoder(ABC):
    """
    Neural multimodal vision encoder contract.

    Replaces the OCR-only ImageCaptioner for all new paths.
    Supports images, video frames, charts, tables, documents.
    """

    @abstractmethod
    async def encode(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        mode: str = "caption",
    ) -> Dict[str, Any]:
        """
        Encode an image into semantic text and/or a vector.

        Args:
            image_path: Absolute path to image file.
            prompt:     Optional task-specific instruction.
                        If None, uses the default for the given mode.
            mode:       Task hint for the encoder:
                        - "caption"   → rich natural-language description
                        - "ocr"       → extracted text characters
                        - "chart"     → data insight from charts/graphs
                        - "embed"     → returns visual embedding vector
                        - "vqa"       → visual question answering
                        - "document"  → structured document understanding

        Returns:
            {
              "caption":     str,
              "ocr_text":    Optional[str],
              "chart_data":  Optional[str],
              "embedding":   Optional[List[float]],
              "is_chart":    bool,
              "is_document": bool,
              "confidence":  float,          # 0.0 – 1.0
              "model_used":  str,
              "tokens_used": Optional[int],
            }
        """
        raise NotImplementedError

    @abstractmethod
    async def encode_batch(
        self,
        image_paths: List[str],
        prompt: Optional[str] = None,
        mode: str = "caption",
    ) -> List[Dict[str, Any]]:
        """
        Batch encode multiple images. Prefer over loop of encode()
        for throughput — GPU providers can parallelise.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the canonical model identifier string."""
        raise NotImplementedError

    @property
    @abstractmethod
    def supports_video_frames(self) -> bool:
        """True if the encoder can accept video frame sequences."""
        ...

    @property
    @abstractmethod
    def supports_batch(self) -> bool:
        """True if encode_batch() is more efficient than N encode() calls."""
        ...

    @property
    @abstractmethod
    def max_image_size_mb(self) -> float:
        """Maximum input image size this encoder handles."""
        ...
