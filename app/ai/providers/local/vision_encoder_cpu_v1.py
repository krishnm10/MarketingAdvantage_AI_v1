# =============================================
# vision_encoder_cpu_v1.py
#
# CPU-safe Multimodal Vision Encoder
#
# Strategy:
#   PRIMARY   → Qwen2.5-VL-3B-Instruct (transformers)
#               Best open-source quality on CPU.
#               Handles captions, OCR, charts, VQA.
#
#   FALLBACK  → moondream2 (via moondream library)
#               Ultra-fast 1.8B, minimal RAM.
#               Activates if Qwen fails to load.
#
# Characteristics:
#   - Lazy model loading (zero startup cost)
#   - Thread-executor offloaded (event-loop safe)
#   - Automatic fallback chain
#   - Batch support (sequential on CPU)
# =============================================

import asyncio
import os
import time
import logging
from typing import Dict, Any, List, Optional

from app.ai.contracts import MultimodalVisionEncoder
from app.utils.logger import log_info, log_warning

logger = logging.getLogger(__name__)


# -------------------------------------------------
# Default prompts per mode
# -------------------------------------------------
_DEFAULT_PROMPTS = {
    "caption": (
        "Describe this image in detail. Include all text visible, "
        "the main subject, colors, layout, and any data or charts shown."
    ),
    "ocr": (
        "Extract ALL text from this image exactly as it appears. "
        "Preserve layout, numbers, and punctuation."
    ),
    "chart": (
        "This image contains a chart or graph. "
        "Describe the data trend, axes, values, title, and key insights. "
        "Format your response as: TITLE: ... TREND: ... KEY_VALUES: ... INSIGHT: ..."
    ),
    "document": (
        "This is a document image. Extract the full text content, "
        "preserving headings, paragraphs, tables, and structure."
    ),
    "vqa": "Answer the question about this image concisely and accurately.",
    "embed": "Describe this image for semantic search indexing.",
}


class QwenVLCPUEncoder(MultimodalVisionEncoder):
    """
    Qwen2.5-VL on CPU via HuggingFace transformers.
    Lazy-loaded. First call takes ~30-60s; subsequent calls are fast.
    Uses torch.float32 on CPU (no BF16 on most CPUs).
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        self._model_name = model_name
        self._model = None
        self._processor = None
        self._load_lock = asyncio.Lock()

    # --------------------------------------------------
    # Contract properties
    # --------------------------------------------------
    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def supports_video_frames(self) -> bool:
        return True

    @property
    def supports_batch(self) -> bool:
        return False  # Sequential on CPU

    @property
    def max_image_size_mb(self) -> float:
        return 20.0

    # --------------------------------------------------
    # Lazy loader
    # --------------------------------------------------
    def _load_model_sync(self):
        """Load Qwen2.5-VL on CPU. Blocks — call via executor."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError as e:
            raise ImportError(
                f"Qwen2.5-VL CPU requires: pip install transformers qwen-vl-utils torch\n"
                f"Error: {e}"
            )

        log_info(f"[QwenVLCPU] Loading {self._model_name} on CPU...")
        t0 = time.monotonic()

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self._model_name,
            dtype=torch.float32,
            device_map="cpu",
        )
        self._model.eval()

        from app.config.ai_config import VISION_MIN_PIXELS
        self._processor = AutoProcessor.from_pretrained(
            self._model_name,
            min_pixels=VISION_MIN_PIXELS,
            max_pixels=512 * 28 * 28,   # smaller for CPU speed
        )

        elapsed = time.monotonic() - t0
        log_info(f"[QwenVLCPU] Loaded in {elapsed:.1f}s")

    # --------------------------------------------------
    # Core inference
    # --------------------------------------------------
    def _infer_sync(self, image_path: str, prompt: str) -> str:
        """Run inference synchronously. Called via executor."""
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    # --------------------------------------------------
    # Contract implementation
    # --------------------------------------------------
    async def encode(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        mode: str = "caption",
    ) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()

        async with self._load_lock:
            if self._model is None:
                await loop.run_in_executor(None, self._load_model_sync)

        effective_prompt = prompt or _DEFAULT_PROMPTS.get(mode, _DEFAULT_PROMPTS["caption"])

        try:
            result_text = await loop.run_in_executor(
                None,
                lambda: self._infer_sync(image_path, effective_prompt),
            )
        except Exception as e:
            log_warning(f"[QwenVLCPU] Inference failed: {e}")
            raise

        is_chart = any(k in result_text.lower() for k in [
            "chart", "graph", "axis", "trend", "percent", "increase", "decrease"
        ])
        is_document = mode == "document" or any(k in result_text.lower() for k in [
            "paragraph", "section", "heading", "table", "column"
        ])

        return {
            "caption":     result_text if mode != "ocr" else "",
            "ocr_text":    result_text if mode == "ocr" else self._extract_ocr(result_text),
            "chart_data":  result_text if mode == "chart" else (result_text if is_chart else None),
            "embedding":   None,
            "is_chart":    is_chart,
            "is_document": is_document,
            "confidence":  0.85,
            "model_used":  self._model_name,
            "tokens_used": None,
        }

    async def encode_batch(
        self,
        image_paths: List[str],
        prompt: Optional[str] = None,
        mode: str = "caption",
    ) -> List[Dict[str, Any]]:
        results = []
        for path in image_paths:
            result = await self.encode(path, prompt=prompt, mode=mode)
            results.append(result)
        return results

    def _extract_ocr(self, text: str) -> Optional[str]:
        """Try to extract OCR-like text from a caption response."""
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        quoted = [ln for ln in lines if '"' in ln or "'" in ln]
        return "\n".join(quoted) if quoted else None


class Moondream2CPUEncoder(MultimodalVisionEncoder):
    """
    moondream2 — ultra-lightweight 1.8B vision model.
    Fallback when Qwen2.5-VL is unavailable.
    pip install moondream
    """

    def __init__(self, model_name: str = "vikhyatk/moondream2"):
        self._model_name = model_name
        self._model = None
        self._load_lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def supports_video_frames(self) -> bool:
        return False

    @property
    def supports_batch(self) -> bool:
        return True

    @property
    def max_image_size_mb(self) -> float:
        return 10.0

    def _load_model_sync(self):
        if self._model is not None:
            return
        try:
            import moondream as md
        except ImportError:
            raise ImportError("pip install moondream")

        log_info(f"[Moondream2CPU] Loading {self._model_name}...")
        self._model = md.vl(model=self._model_name)
        log_info("[Moondream2CPU] Loaded")

    def _infer_sync(self, image_path: str, prompt: str) -> str:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        encoded = self._model.encode_image(img)
        return self._model.query(encoded, prompt)["answer"]

    async def encode(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        mode: str = "caption",
    ) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        async with self._load_lock:
            if self._model is None:
                await loop.run_in_executor(None, self._load_model_sync)

        effective_prompt = prompt or _DEFAULT_PROMPTS.get(mode, _DEFAULT_PROMPTS["caption"])

        result_text = await loop.run_in_executor(
            None,
            lambda: self._infer_sync(image_path, effective_prompt),
        )

        is_chart = any(k in result_text.lower() for k in ["chart", "graph", "trend"])
        return {
            "caption":     result_text,
            "ocr_text":    None,
            "chart_data":  result_text if is_chart else None,
            "embedding":   None,
            "is_chart":    is_chart,
            "is_document": False,
            "confidence":  0.75,
            "model_used":  self._model_name,
            "tokens_used": None,
        }

    async def encode_batch(
        self,
        image_paths: List[str],
        prompt: Optional[str] = None,
        mode: str = "caption",
    ) -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        async with self._load_lock:
            if self._model is None:
                await loop.run_in_executor(None, self._load_model_sync)

        effective_prompt = prompt or _DEFAULT_PROMPTS.get(mode, _DEFAULT_PROMPTS["caption"])

        def _batch_infer():
            from PIL import Image
            results = []
            for path in image_paths:
                img = Image.open(path).convert("RGB")
                encoded = self._model.encode_image(img)
                ans = self._model.query(encoded, effective_prompt)["answer"]
                is_chart = any(k in ans.lower() for k in ["chart", "graph", "trend"])
                results.append({
                    "caption": ans, "ocr_text": None,
                    "chart_data": ans if is_chart else None,
                    "embedding": None, "is_chart": is_chart, "is_document": False,
                    "confidence": 0.75, "model_used": self._model_name, "tokens_used": None,
                })
            return results

        return await loop.run_in_executor(None, _batch_infer)


class VisionEncoderCPUV1(MultimodalVisionEncoder):
    """
    Enterprise CPU vision encoder with automatic fallback chain.

    Priority:
        1. QwenVLCPUEncoder      (Qwen2.5-VL-2B)  — best quality
        2. Moondream2CPUEncoder                     — fastest fallback

    Auto-selects based on available packages.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        self._primary: Optional[MultimodalVisionEncoder] = None
        self._fallback: Optional[MultimodalVisionEncoder] = None
        self._model_name_hint = model_name
        self._active: Optional[MultimodalVisionEncoder] = None
        self._init_lock = asyncio.Lock()

    async def _ensure_encoder(self):
        async with self._init_lock:
            if self._active is not None:
                return

            # If model_name_hint is a moondream model, route directly to moondream
            if "moondream" in self._model_name_hint.lower():
                try:
                    import moondream  # noqa
                    self._fallback = Moondream2CPUEncoder()
                    self._active = self._fallback
                    log_info("[VisionEncoderCPU] Using moondream2 (configured)")
                    return
                except ImportError:
                    pass
                raise RuntimeError(
                    "VISION_MODEL_CPU is set to a moondream model but moondream is not installed.\n"
                    "  pip install moondream"
                )

            # Try primary: Qwen2.5-VL
            try:
                import transformers  # noqa
                import qwen_vl_utils  # noqa
                self._primary = QwenVLCPUEncoder(self._model_name_hint)
                self._active = self._primary
                log_info(f"[VisionEncoderCPU] Using {self._model_name_hint} (primary)")
                return
            except ImportError:
                log_warning("[VisionEncoderCPU] Qwen not available, trying moondream2...")

            # Try fallback: moondream2
            try:
                import moondream  # noqa
                self._fallback = Moondream2CPUEncoder()
                self._active = self._fallback
                log_warning("[VisionEncoderCPU] Using moondream2 (fallback)")
                return
            except ImportError:
                pass

            raise RuntimeError(
                "No CPU vision encoder available.\n"
                "Install one:\n"
                "  pip install transformers qwen-vl-utils torch   # Qwen2.5-VL (best)\n"
                "  pip install moondream                          # moondream2 (fast)"
            )

    @property
    def model_name(self) -> str:
        return self._active.model_name if self._active else self._model_name_hint

    @property
    def supports_video_frames(self) -> bool:
        return self._active.supports_video_frames if self._active else False

    @property
    def supports_batch(self) -> bool:
        return self._active.supports_batch if self._active else False

    @property
    def max_image_size_mb(self) -> float:
        return self._active.max_image_size_mb if self._active else 10.0

    async def encode(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        mode: str = "caption",
    ) -> Dict[str, Any]:
        await self._ensure_encoder()
        return await self._active.encode(image_path, prompt=prompt, mode=mode)

    async def encode_batch(
        self,
        image_paths: List[str],
        prompt: Optional[str] = None,
        mode: str = "caption",
    ) -> List[Dict[str, Any]]:
        await self._ensure_encoder()
        return await self._active.encode_batch(image_paths, prompt=prompt, mode=mode)
