# =============================================
# image_caption_cpu_v1.py
#
# CPU-only Image Caption + OCR provider
#
# Characteristics:
# - CPU safe
# - No GPU required
# - Lazy loading
# - Implements ImageCaptioner contract
# =============================================

import asyncio
from typing import Dict, Any

from app.ai.contracts import ImageCaptioner
from app.utils.logger import log_info


class ImageCaptionerCPU(ImageCaptioner):
    """
    CPU-only image captioner using lightweight OCR + heuristics.
    """

    def __init__(self):
        self._ocr_ready = False

    # -------------------------------------------------
    # Lazy OCR init
    # -------------------------------------------------
    def _init_ocr(self):
        if self._ocr_ready:
            return

        try:
            import pytesseract  # noqa
            from PIL import Image  # noqa

            # Optional (Windows explicit path – uncomment ONLY if needed)
            # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

            self._ocr_ready = True
            log_info("[ImageCaptionerCPU] OCR available")

        except Exception:
            # OCR is optional – do NOT fail ingestion
            self._ocr_ready = False
            log_info(
                "[ImageCaptionerCPU] OCR unavailable, running caption-only mode"
            )

    # -------------------------------------------------
    # Caption image
    # -------------------------------------------------
    async def caption(self, image_path: str) -> Dict[str, Any]:
        """
        Extract OCR text and detect chart-like visuals.
        """
        self._init_ocr()

        loop = asyncio.get_running_loop()

        def _run():
            from PIL import Image
            import pytesseract

            with Image.open(image_path) as img:
                ocr_text = ""
                if self._ocr_ready:
                    try:
                        ocr_text = pytesseract.image_to_string(img)
                    except Exception:
                        ocr_text = ""

                # VERY IMPORTANT:
                # Chart detection is intentionally simple here.
                # LLM explainer will do the real semantic work.
                is_chart = any(
                    token in ocr_text.lower()
                    for token in ["%", "year", "202", "axis", "total"]
                )

                caption = (
                    "This image appears to be a chart or data figure."
                    if is_chart
                    else "This image appears to be a photograph or illustration."
                )

                return {
                    "caption": caption,
                    "ocr_text": ocr_text.strip(),
                    "objects": None,
                    "is_chart": is_chart,
                }

        return await loop.run_in_executor(None, _run)
