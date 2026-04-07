# =====================================================
# document_visual_interceptor_v1.py
#
# Document Visual Interception Layer
#
# Responsibilities:
# - Detect embedded visuals in documents
# - Extract visuals deterministically
# - Convert visuals to semantic explanations
# - Return text-only explanations to be merged
#
# IMPORTANT:
# - No DB writes
# - No chunking
# - No vector operations
# =====================================================

import os
import re
import tempfile
from typing import List, Dict, Any
import time

from app.utils.logger import log_info, log_warning
from app.services.ingestion.media.image_ingestor_v1 import ImageIngestorV1

# -------------------------------------------------------------------
# Visual quality gate — filter out low-value embedded images
# -------------------------------------------------------------------
# Minimum image dimensions (px) to avoid tiny logos / icons
MIN_IMAGE_WIDTH = 80
MIN_IMAGE_HEIGHT = 80

# OCR garble detection — if too many single-char tokens, it's toolbar junk
_GARBLE_PATTERN = re.compile(r"\b\w\b")  # single-character "words"

def _is_low_value_visual(
    caption: str,
    ocr_text: str,
    image_path: str,
) -> bool:
    """
    Returns True if the visual is likely a logo, decorative icon,
    or garbled screenshot with no real information content.
    """
    # ── Check image dimensions ─────────────────────────────
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            w, h = img.size
            if w < MIN_IMAGE_WIDTH or h < MIN_IMAGE_HEIGHT:
                return True
    except Exception:
        pass

    c = (caption or "").lower()
    t = (ocr_text or "").strip()

    # ── Logo / branding detection ──────────────────────────
    if len(t) < 30 and any(
        k in c for k in ["logo", "brand", "watermark", "icon", "photograph or illustration"]
    ):
        return True

    # ── Garbled OCR from toolbar screenshots ───────────────
    if t:
        single_char_tokens = len(_GARBLE_PATTERN.findall(t))
        all_tokens = len(t.split())
        if all_tokens > 5 and single_char_tokens / max(all_tokens, 1) > 0.40:
            return True

    # ── Near-empty OCR on "photo" type ─────────────────────
    if not t and "photograph" in c:
        return True

    return False


class DocumentVisualInterceptorV1:
    """
    Intercepts document ingestion to extract and explain
    embedded visuals (charts, figures, tables, images).
    """

    def __init__(self):
        self.image_ingestor = ImageIngestorV1()

    # -------------------------------------------------
    # Public entrypoint
    # -------------------------------------------------
    async def intercept(
        self,
        file_path: str,
        parsed_output: Dict[str, Any],
        file_type: str,
        file_id: str,
        business_id: str = None,
    ) -> List[str]:
        """
        Detects and processes document visuals.

        Returns:
            List of semantic explanations (text only)
        """
        visual_explanations: List[str] = []

        if file_type == "pdf":
            visuals = await self._extract_pdf_visuals(file_path)

        elif file_type == "docx":
            visuals = await self._extract_docx_visuals(file_path)

        elif file_type in ("xls", "xlsx"):
            visuals = await self._extract_excel_visuals(file_path)

        elif file_type in ("html", "web"):
            visuals = await self._extract_web_visuals(parsed_output)

        else:
            return []

        for visual_path, context in visuals:
            explanation = await self._process_visual(
                visual_path=visual_path,
                context=context,
                file_id=file_id,
                business_id=business_id,
            )
            if explanation:
                visual_explanations.append(explanation)

        return visual_explanations

    async def intercept_explanations_only(
        self,
        file_path: str,
        parsed_output: Dict[str, Any],
        file_type: str,
        max_visuals: int = 24,
    ) -> List[str]:
        """
        Extract semantic explanations from embedded visuals WITHOUT DB writes.

        This method is safe to call from parsers (pre-ingestion). It does not
        create files in IngestedFile, does not call IngestionServiceV2, and
        returns only text explanations to be merged into parser output.
        """
        if file_type not in {"pdf", "docx", "xls", "xlsx", "html", "web"}:
            return []

        visuals: List[tuple] = []
        if file_type == "pdf":
            visuals = await self._extract_pdf_visuals(file_path)
        elif file_type == "docx":
            visuals = await self._extract_docx_visuals(file_path)
        elif file_type in {"xls", "xlsx"}:
            visuals = await self._extract_excel_visuals(file_path)
        else:
            visuals = await self._extract_web_visuals(parsed_output or {})

        if max_visuals > 0 and len(visuals) > max_visuals:
            log_info(
                f"[DocumentVisualInterceptorV1] Trimming visuals "
                f"{len(visuals)} -> {max_visuals} for bounded parser latency"
            )
            visuals = visuals[:max_visuals]

        explanations: List[str] = []
        for visual_path, context in visuals:
            try:
                explanation = await self._explain_visual_without_ingestion(
                    visual_path=visual_path,
                    context=context or {},
                )
                if explanation:
                    explanations.append(explanation)
            except Exception as e:
                log_warning(
                    f"[DocumentVisualInterceptorV1] explain-only visual failed: {e}"
                )
            finally:
                self._safe_cleanup(visual_path)

        return explanations

    # -------------------------------------------------
    # Visual â†’ explanation
    # -------------------------------------------------
    async def _process_visual(
        self,
        visual_path: str,
        context: Dict[str, Any],
        file_id: str,
        business_id: str,
    ) -> str:
        """
        Sends visual through image ingestion pipeline
        and returns semantic explanation.
        """
        try:
            explanation = await self.image_ingestor.ingest(
                #file_id=f"{file_id}::visual::{os.path.basename(visual_path)}",
                file_id=file_id,
                image_path=visual_path,
                business_id=business_id,
            )
            return explanation
        finally:
            self._safe_cleanup(visual_path)

    async def _explain_visual_without_ingestion(
        self,
        visual_path: str,
        context: Dict[str, Any],
    ) -> str:
        """
        Generate semantic explanation directly from captioner output.
        No DB writes, no ingestion pipeline recursion.
        Applies quality gate to skip low-value visuals (logos, garbled OCR).
        """
        result = await self.image_ingestor.captioner.caption(visual_path)
        if not isinstance(result, dict):
            return ""

        caption = (result.get("caption") or "").strip()
        ocr_text = (result.get("ocr_text") or "").strip()
        if not caption and not ocr_text:
            return ""

        # ── Quality gate: skip low-value visuals ──────────────
        if _is_low_value_visual(caption, ocr_text, visual_path):
            page = context.get("page")
            log_info(
                f"[DocumentVisualInterceptorV1] Skipping low-value visual"
                f"{f' on page {page}' if page else ''}: "
                f"caption='{caption[:60]}' ocr_len={len(ocr_text)}"
            )
            return ""

        numeric_ratio = self.image_ingestor._numeric_ratio(ocr_text)
        visual_type = self.image_ingestor._classify_visual(
            caption=caption,
            ocr_text=ocr_text,
            numeric_ratio=numeric_ratio,
            text_density=len(ocr_text),
        )
        semantic_text = await self.image_ingestor._synthesize_text_with_explainer(
            visual_type=visual_type,
            caption=caption,
            ocr_text=ocr_text,
        )

        page = context.get("page")
        if isinstance(page, int) and page > 0:
            return f"[Embedded visual page {page}] {semantic_text}"
        return semantic_text


    # -------------------------------------------------
    # PDF visuals
    # -------------------------------------------------
    async def _extract_pdf_visuals(self, file_path: str):
        """
        Extract embedded images/charts from PDF.
        """
        import fitz  # PyMuPDF

        visuals = []
        doc = fitz.open(file_path)

        for page_index in range(len(doc)):
            page = doc[page_index]
            images = page.get_images(full=True)

            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                temp_path = self._write_temp_image(image_bytes, image_ext)
                visuals.append(
                    (
                        temp_path,
                        {
                            "source": "pdf",
                            "page": page_index + 1,
                        },
                    )
                )

        return visuals

    # -------------------------------------------------
    # DOCX visuals
    # -------------------------------------------------
    async def _extract_docx_visuals(self, file_path: str):
        """
        Extract images from DOCX.
        """
        from docx import Document

        visuals = []
        doc = Document(file_path)

        for rel in doc.part._rels.values():
            if "image" in rel.reltype:
                image_bytes = rel.target_part.blob
                image_ext = rel.target_ref.split(".")[-1]

                temp_path = self._write_temp_image(image_bytes, image_ext)
                visuals.append(
                    (
                        temp_path,
                        {
                            "source": "docx",
                        },
                    )
                )

        return visuals

    # -------------------------------------------------
    # Excel visuals (charts as images)
    # -------------------------------------------------
    async def _extract_excel_visuals(self, file_path: str):
        """
        Extract embedded images from Excel workbooks using openpyxl.
        Handles .xlsx files; .xls files are skipped (binary format).
        """
        visuals = []

        # Only .xlsx is supported (openpyxl cannot read .xls binary)
        if not file_path.lower().endswith(".xlsx"):
            return visuals

        try:
            from openpyxl import load_workbook
        except ImportError:
            log_warning("[DocumentVisualInterceptorV1] openpyxl not available for Excel image extraction")
            return visuals

        try:
            wb = load_workbook(file_path, data_only=True)
        except Exception as e:
            log_warning(f"[DocumentVisualInterceptorV1] Failed to load Excel for images: {e}")
            return visuals

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            for image in getattr(ws, "_images", []):
                try:
                    image_bytes = image._data()
                    if not image_bytes or len(image_bytes) < 1024:
                        continue  # Skip tiny/empty images

                    # Determine extension from content type or default to png
                    ext = "png"
                    content_type = getattr(image, "content_type", "") or ""
                    if "jpeg" in content_type or "jpg" in content_type:
                        ext = "jpg"
                    elif "gif" in content_type:
                        ext = "gif"

                    temp_path = self._write_temp_image(image_bytes, ext)
                    visuals.append(
                        (
                            temp_path,
                            {
                                "source": "excel",
                                "sheet": sheet_name,
                            },
                        )
                    )
                except Exception as e:
                    log_warning(
                        f"[DocumentVisualInterceptorV1] Failed to extract Excel image "
                        f"from sheet '{sheet_name}': {e}"
                    )

        wb.close()
        return visuals

    # -------------------------------------------------
    # Web / HTML visuals
    # -------------------------------------------------
    async def _extract_web_visuals(self, parsed_output: Dict[str, Any]):
        """
        Extract images from scraped web content.
        """
        visuals = []

        images = parsed_output.get("images", [])
        for img in images:
            if img.get("bytes"):
                temp_path = self._write_temp_image(
                    img["bytes"], img.get("ext", "png")
                )
                visuals.append(
                    (
                        temp_path,
                        {
                            "source": "web",
                        },
                    )
                )

        return visuals

    # -------------------------------------------------
    # Temp image writer
    # -------------------------------------------------
    def _write_temp_image(self, image_bytes: bytes, ext: str) -> str:
        fd, path = tempfile.mkstemp(suffix=f".{ext}")
        with os.fdopen(fd, "wb") as f:
            f.write(image_bytes)
        return path
        
        
    def _safe_cleanup(self, path: str, retries: int = 5, delay: float = 0.2):
        """
        Windows-safe temp file cleanup.
        Retries deletion to avoid WinError 32 file locks.
        """
        # Everything inside here must be indented 8 spaces
        for _ in range(retries):
            try:
                if os.path.exists(path):
                    os.remove(path)
                return
            except PermissionError:
                time.sleep(delay)
            except Exception:
                return
