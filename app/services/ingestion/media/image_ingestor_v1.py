# =============================================
# image_ingestor_v1.py
#
# Universal Image → Semantic Text Ingestor
#
# Guarantees:
# - Supports ALL image types
# - Deterministic, explainable logic
# - No vector writes
# - No router dependency
# - IngestionServiceV2 is the ONLY pipeline entry
# =============================================

import os
import re
from datetime import datetime
from typing import Optional
# Add these new imports after the existing imports
from app.services.ingestion.media.media_hash_utils import MediaHashComputer
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from PIL import Image
from app.utils.logger import log_info
from app.db.models.ingested_file_v2 import IngestedFileV2
from app.db.session_v2 import get_async_session
from app.services.ingestion.ingestion_service_v2 import IngestionServiceV2
from app.ai.registry import get_image_captioner, get_visual_explainer, get_vision_encoder
from app.config.ai_config import ENABLE_VISUAL_EXPLANATION


# -------------------------------------------------
# Visual type enum (LOCKED)
# -------------------------------------------------
VISUAL_TYPES = {
    "photo",
    "chart",
    "table",
    "diagram",
    "screenshot",
    "infographic",
    "mixed",
    "unknown",
}


class ImageIngestorV1:
    """
    Universal image ingestion adapter.

    Converts ANY image (photo, chart, table, diagram, screenshot, mixed)
    into semantic text and feeds IngestionServiceV2.

    Uses MultimodalVisionEncoder (Qwen2.5-VL / moondream2 / GPT-4o / Claude)
    when available. Falls back to legacy OCR captioner if no neural encoder
    packages are installed.
    """

    def __init__(self):
        # Try neural vision encoder first
        self._use_neural = False
        self._vision_encoder = None
        try:
            self._vision_encoder = get_vision_encoder()
            self._use_neural = True
            log_info("[ImageIngestorV1] Using neural vision encoder")
        except Exception as e:
            log_info(f"[ImageIngestorV1] Vision encoder unavailable ({e}), using legacy OCR")
        self.captioner = get_image_captioner()

    # -------------------------------------------------
    # Public entrypoint
    # -------------------------------------------------
    async def ingest(
    self,
    file_id: str,
    image_path: str,
    business_id: Optional[str] = None,
    ):
         """
         Ingest an image file or embedded visual with enterprise-grade deduplication.
         file_id:
             - standalone image → image file id
             - embedded visual → parent document file id
         """
         log_info(f"[ImageIngestorV1] 🖼️ Processing image: {image_path}")
         
         # ============================================
         # STEP 0: COMPUTE PERCEPTUAL HASH & CHECK FOR DUPLICATES
         # ============================================
         perceptual_hash, byte_hash = MediaHashComputer.compute_image_hash(image_path)
         
         async with get_async_session() as db:
             # Check if this image already exists (by perceptual hash)
             existing_query = select(IngestedFileV2).where(
                 IngestedFileV2.media_hash == perceptual_hash
             )
             result = await db.execute(existing_query)
             existing_file = result.scalar_one_or_none()
             
             if existing_file:
                 log_info(
                     f"[ImageIngestorV1] ⚠️ DUPLICATE DETECTED → "
                     f"{os.path.basename(image_path)} matches existing file: {existing_file.file_name} "
                     f"(ID: {existing_file.id}, Hash: {perceptual_hash[:12]}...)"
                 )
                 
                 # Return duplicate info (don't process further)
                 return {
                     "status": "duplicate_skipped",
                     "duplicate_of": str(existing_file.id),
                     "original_file": existing_file.file_name,
                     "perceptual_hash": perceptual_hash[:16],
                     "message": f"Image is duplicate of existing file: {existing_file.file_name}"
                 }
             
             # ============================================
             # No duplicate found - proceed with ingestion
             # ============================================
             log_info(f"[ImageIngestorV1] ✅ Unique image confirmed, proceeding with ingestion")
             
             # ----------------------------------------------
             # 1. Ensure IngestedFile record exists
             # ----------------------------------------------
             file_record = await IngestionServiceV2._get_file_record(db, file_id)
             
             if not file_record:
                 file_record = IngestedFileV2(
                     id=file_id,
                     file_name=os.path.basename(image_path),
                     file_type="image",
                     file_path=image_path,
                     business_id=business_id,
                     media_hash=perceptual_hash,  # ⬅️ STORE PERCEPTUAL HASH
                     meta_data={
                         "source_type": "image",
                         "ingested_via": "image_ingestor_v1",
                         "perceptual_hash": perceptual_hash,
                         "byte_hash": byte_hash,
                         "dedup_method": "dhash_256bit",
                     },
                     status="uploaded",
                     created_at=datetime.utcnow(),
                     updated_at=datetime.utcnow(),
                 )
                 db.add(file_record)
                 try:
                     await db.commit()
                 except IntegrityError:
                     await db.rollback()
                     # Another session committed the same media_hash first
                     log_info(
                         f"[ImageIngestorV1] ⚠️ DUPLICATE DETECTED (constraint) → "
                         f"{os.path.basename(image_path)} hash={perceptual_hash[:12]}..."
                     )
                     return {
                         "status": "duplicate_skipped",
                         "perceptual_hash": perceptual_hash[:16],
                         "message": "Image is a duplicate (detected at insert)",
                     }
             else:
                 # Update existing record with media_hash if missing
                 if not file_record.media_hash:
                     file_record.media_hash = perceptual_hash
                     if not file_record.meta_data:
                         file_record.meta_data = {}
                     file_record.meta_data.update({
                         "perceptual_hash": perceptual_hash,
                         "byte_hash": byte_hash,
                         "dedup_method": "dhash_256bit",
                     })
                     await db.commit()
                 
                 log_info(
                     f"[ImageIngestorV1] Reusing existing file record: {file_id}"
                 )
         
             # ----------------------------------------------
             # 2. Load image safely (format-agnostic)
             # ----------------------------------------------
             try:
                 # Verify integrity without leaving the file handle open.
                 with Image.open(image_path) as image:
                     image.verify()
             except Exception as e:
                 log_info(f"[ImageIngestorV1] ❌ Invalid image: {e}")
                 return {
                     "status": "failed",
                     "error": f"Invalid image file: {e}"
                 }
         
             # ----------------------------------------------
             # 3. Caption + OCR (neural encoder or legacy)
             # ----------------------------------------------
             result = await self._encode_image(image_path)
             if not result or not isinstance(result, dict):
                 log_info("[ImageIngestorV1] ❌ Encoder returned no result")
                 return {
                     "status": "failed",
                     "error": "Caption generation failed"
                 }
         
             caption = (result.get("caption") or "").strip()
             ocr_text = (result.get("ocr_text") or "").strip()
         
             if not caption and not ocr_text:
                 log_info("[ImageIngestorV1] ❌ No semantic signal found")
                 return {
                     "status": "failed",
                     "error": "No semantic content extracted"
                 }
         
             # ----------------------------------------------
             # 4. Feature computation (deterministic)
             # ----------------------------------------------
             numeric_ratio = self._numeric_ratio(ocr_text)
             text_density = len(ocr_text)
             visual_type = self._classify_visual(
                 caption=caption,
                 ocr_text=ocr_text,
                 numeric_ratio=numeric_ratio,
                 text_density=text_density,
             )
         
             # ----------------------------------------------
             # 5. Semantic text synthesis (with visual explainer for charts)
             # ----------------------------------------------
             semantic_text = await self._synthesize_text_with_explainer(
                 visual_type=visual_type,
                 caption=caption,
                 ocr_text=ocr_text,
             )
         
             if not semantic_text.strip():
                 log_info("[ImageIngestorV1] ❌ Empty semantic output")
                 return {
                     "status": "failed",
                     "error": "Empty semantic output"
                 }
         
             # ----------------------------------------------
             # 6. Parsed payload (core-compatible)
             # ----------------------------------------------
             # Determine model name for metadata
             _model_name = (
                 self._vision_encoder.model_name
                 if self._use_neural and self._vision_encoder
                 else self.captioner.__class__.__name__
             )
             parsed_payload = {
                 "raw_text": semantic_text,
                 "meta": {
                     "media_type": "image",
                     "visual_type": visual_type,
                     "has_ocr": bool(ocr_text),
                     "caption_model": _model_name,
                     "confidence": result.get("confidence", 0.6),
                     "is_neural_encoder": self._use_neural,
                     "confidence_source": "model",
                     "perceptual_hash": perceptual_hash,
                     "byte_hash": byte_hash,
                 },
             }
         
             # ----------------------------------------------
             # 7. Handoff to ingestion core (SEALED)
             # ----------------------------------------------
             await IngestionServiceV2._run_pipeline(
                 db=db,
                 file_record=file_record,
                 parsed_payload=parsed_payload,
             )
         
             log_info(f"[ImageIngestorV1] ✅ Completed image ingestion: {file_id}")
             
             return {
                 "status": "success",
                 "file_id": file_id,
                 "perceptual_hash": perceptual_hash[:16],
                 "visual_type": visual_type,
                 "message": "Image ingested successfully"
             }


    # -------------------------------------------------
    # Neural encoder / legacy captioner router
    # -------------------------------------------------
    async def _encode_image(self, image_path: str) -> dict:
        """
        Route to neural vision encoder or legacy OCR captioner.
        Neural encoder returns richer output; legacy is OCR-only.
        """
        if self._use_neural and self._vision_encoder:
            # Detect mode from content heuristic
            mode = "caption"
            try:
                with Image.open(image_path) as img:
                    w, h = img.size
                    if w < 800 and h < 600:
                        mode = "ocr"
            except Exception:
                pass

            try:
                result = await self._vision_encoder.encode(image_path, mode=mode)

                # If looks like a chart, re-encode with chart prompt for richer data
                if result.get("is_chart") and mode != "chart":
                    chart_result = await self._vision_encoder.encode(image_path, mode="chart")
                    result["chart_data"] = chart_result.get("caption", "")

                return result
            except (RuntimeError, ImportError) as e:
                # Neural encoder packages not installed — fall back to legacy
                log_info(f"[ImageIngestorV1] Neural encoder unavailable at runtime ({e}), falling back to legacy OCR")
                self._use_neural = False

        # Legacy OCR path
        raw = await self.captioner.caption(image_path)
        return {
            "caption":     raw.get("caption", ""),
            "ocr_text":    raw.get("ocr_text"),
            "chart_data":  raw.get("caption") if raw.get("is_chart") else None,
            "embedding":   None,
            "is_chart":    raw.get("is_chart", False),
            "is_document": False,
            "confidence":  0.6,
            "model_used":  "legacy_ocr",
            "tokens_used": None,
        }

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def _numeric_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        digits = sum(c.isdigit() for c in text)
        return digits / max(len(text), 1)

    def _classify_visual(
        self,
        caption: str,
        ocr_text: str,
        numeric_ratio: float,
        text_density: int,
    ) -> str:
        """
        Deterministic visual intent classification.
        No ML. No LLM. Fully auditable.
        """

        c = caption.lower()
        t = ocr_text.lower()

        if any(k in c for k in ["dashboard", "screenshot", "ui", "interface"]):
            return "screenshot"

        if numeric_ratio > 0.30 and any(
            k in t for k in ["%", "year", "total", "axis", "revenue", "growth"]
        ):
            return "chart"

        if "\n" in ocr_text and re.search(r"\b(row|column)\b", t):
            return "table"

        if any(k in c for k in ["diagram", "flow", "architecture", "process"]):
            return "diagram"

        if any(k in c for k in ["infographic", "visual summary"]):
            return "infographic"

        if numeric_ratio < 0.05 and text_density < 40:
            return "photo"

        if numeric_ratio > 0.10 and text_density > 50:
            return "mixed"

        return "unknown"

    def _synthesize_text(
        self,
        visual_type: str,
        caption: str,
        ocr_text: str,
    ) -> str:
        # Start with caption (already descriptive)
        text = caption

        if ocr_text:
            clean_ocr = self._clean_ocr_for_output(ocr_text)
            if clean_ocr:
                text += "\n\n" + clean_ocr

        return text.strip()

    async def _synthesize_text_with_explainer(
        self,
        visual_type: str,
        caption: str,
        ocr_text: str,
    ) -> str:
        """
        Enhanced synthesis: uses VisualExplainerCPU for chart/table/infographic types
        to produce richer semantic interpretation of OCR data.
        Falls back to basic _synthesize_text if explainer is unavailable.
        """
        # Only use explainer for data-rich visual types
        explainable_types = {"chart", "table", "infographic", "mixed"}

        if visual_type in explainable_types and ocr_text and ENABLE_VISUAL_EXPLANATION:
            try:
                explainer = get_visual_explainer()
                explanation = await explainer.explain(ocr_text)
                if explanation and explanation.strip():
                    # Build structured output
                    text = caption

                    text += "\n\n--- Semantic Analysis ---\n" + explanation

                    # Clean raw OCR before including it
                    clean_ocr = self._clean_ocr_for_output(ocr_text)
                    if clean_ocr:
                        text += "\n\n--- Raw OCR ---\n" + clean_ocr

                    log_info(f"[ImageIngestorV1] ✅ Visual explainer produced {len(explanation)} chars for {visual_type}")
                    return text.strip()
            except Exception as e:
                log_info(f"[ImageIngestorV1] Visual explainer failed (non-fatal): {e}")

        # Fallback to basic synthesis
        return self._synthesize_text(visual_type, caption, ocr_text)

    @staticmethod
    def _clean_ocr_for_output(ocr_text: str) -> str:
        """
        Clean raw OCR text before including in final output.
        Strips: metadata lines, prompt fragments, color region tags,
        extracted data blocks, copyright lines, excessive garble.
        """
        clean_lines = []
        for line in ocr_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # Strip metadata lines
            if stripped.startswith("[Extracted data values:"):
                continue
            if stripped.startswith("[") and stripped.endswith("]") and "region" not in stripped.lower():
                continue
            # Strip color region metadata prefixes but keep value
            if re.match(r"^\[\w+\s+region\]", stripped, re.IGNORECASE):
                # Extract just the value after the tag
                value = re.sub(r"^\[\w+\s+region\]\s*", "", stripped).strip()
                if value and len(value) > 2:
                    clean_lines.append(value)
                continue
            # Strip prompt fragments
            low = stripped.lower()
            if low.startswith("explanation:") and len(stripped) < 15:
                continue
            if "the following content is extracted" in low:
                continue
            # Strip copyright lines
            if re.search(r"[©®™]", stripped):
                continue
            # Strip pipe/underscore leader lines
            if re.match(r"^[\|_\-—–]{2,}", stripped):
                continue
            # Strip garbled OCR noise: lines with high non-ASCII or
            # non-dictionary-word density that look like misread graphics
            if _is_garbled_line(stripped):
                continue
            clean_lines.append(stripped)
        return "\n".join(clean_lines)


def _is_garbled_line(line: str) -> bool:
    """
    Detect OCR garble — misread graphic/stylized text that produces
    nonsense like 'vIY Giephi fel Herne shal Gnat DIV'.

    Core signal: a word with internal lowercase→uppercase transition
    (extremely rare in real English) combined with multiple short
    non-stopword fragments.
    """
    raw_words = line.split()
    if len(raw_words) < 4:
        return False

    # Strip punctuation for cleaner analysis
    clean_words = [re.sub(r"[^a-zA-Z]", "", w) for w in raw_words]
    alpha_words = [w for w in clean_words if len(w) >= 2]

    if len(alpha_words) < 3:
        return False

    # Common English stop words — don't count these as "suspicious short words"
    _STOP = {
        "the", "a", "an", "and", "or", "in", "on", "at", "to", "of",
        "is", "it", "by", "as", "no", "so", "if", "do", "up", "we",
        "he", "be", "my", "its", "was", "are", "for", "not", "but",
        "has", "had", "can", "all", "her", "his", "our", "you",
    }

    # Signal 1: words with internal lowercase→uppercase transition
    # (e.g. 'vIY' — almost never occurs in real English text)
    mid_cap_count = 0
    for w in alpha_words:
        for i in range(1, len(w)):
            if w[i - 1].islower() and w[i].isupper():
                mid_cap_count += 1
                break

    # Signal 2: short (≤3 char) alpha words that aren't stop words
    non_stop_short = sum(
        1 for w in alpha_words
        if len(w) <= 3 and w.lower() not in _STOP
    )

    # Garble = has mid-cap weirdness AND multiple non-stop short fragments
    if mid_cap_count >= 1 and non_stop_short >= 2:
        return True

    # Also flag lines with very low vowel ratio (consonant soup)
    all_alpha = "".join(alpha_words).lower()
    if len(all_alpha) >= 12:
        vowel_ratio = sum(1 for c in all_alpha if c in "aeiou") / len(all_alpha)
        if vowel_ratio < 0.15:
            return True

    return False
