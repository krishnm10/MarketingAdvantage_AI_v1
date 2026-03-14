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
from PIL import Image
from app.utils.logger import log_info
from app.db.models.ingested_file_v2 import IngestedFileV2
from app.db.session_v2 import get_async_session
from app.services.ingestion.ingestion_service_v2 import IngestionServiceV2
from app.ai.registry import get_image_captioner


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
    """

    def __init__(self):
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
                 await db.commit()
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
             # 3. Caption + OCR (via registry)
             # ----------------------------------------------
             result = await self.captioner.caption(image_path)
             if not result or not isinstance(result, dict):
                 log_info("[ImageIngestorV1] ❌ Captioner returned no result")
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
             # 5. Semantic text synthesis
             # ----------------------------------------------
             semantic_text = self._synthesize_text(
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
             parsed_payload = {
                 "raw_text": semantic_text,
                 "meta": {
                     "media_type": "image",
                     "visual_type": visual_type,
                     "has_ocr": bool(ocr_text),
                     "caption_model": self.captioner.__class__.__name__,
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
        prefix_map = {
            "chart": "The image represents a data visualization.",
            "table": "The image shows tabular information.",
            "diagram": "The image illustrates a structured system or process.",
            "screenshot": "The image appears to be a software or dashboard screenshot.",
            "photo": "The image is a real-world photograph.",
            "infographic": "The image combines visual and textual elements.",
            "mixed": "The image contains multiple visual elements.",
            "unknown": "The image contains visual information.",
        }

        text = prefix_map.get(visual_type, "The image contains visual information.")
        text += "\n\n" + caption

        if ocr_text:
            text += "\n\nDetected text:\n" + ocr_text

        return text.strip()
