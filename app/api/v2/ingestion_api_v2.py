# =============================================
# ingestion_api_v2.py — FastAPI Router (Production-Ready + LLM Toggle Support)
# Fully aligned with ingestion_v2 architecture and async-safe DB handling
# =============================================

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.db.session_v2 import get_db
from app.services.ingestion.file_router_v2 import route_file_ingestion, route_external_ingestion
from app.db.models.ingested_file_v2 import IngestedFileV2
from app.db.models.ingested_content_v2 import IngestedContentV2
from app.utils.logger import log_info, log_warning
from app.config import ingestion_settings
from app.services.ingestion.media.media_ingestion_hook_v1 import MediaIngestionHookV1
import uuid
import os

router = APIRouter(prefix="/api/v2/ingestion", tags=["Ingestion v2"])

# -----------------------------------------------------------
# FILE UPLOAD INGESTION ENDPOINT
# -----------------------------------------------------------
@router.post("/upload")
async def ingest_file(
    file: UploadFile = File(...),
    business_id: str = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Handles ingestion of uploaded files (PDF, DOCX, CSV, TXT, etc.)
    Uses async-safe routing via file_router_v2.
    """
    try:
        log_info(f"[ingestion_api_v2] Received upload: {file.filename}")
        response = await route_file_ingestion(file=file, business_id=business_id)
        details_status = response.get("status")
        details_reason = response.get("reason")

        if details_status == "skipped" and details_reason == "db_duplicate":
            return {
                "status": "duplicate_skipped",
                "file_name": file.filename,
                "message": f"Duplicate detected. '{file.filename}' was already ingested.",
                "details": response,
            }

        return {
            "status": "success",
            "file_name": file.filename,
            "message": f"File '{file.filename}' uploaded successfully.",
            "details": response,
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        log_warning(f"[ingestion_api_v2] Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"File ingestion failed: {e}")


# -----------------------------------------------------------
# EXTERNAL SOURCE INGESTION ENDPOINT
# -----------------------------------------------------------
@router.post("/external")
async def ingest_external(
    source_type: str = Form(..., description="Source type: web | rss | api"),
    source_url: str = Form(..., description="URL or API endpoint"),
    business_id: str = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Ingests external data sources — webpages, RSS feeds, or APIs.
    Honors ENABLE_LLM_NORMALIZATION toggle.
    """
    try:
        log_info(f"[ingestion_api_v2] External ingestion triggered: {source_type} → {source_url}")
        if not ingestion_settings.ENABLE_LLM_NORMALIZATION:
            log_info("[ingestion_api_v2] ⚙️ LLM normalization disabled — proceeding without LLM rewrite.")

        response = await route_external_ingestion(
            source_type=source_type,
            source_url=source_url,
            business_id=business_id,
        )
        return {
            "status": "success",
            "source_type": source_type,
            "source_url": source_url,
            "LLM_ENABLED": ingestion_settings.ENABLE_LLM_NORMALIZATION,
            "details": response,
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        log_warning(f"[ingestion_api_v2] External ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"External ingestion failed: {e}")


# -----------------------------------------------------------
# FILE STATUS ENDPOINT
# -----------------------------------------------------------
@router.get("/status/{file_id}")
async def get_file_status(file_id: str, db: AsyncSession = Depends(get_db)):
    """
    Returns ingestion and deduplication statistics for a given file.
    """
    try:
        file_record = await db.get(IngestedFileV2, file_id)
        if not file_record:
            raise HTTPException(status_code=404, detail="File not found.")

        chunk_count = await db.scalar(
            select(func.count()).select_from(IngestedContentV2).where(IngestedContentV2.file_id == file_id)
        )

        return {
            "file_id": str(file_record.id),
            "file_name": file_record.file_name,
            "file_type": file_record.file_type,
            "status": file_record.status,
            "total_chunks": file_record.total_chunks,
            "unique_chunks": file_record.unique_chunks,
            "duplicate_chunks": file_record.duplicate_chunks,
            "dedup_ratio": file_record.dedup_ratio,
            "created_at": file_record.created_at,
            "updated_at": file_record.updated_at,
            "chunk_count": chunk_count,
        }

    except Exception as e:
        log_warning(f"[ingestion_api_v2] Status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving file status: {e}")


# -----------------------------------------------------------
# 🔁 LLM SETTINGS TOGGLES (For UI Live Control)
# -----------------------------------------------------------
@router.get("/llm-settings")
async def get_llm_settings():
    """
    Fetch current LLM rewrite settings for UI toggles.
    """
    try:
        return {
            "ENABLE_LLM_NORMALIZATION": ingestion_settings.ENABLE_LLM_NORMALIZATION,
            "LLM_MODE": ingestion_settings.LLM_MODE,
            "LLM_PROVIDER": ingestion_settings.LLM_PROVIDER,
            "OLLAMA_MODEL": ingestion_settings.OLLAMA_MODEL,
        }
    except Exception as e:
        log_warning(f"[ingestion_api_v2] Failed to fetch LLM settings: {e}")
        raise HTTPException(status_code=500, detail=f"Could not retrieve settings: {e}")


@router.post("/llm-settings")
async def update_llm_settings(
    enable: bool = Form(None, description="Enable or disable LLM normalization"),
    mode: str = Form(None, description="Set LLM mode: factual | creative"),
):
    """
    Allows toggling LLM normalization and mode dynamically via UI.
    Example:
      POST /api/v2/ingestion/llm-settings
      Form Data:
        enable=true
        mode=factual
    """
    try:
        if enable is not None:
            ingestion_settings.ENABLE_LLM_NORMALIZATION = enable
        if mode in ("factual", "creative"):
            ingestion_settings.LLM_MODE = mode

        log_info(f"[ingestion_api_v2] LLM settings updated → enable={enable}, mode={mode}")

        return {
            "ENABLE_LLM_NORMALIZATION": ingestion_settings.ENABLE_LLM_NORMALIZATION,
            "LLM_MODE": ingestion_settings.LLM_MODE,
        }

    except Exception as e:
        log_warning(f"[ingestion_api_v2] Failed to update LLM settings: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating LLM settings: {e}")


# -----------------------------------------------------------
# HEALTH CHECK — handled by ingestion_health.py
# -----------------------------------------------------------

@router.post("/media/upload")
async def ingest_media(
    file: UploadFile = File(...),
    media_kind: str = Form(..., description="audio | image | video"),
    business_id: str = Form(None),
):
    """
    Unified media ingestion endpoint with enterprise-grade deduplication.
    This bypasses file_router_v2 completely.
    
    Returns:
        - status: success | duplicate_skipped | failed
        - file_id: UUID of the ingested file
        - duplicate_of: (if duplicate) UUID of original file
        - message: Human-readable status message
        - perceptual_hash/acoustic_hash: Hash used for deduplication
    """
    try:
        log_info(f"[ingestion_api_v2] Received media upload: {file.filename} ({media_kind})")
        
        file_id = str(uuid.uuid4())
        upload_dir = "static/uploads/media"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, f"{file_id}_{file.filename}")
        
        # Save uploaded file
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # --------------------------------------------------
        # Resolve business_id safely (MEDIA ONLY)
        # --------------------------------------------------
        resolved_business_id = None
        if business_id:
            try:
                resolved_business_id = str(uuid.UUID(business_id))
            except Exception:
                # Swagger default "string" or invalid UUID → ignore
                resolved_business_id = None
        
        # Auto-generate if still missing
        if resolved_business_id is None:
            resolved_business_id = str(uuid.uuid4())
        
        # --------------------------------------------------
        # Call media hook and capture deduplication result
        # --------------------------------------------------
        result = await MediaIngestionHookV1().handle(
            file_id=file_id,
            file_path=file_path,
            file_type=media_kind,
            parsed_output={},
            business_id=resolved_business_id,
            media_kind=media_kind,
        )
        
        # --------------------------------------------------
        # Handle deduplication response
        # --------------------------------------------------
        if result and result.get("status") == "duplicate_skipped":
            log_info(
                f"[ingestion_api_v2] 🔁 Duplicate detected: {file.filename} "
                f"→ matches {result.get('original_file', 'unknown')}"
            )
            
            # Clean up uploaded duplicate file to save storage
            try:
                os.remove(file_path)
                log_info(f"[ingestion_api_v2] Removed duplicate file: {file_path}")
            except Exception as e:
                log_warning(f"[ingestion_api_v2] Failed to remove duplicate: {e}")
            
            return {
                "status": "duplicate_skipped",
                "file_id": file_id,
                "media_kind": media_kind,
                "file_name": file.filename,
                "duplicate_of": result.get("duplicate_of"),
                "original_file": result.get("original_file"),
                "message": result.get("message", "Duplicate media detected"),
                "hash": result.get("perceptual_hash") or result.get("acoustic_hash"),
            }
        
        elif result and result.get("status") == "success":
            log_info(f"[ingestion_api_v2] ✅ Media ingested successfully: {file.filename}")
            
            return {
                "status": "success",
                "file_id": file_id,
                "media_kind": media_kind,
                "file_name": file.filename,
                "message": "Media ingested successfully",
                "visual_type": result.get("visual_type"),  # for images
                "segment_count": result.get("segment_count"),  # for audio
                "hash": result.get("perceptual_hash") or result.get("acoustic_hash"),
            }
        
        else:
            # Failed ingestion
            error_msg = result.get("error", "Unknown error") if result else "No result returned"
            log_warning(f"[ingestion_api_v2] ❌ Media ingestion failed: {error_msg}")
            
            # Clean up failed upload
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass
            
            return {
                "status": "failed",
                "file_id": file_id,
                "media_kind": media_kind,
                "file_name": file.filename,
                "error": error_msg,
                "message": f"Media ingestion failed: {error_msg}",
            }
        
    except Exception as e:
        log_warning(f"[ingestion_api_v2] Media ingestion exception: {e}")
        
        # Clean up on exception
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass
        
        raise HTTPException(status_code=500, detail=f"Media ingestion failed: {e}")
