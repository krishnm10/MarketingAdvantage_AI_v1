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
from app.utils.tenant_validator import validate_business_id, TenantValidationError
import aiofiles
import re
import uuid
import os

router = APIRouter(prefix="/api/v2/ingestion", tags=["Ingestion v2"])


def _is_quota_exceeded(exc: Exception) -> bool:
    """Return True when the OpenAI account has no remaining billing credits."""
    exc_code = getattr(exc, "code", None)
    if exc_code == "insufficient_quota":
        return True
    body = getattr(exc, "body", None) or {}
    if isinstance(body, dict):
        err = body.get("error", {})
        if isinstance(err, dict) and err.get("code") == "insufficient_quota":
            return True
    # Fallback: inspect string representation (older SDK versions)
    return "insufficient_quota" in str(exc)


# -----------------------------------------------------------
# MEDIA UPLOAD CONSTANTS
# -----------------------------------------------------------
_MAX_MEDIA_UPLOAD_BYTES: int = 200 * 1024 * 1024  # 200 MB hard cap
_UPLOAD_CHUNK_SIZE: int = 1024 * 256              # 256 KB streaming chunks

_ALLOWED_MEDIA_KINDS = frozenset({"audio", "image", "video"})

_ALLOWED_EXTENSIONS: dict[str, frozenset[str]] = {
    "audio": frozenset({".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a", ".wma"}),
    "image": frozenset({".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".svg"}),
    "video": frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".flv"}),
}

_ALLOWED_CONTENT_TYPES: dict[str, frozenset[str]] = {
    "audio": frozenset({"audio/mpeg", "audio/wav", "audio/flac", "audio/ogg", "audio/aac", "audio/mp4", "audio/x-ms-wma"}),
    "image": frozenset({"image/jpeg", "image/png", "image/gif", "image/bmp", "image/webp", "image/tiff", "image/svg+xml"}),
    "video": frozenset({"video/mp4", "video/x-msvideo", "video/quicktime", "video/x-matroska", "video/webm", "video/x-ms-wmv", "video/x-flv"}),
}

# Filename sanitisation regex: allow only alphanumerics, hyphens, underscores, and a single dot before the extension.
_UNSAFE_FILENAME_RE = re.compile(r"[^\w.\-]")


def _sanitize_filename(raw: str) -> str:
    """Return a safe, flat filename stripped of path separators and special chars."""
    # Take only the trailing component → defeat ../ and absolute-path tricks
    name = os.path.basename(raw)
    # Collapse any remaining path-separator look-alikes (e.g. backslash on non-Windows)
    name = name.replace("\\", "_").replace("/", "_")
    # Strip non-word characters (keeps [a-zA-Z0-9_], dots, hyphens)
    name = _UNSAFE_FILENAME_RE.sub("_", name)
    # Collapse repeated underscores / dots
    name = re.sub(r"[_.]{2,}", "_", name)
    # Ensure the name is never empty
    return name or "unnamed_upload"

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
        _bctx = validate_business_id(
            business_id, endpoint="ingest_file", allow_default=True,
        )
        business_id = _bctx.tenant_id
    except TenantValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

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
        if _is_quota_exceeded(e):
            log_warning(f"[ingestion_api_v2] OpenAI quota exceeded during upload: {e}")
            raise HTTPException(
                status_code=402,
                detail=(
                    "Embedding failed: your OpenAI account has no remaining credits. "
                    "Please add billing credits at https://platform.openai.com/settings/organization/billing "
                    "or check your project spending limit at https://platform.openai.com/settings/organization/limits"
                ),
            )
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
        _bctx = validate_business_id(
            business_id, endpoint="ingest_external", allow_default=True,
        )
        business_id = _bctx.tenant_id
    except TenantValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        log_info(f"[ingestion_api_v2] External ingestion triggered: {source_type} → {source_url}")
        if not ingestion_settings.ENABLE_LLM_NORMALIZATION:
            log_info("[ingestion_api_v2] ⚙️ LLM normalization disabled — proceeding without LLM rewrite.")

        # ── Celery: queue in background; fall back to inline if broker is down ──
        # Only attempted when CELERY_ENABLED=true in .env.
        try:
            from app.worker.broker_config import is_celery_enabled
            if is_celery_enabled():
                from app.worker.tasks import run_external_ingestion_task
                task = run_external_ingestion_task.delay(source_type, source_url, business_id)
                log_info(f"[ingestion_api_v2] Queued Celery task task_id={task.id} for {source_url}")
                return {
                    "status": "queued",
                    "task_id": task.id,
                    "source_type": source_type,
                    "source_url": source_url,
                    "LLM_ENABLED": ingestion_settings.ENABLE_LLM_NORMALIZATION,
                }
        except Exception:
            pass  # Celery / broker unavailable — fall through to inline

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
        if _is_quota_exceeded(e):
            log_warning(f"[ingestion_api_v2] OpenAI quota exceeded during external ingestion: {e}")
            raise HTTPException(
                status_code=402,
                detail=(
                    "Embedding failed: your OpenAI account has no remaining credits. "
                    "Please add billing credits at https://platform.openai.com/settings/organization/billing"
                ),
            )
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
        _bctx = validate_business_id(
            business_id, endpoint="ingest_media", allow_default=True,
        )
        business_id = _bctx.tenant_id
    except TenantValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        # --------------------------------------------------
        # 1. Validate media_kind
        # --------------------------------------------------
        media_kind = media_kind.strip().lower()
        if media_kind not in _ALLOWED_MEDIA_KINDS:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid media_kind '{media_kind}'. Must be one of: {', '.join(sorted(_ALLOWED_MEDIA_KINDS))}",
            )

        # --------------------------------------------------
        # 2. Validate content-type header
        # --------------------------------------------------
        declared_ct = (file.content_type or "").lower().split(";")[0].strip()
        if declared_ct and declared_ct not in _ALLOWED_CONTENT_TYPES[media_kind]:
            raise HTTPException(
                status_code=415,
                detail=f"Content-Type '{declared_ct}' is not acceptable for media_kind='{media_kind}'.",
            )

        # --------------------------------------------------
        # 3. Validate file extension
        # --------------------------------------------------
        safe_name = _sanitize_filename(file.filename or "upload")
        _, ext = os.path.splitext(safe_name)
        ext = ext.lower()
        if ext not in _ALLOWED_EXTENSIONS[media_kind]:
            raise HTTPException(
                status_code=422,
                detail=f"Extension '{ext}' is not allowed for media_kind='{media_kind}'. "
                       f"Accepted: {', '.join(sorted(_ALLOWED_EXTENSIONS[media_kind]))}.",
            )

        log_info(f"[ingestion_api_v2] Received media upload: {safe_name} ({media_kind})")

        # --------------------------------------------------
        # 4. Stream file to disk asynchronously with size cap
        # --------------------------------------------------
        file_id = str(uuid.uuid4())
        upload_dir = "static/uploads/media"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, f"{file_id}_{safe_name}")

        # Resolve to absolute and verify it stays inside the upload directory
        abs_upload_dir = os.path.realpath(upload_dir)
        abs_file_path = os.path.realpath(file_path)
        if not abs_file_path.startswith(abs_upload_dir + os.sep):
            raise HTTPException(status_code=400, detail="Invalid filename.")

        bytes_written = 0
        async with aiofiles.open(abs_file_path, "wb") as out:
            while True:
                chunk = await file.read(_UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > _MAX_MEDIA_UPLOAD_BYTES:
                    # Abort: remove partial file and reject
                    await out.close()
                    try:
                        os.remove(abs_file_path)
                    except OSError:
                        pass
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {_MAX_MEDIA_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
                    )
                await out.write(chunk)

        file_path = abs_file_path  # use resolved path from here on
        
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

        if _is_quota_exceeded(e):
            raise HTTPException(
                status_code=402,
                detail=(
                    "Embedding failed: your OpenAI account has no remaining credits. "
                    "Please add billing credits at https://platform.openai.com/settings/organization/billing"
                ),
            )
        raise HTTPException(status_code=500, detail=f"Media ingestion failed: {e}")
