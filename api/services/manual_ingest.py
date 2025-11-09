# ==============================================================
# 🚀 manual_ingest.py — Unified Manual / FileWatcher / Excel Smart Ingestion (v5.1)
# ==============================================================

import os
import shutil
import tempfile
import logging
from datetime import datetime
from fastapi import APIRouter, UploadFile

from api.services.intelligent_auto_ingest_router import route_ingestion
from api.services.unified_ingest_helper import clean_text_for_postgres
router = APIRouter(prefix="/ingest/manual", tags=["Manual Ingestion"])
logger = logging.getLogger("manual_ingest")

# ==============================================================
# 🚀 ingest_file_to_vector_and_db — Used by FileWatcher / CLI
# ==============================================================
def ingest_file_to_vector_and_db(file_path: str, source: str = "manual_upload"):
    """
    Unified ingestion for manual uploads or file watcher.
    Uses intelligent_auto_ingest_router to detect and process file type.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_name = os.path.basename(file_path)
        logger.info(f"[IngestStart] Starting ingestion for: {file_name}")

        # Unified intelligent routing
        result = route_ingestion(file_path)

        logger.info(f"[IngestComplete] ✅ {file_name} processed successfully.")
        return {
            "status": "success",
            "source": source,
            "file_name": file_name,
            "result": result
        }

    except Exception as e:
        logger.error(f"[Error] Failed to ingest {file_path}: {e}")
        return {
            "status": "failed",
            "source": source,
            "file_name": os.path.basename(file_path),
            "error": str(e)
        }


# ==============================================================
# 📁 Upload via API / Browser (Unified Endpoint)
# ==============================================================
@router.post("/upload")
async def upload_file(file: UploadFile):
    """
    Handles manual uploads via API or Swagger.
    Automatically detects type (Excel, PDF, DOCX, TXT, JSON, etc.)
    and routes it to the correct intelligent ingestion engine.
    """
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)

    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"[APIUpload] Intelligent ingestion for {file.filename}")
        result = route_ingestion(tmp_path)

        return {
            "status": "success",
            "source": "manual_upload",
            "file": file.filename,
            "result": result
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"[APIUploadError] {e}")
        return {
            "status": "error",
            "file": file.filename,
            "error": str(e)
        }

    finally:
        try:
            file.file.close()
        except Exception:
            pass
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
