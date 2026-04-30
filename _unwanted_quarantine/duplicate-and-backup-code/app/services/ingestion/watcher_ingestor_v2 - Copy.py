import asyncio
import os
import shutil
import time
import uuid
from fastapi import UploadFile
from watchfiles import awatch
from app.services.ingestion.file_router_v2 import route_file_ingestion
from app.utils.logger import log_info, log_warning
from app.services.ingestion.media.media_ingestion_hook_v1 import MediaIngestionHookV1


# Optional broadcast import for live UI updates
try:
    from app.api.v2.ingestion_ws_api import broadcast
except Exception:
    broadcast = None

WATCH_FOLDER = "./static/uploads/manual"
TMP_INGEST_DIR = "./static/tmp_ingest"
SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".csv", ".json", ".txt", ".xml", ".xls", ".xlsx")

_recent_files = {}
_DEBOUNCE_WINDOW = 3.0
_RETRY_COUNT = 3
_RETRY_DELAY = 1.5
_RECENT_PRUNE_AGE = _DEBOUNCE_WINDOW * 100


def _prune_recent_files():
    """Cleans up stale file entries from the debounce cache."""
    now = time.time()
    keys_to_remove = [k for k, ts in _recent_files.items() if now - ts > _RECENT_PRUNE_AGE]
    for k in keys_to_remove:
        _recent_files.pop(k, None)


async def _emit_feed_event(stage: str, status: str, message: str):
    """
    Send structured ingestion progress events to frontend via WebSocket.
    This ensures the dashboard UI can display live ingestion trails.
    """
    if not broadcast:
        return

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "stage": stage,      # e.g. 'detected', 'ingestion', 'startup'
        "status": status,    # e.g. 'pending', 'success', 'failed'
        "message": message,  # descriptive human-readable message
    }

    try:
        await broadcast(payload)
    except Exception:
        pass


async def _process_new_file(file_path: str):
    """
    Handle a new file detected in the watched folder:
    - Debounces duplicate events
    - Copies to temp dir
    - Routes ingestion
    - Emits live feed updates
    """
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        return

    now = time.time()
    last_seen = _recent_files.get(file_path, 0)
    if now - last_seen < _DEBOUNCE_WINDOW:
        log_warning(f"[WatcherIngestorV2] Ignored duplicate trigger for {file_name}")
        return

    _recent_files[file_path] = now
    if len(_recent_files) > 1000:
        _prune_recent_files()

    os.makedirs(TMP_INGEST_DIR, exist_ok=True)
    unique_prefix = uuid.uuid4().hex[:8]
    temp_copy_name = f"{unique_prefix}_{file_name}"
    temp_copy_path = os.path.join(TMP_INGEST_DIR, temp_copy_name)

    for attempt in range(_RETRY_COUNT):
        try:
            shutil.copy2(file_path, temp_copy_path)
            break
        except Exception as e:
            if attempt == _RETRY_COUNT - 1:
                msg = f"❌ Failed to copy {file_name} after retries: {e}"
                log_warning(f"[WatcherIngestorV2] {msg}")
                await _emit_feed_event("copy", "failed", msg)
                raise
            log_warning(f"[WatcherIngestorV2] File busy ({file_name}), retrying... {attempt+1}")
            await asyncio.sleep(_RETRY_DELAY)

    try:
        with open(temp_copy_path, "rb") as f:
            upload_file = UploadFile(filename=file_name, file=f)

            msg = f"📥 Detected new file → {file_name}"
            log_info(f"[WatcherIngestorV2] {msg}")
            await _emit_feed_event("detected", "pending", msg)

            result = await route_file_ingestion(upload_file)

            status = None
            try:
                if isinstance(result, dict):
                    status = result.get("status")
                elif hasattr(result, "status"):
                    status = getattr(result, "status", None)
            except Exception:
                status = None

            msg = f"✅ Auto-ingested {file_name}: status={status}"
            log_info(f"[WatcherIngestorV2] {msg}")
            await _emit_feed_event("ingestion", "success", msg)

    except Exception as e:
        msg = f"❌ Failed to process {file_name}: {e}"
        log_warning(f"[WatcherIngestorV2] {msg}")
        await _emit_feed_event("ingestion", "failed", msg)

    finally:
        if os.path.exists(temp_copy_path):
            try:
                os.remove(temp_copy_path)
            except Exception:
                log_warning(f"[WatcherIngestorV2] Failed to remove temp file: {temp_copy_path}")


async def watch_and_ingest():
    """
    Watches the uploads folder for new files and triggers ingestion.
    Emits a 'startup' event for frontend on launch.
    """
    os.makedirs(WATCH_FOLDER, exist_ok=True)
    abs_path = os.path.abspath(WATCH_FOLDER)

    log_info(f"[WatcherIngestorV2] 👀 Watching folder: {abs_path}")
    await _emit_feed_event("startup", "info", f"👀 Watching folder: {abs_path}")

    async for changes in awatch(WATCH_FOLDER):
        for change, file_path in changes:
            if not os.path.exists(file_path):
                continue
            if os.path.splitext(file_path)[1].lower() not in SUPPORTED_EXTENSIONS:
                continue
            await _process_new_file(file_path)


def start_watcher_background(loop: asyncio.AbstractEventLoop):
    """Starts the watcher in background when FastAPI boots."""
    loop.create_task(watch_and_ingest())
    log_info("[WatcherIngestorV2] Background watcher started with FastAPI server.")
    
_MEDIA_HOOK = MediaIngestionHookV1()

