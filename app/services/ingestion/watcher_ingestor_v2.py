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

# --------------------------------------------------
# Media extensions (audio + video) ✨ UPDATED!
# SAFE: does NOT affect text ingestion
# --------------------------------------------------
AUDIO_EXTENSIONS = (
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
    ".flac",
)

VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".flv",
    ".wmv",
    ".m4v",
)

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".tiff",
    ".svg",
)

MEDIA_EXTENSIONS = AUDIO_EXTENSIONS + VIDEO_EXTENSIONS + IMAGE_EXTENSIONS  # ✨ Combined!

# Optional broadcast import for live UI updates
try:
    from app.api.v2.ingestion_ws_api import broadcast
except Exception:
    broadcast = None

WATCH_FOLDER = "./static/uploads/manual"
TMP_INGEST_DIR = "./static/tmp_ingest"
SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".csv", ".json", ".txt", ".xml", ".xls", ".xlsx")

_ACTIVE_MEDIA_FILES = set()
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
        "stage": stage,  # e.g. 'detected', 'ingestion', 'startup'
        "status": status,  # e.g. 'pending', 'success', 'failed'
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
    # 🔒 STEP-1: HARD STOP — ignore already processed files
    normalized_path = os.path.normpath(file_path)
    if "_processed" in normalized_path.split(os.path.sep):
        return

    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS and ext not in MEDIA_EXTENSIONS:
        return

    # --------------------------------------------------
    # MEDIA FILES → DIRECT MEDIA HOOK ✨ AUDIO + VIDEO!
    # --------------------------------------------------
    if ext in MEDIA_EXTENSIONS:
        # Determine media type
        if ext in AUDIO_EXTENSIONS:
            media_kind = "audio"
            emoji = "🎧"
            media_type_label = "audio"
        elif ext in VIDEO_EXTENSIONS:
            media_kind = "video"
            emoji = "🎬"
            media_type_label = "video"
        elif ext in IMAGE_EXTENSIONS:
            media_kind = "image"
            emoji = "🖼️"
            media_type_label = "image"
        else:
            media_kind = "audio"  # Fallback
            emoji = "🎵"
            media_type_label = "media"
        
        # --------------------------------------------------
        # Guard against duplicate watcher triggers
        # --------------------------------------------------
        if file_path in _ACTIVE_MEDIA_FILES:
            log_warning(f"[WatcherIngestorV2] ⏭️ {media_type_label.capitalize()} already processing → {file_name}")
            return

        _ACTIVE_MEDIA_FILES.add(file_path)
        
        try:
            file_id = str(uuid.uuid4())
            log_info(f"[WatcherIngestorV2] {emoji} Detected {media_type_label} file → {file_name}")

            # Call media hook and capture result
            result = await _MEDIA_HOOK.handle(
                file_id=file_id,
                file_path=file_path,
                file_type=media_kind,  # Pass "audio" or "video"
                parsed_output={},
                business_id=None,
                media_kind=media_kind,
            )
            
            # Handle deduplication response
            if result and result.get("status") == "duplicate_skipped":
                msg = (
                    f"⚠️ Duplicate {media_type_label} skipped → {file_name} "
                    f"(matches: {result.get('original_file', 'unknown')})"
                )
                log_info(f"[WatcherIngestorV2] {msg}")
                await _emit_feed_event("ingestion", "duplicate", msg)
                
            elif result and result.get("status") == "success":
                # Enhanced success message for videos
                if media_kind == "video":
                    scenes = result.get("scenes_analyzed", 0)
                    quality = result.get("text_quality", "standard")
                    msg = f"✅ Auto-ingested video → {file_name} ({scenes} scenes, {quality} quality)"
                else:
                    msg = f"✅ Auto-ingested {media_type_label} → {file_name}"
                
                log_info(f"[WatcherIngestorV2] {msg}")
                await _emit_feed_event("ingestion", "success", msg)
                
            else:
                msg = f"❌ {media_type_label.capitalize()} ingestion failed → {file_name}: {result.get('error', 'Unknown')}"
                log_warning(f"[WatcherIngestorV2] {msg}")
                await _emit_feed_event("ingestion", "failed", msg)
            
            # Move to processed folder (always, even duplicates)
            processed_dir = os.path.join(os.path.dirname(file_path), "_processed")
            os.makedirs(processed_dir, exist_ok=True)
            processed_path = os.path.join(processed_dir, file_name)
            os.replace(file_path, processed_path)
            log_info(f"[WatcherIngestorV2] 📦 Moved to processed → {processed_path}")
            
        except Exception as e:
            log_warning(f"[WatcherIngestorV2] ❌ {media_type_label.capitalize()} ingestion exception → {file_name}: {e}")
            await _emit_feed_event("ingestion", "failed", str(e))
        finally:
            _ACTIVE_MEDIA_FILES.discard(file_path)
        
        return

    # --------------------------------------------------
    # DOCUMENT FILES → Standard pipeline
    # --------------------------------------------------
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

            if isinstance(result, dict) and result.get("status") == "ingested":
                file_id = result.get("file_id")
                saved_path = result.get("path")

                # 🔥 MEDIA HOOK - Handle embedded visuals with deduplication tracking
                try:
                    media_result = await _MEDIA_HOOK.handle(
                        file_id=file_id,
                        file_path=saved_path,
                        file_type=os.path.splitext(saved_path)[1].replace(".", ""),
                        parsed_output={},  # already ingested text, hook handles visuals
                        business_id=None,
                    )

                    # Log deduplication status for embedded media
                    if media_result and media_result.get("status") == "duplicate_skipped":
                        log_info(
                            f"[WatcherIngestorV2] 📎 Embedded media duplicate detected: "
                            f"{media_result.get('original_file', 'unknown')}"
                        )
                except Exception as e:
                    log_warning(f"[WatcherIngestorV2] Media hook failed for {file_name}: {e}")

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

            ext = os.path.splitext(file_path)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS and ext not in MEDIA_EXTENSIONS:
                continue

            await _process_new_file(file_path)

def start_watcher_background(loop: asyncio.AbstractEventLoop):
    """Starts the watcher in background when FastAPI boots."""
    loop.create_task(watch_and_ingest())
    log_info("[WatcherIngestorV2] Background watcher started with FastAPI server.")

_MEDIA_HOOK = MediaIngestionHookV1()
