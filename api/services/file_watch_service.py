# ==========================================================
# 🚀 file_watch_service.py — Enterprise File Watcher for Unified Ingestion (v3.4)
# ==========================================================
"""
Continuously monitors /data/uploads for new files and triggers ingestion
into Postgres + ChromaDB via intelligent_auto_ingest_router.py.

✅ Multi-format: .pdf, .txt, .docx, .csv, .xlsx, .json, .rss
✅ Uses ingestion_config.yaml for dynamic path & format control
✅ Displays structured ingestion summaries in console
✅ Deduplication with persistent tracker
✅ Fault-tolerant UTF-8 log reader (no decode errors)
✅ Logs to /logs/file_watch.log and reads ingestion results from /logs/manual_ingest.log
"""

import os
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Load configuration dynamically
from api.config.ingestion_config import get_ingestion_config
from api.services.intelligent_auto_ingest_router import route_ingestion

# ==========================================================
# ⚙️ CONFIGURATION
# ==========================================================
config = get_ingestion_config()

UPLOAD_DIR = config["paths"].get("upload_dir", os.path.join(os.getcwd(), "data", "uploads"))
LOG_PATH = os.path.join(config["paths"].get("logs_dir", os.path.join(os.getcwd(), "logs")), "file_watch.log")
PROCESSED_TRACKER = os.path.join(UPLOAD_DIR, ".processed_files")

SUPPORTED_EXTENSIONS = config["formats"].get(
    "supported",
    [".pdf", ".txt", ".docx", ".csv", ".xlsx", ".json", ".rss"]
)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# ==========================================================
# 🧾 LOGGING SETUP
# ==========================================================
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("file_watch_service")

# ==========================================================
# 💾 TRACKER FUNCTIONS
# ==========================================================
def load_processed_files():
    """Load list of previously processed files to prevent duplicates."""
    if not os.path.exists(PROCESSED_TRACKER):
        return set()
    with open(PROCESSED_TRACKER, "r", encoding="utf-8", errors="ignore") as f:
        return set(line.strip() for line in f.readlines())


def save_processed_file(file_path: str):
    """Mark a file as processed."""
    with open(PROCESSED_TRACKER, "a", encoding="utf-8", errors="ignore") as f:
        f.write(file_path + "\n")

# ==========================================================
# 🧠 FILE EVENT HANDLER
# ==========================================================
class UploadHandler(FileSystemEventHandler):
    """Handles newly created files in the upload directory."""

    def on_created(self, event):
        if event.is_directory:
            return

        file_path = event.src_path
        _, ext = os.path.splitext(file_path)
        base_name = os.path.basename(file_path)

        # Skip temp, partial, or lock files
        if base_name.startswith("~$") or base_name.endswith(".tmp") or "~" in base_name:
            logger.info(f"[TempSkip] Temporary file skipped: {file_path}")
            return

        if ext.lower() not in SUPPORTED_EXTENSIONS:
            logger.warning(f"[Skip] Unsupported file type: {file_path}")
            return

        processed_files = load_processed_files()
        if file_path in processed_files:
            logger.info(f"[DuplicateSkip] Already processed: {file_path}")
            return

        # Allow some time for file write completion
        time.sleep(2)

        logger.info("───────────────────────────────")
        logger.info(f"[NewFile] Detected {file_path}")
        print(f"📁 New file detected: {base_name}")

        try:
            # 🚀 Route to unified intelligent ingestion
            result = route_ingestion(file_path)
            logger.info(f"[Router] Routed successfully via intelligent_auto_ingest_router")
            logger.info(f"[Result] {result}")
            print("✅ Intelligent ingestion completed successfully.")

            # ✅ Display last lines of ingestion summary (manual_ingest.log)
            manual_log_path = os.path.join(os.getcwd(), "logs", "manual_ingest.log")
            if os.path.exists(manual_log_path):
                try:
                    with open(manual_log_path, "r", encoding="utf-8", errors="replace") as log:
                        lines = [line.strip() for line in log.readlines() if line.strip()]
                        recent_summary = lines[-8:] if len(lines) > 8 else lines
                        print("\n🧾 --- Last Ingestion Summary ---")
                        for l in recent_summary:
                            print(l)
                        print("───────────────────────────────")
                except Exception as e:
                    print(f"⚠️ Could not read ingestion summary: {e}")

            save_processed_file(file_path)
            logger.info(f"[Ingested] ✅ {base_name} processed successfully.")
            logger.info("───────────────────────────────")
            print(f"✅ File processed successfully: {base_name}\n")

        except Exception as e:
            logger.error(f"[Error] Failed to process {file_path}: {e}")
            print(f"❌ Error processing {base_name}: {e}")

# ==========================================================
# 🚀 MAIN WATCHER LOOP
# ==========================================================
def start_watcher():
    logger.info(f"📂 [FileWatcher] Monitoring '{UPLOAD_DIR}' for new files...")
    print(f"👀 [FileWatcher] Monitoring '{UPLOAD_DIR}' for new uploads...")

    event_handler = UploadHandler()
    observer = Observer()
    observer.schedule(event_handler, path=UPLOAD_DIR, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("🛑 [FileWatcher] Stopped manually.")
        print("\n🛑 File watcher stopped manually.")
    observer.join()

# ==========================================================
# 🧩 ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    start_watcher()
