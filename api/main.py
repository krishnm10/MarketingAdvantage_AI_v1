# ==========================================================
# 🚀 main.py — MarketingAdvantage AI Backend (v1.1 — Ingestion Integrated)
# ==========================================================
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import suppress
import os
import threading

# Core modules
from api.routes import strategy, content, admin_cache, admin_ingest_sources
from api.connection import get_db_connection
from api.services.taxonomy_loader import sync_taxonomy
from api.services.rag_service import _client, _collection

# Intelligent ingestion modules
from api.services import manual_ingest
from api.config.ingestion_config import get_ingestion_config

# ==========================================================
# ⚙️ FastAPI App Initialization
# ==========================================================
app = FastAPI(
    title="MarketingAdvantage AI Backend",
    version="1.1.0",
    description="Unified AI-powered marketing intelligence engine (CaaS + RAG + Smart Ingestion + Taxonomy Sync)"
)

# ==========================================================
# 🌐 CORS Configuration
# ==========================================================
origins = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "https://localhost",
    "https://127.0.0.1"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# 📦 Include Routers
# ==========================================================
app.include_router(strategy.router)
app.include_router(content.router)
app.include_router(admin_cache.router)
app.include_router(admin_ingest_sources.router)
app.include_router(manual_ingest.router)  # unified manual/file watcher ingestion

# ==========================================================
# ⚙️ Helper — Taxonomy Sync (Safe Wrapper)
# ==========================================================
def try_sync_taxonomy():
    """Safely sync taxonomy once at startup."""
    from api.connection import get_db_connection
    print("📚 [Taxonomy] Syncing from taxonomy_master.json ...")

    try:
        with get_db_connection() as db:
            sync_taxonomy(db, force_update=False)
        print("✅ [Taxonomy] Synced successfully.")
    except Exception as e:
        print(f"⚠️ [Taxonomy] Sync failed: {e}")

# ==========================================================
# 🔁 Optional File Watcher Auto-Start (Background Thread)
# ==========================================================
def start_file_watcher_background():
    """Start file watcher thread if enabled in config."""
    from api.services.file_watch_service import start_watcher
    try:
        watcher_thread = threading.Thread(target=start_watcher, daemon=True)
        watcher_thread.start()
        print("👀 [FileWatcher] Auto-started in background thread.")
    except Exception as e:
        print(f"⚠️ [FileWatcher] Could not start automatically: {e}")

# ==========================================================
# 🧩 Startup Event — Initialize Systems
# ==========================================================
@app.on_event("startup")
async def on_startup():
    """
    On app startup:
    - Sync taxonomy safely
    - Validate ChromaDB
    - Prepare directories
    - Optionally start File Watcher
    """
    print("🚀 [Startup] Initializing MarketingAdvantage AI services...")
    config = get_ingestion_config()

    # 1️⃣ Taxonomy Sync
    try_sync_taxonomy()

    # 2️⃣ Validate ChromaDB vector store
    try:
        _client.list_collections()
        print(f"✅ [RAG] ChromaDB active — Using collection: '{_collection.name}'")
    except Exception as e:
        print(f"⚠️ [RAG] Failed to connect to ChromaDB: {e}")

    # 3️⃣ Prepare required directories
    upload_dir = config["paths"].get("upload_dir", "./data/uploads")
    rag_dir = "./data/rag_db"
    for path in [upload_dir, rag_dir, "./logs"]:
        os.makedirs(path, exist_ok=True)
    print("✅ [Startup] Environment directories verified.")

    # 4️⃣ Optional File Watcher
    if config.get("file_watcher", {}).get("auto_start", True):
        start_file_watcher_background()

# ==========================================================
# 🧹 Shutdown Hook
# ==========================================================
@app.on_event("shutdown")
async def on_shutdown():
    """Perform cleanup (persist Chroma, close sessions, etc.)"""
    print("🧹 [Shutdown] Cleaning up resources...")
    with suppress(Exception):
        _client.persist()
        print("✅ [RAG] Persisted ChromaDB state.")

# ==========================================================
# 🔍 Root Endpoint
# ==========================================================
@app.get("/")
def root():
    """Health & diagnostics endpoint"""
    return {
        "status": "running",
        "message": "MarketingAdvantage AI Backend is live ✅",
        "services": {
            "taxonomy_sync": True,
            "rag_vector_store": True,
            "ingestion_engine": True,
            "file_watcher": True,
            "routes": ["content", "strategy", "admin_cache", "manual_ingest"]
        }
    }
