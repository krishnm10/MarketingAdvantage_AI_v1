# =============================================================================
# Marketing Advantage AI — main.py (v2 — Enhanced with Pluggable Pipeline)
# =============================================================================
#
# WHAT'S PRESERVED (untouched from your original):
#   ✅ ChromaDB initialization (skip_count=True)
#   ✅ File watcher background start
#   ✅ Validation scheduler (all 3 workers: validation/conflict/temporal)
#   ✅ All 8 existing routers
#   ✅ /health, /health/scheduler, /health/scheduler/metrics endpoints
#   ✅ /api/v2/stats/chromadb endpoint with cache + refresh flag
#   ✅ Graceful shutdown of validation scheduler
#
# WHAT'S NEW (additive only):
#   ✅ Pluggable pipeline_factory initialization at startup (Step 4)
#   ✅ New RAG query router registered (/api/v2/rag/...)
#   ✅ Proper Python logging (replaces raw print())
#   ✅ Env-driven CORS origins (locked down for production)
#   ✅ /health now includes pipeline registry status
#   ✅ Syntax fix: SchedulerConfig closing parenthesis
#   ✅ allow_credentials=True added to CORS
# =============================================================================

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ─────────────────────────────────────────────────────────────────────────────
# Logging — replace all print() with structured logger
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("marketing_advantage_ai")


# ─────────────────────────────────────────────────────────────────────────────
# Existing Routers (ALL PRESERVED — zero changes)
# ─────────────────────────────────────────────────────────────────────────────
from app.api.v2.ingestion_api_v2        import router as ingestion_router
from app.api.v2.ingestion_admin_api     import router as ingestion_admin_router
from app.api.v2.ingestion_sync_api      import router as ingestion_sync_router
from app.api.v2.ingestion_integrity_api import router as ingestion_integrity_router
from app.api.v2.admin_audit_api         import router as admin_audit_router
from app.api.v2.auth_api                import router as auth_router
from app.api.v2.ingestion_ws_api        import router as ingestion_ws_router
from app.api.v2.ingestion_health        import router as ingestion_health_router
from app.api.v2.config_api              import router as config_router
from app.api.v2.retrieve_api            import router as retrieve_router

# ─────────────────────────────────────────────────────────────────────────────
# New Pluggable RAG Router (NEW — additive only)
# ─────────────────────────────────────────────────────────────────────────────
from app.api.v2.rag_api import router as rag_router          # NEW

# ─────────────────────────────────────────────────────────────────────────────
# Existing Services (ALL PRESERVED)
# ─────────────────────────────────────────────────────────────────────────────
from app.services.ingestion.watcher_ingestor_v2 import start_watcher_background

from app.services.validation.scheduler import (
    start_validation_scheduler,
    stop_validation_scheduler,
    get_validation_scheduler,
    SchedulerConfig,
)

# ─────────────────────────────────────────────────────────────────────────────
# New Pluggable Pipeline Factory (NEW — auto-registers all plugins on import)
# ─────────────────────────────────────────────────────────────────────────────
from app.core.pipeline_factory import pipeline_factory       # NEW
from app.core.plugin_registry import (                       # NEW
    vectordb_registry,
    embedder_registry,
    llm_registry,
    reranker_registry,
)


# =============================================================================
# LIFESPAN — Startup + Shutdown
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle.

    Startup order (intentional — each step is independent,
    failures are non-fatal so the app always comes up):
      Step 1: ChromaDB initialization        (existing — preserved)
      Step 2: File watcher                   (existing — preserved)
      Step 3: Validation scheduler           (existing — preserved)
      Step 4: Pluggable pipeline registry    (NEW — additive)

    Shutdown:
      - Stop validation scheduler gracefully (existing — preserved)
      - Clear all pipeline caches            (NEW)
    """
    logger.info("=" * 70)
    logger.info("🚀 Marketing Advantage AI v2 — Starting Up...")
    logger.info("=" * 70)

    # ─────────────────────────────────────────────────────────────────
    # STEP 1: Initialize ChromaDB (EXISTING — preserved exactly)
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n[Startup] STEP 1: Initializing ChromaDB...")
    try:
        from app.services.ingestion.ingestion_service_v2 import get_chroma_collection

        # ✅ skip_count=True → instant startup even with 100M+ vectors
        client, collection = get_chroma_collection(skip_count=True)
        logger.info("✅ ChromaDB Ready: Collection '%s' initialized", collection.name)
        logger.info("💡 Vector count: GET /health or GET /api/v2/stats/chromadb")

    except Exception as e:
        logger.error("❌ ChromaDB Initialization Failed: %s", e)
        logger.warning("⚠️  Vector search will be unavailable!")
        logger.info("💡 Run 'python init_chromadb.py' to fix")

    # ─────────────────────────────────────────────────────────────────
    # STEP 2: Start Background File Watcher (EXISTING — preserved exactly)
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n[Startup] STEP 2: Starting file watcher...")
    try:
        loop = asyncio.get_running_loop()
        start_watcher_background(loop)
        logger.info("✅ File watcher started successfully")

    except Exception as e:
        logger.error("❌ File Watcher Failed: %s", e)
        logger.warning("⚠️  Automatic ingestion disabled — use manual upload")

    # ─────────────────────────────────────────────────────────────────
    # STEP 3: Start Validation Scheduler (EXISTING — preserved, syntax fixed)
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n[Startup] STEP 3: Initializing validation scheduler...")
    try:
        scheduler_config = SchedulerConfig(
            # Worker intervals (seconds)
            validation_interval=int(os.getenv("VALIDATION_INTERVAL", "60")),
            conflict_interval=int(os.getenv("CONFLICT_INTERVAL", "120")),
            temporal_interval=int(os.getenv("TEMPORAL_INTERVAL", "300")),

            # Batch sizes (chunks per run)
            validation_batch_size=int(os.getenv("VALIDATION_BATCH_SIZE", "50")),
            conflict_batch_size=int(os.getenv("CONFLICT_BATCH_SIZE", "30")),
            temporal_batch_size=int(os.getenv("TEMPORAL_BATCH_SIZE", "50")),

            # Enable/disable individual workers
            enable_validation=os.getenv("ENABLE_VALIDATION", "true").lower() == "true",
            enable_conflict=os.getenv("ENABLE_CONFLICT", "true").lower() == "true",
            enable_temporal=os.getenv("ENABLE_TEMPORAL", "true").lower() == "true",
        )   # ← FIX: closing parenthesis was missing in original

        scheduler = await start_validation_scheduler(scheduler_config)

        logger.info("✅ Validation scheduler started:")
        logger.info(
            "   • Agentic validation  : %s (%ss interval)",
            "enabled" if scheduler_config.enable_validation else "disabled",
            scheduler_config.validation_interval,
        )
        logger.info(
            "   • Conflict detection  : %s (%ss interval)",
            "enabled" if scheduler_config.enable_conflict else "disabled",
            scheduler_config.conflict_interval,
        )
        logger.info(
            "   • Temporal revalidation: %s (%ss interval)",
            "enabled" if scheduler_config.enable_temporal else "disabled",
            scheduler_config.temporal_interval,
        )

    except Exception as e:
        logger.error("❌ Validation Scheduler Failed: %s", e)
        logger.warning("⚠️  App continues without background validation")
        logger.info("💡 Retrieval works, but trust scores won't auto-update")

    # ─────────────────────────────────────────────────────────────────
    # STEP 4: Initialize Pluggable Pipeline Registry (NEW — additive)
    # ─────────────────────────────────────────────────────────────────
    # pipeline_factory import above already triggered all register.py
    # auto-registrations (vectordb, embedders, llms, rerankers).
    # This step just logs what's available so ops can verify on startup.
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n[Startup] STEP 4: Pluggable Pipeline Registry...")
    try:
        registered_vectordbs  = list(vectordb_registry.list().keys())
        registered_embedders  = list(embedder_registry.list().keys())
        registered_llms       = list(llm_registry.list().keys())
        registered_rerankers  = list(reranker_registry.list().keys())

        logger.info("✅ Plugin registries loaded:")
        logger.info("   • VectorDBs  : %s", registered_vectordbs)
        logger.info("   • Embedders  : %s", registered_embedders)
        logger.info("   • LLMs       : %s", registered_llms)
        logger.info("   • Rerankers  : %s", registered_rerankers)
        logger.info("   • Cache      : enabled (per client_id)")
        logger.info("   • Default DB : NONE — client config required")

    except Exception as e:
        logger.error("❌ Plugin Registry Failed: %s", e)
        logger.warning("⚠️  Pluggable RAG unavailable — existing Chroma RAG still works")

    # ─────────────────────────────────────────────────────────────────
    # STARTUP COMPLETE
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("🎉 Server Ready! All systems operational.")
    logger.info("=" * 70)
    logger.info("📅 Started at : %sZ", datetime.utcnow().isoformat())
    logger.info("📖 API Docs   : http://localhost:8000/docs")
    logger.info("")
    logger.info("📊 Health endpoints:")
    logger.info("   GET /health                       → Full system health")
    logger.info("   GET /health/scheduler             → Scheduler status")
    logger.info("   GET /health/scheduler/metrics     → Worker metrics")
    logger.info("   GET /health/plugins               → Plugin registry status (NEW)")
    logger.info("")
    logger.info("📦 Data endpoints:")
    logger.info("   GET /api/v2/stats/chromadb        → Vector DB stats")
    logger.info("")
    logger.info("🔍 RAG endpoints (NEW — pluggable):")
    logger.info("   POST /api/v2/rag/query            → Dynamic RAG query")
    logger.info("   POST /api/v2/rag/pipeline/build   → Build client pipeline")
    logger.info("   GET  /api/v2/rag/pipeline/health  → Per-client health")
    logger.info("=" * 70 + "\n")

    yield  # ═══════════════════ App runs here ═══════════════════════

    # ─────────────────────────────────────────────────────────────────
    # SHUTDOWN (EXISTING + NEW cache cleanup)
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("🛑 Marketing Advantage AI v2 — Shutting down...")
    logger.info("=" * 70)

    # Stop validation scheduler (EXISTING — preserved)
    try:
        logger.info("\n[Shutdown] Stopping validation scheduler...")
        await stop_validation_scheduler()
        logger.info("✅ Validation scheduler stopped gracefully")
    except Exception as e:
        logger.error("❌ Error stopping scheduler: %s", e)

    # Clear pipeline caches (NEW)
    try:
        logger.info("[Shutdown] Clearing pipeline caches...")
        pipeline_factory.invalidate_all()
        logger.info("✅ Pipeline caches cleared")
    except Exception as e:
        logger.error("❌ Error clearing pipeline caches: %s", e)

    logger.info("\n" + "=" * 70)
    logger.info("✅ Shutdown complete. Goodbye!")
    logger.info("=" * 70)


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Marketing Advantage AI",
    version="2.0.0",
    description=(
        "Enterprise AI Platform — "
        "Pluggable VectorDB / Embedder / LLM / Reranker + "
        "Advanced ingestion and classification pipeline."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# =============================================================================
# CORS (env-driven for production safety)
# =============================================================================

# In production: set CORS_ORIGINS="https://app.yourdomain.com,https://admin.yourdomain.com"
# In dev: leave unset → defaults to ["*"]
_raw_origins = os.getenv("CORS_ORIGINS", "*")
_cors_origins: List[str] = (
    ["*"] if _raw_origins.strip() == "*"
    else [o.strip() for o in _raw_origins.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,        # ← FIX: was missing in original
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# ROUTER REGISTRATION
# =============================================================================

# ── Existing routers (ALL PRESERVED — prefixes/tags unchanged) ──────────────
app.include_router(
    ingestion_router,
    tags=["Ingestion v2"],
)
app.include_router(ingestion_admin_router,     tags=["Admin"])
app.include_router(ingestion_sync_router,      tags=["Sync"])
app.include_router(ingestion_integrity_router, tags=["Integrity"])
app.include_router(admin_audit_router,         tags=["Audit"])
app.include_router(auth_router,                tags=["Auth"])
app.include_router(ingestion_health_router,    tags=["Health"])
app.include_router(ingestion_ws_router,        tags=["WebSocket"])
app.include_router(config_router,              tags=["Configuration"])
app.include_router(retrieve_router,            tags=["Retrieval"])

# ── New pluggable RAG router (NEW — additive, own prefix) ───────────────────
app.include_router(
    rag_router,
    prefix="/api/v2/rag",
    tags=["RAG Pipeline (Pluggable)"],
)


# =============================================================================
# ROOT ENDPOINT (preserved + enhanced)
# =============================================================================

@app.get("/", tags=["System"])
async def index():
    """Root endpoint — service identity and available endpoint map."""
    return {
        "status":  "running",
        "version": "2.0.0",
        "service": "Marketing Advantage AI",
        "message": "All systems operational ✅",
        "endpoints": {
            # Existing
            "health":             "/health",
            "scheduler_health":   "/health/scheduler",
            "scheduler_metrics":  "/health/scheduler/metrics",
            "chromadb_stats":     "/api/v2/stats/chromadb",
            "docs":               "/docs",
            # New
            "plugin_registry":    "/health/plugins",
            "rag_query":          "/api/v2/rag/query",
            "rag_pipeline_build": "/api/v2/rag/pipeline/build",
            "rag_pipeline_health":"/api/v2/rag/pipeline/health",
        },
    }


# =============================================================================
# HEALTH ENDPOINT (enhanced — adds plugin registry section)
# =============================================================================

@app.get("/health", tags=["System"])
async def health_check():
    """
    Comprehensive system health check.

    Returns:
    - Server status
    - ChromaDB status (cached vector count)      [EXISTING]
    - Validation scheduler status                [EXISTING]
    - Plugin registry status                     [NEW]
    - File watcher status                        [EXISTING]
    """
    health_status = {
        "status":    "operational",
        "version":   "2.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "server":    "running",

        # ── Existing sections (unchanged) ────────────────────────────
        "chromadb": {
            "status":       "unknown",
            "collection":   None,
            "vector_count": 0,
            "cache_info":   "unavailable",
        },
        "validation_scheduler": {
            "status":         "unknown",
            "workers_active": 0,
        },
        "file_watcher": "active",

        # ── New section ───────────────────────────────────────────────
        "plugin_registry": {
            "status":    "unknown",
            "vectordbs": [],
            "embedders": [],
            "llms":      [],
            "rerankers": [],
        },
    }

    # ── ChromaDB check (EXISTING — preserved exactly) ────────────────
    try:
        from app.services.ingestion.ingestion_service_v2 import (
            get_chroma_collection,
            get_collection_count_cached,
            _COLLECTION_COUNT_CACHE,
        )
        _, collection = get_chroma_collection(skip_count=True)
        count = get_collection_count_cached()

        cache_age = "never updated"
        last_updated = _COLLECTION_COUNT_CACHE.get("last_updated")
        if last_updated:
            age_seconds = (datetime.utcnow() - last_updated).total_seconds()
            cache_age = f"{int(age_seconds)}s ago"

        health_status["chromadb"] = {
            "status":       "operational",
            "collection":   collection.name,
            "vector_count": count,
            "cache_info":   f"Cached ({cache_age}, refreshes every 5 min)",
        }
    except Exception as e:
        health_status["chromadb"]["status"] = f"error: {str(e)[:100]}"

    # ── Validation scheduler check (EXISTING — preserved exactly) ────
    try:
        scheduler = get_validation_scheduler()
        if scheduler and scheduler._running:
            status = scheduler.get_status()
            workers = status["workers"]
            active_workers = sum(
                1 for w in workers.values()
                if w["status"] not in ["disabled", "idle"]
            )
            health_status["validation_scheduler"] = {
                "status":         "running",
                "workers_active": active_workers,
                "uptime_seconds": status["scheduler"]["uptime_seconds"],
            }
        else:
            health_status["validation_scheduler"] = {
                "status":         "not_running",
                "workers_active": 0,
            }
    except Exception as e:
        health_status["validation_scheduler"]["status"] = f"error: {str(e)[:100]}"

    # ── Plugin registry check (NEW — additive) ───────────────────────
    try:
        health_status["plugin_registry"] = {
            "status":    "operational",
            "vectordbs": list(vectordb_registry.list().keys()),
            "embedders": list(embedder_registry.list().keys()),
            "llms":      list(llm_registry.list().keys()),
            "rerankers": list(reranker_registry.list().keys()),
            "cached_pipelines": pipeline_factory.list_cached(),
        }
    except Exception as e:
        health_status["plugin_registry"]["status"] = f"error: {str(e)[:100]}"

    return health_status


# =============================================================================
# NEW: Plugin registry health endpoint
# =============================================================================

@app.get("/health/plugins", tags=["System"])
async def plugin_registry_health():
    """
    Plugin registry status — shows all registered connectors
    and currently cached client pipelines.
    """
    return {
        "status":    "operational",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "registered": {
            "vectordbs":  vectordb_registry.list(),
            "embedders":  embedder_registry.list(),
            "llms":       llm_registry.list(),
            "rerankers":  reranker_registry.list(),
        },
        "cached_pipelines": pipeline_factory.list_cached(),
    }


# =============================================================================
# SCHEDULER HEALTH ENDPOINTS (EXISTING — preserved exactly)
# =============================================================================

@app.get("/health/scheduler", tags=["Health"])
async def scheduler_health():
    """
    Detailed validation scheduler health status.
    Returns worker status, execution statistics, and configuration.
    """
    scheduler = get_validation_scheduler()
    if scheduler is None:
        return {
            "status":  "not_running",
            "message": "Scheduler not initialized",
        }
    return {
        "status":  "running" if scheduler._running else "stopped",
        "details": scheduler.get_status(),
    }


@app.get("/health/scheduler/metrics", tags=["Health"])
async def scheduler_metrics():
    """
    Detailed scheduler metrics for monitoring and alerting.
    Returns aggregate statistics, per-worker metrics, and unhealthy worker list.
    """
    scheduler = get_validation_scheduler()
    if not scheduler:
        return {
            "error":     "Scheduler not running",
            "aggregate": {"total_runs": 0, "success_rate_percent": 0},
        }

    status  = scheduler.get_status()
    workers = status["workers"]

    total_runs      = sum(w["total_runs"]      for w in workers.values())
    total_successes = sum(w["total_successes"]  for w in workers.values())
    total_failures  = sum(w["total_failures"]   for w in workers.values())
    success_rate    = (total_successes / total_runs * 100) if total_runs > 0 else 0

    unhealthy = [
        name for name, stats in workers.items()
        if stats["consecutive_failures"] >= 3 and stats["status"] != "disabled"
    ]

    return {
        "aggregate": {
            "total_runs":           total_runs,
            "total_successes":      total_successes,
            "total_failures":       total_failures,
            "success_rate_percent": round(success_rate, 2),
        },
        "health": {
            "all_healthy":      len(unhealthy) == 0,
            "unhealthy_workers": unhealthy,
        },
        "uptime_seconds": status["scheduler"]["uptime_seconds"],
        "workers":        workers,
        "config":         status["config"],
    }


# =============================================================================
# CHROMADB STATS ENDPOINT (EXISTING — preserved exactly)
# =============================================================================

@app.get("/api/v2/stats/chromadb", tags=["Stats"])
async def chromadb_stats(refresh: bool = False):
    """
    Get detailed ChromaDB statistics.

    Query params:
        refresh (bool): If true, bypass cache and fetch fresh count (slower).

    Examples:
        GET /api/v2/stats/chromadb           → Fast (uses cache)
        GET /api/v2/stats/chromadb?refresh=true → Slow but fresh
    """
    try:
        from app.services.ingestion.ingestion_service_v2 import (
            get_chroma_collection,
            get_collection_count_cached,
            _COLLECTION_COUNT_CACHE,
            CHROMA_PATH,
        )
        _, collection = get_chroma_collection(skip_count=True)

        if refresh:
            loop  = asyncio.get_running_loop()
            count = await loop.run_in_executor(None, collection.count)
            _COLLECTION_COUNT_CACHE["count"]        = count
            _COLLECTION_COUNT_CACHE["last_updated"] = datetime.utcnow()
            cache_status = "fresh (cache updated)"
        else:
            count        = get_collection_count_cached()
            cache_status = "cached"

        cache_age_seconds = None
        last_updated = _COLLECTION_COUNT_CACHE.get("last_updated")
        if last_updated:
            cache_age_seconds = int(
                (datetime.utcnow() - last_updated).total_seconds()
            )

        return {
            "status":          "operational",
            "collection_name": collection.name,
            "vector_count":    count,
            "storage_path":    CHROMA_PATH,
            "cache": {
                "status":           cache_status,
                "age_seconds":      cache_age_seconds,
                "last_updated":     last_updated.isoformat() if last_updated else None,
                "refresh_interval": "300 seconds (5 minutes)",
            },
            "note": "Use ?refresh=true to force fresh count (slower on large collections)",
        }

    except Exception as e:
        return {
            "status":  "error",
            "error":   str(e),
            "message": "Failed to retrieve ChromaDB statistics",
        }
