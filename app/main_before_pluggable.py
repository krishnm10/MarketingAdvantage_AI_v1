# =============================================
# main.py — Marketing Advantage AI (v2)
# Updated for ingestion_v2 + validation scheduler
# =============================================

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers
from app.api.v2.ingestion_api_v2 import router as ingestion_router
from app.api.v2.ingestion_admin_api import router as ingestion_admin_router
from app.api.v2.ingestion_sync_api import router as ingestion_sync_router
from app.api.v2.ingestion_integrity_api import router as ingestion_integrity_router
from app.api.v2.admin_audit_api import router as admin_audit_router
from app.api.v2.auth_api import router as auth_router
from app.api.v2.ingestion_ws_api import router as ingestion_ws_router
from app.api.v2.ingestion_health import router as ingestion_health_router

# Import services
from app.services.ingestion.watcher_ingestor_v2 import start_watcher_background

# Import validation scheduler
from app.services.validation.scheduler import (
    start_validation_scheduler,
    stop_validation_scheduler,
    get_validation_scheduler,
    SchedulerConfig
)


# -----------------------------------------------------------
# LIFESPAN: Startup and Shutdown Logic
# -----------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown logic.
    
    Startup:
    1. Initialize ChromaDB (fast, skip expensive count)
    2. Start file watcher for ingestion
    3. Start validation scheduler for background validation
    
    Shutdown:
    1. Stop validation scheduler gracefully
    2. Cleanup resources
    """
    print("=" * 80)
    print("🚀 Marketing Advantage AI v2 - Starting Up...")
    print("=" * 80)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 1: Initialize ChromaDB Collection
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    try:
        from app.services.ingestion.ingestion_service_v2 import get_chroma_collection
        print("\n[Startup] Initializing ChromaDB...")
        
        # ✅ skip_count=True for instant startup even with 100M+ vectors
        client, collection = get_chroma_collection(skip_count=True)
        print(f"✅ ChromaDB Ready: Collection '{collection.name}' initialized")
        print("💡 Vector count available at GET /health or GET /api/v2/stats/chromadb")
    except Exception as e:
        print(f"❌ ChromaDB Initialization Failed: {e}")
        print("⚠️  Vector search will be unavailable!")
        print("💡 Run 'python init_chromadb.py' to fix")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 2: Start Background File Watcher
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    try:
        print("\n[Startup] Starting file watcher...")
        loop = asyncio.get_running_loop()
        start_watcher_background(loop)
        print("✅ File watcher started successfully")
    except Exception as e:
        print(f"❌ File Watcher Failed: {e}")
        print("⚠️  Automatic ingestion disabled, use manual upload")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 3: Start Validation Scheduler (Background Workers)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    try:
        print("\n[Startup] Initializing validation scheduler...")
        
        # Load configuration from environment
        config = SchedulerConfig(
            # Worker intervals (seconds)
            validation_interval=int(os.getenv("VALIDATION_INTERVAL", "60")),
            conflict_interval=int(os.getenv("CONFLICT_INTERVAL", "120")),
            temporal_interval=int(os.getenv("TEMPORAL_INTERVAL", "300")),
            
            # Batch sizes (chunks per run)
            validation_batch_size=int(os.getenv("VALIDATION_BATCH_SIZE", "50")),
            conflict_batch_size=int(os.getenv("CONFLICT_BATCH_SIZE", "30")),
            temporal_batch_size=int(os.getenv("TEMPORAL_BATCH_SIZE", "50")),
            
            # Enable/disable workers
            enable_validation=os.getenv("ENABLE_VALIDATION", "true").lower() == "true",
            enable_conflict=os.getenv("ENABLE_CONFLICT", "true").lower() == "true",
            enable_temporal=os.getenv("ENABLE_TEMPORAL", "true").lower() == "true",
        )
        
        scheduler = await start_validation_scheduler(config)
        
        print(f"✅ Validation scheduler started:")
        print(f"   • Agentic validation: {'enabled' if config.enable_validation else 'disabled'} ({config.validation_interval}s interval)")
        print(f"   • Conflict detection: {'enabled' if config.enable_conflict else 'disabled'} ({config.conflict_interval}s interval)")
        print(f"   • Temporal revalidation: {'enabled' if config.enable_temporal else 'disabled'} ({config.temporal_interval}s interval)")
        
    except Exception as e:
        print(f"❌ Validation Scheduler Failed: {e}")
        print("⚠️  App will continue without background validation")
        print("💡 Retrieval still works, but trust scores won't update automatically")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STARTUP COMPLETE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "=" * 80)
    print("🎉 Server Ready! All systems operational.")
    print("=" * 80)
    print(f"📅 Started at: {datetime.utcnow().isoformat()}Z")
    print("📊 Health endpoints:")
    print("   • GET /health                    → System health")
    print("   • GET /health/scheduler          → Validation scheduler status")
    print("   • GET /health/scheduler/metrics  → Detailed worker metrics")
    print("   • GET /api/v2/stats/chromadb     → Vector database stats")
    print("=" * 80 + "\n")
    
    yield  # ═══════════════════════════════════════════════════════════
           # Server runs here
           # ═══════════════════════════════════════════════════════════
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SHUTDOWN: Cleanup
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "=" * 80)
    print("🛑 Marketing Advantage AI v2 - Shutting down...")
    print("=" * 80)
    
    # Stop validation scheduler gracefully
    try:
        print("\n[Shutdown] Stopping validation scheduler...")
        await stop_validation_scheduler()
        print("✅ Validation scheduler stopped gracefully")
    except Exception as e:
        print(f"❌ Error stopping scheduler: {e}")
    
    print("\n" + "=" * 80)
    print("✅ Shutdown complete. Goodbye!")
    print("=" * 80)


# -----------------------------------------------------------
# FASTAPI APP INITIALIZATION
# -----------------------------------------------------------
app = FastAPI(
    title="Marketing Advantage AI — Ingestion v2",
    version="2.0",
    description="Advanced ingestion and classification pipeline for business intelligence.",
    lifespan=lifespan
)


# -----------------------------------------------------------
# CORS Configuration
# -----------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------
# ROUTER REGISTRATION
# -----------------------------------------------------------
app.include_router(ingestion_router, prefix="/api/v2", tags=["Ingestion v2"])
app.include_router(ingestion_admin_router)
app.include_router(ingestion_sync_router)
app.include_router(ingestion_integrity_router)
app.include_router(admin_audit_router)
app.include_router(auth_router)
app.include_router(ingestion_health_router)
app.include_router(ingestion_ws_router)


# -----------------------------------------------------------
# ROOT ENDPOINT
# -----------------------------------------------------------
@app.get("/")
async def index():
    """Health check endpoint."""
    return {
        "status": "running",
        "version": "2.0",
        "service": "Marketing Advantage AI",
        "message": "All systems operational ✅",
        "endpoints": {
            "health": "/health",
            "scheduler_health": "/health/scheduler",
            "scheduler_metrics": "/health/scheduler/metrics",
            "chromadb_stats": "/api/v2/stats/chromadb",
            "docs": "/docs"
        }
    }


# -----------------------------------------------------------
# HEALTH CHECK ENDPOINT (Enhanced with Scheduler Status)
# -----------------------------------------------------------
@app.get("/health")
async def health_check():
    """
    Comprehensive health check including ChromaDB and validation scheduler.
    
    Returns:
        - Server status
        - ChromaDB status (cached vector count)
        - Validation scheduler status
        - File watcher status
    """
    health_status = {
        "status": "operational",
        "version": "2.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "server": "running",
        "chromadb": {
            "status": "unknown",
            "collection": None,
            "vector_count": 0,
            "cache_info": "unavailable"
        },
        "validation_scheduler": {
            "status": "unknown",
            "workers_active": 0
        },
        "file_watcher": "active"
    }
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Check ChromaDB (non-blocking with cached count)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    try:
        from app.services.ingestion.ingestion_service_v2 import (
            get_chroma_collection, 
            get_collection_count_cached,
            _COLLECTION_COUNT_CACHE
        )
        
        # Get collection (fast - no count operation)
        _, collection = get_chroma_collection(skip_count=True)
        
        # Get cached count (fast - only refreshes if >5 min old)
        count = get_collection_count_cached()
        
        # Calculate cache age
        cache_age = "never updated"
        last_updated = _COLLECTION_COUNT_CACHE.get("last_updated")
        if last_updated:
            age_seconds = (datetime.utcnow() - last_updated).total_seconds()
            cache_age = f"{int(age_seconds)}s ago"
        
        health_status["chromadb"] = {
            "status": "operational",
            "collection": collection.name,
            "vector_count": count,
            "cache_info": f"Cached ({cache_age}, refreshes every 5 min)"
        }
        
    except Exception as e:
        health_status["chromadb"]["status"] = f"error: {str(e)[:100]}"
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Check Validation Scheduler
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    try:
        scheduler = get_validation_scheduler()
        
        if scheduler and scheduler._running:
            status = scheduler.get_status()
            workers = status["workers"]
            
            # Count active workers
            active_workers = sum(
                1 for w in workers.values() 
                if w["status"] not in ["disabled", "idle"]
            )
            
            health_status["validation_scheduler"] = {
                "status": "running",
                "workers_active": active_workers,
                "uptime_seconds": status["scheduler"]["uptime_seconds"]
            }
        else:
            health_status["validation_scheduler"] = {
                "status": "not_running",
                "workers_active": 0
            }
            
    except Exception as e:
        health_status["validation_scheduler"]["status"] = f"error: {str(e)[:100]}"
    
    return health_status


# -----------------------------------------------------------
# SCHEDULER HEALTH ENDPOINTS
# -----------------------------------------------------------
@app.get("/health/scheduler")
async def scheduler_health():
    """
    Detailed validation scheduler health status.
    
    Returns:
        - Worker status for each validation type
        - Execution statistics
        - Configuration details
    """
    scheduler = get_validation_scheduler()
    
    if scheduler is None:
        return {
            "status": "not_running",
            "message": "Scheduler not initialized"
        }
    
    return {
        "status": "running" if scheduler._running else "stopped",
        "details": scheduler.get_status()
    }


@app.get("/health/scheduler/metrics")
async def scheduler_metrics():
    """
    Detailed scheduler metrics for monitoring/alerting.
    
    Returns:
        - Aggregate statistics (total runs, success rate)
        - Per-worker metrics
        - Health indicators (unhealthy workers)
    """
    scheduler = get_validation_scheduler()
    
    if not scheduler:
        return {
            "error": "Scheduler not running",
            "aggregate": {
                "total_runs": 0,
                "success_rate_percent": 0
            }
        }
    
    status = scheduler.get_status()
    workers = status["workers"]
    
    # Calculate aggregate metrics
    total_runs = sum(w["total_runs"] for w in workers.values())
    total_successes = sum(w["total_successes"] for w in workers.values())
    total_failures = sum(w["total_failures"] for w in workers.values())
    
    success_rate = (total_successes / total_runs * 100) if total_runs > 0 else 0
    
    # Get unhealthy workers
    unhealthy = [
        name for name, stats in workers.items()
        if stats["consecutive_failures"] >= 3 and stats["status"] != "disabled"
    ]
    
    return {
        "aggregate": {
            "total_runs": total_runs,
            "total_successes": total_successes,
            "total_failures": total_failures,
            "success_rate_percent": round(success_rate, 2),
        },
        "health": {
            "all_healthy": len(unhealthy) == 0,
            "unhealthy_workers": unhealthy,
        },
        "uptime_seconds": status["scheduler"]["uptime_seconds"],
        "workers": workers,
        "config": status["config"]
    }


# -----------------------------------------------------------
# CHROMADB STATS ENDPOINT (Enhanced)
# -----------------------------------------------------------
@app.get("/api/v2/stats/chromadb")
async def chromadb_stats(refresh: bool = False):
    """
    Get detailed ChromaDB statistics.
    
    Query params:
        refresh (bool): If true, bypass cache and fetch fresh count (slower)
    
    Examples:
        GET /api/v2/stats/chromadb          → Fast (uses cache)
        GET /api/v2/stats/chromadb?refresh=true → Slow but fresh (bypasses cache)
    """
    try:
        from app.services.ingestion.ingestion_service_v2 import (
            get_chroma_collection,
            get_collection_count_cached,
            _COLLECTION_COUNT_CACHE,
            CHROMA_PATH
        )
        
        # Get collection
        _, collection = get_chroma_collection(skip_count=True)
        
        # Get count (force refresh if requested)
        if refresh:
            # Bypass cache and get fresh count
            loop = asyncio.get_running_loop()
            count = await loop.run_in_executor(None, collection.count)
            
            # Update cache
            _COLLECTION_COUNT_CACHE["count"] = count
            _COLLECTION_COUNT_CACHE["last_updated"] = datetime.utcnow()
            
            cache_status = "fresh (cache updated)"
        else:
            # Use cached count
            count = get_collection_count_cached()
            cache_status = "cached"
        
        # Calculate cache age
        cache_age_seconds = None
        last_updated = _COLLECTION_COUNT_CACHE.get("last_updated")
        if last_updated:
            cache_age_seconds = int((datetime.utcnow() - last_updated).total_seconds())
        
        return {
            "status": "operational",
            "collection_name": collection.name,
            "vector_count": count,
            "storage_path": CHROMA_PATH,
            "cache": {
                "status": cache_status,
                "age_seconds": cache_age_seconds,
                "last_updated": last_updated.isoformat() if last_updated else None,
                "refresh_interval": "300 seconds (5 minutes)"
            },
            "note": "Use ?refresh=true to force fresh count (slower on large collections)"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to retrieve ChromaDB statistics"
        }
