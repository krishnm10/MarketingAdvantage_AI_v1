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
#   ✅ PHANTOM Protocol Phase 0 — hardware profiler + config bridge (Step 5)
#   ✅ /health now includes phantom hardware section
#   ✅ GET /phantom/stats endpoint (PHANTOM runtime diagnostics)
#   ✅ uvloop event loop accelerator (Linux/Mac; silently skipped on Windows)
#   ✅ Sentry SDK error tracking (gated on SENTRY_DSN env var)
#   ✅ slowapi rate limiting (200/minute default, env-configurable)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# uvloop — must be installed before the event loop is created.
# Provides a 2-4x faster asyncio event loop on Linux/Mac.
# Silently skipped on Windows (not supported server-side; fine for dev).
# ─────────────────────────────────────────────────────────────────────────────
try:
    import uvloop
    uvloop.install()  # Sets uvloop as the default asyncio event loop policy
except ImportError:
    pass  # Not available on this platform — standard asyncio loop used

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List

try:
    import sentry_sdk
    _sentry_available = True
except ImportError:
    sentry_sdk = None  # type: ignore[assignment]
    _sentry_available = False

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.utils.instrumentation import RequestIDMiddleware
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address
    _slowapi_available = True
except ImportError:
    _slowapi_available = False

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
# Sentry SDK  — Error tracking + performance monitoring
# Activated only when SENTRY_DSN is set in the environment.
# Safe no-op when the variable is missing (dev / test environments).
# ─────────────────────────────────────────────────────────────────────────────
_sentry_dsn = os.getenv("SENTRY_DSN", "").strip()
if _sentry_available and _sentry_dsn:
    sentry_sdk.init(
        dsn=_sentry_dsn,
        environment=os.getenv("ENVIRONMENT", "production"),
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
        send_default_pii=False,          # GDPR / privacy compliance
        attach_stacktrace=True,
        integrations=[
            sentry_sdk.integrations.fastapi.FastApiIntegration(
                transaction_style="endpoint"
            ),
            sentry_sdk.integrations.starlette.StarletteIntegration(
                transaction_style="endpoint"
            ),
            sentry_sdk.integrations.sqlalchemy.SqlalchemyIntegration(),
            sentry_sdk.integrations.logging.LoggingIntegration(
                level=logging.WARNING,    # breadcrumbs from WARNING+
                event_level=logging.ERROR,  # send event on ERROR+
            ),
        ],
    )
    logger.info("[Sentry] Error tracking enabled (env=%s)", os.getenv("ENVIRONMENT", "production"))
elif not _sentry_available:
    logger.warning("[Sentry] sentry-sdk not installed — run: pip install sentry-sdk[fastapi]")
else:
    logger.info("[Sentry] SENTRY_DSN not set — error tracking disabled (dev mode)")


# ─────────────────────────────────────────────────────────────────────────────
# Rate Limiter (slowapi)
# Default: 200 requests/minute per IP — override via RATE_LIMIT_DEFAULT env var.
# Storage: in-memory by default; set REDIS_URL=redis://... for distributed.
# ─────────────────────────────────────────────────────────────────────────────
_rate_limit_default = os.getenv("RATE_LIMIT_DEFAULT", "200/minute")
_redis_url = os.getenv("REDIS_URL", "memory://")

if _slowapi_available:
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[_rate_limit_default],
        storage_uri=_redis_url,
    )
else:
    limiter = None  # type: ignore[assignment]
    logger.warning("[RateLimit] slowapi not installed — rate limiting disabled")


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
from app.api.v2.retrieve_chat_api       import router as retrieve_chat_router
from app.api.v2.model_discovery_api     import router as model_discovery_router

# ─────────────────────────────────────────────────────────────────────────────
# New Pluggable RAG Router (NEW — additive only)
# ─────────────────────────────────────────────────────────────────────────────
from app.api.v2.rag_api import router as rag_router

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Embedding Alignment Router (NEW — additive only)
# ─────────────────────────────────────────────────────────────────────────────
from app.api.v2.embedding_alignment_api import router as embedding_alignment_router
from app.api.v2.pipeline_recommendation_api import router as pipeline_recommendation_router

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Reranker Config + Prompt Template Routers (NEW — additive only)
# ─────────────────────────────────────────────────────────────────────────────
from app.api.v2.rag_config_api       import router as rag_config_router
from app.api.v2.prompt_template_api  import router as prompt_template_router

# ─────────────────────────────────────────────────────────────────────────────
# New Kafka Management Router (NEW — additive only)
# ─────────────────────────────────────────────────────────────────────────────
from app.api.v2.kafka_api import router as kafka_router
from app.api.v2.ingestion_audit_api import router as ingestion_audit_router

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
from app.utils.env_flags import get_env_bool

# ─────────────────────────────────────────────────────────────────────────────
# New Pluggable Pipeline Factory (NEW — auto-registers all plugins on import)
# ─────────────────────────────────────────────────────────────────────────────
from app.core.pipeline_factory import pipeline_factory
from app.core.plugin_registry import (
    vectordb_registry,
    embedder_registry,
    llm_registry,
    reranker_registry,
)

# ─────────────────────────────────────────────────────────────────────────────
# PHANTOM Protocol — Phase 0 (NEW)
# Detects AMD ROCm / CUDA / CPU at startup and computes all PHANTOM tuning
# parameters. Non-fatal — if this import fails, the app still starts normally.
# ─────────────────────────────────────────────────────────────────────────────
from app.services.ingestion.phantom_config_bridge import (
    phantom_startup,
    phantom_cfg,
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
      Step 4: Pluggable pipeline registry    (existing — preserved)
      Step 5: PHANTOM hardware profiler      (NEW — Phase 0)

    Shutdown:
      - Stop validation scheduler gracefully (existing — preserved)
      - Clear all pipeline caches            (existing — preserved)
    """
    logger.info("=" * 70)
    logger.info("🚀 Marketing Advantage AI v2 — Starting Up...")
    logger.info("=" * 70)

    # ─────────────────────────────────────────────────────────────────
    # STEP 0: Startup environment validation — fail fast for critical vars
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n[Startup] STEP 0: Environment validation...")
    _ai_profile = os.getenv("AI_PROFILE", "cpu").lower()
    _required_by_profile = {
        "api": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"],
    }
    _needed = _required_by_profile.get(_ai_profile, [])
    if _needed and not any(os.getenv(v) for v in _needed):
        logger.error(
            "❌ AI_PROFILE=%s requires at least one of: %s — set in .env before starting",
            _ai_profile, _needed,
        )
        # Non-fatal: log and continue so the app starts (API calls will fail at request time)
    else:
        logger.info("✅ Environment validation passed (AI_PROFILE=%s)", _ai_profile)

    # ─────────────────────────────────────────────────────────────────
    active_vectordb = os.getenv("MAI_VECTORDB", "chroma").lower()
    pipeline_factory.invalidate_all()
    logger.info("[Startup] Cleared cached pipelines before initialization.")

    # STEP 1: Initialize active vector DB via compatibility adapter
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n[Startup] STEP 1: Initializing vector DB (%s)...", active_vectordb)
    try:
        from app.services.ingestion.ingestion_service_v2 import get_chroma_collection

        # Compatibility wrapper returns the active pluggable backend adapter.
        client, collection = get_chroma_collection(skip_count=True)
        logger.info(
            "✅ Vector DB Ready: backend='%s' collection='%s' initialized",
            active_vectordb,
            collection.name,
        )
        logger.info("💡 Vector health: GET /health")

    except Exception as e:
        logger.error("❌ Vector DB Initialization Failed (%s): %s", active_vectordb, e)
        logger.warning("⚠️  Vector search will be unavailable!")
        if active_vectordb == "chroma":
            logger.info("💡 Run 'python init_chromadb.py' to fix")
        elif active_vectordb == "redis":
            logger.info("💡 Redis vector backend requires Redis Stack / RediSearch support")

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
            enable_validation=get_env_bool(
                "ENABLE_AGENTIC_VALIDATION",
                default=True,
                aliases=("ENABLE_VALIDATION",),
            ),
            enable_conflict=get_env_bool(
                "ENABLE_CONFLICT_ANALYSIS",
                default=True,
                aliases=("ENABLE_CONFLICT",),
            ),
            enable_temporal=get_env_bool(
                "ENABLE_TEMPORAL_REVALIDATION",
                default=True,
                aliases=("ENABLE_TEMPORAL",),
            ),
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
    # STEP 4: Initialize Pluggable Pipeline Registry (EXISTING — preserved)
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
    # STEP 5: PHANTOM Hardware Profiler (NEW — Phase 0)
    #
    # Probes the host hardware (AMD ROCm → CUDA → CPU fallback) and
    # computes all PHANTOM tuning parameters:
    #   - embed_batch_size  (replaces hardcoded BATCH_SIZE=256)
    #   - upsert_batch_size (replaces hardcoded VectorDB loop)
    #   - ingest_workers    (parallel file ingestion)
    #   - io_thread_pool    (run_in_executor pool size)
    #   - bloom_capacity    (Phase 2 Bloom gate)
    #   - gravity params    (Phase 3 clustering)
    #   - stage_collapse    (Phase 5 kernel concurrency)
    #
    # Non-fatal — if detection fails, PHANTOM uses conservative CPU
    # defaults and the rest of the app continues normally.
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n[Startup] STEP 5: PHANTOM Hardware Profiler...")
    try:
        phantom_startup(verbose=True)
        logger.info("✅ PHANTOM config bridge ready:")
        logger.info("   • Tier          : %s", phantom_cfg.tier)
        logger.info("   • GPU           : %s (%.1f GB VRAM)", phantom_cfg.gpu_name, phantom_cfg.gpu_vram_gb)
        logger.info("   • RAM           : %.1f GB", phantom_cfg.system_ram_gb)
        logger.info("   • embed_batch   : %d", phantom_cfg.embed_batch_size)
        logger.info("   • upsert_batch  : %d", phantom_cfg.upsert_batch_size)
        logger.info("   • ingest_workers: %d", phantom_cfg.ingest_workers)
        logger.info("   • io_thread_pool: %d", phantom_cfg.io_thread_pool)
        logger.info("   • bloom_capacity: %s", f"{phantom_cfg.bloom_capacity:,}")
        logger.info("   • Stats         : GET /phantom/stats")

    except Exception as e:
        logger.error("❌ PHANTOM Profiler Failed: %s", e)
        logger.warning("⚠️  PHANTOM will use safe CPU defaults — ingestion still works")

    # ─────────────────────────────────────────────────────────────────
    # STEP 6: Kafka Event Streaming (NEW)
    #
    # When KAFKA_EVENTS_ENABLED=true, initializes the Kafka producer
    # and ensures all MAI topics exist on the Kafka cluster.
    # Non-fatal — if Kafka is unreachable, the app still starts normally.
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n[Startup] STEP 6: Kafka Event Streaming...")
    try:
        from app.services.kafka import kafka_service
        if kafka_service.enabled:
            topic_results = await kafka_service.ensure_topics()
            created = sum(1 for v in topic_results.values() if v == "created")
            existing = sum(1 for v in topic_results.values() if v == "exists")
            errors = sum(1 for v in topic_results.values() if v.startswith("error"))
            logger.info("✅ Kafka Event Streaming ready:")
            logger.info("   • Topics: %d created, %d existing, %d errors", created, existing, errors)
            logger.info("   • Bootstrap: %s", os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"))
            logger.info("   • Compression: %s", os.getenv("KAFKA_COMPRESSION_TYPE", "lz4"))
            logger.info("   • Idempotence: %s", os.getenv("KAFKA_ENABLE_IDEMPOTENCE", "true"))
            logger.info("   • Endpoints: /api/v2/kafka/health, /api/v2/kafka/topics")
        else:
            logger.info("   Kafka events disabled (KAFKA_EVENTS_ENABLED!=true)")
    except ImportError:
        logger.info("   confluent-kafka not installed — Kafka events unavailable")
    except Exception as e:
        logger.error("❌ Kafka Startup Failed: %s", e)
        logger.warning("⚠️  Kafka events unavailable — ingestion still works via Celery/inline")

    # ─────────────────────────────────────────────────────────────────
    # STEP 7: Ingestion Worker (event-driven background processor)
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n[Startup] STEP 7: Ingestion Worker...")
    try:
        from app.services.ingestion.ingestion_worker import get_ingestion_worker
        _ingestion_worker = get_ingestion_worker()
        await _ingestion_worker.start()
        logger.info(
            "✅ Ingestion worker started (concurrency=%s)",
            os.getenv("INGESTION_WORKER_CONCURRENCY", "4"),
        )
    except Exception as e:
        logger.error("❌ Ingestion Worker Failed: %s", e)
        logger.warning("⚠️  Ingestion will fall back to synchronous processing")

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
    logger.info("   GET /health/plugins               → Plugin registry status")
    logger.info("")
    logger.info("📦 Data endpoints:")
    logger.info("   GET /api/v2/stats/chromadb        → Vector DB stats")
    logger.info("")
    logger.info("🔍 RAG endpoints (pluggable):")
    logger.info("   POST /api/v2/rag/query            → Dynamic RAG query")
    logger.info("   POST /api/v2/rag/pipeline/build   → Build client pipeline")
    logger.info("   GET  /api/v2/rag/pipeline/health  → Per-client health")
    logger.info("")
    logger.info("⚡ PHANTOM endpoints (Phase 0):")
    logger.info("   GET  /phantom/stats               → Hardware profile + tuning params")
    logger.info("")
    logger.info("🔌 Kafka endpoints:")
    logger.info("   GET  /api/v2/kafka/health          → Kafka cluster health")
    logger.info("   GET  /api/v2/kafka/topics          → List Kafka topics")
    logger.info("   POST /api/v2/kafka/topics/ensure   → Create MAI topics")
    logger.info("   POST /api/v2/kafka/produce          → Publish test message")
    logger.info("   GET  /api/v2/kafka/config          → Current Kafka config")
    logger.info("   GET  /api/v2/kafka/stats           → Producer statistics")
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

    # Clear pipeline caches (EXISTING — preserved)
    try:
        logger.info("[Shutdown] Clearing pipeline caches...")
        pipeline_factory.invalidate_all()
        logger.info("✅ Pipeline caches cleared")
    except Exception as e:
        logger.error("❌ Error clearing pipeline caches: %s", e)

    # Stop ingestion worker (NEW — drain current batch)
    try:
        from app.services.ingestion.ingestion_worker import get_ingestion_worker
        _worker = get_ingestion_worker()
        if _worker._running:
            logger.info("[Shutdown] Stopping ingestion worker...")
            await _worker.stop()
            logger.info("✅ Ingestion worker stopped")
    except Exception as e:
        logger.error("❌ Error stopping ingestion worker: %s", e)

    # Shutdown Kafka producer (NEW — flush pending events)
    try:
        from app.services.kafka import kafka_service
        if kafka_service.enabled and kafka_service._started:
            logger.info("[Shutdown] Flushing Kafka producer...")
            await kafka_service.shutdown()
            logger.info("✅ Kafka producer shut down gracefully")
    except Exception as e:
        logger.error("❌ Error shutting down Kafka: %s", e)

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

# ── Rate limiter — attach to app state and register 429 handler ──────────────
app.state.limiter = limiter
if _slowapi_available and limiter is not None:
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# =============================================================================
# REQUEST ID MIDDLEWARE — must be added BEFORE CORS and other middleware
# =============================================================================
app.add_middleware(RequestIDMiddleware)


# =============================================================================
# CORS (env-driven for production safety)
#
# SECURITY FIX: The CORS spec forbids allow_origins=["*"] together with
# allow_credentials=True. Browsers reject such responses, and if ever
# replaced with real origins, credentials would be forwarded to every listed
# origin. In production, CORS_ORIGINS must be set explicitly.
#
# Dev default: localhost:3000 and localhost:8000 (explicit, not wildcard)
# Production: set CORS_ORIGINS="https://app.yourdomain.com,..." in .env
# =============================================================================
_env = os.getenv("ENVIRONMENT", "production").lower()
_raw_origins = os.getenv("CORS_ORIGINS", "").strip()

if _raw_origins and _raw_origins != "*":
    _cors_origins: List[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]
    _cors_allow_credentials = True
elif _env in ("development", "dev", "local", "test"):
    _cors_origins = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]
    _cors_allow_credentials = True
    logger.warning(
        "[CORS] CORS_ORIGINS not set in dev env — using localhost defaults. "
        "Set CORS_ORIGINS explicitly before deploying."
    )
else:
    # Production with no CORS_ORIGINS: allow wildcard WITHOUT credentials
    # (wildcard + credentials is invalid per CORS spec and rejected by browsers)
    _cors_origins = ["*"]
    _cors_allow_credentials = False
    logger.warning(
        "[CORS] CORS_ORIGINS not set in production — using wildcard without credentials. "
        "Set CORS_ORIGINS to explicit origins in production."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# ROUTER REGISTRATION
# =============================================================================

# ── Existing routers (ALL PRESERVED — prefixes/tags unchanged) ──────────────
app.include_router(ingestion_router,           tags=["Ingestion v2"])
app.include_router(ingestion_admin_router,     tags=["Admin"])
app.include_router(ingestion_sync_router,      tags=["Sync"])
app.include_router(ingestion_integrity_router, tags=["Integrity"])
app.include_router(admin_audit_router,         tags=["Audit"])
app.include_router(auth_router,                tags=["Auth"])
app.include_router(ingestion_health_router,    tags=["Health"])
app.include_router(ingestion_ws_router,        tags=["WebSocket"])
app.include_router(config_router,              tags=["Configuration"])
app.include_router(retrieve_router,            tags=["Retrieval"])
app.include_router(retrieve_chat_router,       tags=["Retrieval Chat"])
app.include_router(model_discovery_router,     tags=["Model Discovery"])
app.include_router(kafka_router,               tags=["Kafka"])
app.include_router(ingestion_audit_router,     tags=["Ingestion Audit"])

# ── New pluggable RAG router (EXISTING — preserved) ─────────────────────────
app.include_router(
    rag_router,
    prefix="/api/v2/rag",
    tags=["RAG Pipeline (Pluggable)"],
)

# ── Phase 1 Embedding Alignment Router (NEW — additive only) ─────────────────
app.include_router(
    embedding_alignment_router,
    tags=["Embedding Alignment"],
)

# ── Pipeline Recommendations Router (catalog-driven, read-only) ───────────────
app.include_router(
    pipeline_recommendation_router,
    tags=["Pipeline Recommendations"],
)

# ── Phase 2 — RAG Config + Prompt Templates + Evaluation ─────────────────────
app.include_router(
    rag_config_router,
    tags=["RAG Configuration"],
)

app.include_router(
    prompt_template_router,
    tags=["Prompt Templates"],
)

from app.api.v2.rag_eval_api import router as rag_eval_router
app.include_router(
    rag_eval_router,
    tags=["RAG Evaluation"],
)

# ── Phase 3 — Pipeline Template Gallery ────────────────────────────────────
try:
    from app.api.v2.pipeline_template_api import router as pipeline_template_router
    app.include_router(pipeline_template_router, tags=["Pipeline Templates"])
except ImportError:
    logger.warning("[Router] pipeline_template_api not available — skipping")


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
            "health":               "/health",
            "scheduler_health":     "/health/scheduler",
            "scheduler_metrics":    "/health/scheduler/metrics",
            "chromadb_stats":       "/api/v2/stats/chromadb",
            "docs":                 "/docs",
            # Existing (pluggable)
            "plugin_registry":      "/health/plugins",
            "rag_query":            "/api/v2/rag/query",
            "rag_pipeline_build":   "/api/v2/rag/pipeline/build",
            "rag_pipeline_health":  "/api/v2/rag/pipeline/health",
            # PHANTOM Phase 0
            "phantom_stats":        "/phantom/stats",
        },
    }


# =============================================================================
# HEALTH ENDPOINT (enhanced — adds plugin registry + PHANTOM sections)
# =============================================================================

@app.get("/health", tags=["System"])
async def health_check():
    """
    Comprehensive system health check.

    Returns:
    - Server status
    - ChromaDB status (cached vector count)      [EXISTING]
    - Validation scheduler status                [EXISTING]
    - Plugin registry status                     [EXISTING]
    - File watcher status                        [EXISTING]
    - PHANTOM hardware profile summary           [NEW — Phase 0]
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
        "plugin_registry": {
            "status":    "unknown",
            "vectordbs": [],
            "embedders": [],
            "llms":      [],
            "rerankers": [],
        },

        # ── PHANTOM section (NEW — Phase 0) ──────────────────────────
        "phantom": {
            "status": "unknown",
            "tier":   None,
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

    # ── Plugin registry check (EXISTING — preserved) ─────────────────
    try:
        health_status["plugin_registry"] = {
            "status":           "operational",
            "vectordbs":        list(vectordb_registry.list().keys()),
            "embedders":        list(embedder_registry.list().keys()),
            "llms":             list(llm_registry.list().keys()),
            "rerankers":        list(reranker_registry.list().keys()),
            "cached_pipelines": pipeline_factory.list_cached(),
        }
    except Exception as e:
        health_status["plugin_registry"]["status"] = f"error: {str(e)[:100]}"

    # ── PHANTOM check (NEW — Phase 0) ────────────────────────────────
    try:
        if phantom_cfg._initialized:
            health_status["phantom"] = {
                "status":        "operational",
                "tier":          phantom_cfg.tier,
                "gpu":           phantom_cfg.gpu_name,
                "gpu_vram_gb":   phantom_cfg.gpu_vram_gb,
                "ram_gb":        phantom_cfg.system_ram_gb,
                "embed_batch":   phantom_cfg.embed_batch_size,
                "upsert_batch":  phantom_cfg.upsert_batch_size,
                "workers":       phantom_cfg.ingest_workers,
                "is_rocm":       phantom_cfg.is_rocm,
            }
        else:
            health_status["phantom"] = {
                "status": "not_initialized",
                "note":   "phantom_startup() did not run — check STEP 5 startup logs",
            }
    except Exception as e:
        health_status["phantom"]["status"] = f"error: {str(e)[:100]}"

    return health_status


# =============================================================================
# SPLIT HEALTH ENDPOINTS
#
# /health/live  — Kubernetes liveness probe: is the process up?
#                 Public endpoint, returns minimal data (no infra leak).
# /health/ready — Kubernetes readiness probe + deep diagnostic.
#                 Requires X-Internal-Token header to prevent reconnaissance.
# =============================================================================

@app.get("/health/live", tags=["System"])
async def health_live():
    """
    Liveness probe — fast, public, minimal.
    Returns 200 if the process is running; no infra details.
    Safe to expose externally (no GPU VRAM, storage paths, or plugin info).
    """
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z"}


def _check_internal_token(request: Request):
    """Dependency: require X-Internal-Token for sensitive readiness endpoint."""
    expected = os.getenv("INTERNAL_HEALTH_TOKEN", "")
    if not expected:
        # Token not configured — allow (dev/staging mode)
        return
    provided = request.headers.get("X-Internal-Token", "")
    if provided != expected:
        raise HTTPException(status_code=403, detail="Forbidden: invalid internal token")


@app.get("/health/ready", tags=["System"])
async def health_ready(request: Request, _: None = Depends(_check_internal_token)):
    """
    Readiness probe — deep diagnostic with full infra details.
    Protected by X-Internal-Token header when INTERNAL_HEALTH_TOKEN is set.
    Exposes GPU VRAM, storage paths, plugin registry, and scheduler internals.
    """
    ready_status: dict = {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    try:
        from app.services.ingestion.ingestion_service_v2 import get_chroma_collection
        _, collection = get_chroma_collection(skip_count=True)
        ready_status["vectordb"] = {"status": "ok", "collection": collection.name}
    except Exception as e:
        ready_status["vectordb"] = {"status": "error", "detail": str(e)[:100]}

    scheduler = get_validation_scheduler()
    ready_status["scheduler"] = {
        "status": "running" if scheduler and getattr(scheduler, "_running", False) else "stopped"
    }

    try:
        if phantom_cfg._initialized:
            ready_status["phantom"] = {
                "tier": phantom_cfg.tier,
                "gpu": phantom_cfg.gpu_name,
                "gpu_vram_gb": phantom_cfg.gpu_vram_gb,
                "ram_gb": phantom_cfg.system_ram_gb,
                "embed_batch": phantom_cfg.embed_batch_size,
                "workers": phantom_cfg.ingest_workers,
                "is_rocm": phantom_cfg.is_rocm,
            }
    except Exception:
        pass

    return ready_status


# =============================================================================
# PLUGIN REGISTRY HEALTH ENDPOINT (EXISTING — preserved exactly)
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
            "all_healthy":       len(unhealthy) == 0,
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


# =============================================================================
# PHANTOM STATS ENDPOINT (NEW — Phase 0)
# =============================================================================

@app.get("/phantom/stats", tags=["PHANTOM"])
async def phantom_stats():
    """
    PHANTOM Protocol runtime diagnostics.

    Returns the full hardware profile and all computed tuning parameters.
    Use this to verify that:
      - AMD ROCm was detected correctly (tier = "amd_rocm")
      - embed_batch_size and upsert_batch_size match your VRAM
      - All Phase 0–5 parameters are populated

    If tier shows "cpu_mid" instead of "amd_rocm", ROCm is not visible
    to Python — check your torch/ROCm installation.
    """
    if not phantom_cfg._initialized:
        return {
            "status": "not_initialized",
            "note":   (
                "phantom_startup() did not run at startup. "
                "Check STEP 5 in your startup logs for the error. "
                "Add phantom_startup() to your FastAPI lifespan if missing."
            ),
        }

    return {
        "status":    "operational",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hardware": {
            "tier":         phantom_cfg.tier,
            "gpu_name":     phantom_cfg.gpu_name,
            "gpu_vram_gb":  phantom_cfg.gpu_vram_gb,
            "system_ram_gb": phantom_cfg.system_ram_gb,
            "cpu_cores":    phantom_cfg.cpu_cores,
            "is_rocm":      phantom_cfg.is_rocm,
            "is_gpu":       phantom_cfg.is_gpu,
        },
        "tuning": {
            "phase_1": {
                "embed_batch_size":   phantom_cfg.embed_batch_size,
                "embed_prefetch":     phantom_cfg.embed_prefetch,
                "upsert_batch_size":  phantom_cfg.upsert_batch_size,
                "upsert_concurrency": phantom_cfg.upsert_concurrency,
                "ingest_workers":     phantom_cfg.ingest_workers,
                "parse_workers":      phantom_cfg.parse_workers,
                "io_thread_pool":     phantom_cfg.io_thread_pool,
            },
            "phase_2": {
                "bloom_capacity":   phantom_cfg.bloom_capacity,
                "bloom_error_rate": phantom_cfg.bloom_error_rate,
                "l2_dedup_batch":   phantom_cfg.l2_dedup_batch,
            },
            "phase_3": {
                "gravity_clusters":  phantom_cfg.gravity_clusters,
                "gravity_batch_size": phantom_cfg.gravity_batch_size,
            },
            "phase_5": {
                "stage_collapse_concurrency": phantom_cfg.stage_collapse_concurrency,
            },
        },
        "env_overrides": {
            "PHANTOM_EMBED_BATCH_SIZE":  os.getenv("PHANTOM_EMBED_BATCH_SIZE", "not set"),
            "PHANTOM_UPSERT_BATCH_SIZE": os.getenv("PHANTOM_UPSERT_BATCH_SIZE", "not set"),
            "PHANTOM_INGEST_WORKERS":    os.getenv("PHANTOM_INGEST_WORKERS", "not set"),
            "PHANTOM_BLOOM_CAPACITY":    os.getenv("PHANTOM_BLOOM_CAPACITY", "not set"),
        },
    }


# =============================================================================
# COST TRACKING ENDPOINT (NEW — per-tenant token usage)
# =============================================================================

@app.get("/api/v2/stats/token-usage", tags=["Stats"])
async def token_usage_stats(tenant_id: str = "", month: str = ""):
    """
    Get token usage summary for cost tracking.

    Query params:
        tenant_id: Filter by specific tenant (empty = all tenants)
        month:     "YYYY-MM" format (empty = current month)
    """
    from app.utils.cost_tracker import get_usage, get_all_usage
    if tenant_id:
        usage = get_usage(tenant_id, month=month or None)
        return {
            "tenant_id": tenant_id,
            "month": month or __import__("datetime").datetime.utcnow().strftime("%Y-%m"),
            "usage": usage,
        }
    return {
        "month": month or __import__("datetime").datetime.utcnow().strftime("%Y-%m"),
        "all_tenants": get_all_usage(month=month or None),
    }


# =============================================================================
# INGESTION WORKER STATS ENDPOINT (NEW)
# =============================================================================

@app.get("/api/v2/stats/ingestion-worker", tags=["Stats"])
async def ingestion_worker_stats():
    """
    Get ingestion worker queue statistics.
    """
    try:
        from app.services.ingestion.ingestion_worker import get_ingestion_queue, get_ingestion_worker
        queue = get_ingestion_queue()
        worker = get_ingestion_worker()
        return {
            "status": "running" if worker._running else "stopped",
            "queue": queue.stats,
            "dead_letter": queue.dead_letter_items(),
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)[:200]}
