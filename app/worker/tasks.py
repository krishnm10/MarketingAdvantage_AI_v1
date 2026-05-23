# =============================================================================
# app/worker/tasks.py — Celery Task Definitions
# =============================================================================
#
# Tasks are imported lazily inside each function body so that the Celery worker
# process does NOT load every single app module at startup — only what each task
# actually needs.  This keeps worker boot time short and avoids circular imports.
#
# INGESTION TASKS
#   run_ingestion_pipeline(file_id, saved_path, file_ext)
#       → Parses a file that is already saved on disk and pushes it through
#         IngestionOrchestrator (PII sanitization → IngestionServiceV2).
#         Dispatched by file_router_v2 after the file is saved and the DB
#         record is created (status='uploaded').
#
#   run_external_ingestion_task(source_type, source_url, business_id)
#       → Runs the full route_external_ingestion() flow (fetch → parse → embed)
#         for a web page, RSS feed, or API source.
#
# VALIDATION TASKS  (also driven by Celery Beat — see celery_app.beat_schedule)
#   run_agentic_validation()
#   run_conflict_detection()
#   run_temporal_revalidation()
#       → Equivalent to the three asyncio workers already in scheduler.py,
#         but now executed by a Celery worker so they scale horizontally.
# =============================================================================

import asyncio

from app.worker.broker_config import is_celery_enabled

# Guard: only import the real Celery app when Celery is enabled.
# When disabled, this module should never be imported (all call-sites
# check is_celery_enabled() first), but if it is, the task functions
# below will be plain functions rather than Celery task objects.
if is_celery_enabled():
    from app.worker.celery_app import celery_app
else:
    celery_app = None  # type: ignore[assignment]


def _run(coro):
    """
    Run an async coroutine in a fresh event loop (Celery workers are sync).

    IMPORTANT: asyncpg connections inside SQLAlchemy's pool are bound to the
    event loop that created them.  When we close the loop at the end of one
    task, those connections become zombies — the next task gets a new loop but
    the pool hands out connections from the dead loop, causing:
        'NoneType' object has no attribute 'send'

    Fix: dispose the async engine's pool after every task so that the next
    task creates fresh connections on its own fresh loop.

    PIPELINE CACHE: The ingestion pipeline (including ChromaDB PersistentClient)
    is cached via @lru_cache.  When multiple processes (FastAPI + Celery) share
    the same ChromaDB persist_directory, the in-memory HNSW index can become
    stale.  Clearing the cache after each task ensures each task starts with a
    fresh ChromaDB client that reads the latest on-disk state, preventing
    silent data loss from concurrent-writer segment overwrites.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        # Drain asyncpg pool so stale connections don't poison the next task
        try:
            from app.db.session_v2 import async_engine
            loop.run_until_complete(async_engine.dispose())
        except Exception:
            pass
        # Release cached pipelines so ChromaDB PersistentClients are
        # properly closed and their WAL state is flushed to disk.
        try:
            from app.services.ingestion.ingestion_service_v2 import (
                clear_ingestion_pipeline_cache,
            )

            clear_ingestion_pipeline_cache()
        except Exception:
            pass
        loop.close()


# ─────────────────────────────────────────────────────────────────────────────
# TASK DEFINITIONS — only registered when Celery is enabled.
# When CELERY_ENABLED=false, this entire block is skipped and no task objects
# are created.  All call-sites already guard `from app.worker.tasks import ...`
# behind an `is_celery_enabled()` check, so these symbols are never needed
# when Celery is off.
# ─────────────────────────────────────────────────────────────────────────────

if celery_app is not None:

    # ── INGESTION TASKS ──────────────────────────────────────────────────────

    @celery_app.task(
        bind=True,
        name="tasks.run_ingestion_pipeline",
        max_retries=50,
        default_retry_delay=60,
        queue="ingestion",
    )
    def run_ingestion_pipeline(self, file_id: str, saved_path: str, file_ext: str, business_id: str = None):
        """
        Parse a file that has already been saved to disk and ingest it.

        Arguments:
            file_id      — UUID string matching the IngestedFileV2 row already created.
            saved_path   — Absolute or relative path to the saved file on disk.
            file_ext     — Lowercase extension including dot, e.g. ".pdf".
            business_id  — Validated tenant ID (recovered from DB record if not provided).
        """
        from app.services.ingestion.file_router_v2 import PARSER_MAP
        from app.services.ingestion.ingestion_orchestrator import IngestionOrchestrator
        from app.services.ingestion.tenant_guard import resolve_ingestion_tenant
        from app.utils.logger import log_info, log_warning

        tenant_id = business_id
        try:
            # Resolve tenant: prefer explicit param, fall back to DB recovery
            tenant_ctx = resolve_ingestion_tenant(
                business_id=business_id,
                file_id=file_id,
                source="celery_task",
                allow_default=True,
            )
            tenant_id = tenant_ctx.tenant_id

            parser_func = PARSER_MAP.get(file_ext)
            if parser_func is None:
                raise ValueError(f"No parser registered for extension: {file_ext}")

            log_info(f"[celery/ingestion] Parsing file_id={file_id}  ext={file_ext}  tenant={tenant_ctx.tenant_id}")
            parsed_output = _run(parser_func(saved_path))

            log_info(f"[celery/ingestion] Running ingestion pipeline for file_id={file_id}")
            _run(IngestionOrchestrator().ingest_parsed_output(
                file_id=file_id, parsed=parsed_output,
                client_id=tenant_ctx.tenant_id,
            ))

            log_info(f"[celery/ingestion] Done — file_id={file_id}")
            return {"status": "ingested", "file_id": file_id, "tenant_id": tenant_ctx.tenant_id}

        except Exception as exc:
            log_warning(f"[celery/ingestion] Task failed for file_id={file_id}: {exc}")
            delay = 60
            cap = 3
            try:
                if tenant_id:
                    from app.core.config.client_config_resolver import get_client_config_for_ingestion

                    _cd = get_client_config_for_ingestion(str(tenant_id)).celery_dispatch
                    delay = int(_cd.retry_delay_seconds)
                    cap = int(_cd.max_retries)
            except Exception:
                pass
            if self.request.retries < cap:
                raise self.retry(exc=exc, countdown=delay)
            raise

    @celery_app.task(
        bind=True,
        name="tasks.run_external_ingestion_task",
        max_retries=50,
        default_retry_delay=120,
        queue="ingestion",
    )
    def run_external_ingestion_task(
        self,
        source_type: str,
        source_url: str,
        business_id,            # str | None — kept untyped for JSON serialisation
    ):
        """
        Fetch and ingest an external source (web page, RSS feed, or API endpoint).

        Arguments:
            source_type  — "web", "rss", or "api"
            source_url   — URL to fetch
            business_id  — Optional tenant UUID string (or None)
        """
        from app.services.ingestion.file_router_v2 import route_external_ingestion
        from app.utils.logger import log_info, log_warning

        tenant_id = business_id
        try:
            log_info(f"[celery/external] Ingesting {source_type.upper()} → {source_url}")
            result = _run(
                route_external_ingestion(
                    source_type=source_type,
                    source_url=source_url,
                    business_id=business_id,
                )
            )
            log_info(f"[celery/external] ✅ Done — {source_url}")
            return result
        except Exception as exc:
            log_warning(f"[celery/external] Task failed for {source_url}: {exc}")
            delay = 120
            cap = 3
            try:
                if tenant_id:
                    from app.core.config.client_config_resolver import get_client_config_for_ingestion

                    _cd = get_client_config_for_ingestion(str(tenant_id)).celery_dispatch
                    delay = int(_cd.retry_delay_seconds)
                    cap = int(_cd.max_retries)
            except Exception:
                pass
            if self.request.retries < cap:
                raise self.retry(exc=exc, countdown=delay)
            raise

    # ── VALIDATION TASKS (driven by Celery Beat) ─────────────────────────────

    @celery_app.task(
        name="tasks.run_agentic_validation",
        queue="validation",
        ignore_result=True,
    )
    def run_agentic_validation():
        """Run one batch of agentic validation (equivalent to the scheduler worker)."""
        from scripts.run_agentic_validation import run_validation
        _run(run_validation())

    @celery_app.task(
        name="tasks.run_conflict_detection",
        queue="validation",
        ignore_result=True,
    )
    def run_conflict_detection():
        """Run one batch of conflict detection."""
        from scripts.run_conflict_detection import run_conflict_detection as _detect
        _run(_detect())

    @celery_app.task(
        name="tasks.run_temporal_revalidation",
        queue="validation",
        ignore_result=True,
    )
    def run_temporal_revalidation():
        """Run one batch of temporal revalidation."""
        from scripts.run_temporal_revalidation import run_temporal_revalidation as _revalidate
        _run(_revalidate())
