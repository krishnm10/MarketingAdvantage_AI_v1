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
#         IngestionServiceV2.ingest_parsed_output().
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
from app.worker.celery_app import celery_app


def _run(coro):
    """Run an async coroutine in a fresh event loop (Celery workers are sync)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ─────────────────────────────────────────────────────────────────────────────
# INGESTION TASKS
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="tasks.run_ingestion_pipeline",
    max_retries=3,
    default_retry_delay=60,
    queue="ingestion",
)
def run_ingestion_pipeline(self, file_id: str, saved_path: str, file_ext: str):
    """
    Parse a file that has already been saved to disk and ingest it.

    Arguments:
        file_id     — UUID string matching the IngestedFileV2 row already created.
        saved_path  — Absolute or relative path to the saved file on disk.
        file_ext    — Lowercase extension including dot, e.g. ".pdf".
    """
    from app.services.ingestion.file_router_v2 import PARSER_MAP
    from app.services.ingestion.ingestion_service_v2 import IngestionServiceV2
    from app.utils.logger import log_info, log_warning

    try:
        parser_func = PARSER_MAP.get(file_ext)
        if parser_func is None:
            raise ValueError(f"No parser registered for extension: {file_ext}")

        log_info(f"[celery/ingestion] Parsing file_id={file_id}  ext={file_ext}")
        parsed_output = _run(parser_func(saved_path))

        log_info(f"[celery/ingestion] Running ingestion pipeline for file_id={file_id}")
        _run(IngestionServiceV2.ingest_parsed_output(file_id, parsed_output))

        log_info(f"[celery/ingestion] ✅ Done — file_id={file_id}")
        return {"status": "ingested", "file_id": file_id}

    except Exception as exc:
        log_warning(f"[celery/ingestion] Task failed for file_id={file_id}: {exc}")
        raise self.retry(exc=exc)


@celery_app.task(
    bind=True,
    name="tasks.run_external_ingestion_task",
    max_retries=3,
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
        raise self.retry(exc=exc)


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION TASKS  (driven by Celery Beat — replaces the internal asyncio loop)
# ─────────────────────────────────────────────────────────────────────────────

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
