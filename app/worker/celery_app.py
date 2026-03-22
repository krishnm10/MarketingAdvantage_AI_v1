# =============================================================================
# app/worker/celery_app.py — Celery Application Instance (Pluggable Broker)
# =============================================================================
#
# HOW TO LAUNCH A WORKER:
#   cd C:\ProjectK\MarketingAdvantage_AI_v1
#   .\.venv\Scripts\celery.exe -A app.worker worker --loglevel=info -Q ingestion,validation
#
# WINDOWS NOTE: The default pool is auto-set to "solo" (prefork crashes on Windows).
#   You can override with CELERY_WORKER_POOL=threads in .env if you need concurrency.
#
# HOW TO LAUNCH CELERY BEAT (periodic tasks — replaces the internal scheduler):
#   .\.venv\Scripts\celery.exe -A app.worker beat --loglevel=info
#
# QUEUES:
#   ingestion   — file upload & external-source ingestion tasks
#   validation  — agentic validation, conflict detection, temporal revalidation
#
# BROKER SELECTION — set in .env:
#   CELERY_ENABLED=true                    # false = inline ingestion (no broker)
#   CELERY_BROKER=redis                    # redis | rabbitmq | kafka | redpanda
#                                          # pulsar | nats | sqs | kinesis | ...
#   See broker_config.py for full list and per-broker env vars.
# =============================================================================

import os
import sys

from celery import Celery
from app.worker.broker_config import get_broker_config, is_celery_enabled

_BROKER_URL, _RESULT_BACKEND, _EXTRA_CONF = get_broker_config()

# ── Worker pool selection ────────────────────────────────────────────────────
# "prefork" (default on Linux/Mac) uses billiard sub-processes. It CRASHES on
# Windows with PermissionError in billiard's shared-memory semaphores.
# On Windows we default to "solo" (single-threaded) which is safe.
# Override via CELERY_WORKER_POOL env var.
_IS_WINDOWS = sys.platform == "win32"
_DEFAULT_POOL = "solo" if _IS_WINDOWS else "prefork"
_WORKER_POOL = os.getenv("CELERY_WORKER_POOL", _DEFAULT_POOL)
_WORKER_CONCURRENCY = int(os.getenv("CELERY_WORKER_CONCURRENCY", "4" if not _IS_WINDOWS else "1"))
_WORKER_LOGLEVEL = os.getenv("CELERY_WORKER_LOGLEVEL", "INFO").upper()
_WORKER_MAX_TASKS = int(os.getenv("CELERY_WORKER_MAX_TASKS_PER_CHILD", "0"))  # 0 = no limit
_WORKER_PREFETCH = int(os.getenv("CELERY_WORKER_PREFETCH_MULTIPLIER", "1"))
_TASK_SOFT_LIMIT = int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "0"))  # 0 = disabled
_TASK_HARD_LIMIT = int(os.getenv("CELERY_TASK_HARD_TIME_LIMIT", "0"))  # 0 = disabled
_TASK_MAX_RETRIES = int(os.getenv("CELERY_TASK_MAX_RETRIES", "3"))
_TASK_RETRY_DELAY = int(os.getenv("CELERY_TASK_RETRY_DELAY", "60"))
_RESULT_EXPIRES = int(os.getenv("CELERY_RESULT_EXPIRES", "86400"))
_WORKER_HEARTBEAT = os.getenv("CELERY_WORKER_DISABLE_HEARTBEAT", "true").lower() == "true"
_WORKER_GOSSIP = os.getenv("CELERY_WORKER_DISABLE_GOSSIP", "true").lower() == "true"
_WORKER_MINGLE = os.getenv("CELERY_WORKER_DISABLE_MINGLE", "true").lower() == "true"
_WORKER_QUEUES = os.getenv("CELERY_WORKER_QUEUES", "ingestion,validation")

celery_app = Celery(
    "mai_worker",
    broker=_BROKER_URL,
    backend=_RESULT_BACKEND,
    include=["app.worker.tasks"],
)

# Apply any broker-specific settings from the connector
if _EXTRA_CONF:
    celery_app.conf.update(**_EXTRA_CONF)

_conf: dict = {
    # ── Serialization ────────────────────────────────────────────────────────
    "task_serializer": "json",
    "result_serializer": "json",
    "accept_content": ["json"],

    # ── Time ─────────────────────────────────────────────────────────────────
    "timezone": "UTC",
    "enable_utc": True,

    # ── Worker pool ──────────────────────────────────────────────────────────
    "worker_pool": _WORKER_POOL,
    "worker_concurrency": _WORKER_CONCURRENCY,
    "worker_prefetch_multiplier": _WORKER_PREFETCH,
    "worker_disable_rate_limits": True,

    # ── Reliability ──────────────────────────────────────────────────────────
    "task_track_started": True,
    "task_acks_late": True,

    # ── Result expiry ────────────────────────────────────────────────────────
    "result_expires": _RESULT_EXPIRES,

    # ── Queue routing ────────────────────────────────────────────────────────
    "task_routes": {
        "tasks.run_ingestion_pipeline":       {"queue": "ingestion"},
        "tasks.run_external_ingestion_task":  {"queue": "ingestion"},
        "tasks.run_agentic_validation":       {"queue": "validation"},
        "tasks.run_conflict_detection":       {"queue": "validation"},
        "tasks.run_temporal_revalidation":    {"queue": "validation"},
    },

    # ── Beat schedule (periodic tasks — replaces the internal asyncio scheduler)
    # Enable by running: celery -A app.worker beat
    "beat_schedule": {
        "agentic-validation-every-60s": {
            "task": "tasks.run_agentic_validation",
            "schedule": 60.0,
            "options": {"queue": "validation"},
        },
        "conflict-detection-every-120s": {
            "task": "tasks.run_conflict_detection",
            "schedule": 120.0,
            "options": {"queue": "validation"},
        },
        "temporal-revalidation-every-300s": {
            "task": "tasks.run_temporal_revalidation",
            "schedule": 300.0,
            "options": {"queue": "validation"},
        },
    },
}

# ── Optional limits ──────────────────────────────────────────────────────────
if _WORKER_MAX_TASKS > 0:
    _conf["worker_max_tasks_per_child"] = _WORKER_MAX_TASKS
if _TASK_SOFT_LIMIT > 0:
    _conf["task_soft_time_limit"] = _TASK_SOFT_LIMIT
if _TASK_HARD_LIMIT > 0:
    _conf["task_time_limit"] = _TASK_HARD_LIMIT

celery_app.conf.update(**_conf)
