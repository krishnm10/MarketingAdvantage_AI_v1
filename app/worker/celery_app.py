# =============================================================================
# app/worker/celery_app.py — Celery Application Instance (Pluggable Broker)
# =============================================================================
#
# HOW TO LAUNCH A WORKER:
#   cd C:\ProjectK\MarketingAdvantage_AI_v1
#   .\.venv\Scripts\celery.exe -A app.worker worker --loglevel=info -Q ingestion,validation
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

from celery import Celery
from app.worker.broker_config import get_broker_config, is_celery_enabled

_BROKER_URL, _RESULT_BACKEND, _EXTRA_CONF = get_broker_config()

celery_app = Celery(
    "mai_worker",
    broker=_BROKER_URL,
    backend=_RESULT_BACKEND,
    include=["app.worker.tasks"],
)

# Apply any broker-specific settings from the connector
if _EXTRA_CONF:
    celery_app.conf.update(**_EXTRA_CONF)

celery_app.conf.update(
    # ── Serialization ────────────────────────────────────────────────────────
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # ── Time ─────────────────────────────────────────────────────────────────
    timezone="UTC",
    enable_utc=True,

    # ── Reliability ──────────────────────────────────────────────────────────
    task_track_started=True,
    task_acks_late=True,                   # ACK only after the task completes
    worker_prefetch_multiplier=1,          # One active task per worker = fair

    # ── Result expiry ────────────────────────────────────────────────────────
    result_expires=86_400,                 # Keep task results for 24 h in Redis

    # ── Queue routing ────────────────────────────────────────────────────────
    task_routes={
        "tasks.run_ingestion_pipeline":       {"queue": "ingestion"},
        "tasks.run_external_ingestion_task":  {"queue": "ingestion"},
        "tasks.run_agentic_validation":       {"queue": "validation"},
        "tasks.run_conflict_detection":       {"queue": "validation"},
        "tasks.run_temporal_revalidation":    {"queue": "validation"},
    },

    # ── Beat schedule (periodic tasks — replaces the internal asyncio scheduler)
    # Enable by running: celery -A app.worker beat
    beat_schedule={
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
)
