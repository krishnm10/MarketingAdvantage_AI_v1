# app/worker — Celery distributed task queue workers
#
# When CELERY_ENABLED=true  → imports the real Celery app & tasks so that
#   `celery -A app.worker worker` works automatically.
# When CELERY_ENABLED=false → exports None for celery_app; no broker
#   connection is attempted at import time.

from app.worker.broker_config import is_celery_enabled  # noqa: F401

if is_celery_enabled():
    from app.worker.celery_app import celery_app  # noqa: F401
else:
    celery_app = None  # type: ignore[assignment]  # noqa: F401
