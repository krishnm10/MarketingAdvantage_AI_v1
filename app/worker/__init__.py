# app/worker — Celery distributed task queue workers
# Import the app here so `celery -A app.worker worker` works automatically.
from app.worker.celery_app import celery_app  # noqa: F401
from app.worker.broker_config import is_celery_enabled  # noqa: F401
