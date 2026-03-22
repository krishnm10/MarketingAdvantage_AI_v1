# =============================================================================
# app/core/settings.py — Centralized Application Settings
# =============================================================================
#
# Uses pydantic-settings BaseSettings for type-safe, validated configuration
# loaded from environment variables and .env file.
#
# BACKWARD COMPATIBLE — all existing os.getenv() calls continue to work.
# This module is additive: new code can import `settings` for type safety;
# existing code is not required to migrate immediately.
#
# Usage:
#   from app.core.settings import settings
#   settings.jwt_secret_key   # type: str, validated at startup
#
# Environment variables are loaded from `.env` file at project root.
# All variable names match the existing os.getenv() keys exactly.
# =============================================================================

from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """
    Single source of truth for all application configuration.
    Every field maps 1-to-1 with an existing os.getenv() call in the codebase.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,   # MAI_VECTORDB and mai_vectordb both work
        extra="ignore",         # Unknown env vars are silently ignored
    )

    # ── JWT / Auth ────────────────────────────────────────────────────────────
    jwt_secret_key: str = Field(default="", alias="JWT_SECRET_KEY")
    jwt_secret: str = Field(default="", alias="JWT_SECRET")          # legacy alias
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, alias="ACCESS_TOKEN_EXPIRE_MINUTES")

    @model_validator(mode="after")
    def _validate_jwt_secret(self) -> "AppSettings":
        """Mirror the same validation logic already in generate_token.py."""
        secret = self.jwt_secret_key or self.jwt_secret
        banned = {
            "supersecretkey", "your_strong_secret_here", "changeme",
            "REPLACE_WITH_GENERATED_SECRET", "your_jwt_secret_here",
        }
        if secret and secret not in banned:
            # Valid — write back to the canonical field
            self.jwt_secret_key = secret
        return self

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://localhost/marketingai",
        alias="DATABASE_URL",
    )

    # ── Vector DB ─────────────────────────────────────────────────────────────
    mai_vectordb: str = Field(default="chroma", alias="MAI_VECTORDB")

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

    # ── Validation Scheduler ─────────────────────────────────────────────────
    validation_interval: int = Field(default=60, alias="VALIDATION_INTERVAL")
    conflict_interval: int = Field(default=120, alias="CONFLICT_INTERVAL")
    temporal_interval: int = Field(default=300, alias="TEMPORAL_INTERVAL")
    validation_batch_size: int = Field(default=50, alias="VALIDATION_BATCH_SIZE")
    conflict_batch_size: int = Field(default=30, alias="CONFLICT_BATCH_SIZE")
    temporal_batch_size: int = Field(default=50, alias="TEMPORAL_BATCH_SIZE")

    # Scheduler feature flags
    enable_agentic_validation: bool = Field(default=True, alias="ENABLE_AGENTIC_VALIDATION")
    enable_conflict_analysis: bool = Field(default=True, alias="ENABLE_CONFLICT_ANALYSIS")
    enable_temporal_revalidation: bool = Field(default=True, alias="ENABLE_TEMPORAL_REVALIDATION")

    # ── Observability — Sentry ────────────────────────────────────────────────
    sentry_dsn: str = Field(default="", alias="SENTRY_DSN")
    environment: str = Field(default="production", alias="ENVIRONMENT")
    sentry_traces_sample_rate: float = Field(default=0.1, alias="SENTRY_TRACES_SAMPLE_RATE")

    # ── Structured Logging ────────────────────────────────────────────────────
    # Set LOG_FORMAT=json in production for JSON lines (Datadog / Splunk / CloudWatch)
    # Leave unset (or "text") for the existing colored console output
    log_format: str = Field(default="text", alias="LOG_FORMAT")

    # ── Rate Limiting (slowapi) ───────────────────────────────────────────────
    # Default rate limit applied to all endpoints.
    # Format: "<count>/<period>"  e.g. "200/minute" "1000/hour"
    rate_limit_default: str = Field(default="200/minute", alias="RATE_LIMIT_DEFAULT")
    # Redis URL for distributed rate-limit counters.
    # Use "memory://" (default) for single-instance deployments.
    # Use "redis://localhost:6379" when Redis is available.
    redis_url: str = Field(default="memory://", alias="REDIS_URL")
    # ── Celery (Distributed Task Queue) ───────────────────────────────────────
    # Master switch: set to true to offload work to Celery workers.
    # When false (default) the server uses inline ingestion — slower but needs no broker.
    celery_enabled: bool = Field(default=False, alias="CELERY_ENABLED")

    # Broker type — determines which message broker Celery connects to.
    # Supported: redis | rabbitmq | kafka | redpanda | pulsar | nats | sqs
    #            kinesis | pubsub | eventhubs | upstash | redis_streams
    #            tinybird | glassflow | streamnative | aiven | warpstream
    # See app/worker/broker_config.py for per-broker env vars & pip hints.
    celery_broker: str = Field(default="redis", alias="CELERY_BROKER")

    # Legacy / explicit override — if set, these take precedence over CELERY_BROKER
    celery_broker_url: str = Field(
        default="", alias="CELERY_BROKER_URL"
    )
    celery_result_backend: str = Field(
        default="", alias="CELERY_RESULT_BACKEND"
    )

    # ── Broker-specific connection vars ───────────────────────────────────────
    # Redis (also used by redis_streams)
    celery_redis_url: str = Field(default="", alias="CELERY_REDIS_URL")

    # RabbitMQ
    rabbitmq_url: str = Field(default="amqp://guest:guest@localhost:5672//", alias="RABBITMQ_URL")

    # Kafka / Redpanda / WarpStream / Aiven / Tinybird / GlassFlow / EventHubs / Kinesis
    kafka_bootstrap_servers: str = Field(default="localhost:9092", alias="KAFKA_BOOTSTRAP_SERVERS")
    kafka_security_protocol: str = Field(default="PLAINTEXT", alias="KAFKA_SECURITY_PROTOCOL")
    kafka_sasl_mechanism: str = Field(default="", alias="KAFKA_SASL_MECHANISM")
    kafka_sasl_username: str = Field(default="", alias="KAFKA_SASL_USERNAME")
    kafka_sasl_password: str = Field(default="", alias="KAFKA_SASL_PASSWORD")

    # NATS JetStream
    nats_url: str = Field(default="nats://localhost:4222", alias="NATS_URL")

    # Apache Pulsar / StreamNative
    pulsar_url: str = Field(default="pulsar://localhost:6650", alias="PULSAR_URL")

    # Amazon SQS
    sqs_region: str = Field(default="us-east-1", alias="SQS_REGION")
    sqs_queue_prefix: str = Field(default="mai-", alias="SQS_QUEUE_PREFIX")

    # Google Cloud Pub/Sub
    google_cloud_project: str = Field(default="", alias="GOOGLE_CLOUD_PROJECT")
    pubsub_subscription_prefix: str = Field(default="mai-", alias="PUBSUB_SUBSCRIPTION_PREFIX")

    # Upstash (redis or kafka)
    upstash_broker_type: str = Field(default="redis", alias="UPSTASH_BROKER_TYPE")

    # ── Celery Worker Configuration ───────────────────────────────────────────
    # Pool type: solo (safe on Windows), prefork (default Linux/Mac), threads, gevent, eventlet
    celery_worker_pool: str = Field(default="solo", alias="CELERY_WORKER_POOL")
    celery_worker_concurrency: int = Field(default=4, alias="CELERY_WORKER_CONCURRENCY")
    celery_worker_loglevel: str = Field(default="INFO", alias="CELERY_WORKER_LOGLEVEL")
    celery_worker_queues: str = Field(default="ingestion,validation", alias="CELERY_WORKER_QUEUES")
    celery_worker_max_tasks_per_child: int = Field(default=0, alias="CELERY_WORKER_MAX_TASKS_PER_CHILD")
    celery_worker_prefetch_multiplier: int = Field(default=1, alias="CELERY_WORKER_PREFETCH_MULTIPLIER")
    celery_task_soft_time_limit: int = Field(default=0, alias="CELERY_TASK_SOFT_TIME_LIMIT")
    celery_task_hard_time_limit: int = Field(default=0, alias="CELERY_TASK_HARD_TIME_LIMIT")
    celery_task_max_retries: int = Field(default=3, alias="CELERY_TASK_MAX_RETRIES")
    celery_task_retry_delay: int = Field(default=60, alias="CELERY_TASK_RETRY_DELAY")
    celery_result_expires: int = Field(default=86400, alias="CELERY_RESULT_EXPIRES")
    celery_worker_disable_heartbeat: bool = Field(default=True, alias="CELERY_WORKER_DISABLE_HEARTBEAT")
    celery_worker_disable_gossip: bool = Field(default=True, alias="CELERY_WORKER_DISABLE_GOSSIP")
    celery_worker_disable_mingle: bool = Field(default=True, alias="CELERY_WORKER_DISABLE_MINGLE")

    # ── PHANTOM Hardware Tuning ───────────────────────────────────────────────
    phantom_embed_batch_size: str = Field(default="", alias="PHANTOM_EMBED_BATCH_SIZE")
    phantom_upsert_batch_size: str = Field(default="", alias="PHANTOM_UPSERT_BATCH_SIZE")
    phantom_ingest_workers: str = Field(default="", alias="PHANTOM_INGEST_WORKERS")
    phantom_bloom_capacity: str = Field(default="", alias="PHANTOM_BLOOM_CAPACITY")


# ---------------------------------------------------------------------------
# Singleton — imported once at module load time.
# The .env file is read exactly once; subsequent imports use the cached object.
# ---------------------------------------------------------------------------
settings = AppSettings()
