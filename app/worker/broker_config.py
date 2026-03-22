# =============================================================================
# app/worker/broker_config.py — Pluggable Broker Connector for Celery
# =============================================================================
#
# Works like your pluggable VectorDB system: change ONE env var to swap broker.
#
#   CELERY_ENABLED=true          # false = skip Celery, use inline ingestion
#   CELERY_BROKER=redis          # redis | rabbitmq | kafka | redpanda | ...
#
# Each broker type reads its own env vars for host/port/credentials.
# The connector builds the correct broker URL and any extra Celery config
# that the specific transport requires.
#
# SUPPORTED BROKERS:
#   Tier 1 — Native Celery/Kombu transports (zero extra libs):
#     redis        Redis  (default — already running for your vector cache)
#     rabbitmq     RabbitMQ / CloudAMQP / Amazon MQ  (AMQP 0-9-1)
#     sqs          Amazon SQS  (via kombu[sqs])
#
#   Tier 2 — Community transports (install one extra pip package):
#     kafka        Apache Kafka / Confluent Cloud
#     redpanda     Redpanda  (Kafka-API-compatible → same transport)
#     nats         NATS JetStream
#     pulsar       Apache Pulsar
#     warpstream   WarpStream  (Kafka-API-compatible → same transport)
#
#   Tier 3 — Cloud managed (route through their Kafka/AMQP adapter):
#     kinesis          Amazon Kinesis  (via Kafka-compatible consumer)
#     pubsub           Google Cloud Pub/Sub
#     eventhubs        Azure Event Hubs  (via AMQP 1.0 / Kafka adapter)
#     upstash          Upstash Redis / Kafka
#     redis_streams    Redis Streams  (same Redis transport, version 5+)
#     tinybird         Tinybird  (Kafka-compatible ingest endpoint)
#     glassflow        GlassFlow  (Kafka-compatible)
#     streamnative     StreamNative  (Pulsar-compatible)
#     aiven            Aiven for Kafka  (Kafka-compatible)
#
# ADDING A NEW BROKER:
#   1. Add a function `_build_<name>()` that returns (broker_url, result_backend, extra_conf)
#   2. Register it in the BROKER_BUILDERS dict at the bottom of this file
#   3. Add env var docs in .env — done.
# =============================================================================

from __future__ import annotations

import os
import logging
from typing import Any

from dotenv import load_dotenv

# Load .env so that CLI-launched workers (celery -A app.worker ...) pick up
# the same env vars that FastAPI reads via pydantic BaseSettings.
load_dotenv()

logger = logging.getLogger("marketing_advantage_ai")


# ---------------------------------------------------------------------------
# Helper: read an env var with a fallback
# ---------------------------------------------------------------------------
def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


# ---------------------------------------------------------------------------
# Per-broker builder functions
# Each returns: (broker_url, result_backend, extra_celery_conf_dict)
# ---------------------------------------------------------------------------

def _build_redis() -> tuple[str, str, dict[str, Any]]:
    """
    Redis — the default.  Reads the standard REDIS_* env vars you already have.

    Example .env:
        CELERY_BROKER=redis
        CELERY_REDIS_URL=redis://localhost:6379/0
    """
    url = _env("CELERY_REDIS_URL") or _env("REDIS_URL", "redis://localhost:6379/0")
    backend = _env("CELERY_RESULT_BACKEND", url)
    return url, backend, {}


def _build_redis_streams() -> tuple[str, str, dict[str, Any]]:
    """
    Redis Streams — same Redis transport but opt-in to Streams API.
    Requires Redis >= 5.0.

    Example .env:
        CELERY_BROKER=redis_streams
        CELERY_REDIS_URL=redis://localhost:6379/0
    """
    url = _env("CELERY_REDIS_URL") or _env("REDIS_URL", "redis://localhost:6379/0")
    backend = _env("CELERY_RESULT_BACKEND", url)
    return url, backend, {}


def _build_rabbitmq() -> tuple[str, str, dict[str, Any]]:
    """
    RabbitMQ — native AMQP transport.  Also works with CloudAMQP and Amazon MQ.

    Example .env:
        CELERY_BROKER=rabbitmq
        RABBITMQ_URL=amqp://guest:guest@localhost:5672//
    """
    url = _env("RABBITMQ_URL", "amqp://guest:guest@localhost:5672//")
    backend = _env("CELERY_RESULT_BACKEND") or _build_redis()[1]  # results still in Redis
    return url, backend, {
        "broker_connection_retry_on_startup": True,
    }


def _build_sqs() -> tuple[str, str, dict[str, Any]]:
    """
    Amazon SQS — via kombu SQS transport.
    pip install celery[sqs]

    Example .env:
        CELERY_BROKER=sqs
        AWS_ACCESS_KEY_ID=...
        AWS_SECRET_ACCESS_KEY=...
        SQS_REGION=us-east-1
        SQS_QUEUE_PREFIX=mai-
    """
    region = _env("SQS_REGION", "us-east-1")
    key_id = _env("AWS_ACCESS_KEY_ID")
    secret = _env("AWS_SECRET_ACCESS_KEY")
    prefix = _env("SQS_QUEUE_PREFIX", "mai-")

    if key_id and secret:
        url = f"sqs://{key_id}:{secret}@"
    else:
        url = "sqs://"  # uses IAM role from instance profile
    backend = _env("CELERY_RESULT_BACKEND") or _build_redis()[1]
    return url, backend, {
        "broker_transport_options": {
            "region": region,
            "queue_name_prefix": prefix,
        },
    }


def _build_kafka() -> tuple[str, str, dict[str, Any]]:
    """
    Apache Kafka / Confluent Cloud.
    pip install celery-kafka

    Example .env:
        CELERY_BROKER=kafka
        KAFKA_BOOTSTRAP_SERVERS=localhost:9092
        KAFKA_SECURITY_PROTOCOL=PLAINTEXT
        KAFKA_SASL_MECHANISM=
        KAFKA_SASL_USERNAME=
        KAFKA_SASL_PASSWORD=
    """
    servers = _env("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    url = f"confluentkafka://{servers}"
    backend = _env("CELERY_RESULT_BACKEND") or _build_redis()[1]

    transport_opts: dict[str, Any] = {
        "bootstrap.servers": servers,
        "security.protocol": _env("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"),
    }
    mech = _env("KAFKA_SASL_MECHANISM")
    if mech:
        transport_opts["sasl.mechanism"] = mech
        transport_opts["sasl.username"] = _env("KAFKA_SASL_USERNAME")
        transport_opts["sasl.password"] = _env("KAFKA_SASL_PASSWORD")

    return url, backend, {"broker_transport_options": transport_opts}


def _build_redpanda() -> tuple[str, str, dict[str, Any]]:
    """
    Redpanda — Kafka-API-compatible, uses the same Kafka transport.
    pip install celery-kafka

    Example .env:
        CELERY_BROKER=redpanda
        KAFKA_BOOTSTRAP_SERVERS=localhost:9092
    """
    return _build_kafka()  # Redpanda speaks the Kafka protocol


def _build_warpstream() -> tuple[str, str, dict[str, Any]]:
    """
    WarpStream — Kafka-API-compatible, uses the same Kafka transport.
    pip install celery-kafka

    Example .env:
        CELERY_BROKER=warpstream
        KAFKA_BOOTSTRAP_SERVERS=serverless.warpstream.com:9092
        KAFKA_SASL_MECHANISM=PLAIN
        KAFKA_SASL_USERNAME=...
        KAFKA_SASL_PASSWORD=...
    """
    return _build_kafka()  # WarpStream speaks the Kafka protocol


def _build_nats() -> tuple[str, str, dict[str, Any]]:
    """
    NATS JetStream.
    pip install celery-nats

    Example .env:
        CELERY_BROKER=nats
        NATS_URL=nats://localhost:4222
    """
    url = _env("NATS_URL", "nats://localhost:4222")
    backend = _env("CELERY_RESULT_BACKEND") or _build_redis()[1]
    return url, backend, {}


def _build_pulsar() -> tuple[str, str, dict[str, Any]]:
    """
    Apache Pulsar.
    pip install celery-pulsar

    Example .env:
        CELERY_BROKER=pulsar
        PULSAR_URL=pulsar://localhost:6650
    """
    url = _env("PULSAR_URL", "pulsar://localhost:6650")
    backend = _env("CELERY_RESULT_BACKEND") or _build_redis()[1]
    return url, backend, {}


def _build_streamnative() -> tuple[str, str, dict[str, Any]]:
    """
    StreamNative (managed Pulsar) — uses the Pulsar transport.

    Example .env:
        CELERY_BROKER=streamnative
        PULSAR_URL=pulsar+ssl://your-cluster.streamnative.cloud:6651
    """
    return _build_pulsar()


def _build_kinesis() -> tuple[str, str, dict[str, Any]]:
    """
    Amazon Kinesis — route through Kafka-compatible consumer adapter.
    pip install celery-kafka   (uses Enhanced Fan-Out → Kafka bridge)

    Example .env:
        CELERY_BROKER=kinesis
        KAFKA_BOOTSTRAP_SERVERS=<kinesis-kafka-bridge>:9092
    """
    return _build_kafka()


def _build_pubsub() -> tuple[str, str, dict[str, Any]]:
    """
    Google Cloud Pub/Sub — via kombu transport.
    pip install kombu[gcpubsub]

    Example .env:
        CELERY_BROKER=pubsub
        GOOGLE_CLOUD_PROJECT=your-project-id
        PUBSUB_SUBSCRIPTION_PREFIX=mai-
    """
    project = _env("GOOGLE_CLOUD_PROJECT", "my-project")
    url = f"gcpubsub://projects/{project}"
    backend = _env("CELERY_RESULT_BACKEND") or _build_redis()[1]
    return url, backend, {
        "broker_transport_options": {
            "queue_name_prefix": _env("PUBSUB_SUBSCRIPTION_PREFIX", "mai-"),
        },
    }


def _build_eventhubs() -> tuple[str, str, dict[str, Any]]:
    """
    Azure Event Hubs — via Kafka-compatible endpoint.
    pip install celery-kafka

    Example .env:
        CELERY_BROKER=eventhubs
        KAFKA_BOOTSTRAP_SERVERS=<namespace>.servicebus.windows.net:9093
        KAFKA_SECURITY_PROTOCOL=SASL_SSL
        KAFKA_SASL_MECHANISM=PLAIN
        KAFKA_SASL_USERNAME=$ConnectionString
        KAFKA_SASL_PASSWORD=Endpoint=sb://...
    """
    return _build_kafka()


def _build_upstash() -> tuple[str, str, dict[str, Any]]:
    """
    Upstash — can be used as Redis (serverless) or Kafka.
    Auto-detects from UPSTASH_BROKER_TYPE env var.

    Example .env — Redis mode:
        CELERY_BROKER=upstash
        UPSTASH_BROKER_TYPE=redis
        CELERY_REDIS_URL=rediss://default:xxxx@us1-xxx.upstash.io:6379

    Example .env — Kafka mode:
        CELERY_BROKER=upstash
        UPSTASH_BROKER_TYPE=kafka
        KAFKA_BOOTSTRAP_SERVERS=xxx.upstash.io:9092
        KAFKA_SASL_MECHANISM=SCRAM-SHA-256
        KAFKA_SASL_USERNAME=...
        KAFKA_SASL_PASSWORD=...
    """
    mode = _env("UPSTASH_BROKER_TYPE", "redis").lower()
    if mode == "kafka":
        return _build_kafka()
    return _build_redis()


def _build_tinybird() -> tuple[str, str, dict[str, Any]]:
    """
    Tinybird — Kafka-compatible ingest endpoint.
    pip install celery-kafka

    Example .env:
        CELERY_BROKER=tinybird
        KAFKA_BOOTSTRAP_SERVERS=kafka.tinybird.co:9092
        KAFKA_SASL_MECHANISM=PLAIN
        KAFKA_SASL_USERNAME=...
        KAFKA_SASL_PASSWORD=<TINYBIRD_TOKEN>
    """
    return _build_kafka()


def _build_glassflow() -> tuple[str, str, dict[str, Any]]:
    """
    GlassFlow — Kafka-compatible.
    pip install celery-kafka

    Example .env:
        CELERY_BROKER=glassflow
        KAFKA_BOOTSTRAP_SERVERS=<glassflow-endpoint>:9092
    """
    return _build_kafka()


def _build_aiven() -> tuple[str, str, dict[str, Any]]:
    """
    Aiven for Kafka — managed Kafka service.
    pip install celery-kafka

    Example .env:
        CELERY_BROKER=aiven
        KAFKA_BOOTSTRAP_SERVERS=<service>.aivencloud.com:12345
        KAFKA_SECURITY_PROTOCOL=SSL
    """
    return _build_kafka()


# ---------------------------------------------------------------------------
# BROKER REGISTRY — maps env-var names to builder functions
# ---------------------------------------------------------------------------

BROKER_BUILDERS: dict[str, callable] = {
    # Tier 1 — native Celery/Kombu
    "redis":            _build_redis,
    "rabbitmq":         _build_rabbitmq,
    "sqs":              _build_sqs,

    # Tier 2 — community transports
    "kafka":            _build_kafka,
    "redpanda":         _build_redpanda,
    "warpstream":       _build_warpstream,
    "nats":             _build_nats,
    "pulsar":           _build_pulsar,

    # Tier 3 — cloud managed / Kafka-compatible
    "kinesis":          _build_kinesis,
    "pubsub":           _build_pubsub,
    "eventhubs":        _build_eventhubs,
    "upstash":          _build_upstash,
    "redis_streams":    _build_redis_streams,
    "tinybird":         _build_tinybird,
    "glassflow":        _build_glassflow,
    "streamnative":     _build_streamnative,
    "aiven":            _build_aiven,
}

SUPPORTED_BROKERS = sorted(BROKER_BUILDERS.keys())

# Human-readable pip install hints for each broker
BROKER_PIP_HINTS: dict[str, str] = {
    "redis":            "celery[redis]          (already installed)",
    "rabbitmq":         "celery[librabbitmq]    or   pip install amqp",
    "sqs":              "celery[sqs]",
    "kafka":            "celery-kafka           or   confluent-kafka",
    "redpanda":         "celery-kafka           (Kafka-compatible)",
    "warpstream":       "celery-kafka           (Kafka-compatible)",
    "nats":             "celery-nats",
    "pulsar":           "celery-pulsar          or   pulsar-client",
    "kinesis":          "celery-kafka           (Kafka bridge required)",
    "pubsub":           "kombu[gcpubsub]",
    "eventhubs":        "celery-kafka           (Kafka endpoint)",
    "upstash":          "celery[redis] or celery-kafka (depends on UPSTASH_BROKER_TYPE)",
    "redis_streams":    "celery[redis]          (Redis >= 5.0)",
    "tinybird":         "celery-kafka           (Kafka-compatible)",
    "glassflow":        "celery-kafka           (Kafka-compatible)",
    "streamnative":     "celery-pulsar          (Pulsar-compatible)",
    "aiven":            "celery-kafka           (Kafka-compatible)",
}


# ---------------------------------------------------------------------------
# PUBLIC API — called by celery_app.py
# ---------------------------------------------------------------------------

def is_celery_enabled() -> bool:
    """
    Returns True only when Celery is explicitly enabled.
    Default is False — your inline ingestion runs instead (slower, but no broker needed).
    """
    return _env("CELERY_ENABLED", "false").lower() in ("true", "1", "yes")


def get_broker_config() -> tuple[str, str, dict[str, Any]]:
    """
    Read CELERY_BROKER from .env and return (broker_url, result_backend, extra_conf).
    Falls back to raw CELERY_BROKER_URL if CELERY_BROKER is not set.

    Raises ValueError if the broker name isn't recognized.
    """
    # Shortcut: if user set an explicit URL, honour it (backward-compatible)
    explicit_url = _env("CELERY_BROKER_URL")
    if explicit_url and not _env("CELERY_BROKER"):
        backend = _env("CELERY_RESULT_BACKEND", explicit_url)
        return explicit_url, backend, {}

    broker_name = _env("CELERY_BROKER", "redis").lower()
    builder = BROKER_BUILDERS.get(broker_name)

    if builder is None:
        raise ValueError(
            f"Unknown CELERY_BROKER={broker_name!r}.  "
            f"Supported: {', '.join(SUPPORTED_BROKERS)}"
        )

    broker_url, result_backend, extra = builder()

    logger.info(
        "[broker_config] CELERY_BROKER=%s  →  broker_url=%s  result_backend=%s",
        broker_name,
        # Mask credentials in logs
        broker_url.split("@")[-1] if "@" in broker_url else broker_url,
        result_backend.split("@")[-1] if "@" in result_backend else result_backend,
    )
    pip_hint = BROKER_PIP_HINTS.get(broker_name, "")
    if pip_hint:
        logger.info("[broker_config] Required package: %s", pip_hint)

    return broker_url, result_backend, extra
