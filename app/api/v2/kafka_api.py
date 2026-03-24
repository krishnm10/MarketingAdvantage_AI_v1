# =============================================================================
# app/api/v2/kafka_api.py — Kafka Management REST API
# =============================================================================
#
# Provides endpoints to manage, monitor, and interact with Kafka integration:
#
#   GET  /api/v2/kafka/health          → Kafka cluster health + producer stats
#   GET  /api/v2/kafka/topics          → List all topics with partition info
#   POST /api/v2/kafka/topics/ensure   → Create missing MAI topics
#   POST /api/v2/kafka/produce         → Publish a test/manual message
#   GET  /api/v2/kafka/config          → Current Kafka configuration (masked)
#   GET  /api/v2/kafka/stats           → Producer delivery statistics
#
# All endpoints are admin-only (require_role("admin")).
# =============================================================================

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.auth.guards import require_role

logger = logging.getLogger("marketing_advantage_ai.kafka_api")

router = APIRouter(
    prefix="/api/v2/kafka",
    tags=["Kafka"],
)


# ── Request / Response Models ────────────────────────────────────────────────

class ProduceRequest(BaseModel):
    topic: str
    value: Dict[str, Any]
    key: Optional[str] = None
    headers: Optional[Dict[str, str]] = None


class EnsureTopicsRequest(BaseModel):
    topics: Optional[List[str]] = None
    num_partitions: int = 0
    replication_factor: int = 0


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/health")
async def kafka_health(_=Depends(require_role("admin"))):
    """
    Comprehensive Kafka health check — broker connectivity, latency,
    topic count, and producer delivery statistics.
    """
    from app.services.kafka import kafka_service

    result = await kafka_service.health_check()
    return result


@router.get("/topics")
async def kafka_topics(_=Depends(require_role("admin"))):
    """List all non-internal Kafka topics with partition and replica details."""
    from app.services.kafka import kafka_service

    if not kafka_service.enabled:
        return {"status": "disabled", "message": "KAFKA_EVENTS_ENABLED is not true"}

    topics = await kafka_service.list_topics()
    return {
        "topic_count": len(topics),
        "topics": topics,
    }


@router.post("/topics/ensure")
async def kafka_ensure_topics(
    req: EnsureTopicsRequest,
    _=Depends(require_role("admin")),
):
    """
    Ensure all MAI Kafka topics exist.  Creates any missing topics with
    the specified partition count and replication factor.
    """
    from app.services.kafka import kafka_service

    if not kafka_service.enabled:
        raise HTTPException(400, "KAFKA_EVENTS_ENABLED is not true")

    result = await kafka_service.ensure_topics(
        topics=req.topics,
        num_partitions=req.num_partitions,
        replication_factor=req.replication_factor,
    )
    return {"results": result}


@router.post("/produce")
async def kafka_produce(
    req: ProduceRequest,
    _=Depends(require_role("admin")),
):
    """
    Publish a test/manual message to a Kafka topic.
    Useful for integration testing and debugging.
    """
    from app.services.kafka import kafka_service

    if not kafka_service.enabled:
        raise HTTPException(400, "KAFKA_EVENTS_ENABLED is not true")

    success = await kafka_service.produce(
        topic=req.topic,
        value=req.value,
        key=req.key,
        headers=req.headers,
    )
    if success:
        return {"status": "enqueued", "topic": req.topic}
    raise HTTPException(500, "Failed to enqueue message")


@router.get("/stats")
async def kafka_stats(_=Depends(require_role("admin"))):
    """Producer delivery statistics — messages produced, failed, DLQ count."""
    from app.services.kafka import kafka_service
    return kafka_service.get_stats()


@router.get("/config")
async def kafka_config(_=Depends(require_role("admin"))):
    """
    Return the current Kafka configuration (sensitive values masked).
    """
    def _env(key: str, default: str = "") -> str:
        return os.getenv(key, default).strip()

    def _mask(val: str) -> str:
        if not val or len(val) <= 4:
            return "****" if val else ""
        return val[:2] + "****" + val[-2:]

    return {
        "kafka_events_enabled": _env("KAFKA_EVENTS_ENABLED", "false"),
        "bootstrap_servers": _env("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        "security_protocol": _env("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"),
        "sasl_mechanism": _env("KAFKA_SASL_MECHANISM"),
        "sasl_username": _env("KAFKA_SASL_USERNAME"),
        "sasl_password": _mask(_env("KAFKA_SASL_PASSWORD")),
        "client_id": _env("KAFKA_CLIENT_ID", "mai-producer"),
        "compression_type": _env("KAFKA_COMPRESSION_TYPE", "lz4"),
        "enable_idempotence": _env("KAFKA_ENABLE_IDEMPOTENCE", "true"),
        "acks": _env("KAFKA_ACKS", "all"),
        "linger_ms": _env("KAFKA_LINGER_MS", "5"),
        "batch_size": _env("KAFKA_BATCH_SIZE", "65536"),
        "schema_registry_url": _env("KAFKA_SCHEMA_REGISTRY_URL"),
        "ssl_ca_location": _env("KAFKA_SSL_CA_LOCATION"),
        "ssl_certificate_location": _env("KAFKA_SSL_CERTIFICATE_LOCATION"),
        "consumer_group_id": _env("KAFKA_CONSUMER_GROUP_ID", "mai-consumer-group"),
        "auto_offset_reset": _env("KAFKA_AUTO_OFFSET_RESET", "earliest"),
        "default_partitions": _env("KAFKA_DEFAULT_PARTITIONS", "3"),
        "default_replication_factor": _env("KAFKA_DEFAULT_REPLICATION_FACTOR", "1"),
        "topic_retention_ms": _env("KAFKA_TOPIC_RETENTION_MS", "604800000"),
    }
