# =============================================================================
# app/services/kafka/kafka_events.py — Typed Event Schemas for Kafka Topics
# =============================================================================
#
# Every message published to Kafka is a KafkaEvent subclass with a strict
# JSON-serializable schema.  This guarantees downstream consumers never need
# to guess the payload format.
#
# TOPICS MAP:
#   mai.ingestion.started      → IngestionEvent(stage="started")
#   mai.ingestion.completed    → IngestionEvent(stage="completed")
#   mai.ingestion.failed       → IngestionEvent(stage="failed")
#   mai.ingestion.chunked      → IngestionEvent(stage="chunked")
#   mai.validation.started     → ValidationEvent(stage="started")
#   mai.validation.completed   → ValidationEvent(stage="completed")
#   mai.validation.failed      → ValidationEvent(stage="failed")
#   mai.dlq                    → Dead-letter events (any failed delivery)
# =============================================================================

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class KafkaEvent:
    """Base event published to any Kafka topic."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    timestamp: float = field(default_factory=time.time)
    source: str = "marketing_advantage_ai"
    version: str = "1.0"
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IngestionEvent(KafkaEvent):
    """Event emitted at each stage of the ingestion pipeline."""

    event_type: str = "ingestion"
    file_id: str = ""
    file_name: str = ""
    file_ext: str = ""
    stage: str = ""  # started | chunked | completed | failed
    chunk_count: int = 0
    dedup_stats: Dict[str, Any] = field(default_factory=dict)
    business_id: Optional[str] = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None


@dataclass(frozen=True)
class ValidationEvent(KafkaEvent):
    """Event emitted during agentic validation / conflict detection."""

    event_type: str = "validation"
    stage: str = ""  # started | completed | failed
    validation_type: str = ""  # agentic | conflict | temporal
    batch_size: int = 0
    results_count: int = 0
    error: Optional[str] = None
    duration_ms: Optional[float] = None


@dataclass(frozen=True)
class ConnectorEvent(KafkaEvent):
    """Event emitted when an external connector ingests data."""

    event_type: str = "connector"
    connector_type: str = ""  # web | rss | api | kafka
    source_url: str = ""
    stage: str = ""  # started | completed | failed
    chunk_count: int = 0
    error: Optional[str] = None


# ─── Topic name helpers ──────────────────────────────────────────────────────

TOPIC_PREFIX = "mai"

TOPICS = {
    "ingestion.started":      f"{TOPIC_PREFIX}.ingestion.started",
    "ingestion.chunked":      f"{TOPIC_PREFIX}.ingestion.chunked",
    "ingestion.completed":    f"{TOPIC_PREFIX}.ingestion.completed",
    "ingestion.failed":       f"{TOPIC_PREFIX}.ingestion.failed",
    "validation.started":     f"{TOPIC_PREFIX}.validation.started",
    "validation.completed":   f"{TOPIC_PREFIX}.validation.completed",
    "validation.failed":      f"{TOPIC_PREFIX}.validation.failed",
    "connector.completed":    f"{TOPIC_PREFIX}.connector.completed",
    "connector.failed":       f"{TOPIC_PREFIX}.connector.failed",
    "dlq":                    f"{TOPIC_PREFIX}.dlq",
}

ALL_TOPICS: List[str] = list(TOPICS.values())
