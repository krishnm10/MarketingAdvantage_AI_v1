# =============================================================================
# app/services/kafka — Enterprise Apache Kafka Integration
# =============================================================================
#
# Production-grade Kafka producer, consumer, and admin management.
# Follows the same pluggable connector pattern as VectorDB/Embedder/LLM.
#
# Public API:
#   from app.services.kafka import kafka_service, KafkaEvent
#   await kafka_service.produce("ingestion.completed", payload)
#   await kafka_service.health_check()
# =============================================================================

from app.services.kafka.kafka_service import kafka_service, KafkaService
from app.services.kafka.kafka_events import KafkaEvent, IngestionEvent, ValidationEvent

__all__ = [
    "kafka_service",
    "KafkaService",
    "KafkaEvent",
    "IngestionEvent",
    "ValidationEvent",
]
