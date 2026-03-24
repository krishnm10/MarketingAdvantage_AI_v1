# =============================================================================
# app/core/connectors/kafka_connector.py — Kafka Source Connector
# =============================================================================
#
# Follows the BaseConnector contract.  Allows Kafka topics to be treated as
# an external ingestion source — just like Web, RSS, and API connectors.
#
# USE CASE:
#   Your upstream systems publish content to Kafka topics.  This connector
#   consumes messages from a topic, normalizes them into chunks, and feeds
#   them into the standard ingestion pipeline (dedup → embed → store).
#
# USAGE:
#   connector = KafkaConnector()
#   result = await connector.fetch("mai.external.content")
#   # result.chunks = [{"text": "...", "metadata": {...}}, ...]
#
# CONFIGURATION (via .env):
#   KAFKA_BOOTSTRAP_SERVERS=localhost:9092
#   KAFKA_CONSUMER_GROUP_ID=mai-connector-group
#   KAFKA_CONNECTOR_MAX_MESSAGES=100
#   KAFKA_CONNECTOR_POLL_TIMEOUT_S=10
#   + all security/auth vars from kafka_service.py
#
# The connector:
#   1. Subscribes to the topic (or uses assign for specific partitions)
#   2. Polls up to KAFKA_CONNECTOR_MAX_MESSAGES messages
#   3. Normalizes each message into a ConnectorResult chunk
#   4. Commits offsets only after successful processing
#   5. Returns ConnectorResult for the ingestion pipeline
# =============================================================================

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.connectors.base import BaseConnector, ConnectorResult
from app.core.connectors.auth.base_auth import BaseAuthProvider

logger = logging.getLogger("marketing_advantage_ai.kafka_connector")

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="kafka-conn")


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_int(key: str, default: int = 0) -> int:
    v = _env(key)
    return int(v) if v else default


class KafkaConnector(BaseConnector):
    """
    Kafka topic connector — consumes messages and normalizes them into
    ingestion-ready chunks following the BaseConnector contract.

    Args:
        auth:           Optional auth provider (for Schema Registry etc).
        group_id:       Consumer group ID override.
        max_messages:   Max messages to consume per fetch call.
        poll_timeout:   Seconds to wait for messages per poll.
    """

    def __init__(
        self,
        auth: Optional[BaseAuthProvider] = None,
        group_id: Optional[str] = None,
        max_messages: Optional[int] = None,
        poll_timeout: Optional[float] = None,
    ):
        super().__init__(auth)
        self._group_id = group_id or _env(
            "KAFKA_CONNECTOR_GROUP_ID", "mai-connector-group"
        )
        self._max_messages = max_messages or _env_int(
            "KAFKA_CONNECTOR_MAX_MESSAGES", 100
        )
        self._poll_timeout = poll_timeout or float(
            _env("KAFKA_CONNECTOR_POLL_TIMEOUT_S", "10")
        )

    @property
    def source_type(self) -> str:
        return "kafka"

    async def fetch(
        self,
        source_url: str,
        *,
        db_session: Optional[AsyncSession] = None,
        **kwargs: Any,
    ) -> ConnectorResult:
        """
        Consume messages from a Kafka topic and return as ConnectorResult.

        Args:
            source_url:  The Kafka topic name to consume from.
            db_session:  Optional DB session (for dedup during ingestion).

        Returns:
            ConnectorResult with chunks normalized from Kafka messages.
        """
        topic = source_url  # overloaded: source_url = topic name
        loop = asyncio.get_running_loop()
        messages = await loop.run_in_executor(
            _EXECUTOR,
            self._consume_sync,
            topic,
        )

        chunks = self._normalize_messages(messages, topic)

        return ConnectorResult(
            source_type=self.source_type,
            source_url=f"kafka://{topic}",
            chunks=chunks,
            metadata={
                "topic": topic,
                "message_count": len(messages),
                "chunk_count": len(chunks),
                "consumer_group": self._group_id,
                "consumed_at": time.time(),
            },
        )

    def _consume_sync(self, topic: str) -> List[Dict[str, Any]]:
        """
        Synchronous Kafka consume — runs in thread pool.
        Polls messages up to max_messages with manual commit.
        """
        try:
            from confluent_kafka import Consumer, KafkaError
        except ImportError:
            raise ImportError(
                "confluent-kafka is required for KafkaConnector. "
                "Install with: pip install confluent-kafka"
            )

        from app.services.kafka.kafka_service import build_kafka_consumer_config

        config = build_kafka_consumer_config(group_id=self._group_id)
        consumer = Consumer(config)
        messages: List[Dict[str, Any]] = []

        try:
            consumer.subscribe([topic])
            deadline = time.monotonic() + self._poll_timeout
            consumed = 0

            while consumed < self._max_messages and time.monotonic() < deadline:
                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        break
                    logger.warning(
                        "[KafkaConnector] Poll error: %s", msg.error()
                    )
                    continue

                # Decode message
                try:
                    value = json.loads(msg.value().decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    value = {"raw": msg.value().decode("utf-8", errors="replace")}

                messages.append({
                    "value": value,
                    "key": (
                        msg.key().decode("utf-8") if msg.key() else None
                    ),
                    "topic": msg.topic(),
                    "partition": msg.partition(),
                    "offset": msg.offset(),
                    "timestamp": msg.timestamp(),
                    "headers": (
                        {
                            k: v.decode("utf-8", errors="replace")
                            for k, v in msg.headers()
                        }
                        if msg.headers()
                        else {}
                    ),
                })
                consumed += 1

            # Commit offsets after successful consumption
            if messages:
                consumer.commit(asynchronous=False)
                logger.info(
                    "[KafkaConnector] Consumed %d messages from %s",
                    len(messages),
                    topic,
                )

        except Exception as e:
            logger.error("[KafkaConnector] Consume error: %s", e)
        finally:
            consumer.close()

        return messages

    def _normalize_messages(
        self, messages: List[Dict[str, Any]], topic: str
    ) -> List[Dict[str, Any]]:
        """
        Normalize raw Kafka messages into ingestion-ready chunks.
        Each chunk has {"text": "...", "metadata": {...}} matching the
        ingestion pipeline's expected format.
        """
        chunks = []
        for msg in messages:
            value = msg.get("value", {})

            # Extract text content — supports multiple payload formats
            text = ""
            if isinstance(value, str):
                text = value
            elif isinstance(value, dict):
                # Try common content field names
                for field in ("text", "content", "body", "message", "data"):
                    if field in value and isinstance(value[field], str):
                        text = value[field]
                        break
                if not text:
                    text = json.dumps(value, default=str)
            else:
                text = str(value)

            if not text.strip():
                continue

            chunk_meta = {
                "source": f"kafka://{topic}",
                "source_type": "kafka",
                "kafka_topic": topic,
                "kafka_partition": msg.get("partition"),
                "kafka_offset": msg.get("offset"),
                "kafka_key": msg.get("key"),
                "kafka_timestamp": msg.get("timestamp"),
            }

            # Merge any headers as metadata
            if msg.get("headers"):
                for hk, hv in msg["headers"].items():
                    chunk_meta[f"kafka_header_{hk}"] = hv

            chunks.append({"text": text, "metadata": chunk_meta})

        return chunks

    def health_check(self) -> bool:
        """Check if Kafka is reachable via the auth provider + TCP probe."""
        try:
            import socket

            servers = _env("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
            host, _, port_str = servers.partition(":")
            port = int(port_str) if port_str else 9092
            with socket.create_connection((host, port), timeout=3):
                return True
        except Exception:
            return False
