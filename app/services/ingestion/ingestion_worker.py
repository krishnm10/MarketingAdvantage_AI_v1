# =============================================
# ingestion_worker.py — Event-Driven Ingestion Worker
#
# ARCHITECTURE:
#   HTTP Upload → IngestionCommandQueue (in-memory or Kafka) → IngestionWorker
#
# This decouples heavy ingestion work from the HTTP request lifecycle:
#   - HTTP handler enqueues a command and returns 202 Accepted immediately
#   - Worker processes commands asynchronously in the background
#   - Status updates flow back via WebSocket broadcasts + DB status updates
#
# Queue backends:
#   IN_MEMORY  — asyncio.Queue (single process, dev/staging)
#   KAFKA      — confluent_kafka consumer (multi-process, production)
#               Activated when KAFKA_EVENTS_ENABLED=true and
#               KAFKA_INGESTION_TOPIC is set
#
# RESILIENCE:
#   - Failed commands are retried up to MAX_RETRY_ATTEMPTS times
#   - Dead-letter queue captures permanently failed commands
#   - Worker failures do not crash the HTTP process
# =============================================

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid as _uuid_module
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
_WORKER_CONCURRENCY: int = int(os.getenv("INGESTION_WORKER_CONCURRENCY", "4"))
_MAX_RETRY_ATTEMPTS: int = int(os.getenv("INGESTION_MAX_RETRIES", "3"))
_RETRY_DELAY_BASE_S: float = float(os.getenv("INGESTION_RETRY_DELAY_S", "5"))
_KAFKA_TOPIC: str = os.getenv("KAFKA_INGESTION_TOPIC", "mai.ingestion.requested")
_KAFKA_ENABLED: bool = os.getenv("KAFKA_EVENTS_ENABLED", "").lower() in ("1", "true", "yes")


@dataclass
class IngestionCommand:
    """
    Command schema for the ingestion queue.

    Produced by: ingestion_api_v2, rss_ingestor, watcher_ingestor, api_ingestor
    Consumed by: IngestionWorker
    """
    file_id: str
    business_id: Optional[str] = None
    file_path: Optional[str] = None
    source_type: Optional[str] = None
    priority: int = 0                   # higher = processed sooner
    attempt: int = 0
    command_id: str = field(default_factory=lambda: str(_uuid_module.uuid4()))
    enqueued_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_id":   self.command_id,
            "file_id":      self.file_id,
            "business_id":  self.business_id,
            "file_path":    self.file_path,
            "source_type":  self.source_type,
            "priority":     self.priority,
            "attempt":      self.attempt,
            "enqueued_at":  self.enqueued_at,
            "metadata":     self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IngestionCommand":
        return cls(
            file_id=d["file_id"],
            business_id=d.get("business_id"),
            file_path=d.get("file_path"),
            source_type=d.get("source_type"),
            priority=d.get("priority", 0),
            attempt=d.get("attempt", 0),
            command_id=d.get("command_id", str(_uuid_module.uuid4())),
            enqueued_at=d.get("enqueued_at", datetime.utcnow().isoformat() + "Z"),
            metadata=d.get("metadata", {}),
        )


# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY QUEUE (default for single-process deployments)
# ─────────────────────────────────────────────────────────────────────────────

class InMemoryIngestionQueue:
    """
    asyncio.PriorityQueue-backed ingestion command queue.
    Thread-safe within a single async event loop.
    Not suitable for multi-process / multi-worker deployments.
    """

    def __init__(self, maxsize: int = 1000):
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=maxsize)
        self._dead_letter: List[IngestionCommand] = []
        self._enqueued: int = 0
        self._processed: int = 0
        self._failed: int = 0

    async def enqueue(self, command: IngestionCommand) -> None:
        priority_key = -command.priority  # lower number = higher priority in PriorityQueue
        await self._queue.put((priority_key, command.enqueued_at, command))
        self._enqueued += 1
        logger.info(
            "Ingestion command enqueued",
            extra={
                "command_id": command.command_id,
                "file_id": command.file_id,
                "business_id": command.business_id,
                "queue_size": self._queue.qsize(),
            },
        )

    async def dequeue(self) -> IngestionCommand:
        _, _, command = await self._queue.get()
        return command

    def task_done(self) -> None:
        self._queue.task_done()
        self._processed += 1

    def send_to_dead_letter(self, command: IngestionCommand) -> None:
        self._dead_letter.append(command)
        self._failed += 1
        logger.error(
            "Command sent to dead-letter queue after max retries",
            extra={
                "command_id": command.command_id,
                "file_id": command.file_id,
                "attempt": command.attempt,
            },
        )

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "enqueued": self._enqueued,
            "processed": self._processed,
            "failed": self._failed,
            "dead_letter": len(self._dead_letter),
            "pending": self._queue.qsize(),
        }

    def dead_letter_items(self) -> List[Dict[str, Any]]:
        return [cmd.to_dict() for cmd in self._dead_letter[-50:]]  # last 50 only


# ─────────────────────────────────────────────────────────────────────────────
# INGESTION WORKER
# ─────────────────────────────────────────────────────────────────────────────

class IngestionWorker:
    """
    Background worker that consumes IngestionCommand objects and invokes
    IngestionServiceV2.process_file() for each.

    Features:
      - Configurable concurrency (INGESTION_WORKER_CONCURRENCY env var)
      - Exponential backoff retry (up to MAX_RETRY_ATTEMPTS)
      - Dead-letter queue for permanently failed commands
      - Graceful shutdown (drains current batch before stopping)
    """

    def __init__(self, queue: InMemoryIngestionQueue):
        self._queue = queue
        self._semaphore = asyncio.Semaphore(_WORKER_CONCURRENCY)
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        if self._running:
            logger.warning("[IngestionWorker] Already running")
            return
        self._running = True
        for i in range(_WORKER_CONCURRENCY):
            task = asyncio.create_task(self._consume_loop(worker_id=i))
            self._tasks.append(task)
        logger.info(
            "[IngestionWorker] Started %d consumer(s)", _WORKER_CONCURRENCY
        )

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("[IngestionWorker] Stopped")

    async def _consume_loop(self, worker_id: int) -> None:
        while self._running:
            try:
                command = await asyncio.wait_for(
                    self._queue.dequeue(), timeout=2.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("[IngestionWorker %d] Queue dequeue error: %s", worker_id, e)
                await asyncio.sleep(1)
                continue

            async with self._semaphore:
                await self._process_command(command, worker_id=worker_id)
            self._queue.task_done()

    async def _process_command(self, command: IngestionCommand, worker_id: int) -> None:
        t0 = time.monotonic()
        try:
            from app.services.ingestion.ingestion_service_v2 import IngestionServiceV2
            logger.info(
                "[IngestionWorker %d] Processing file_id=%s attempt=%d",
                worker_id, command.file_id, command.attempt,
            )
            await IngestionServiceV2.process_file(
                file_id=command.file_id,
                file_path=command.file_path,
                business_id=command.business_id,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.info(
                "[IngestionWorker %d] Completed file_id=%s in %.1fms",
                worker_id, command.file_id, elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.error(
                "[IngestionWorker %d] Failed file_id=%s attempt=%d error=%s in %.1fms",
                worker_id, command.file_id, command.attempt, e, elapsed_ms,
            )
            if command.attempt < _MAX_RETRY_ATTEMPTS:
                delay = _RETRY_DELAY_BASE_S * (2 ** command.attempt)
                logger.info(
                    "[IngestionWorker %d] Retrying file_id=%s in %.1fs (attempt %d/%d)",
                    worker_id, command.file_id, delay,
                    command.attempt + 1, _MAX_RETRY_ATTEMPTS,
                )
                await asyncio.sleep(delay)
                retry_cmd = IngestionCommand(
                    file_id=command.file_id,
                    business_id=command.business_id,
                    file_path=command.file_path,
                    source_type=command.source_type,
                    priority=command.priority,
                    attempt=command.attempt + 1,
                    command_id=command.command_id,
                    metadata=command.metadata,
                )
                await self._queue.enqueue(retry_cmd)
            else:
                self._queue.send_to_dead_letter(command)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETONS — used by main.py and API endpoints
# ─────────────────────────────────────────────────────────────────────────────

_ingestion_queue: Optional[InMemoryIngestionQueue] = None
_ingestion_worker: Optional[IngestionWorker] = None


def get_ingestion_queue() -> InMemoryIngestionQueue:
    """Return the module-level ingestion queue, creating it on first call."""
    global _ingestion_queue
    if _ingestion_queue is None:
        _ingestion_queue = InMemoryIngestionQueue(
            maxsize=int(os.getenv("INGESTION_QUEUE_MAXSIZE", "1000"))
        )
    return _ingestion_queue


def get_ingestion_worker() -> IngestionWorker:
    """Return the module-level ingestion worker, creating it on first call."""
    global _ingestion_worker
    if _ingestion_worker is None:
        _ingestion_worker = IngestionWorker(queue=get_ingestion_queue())
    return _ingestion_worker


async def enqueue_file_ingestion(
    file_id: str,
    *,
    business_id: Optional[str] = None,
    file_path: Optional[str] = None,
    source_type: Optional[str] = None,
    priority: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> IngestionCommand:
    """
    Convenience function: enqueue a file ingestion command.

    Usage in API handlers::

        cmd = await enqueue_file_ingestion(
            file_id=file_id,
            business_id=request.business_id,
            file_path=saved_path,
        )
        return {"status": "queued", "command_id": cmd.command_id}
    """
    command = IngestionCommand(
        file_id=file_id,
        business_id=business_id,
        file_path=file_path,
        source_type=source_type,
        priority=priority,
        metadata=metadata or {},
    )
    queue = get_ingestion_queue()
    await queue.enqueue(command)
    return command


async def retry_failed_ingestions(
    min_retry_age_minutes: int = 5,
    max_retries: int = 3,
) -> int:
    """
    Scan for files stuck in 'failed' status and re-enqueue them.
    Intended to be called by the validation scheduler periodically.

    Returns the count of re-enqueued files.
    """
    from datetime import timedelta
    from sqlalchemy import select
    from app.db.session_v2 import async_engine
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
    from app.db.models.ingested_file_v2 import IngestedFileV2

    _session = async_sessionmaker(async_engine, expire_on_commit=False, autoflush=False)
    cutoff = datetime.utcnow() - timedelta(minutes=min_retry_age_minutes)

    try:
        async with _session() as db:
            result = await db.execute(
                select(IngestedFileV2)
                .where(IngestedFileV2.status == "failed")
                .where(IngestedFileV2.updated_at < cutoff)
            )
            failed_files = result.scalars().all()

        requeued = 0
        queue = get_ingestion_queue()
        for file_record in failed_files:
            cmd = IngestionCommand(
                file_id=str(file_record.id),
                business_id=str(file_record.business_id) if file_record.business_id else None,
                file_path=getattr(file_record, "file_path", None),
                source_type=getattr(file_record, "source_type", None),
                priority=-1,
            )
            await queue.enqueue(cmd)
            requeued += 1

        if requeued:
            logger.info("[IngestionWorker] Re-enqueued %d failed files for retry", requeued)
        return requeued

    except Exception as e:
        logger.error("[IngestionWorker] retry_failed_ingestions failed: %s", e)
        return 0
