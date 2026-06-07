"""
Postgres-backed ingestion DLQ — canonical dead-letter store.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from opentelemetry import trace

from app.db.models.ingestion_dlq import IngestionDlq
from app.db.session_v2 import async_engine
from app.observability.metrics import record_dlq_write

logger = logging.getLogger(__name__)

_dlq_session_factory = async_sessionmaker(
    async_engine, expire_on_commit=False, autoflush=False
)

_MAX_ERROR_MESSAGE_LEN = 4000
_MAX_PAYLOAD_BYTES = 64_000


def _coerce_uuid(value: Union[str, uuid.UUID, None]) -> Optional[uuid.UUID]:
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError):
        return None


def _json_safe(value: Any, *, _depth: int = 0) -> Any:
    """Recursively coerce values to JSON-serializable primitives."""
    if _depth > 12:
        return "<max_depth>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (uuid.UUID, datetime)):
        return str(value)
    if isinstance(value, dict):
        return {
            str(k): _json_safe(v, _depth=_depth + 1)
            for k, v in value.items()
            if not str(k).startswith("_")
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v, _depth=_depth + 1) for v in value]
    return str(value)


def _sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    safe = _json_safe(payload)
    if not isinstance(safe, dict):
        safe = {"value": safe}
    try:
        encoded = json.dumps(safe, default=str)
        if len(encoded) > _MAX_PAYLOAD_BYTES:
            safe = {
                "truncated": True,
                "preview": encoded[:_MAX_PAYLOAD_BYTES],
            }
    except (TypeError, ValueError):
        safe = {"unserializable": True}
    return safe


async def record_ingestion_failure(
    *,
    business_id: Union[str, uuid.UUID],
    file_id: Union[str, uuid.UUID],
    stage: str,
    error: BaseException,
    payload_snapshot: Optional[Dict[str, Any]] = None,
    retry_count: int = 0,
    window_index: Optional[int] = None,
    status: str = "failed",
    db: Optional[AsyncSession] = None,
) -> Optional[uuid.UUID]:
    """
    Persist a durable DLQ tombstone. Commits when using the internal session.

    Returns the new row id, or None if persistence failed (errors are logged).
    """
    biz_uuid = _coerce_uuid(business_id)
    file_uuid = _coerce_uuid(file_id)
    if biz_uuid is None or file_uuid is None:
        logger.error(
            "DLQ write skipped: invalid tenant/file UUID business_id=%s file_id=%s stage=%s",
            business_id,
            file_id,
            stage,
        )
        return None

    row = IngestionDlq(
        business_id=biz_uuid,
        file_id=file_uuid,
        stage=str(stage)[:128],
        error_type=type(error).__name__[:128],
        error_message=str(error)[:_MAX_ERROR_MESSAGE_LEN],
        status=str(status)[:32],
        payload_snapshot=_sanitize_payload(payload_snapshot or {}),
        retry_count=int(retry_count),
        window_index=window_index,
        last_seen_at=datetime.utcnow(),
    )

    owns_session = db is None
    session = db
    if owns_session:
        session = _dlq_session_factory()

    assert session is not None
    try:
        session.add(row)
        await session.flush()
        if owns_session:
            await session.commit()
        # metrics + tracing (best-effort only)
        try:
            stage_label = str(stage)[:128]
            record_dlq_write(stage_label)
            tracer = trace.get_tracer("mai.ingestion.dlq")
            with tracer.start_as_current_span("ingestion.dlq.record_failure") as span:
                span.set_attribute("mai.stage", stage_label)
                if window_index is not None:
                    span.set_attribute("mai.window_index", int(window_index))
                span.set_attribute("mai.error_type", type(error).__name__)
        except Exception:
            pass
        logger.error(
            "Ingestion DLQ tombstone recorded business_id=%s file_id=%s stage=%s "
            "window_index=%s retry_count=%s error_type=%s",
            biz_uuid,
            file_uuid,
            stage,
            window_index,
            retry_count,
            type(error).__name__,
        )
        return row.id
    except Exception as dlq_err:
        if owns_session:
            await session.rollback()
        logger.error(
            "DLQ persistence failed business_id=%s file_id=%s stage=%s window_index=%s: %s",
            biz_uuid,
            file_uuid,
            stage,
            window_index,
            dlq_err,
        )
        return None
    finally:
        if owns_session:
            await session.close()


async def count_dlq_entries(*, business_id: Optional[Union[str, uuid.UUID]] = None) -> int:
    async with _dlq_session_factory() as db:
        query = select(func.count()).select_from(IngestionDlq)
        biz = _coerce_uuid(business_id)
        if biz is not None:
            query = query.where(IngestionDlq.business_id == biz)
        result = await db.execute(query)
        return int(result.scalar() or 0)


async def fetch_recent_dlq_entries(
    *,
    limit: int = 50,
    business_id: Optional[Union[str, uuid.UUID]] = None,
) -> List[Dict[str, Any]]:
    """Bounded recent failures for ops dashboards (newest first)."""
    limit = max(1, min(int(limit), 200))
    async with _dlq_session_factory() as db:
        query = (
            select(IngestionDlq)
            .order_by(IngestionDlq.created_at.desc())
            .limit(limit)
        )
        biz = _coerce_uuid(business_id)
        if biz is not None:
            query = query.where(IngestionDlq.business_id == biz)
        result = await db.execute(query)
        rows = result.scalars().all()
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "id": str(row.id),
                "business_id": str(row.business_id),
                "file_id": str(row.file_id),
                "stage": row.stage,
                "error_type": row.error_type,
                "error_message": row.error_message,
                "status": row.status,
                "payload_snapshot": row.payload_snapshot or {},
                "retry_count": row.retry_count,
                "window_index": row.window_index,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                "last_seen_at": row.last_seen_at.isoformat() if row.last_seen_at else None,
            }
        )
    return out
