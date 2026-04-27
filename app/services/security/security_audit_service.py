"""
SecurityAuditService — async service for recording PII/security events.

All business_id values are sanitized before storage. Entity detection results
are stored as type names only — raw PII is never persisted or logged.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

from sqlalchemy import select

from app.db.models.security_audit_log import SecurityAuditLog
from app.db.session_v2 import get_async_session
from app.utils.path_sanitizer import sanitize_client_id

logger = logging.getLogger(__name__)


class SecurityAuditService:
    """Async service for security audit log CRUD operations."""

    async def log_pii_event(
        self,
        business_id: str,
        pipeline_id: Optional[str] = None,
        node_id: Optional[str] = None,
        position: Optional[str] = None,
        entities_detected: Optional[List[str]] = None,
        action_taken: str = "REDACT",
        severity_max: Optional[str] = None,
        chunk_ids_affected: Optional[List[str]] = None,
        latency_ms: Optional[float] = None,
        meta_data: Optional[dict] = None,
    ) -> SecurityAuditLog:
        safe_biz_id = sanitize_client_id(business_id)

        record = SecurityAuditLog(
            business_id=safe_biz_id,
            pipeline_id=pipeline_id,
            node_id=node_id,
            position=position,
            entities_detected=json.dumps(entities_detected) if entities_detected else None,
            action_taken=action_taken,
            severity_max=severity_max,
            chunk_ids_affected=json.dumps(chunk_ids_affected) if chunk_ids_affected else None,
            latency_ms=latency_ms,
            meta_data=json.dumps(meta_data) if meta_data else None,
        )

        try:
            async with get_async_session() as session:
                session.add(record)
                await session.commit()
                await session.refresh(record)
                logger.info(
                    "Security audit event recorded: business=%s action=%s entities=%s",
                    safe_biz_id, action_taken,
                    entities_detected or [],
                )
        except Exception:
            logger.exception(
                "Failed to persist security audit event for business=%s",
                safe_biz_id,
            )
            raise

        return record

    async def get_recent_events(
        self, business_id: str, limit: int = 50,
    ) -> List[dict[str, Any]]:
        safe_biz_id = sanitize_client_id(business_id)

        async with get_async_session() as session:
            stmt = (
                select(SecurityAuditLog)
                .where(SecurityAuditLog.business_id == safe_biz_id)
                .order_by(SecurityAuditLog.timestamp.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

        return [
            {
                "id": row.id,
                "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                "business_id": row.business_id,
                "pipeline_id": row.pipeline_id,
                "node_id": row.node_id,
                "position": row.position,
                "entities_detected": json.loads(row.entities_detected)
                if row.entities_detected
                else [],
                "action_taken": row.action_taken,
                "severity_max": row.severity_max,
                "chunk_ids_affected": json.loads(row.chunk_ids_affected)
                if row.chunk_ids_affected
                else [],
                "latency_ms": row.latency_ms,
                "meta_data": json.loads(row.meta_data) if row.meta_data else None,
            }
            for row in rows
        ]
