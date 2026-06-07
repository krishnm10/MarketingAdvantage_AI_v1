"""
Durable dead-letter queue for permanently failed ingestion work items.
"""

from __future__ import annotations

import uuid

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func

from app.db.base import Base


class IngestionDlq(Base):
    __tablename__ = "ingestion_dlq"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    business_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    file_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    stage = Column(String(128), nullable=False)
    error_type = Column(String(128), nullable=False)
    error_message = Column(Text, nullable=False)
    status = Column(String(32), nullable=False, default="failed", server_default="failed")

    payload_snapshot = Column(JSONB, nullable=False, default=dict)
    retry_count = Column(Integer, nullable=False, default=0, server_default="0")
    window_index = Column(Integer, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    last_seen_at = Column(DateTime(timezone=True), server_default=func.now())
