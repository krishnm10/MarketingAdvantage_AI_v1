"""
SecurityAuditLog — records PII detection and security middleware events.

Stores only entity TYPE names (e.g. "email", "ssn_us"), never raw PII values.
"""

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, func

from app.db.base import Base


class SecurityAuditLog(Base):
    __tablename__ = "security_audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(
        DateTime(timezone=True), server_default=func.now(), index=True,
    )
    business_id = Column(String(64), nullable=False, index=True)
    pipeline_id = Column(String(36), nullable=True)
    node_id = Column(String(36), nullable=True)
    position = Column(String(20), nullable=True)
    entities_detected = Column(Text, nullable=True)
    action_taken = Column(String(20), nullable=False)
    severity_max = Column(String(10), nullable=True)
    chunk_ids_affected = Column(Text, nullable=True)
    latency_ms = Column(Float, nullable=True)
    meta_data = Column(Text, nullable=True)
