import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    TIMESTAMP,
    JSON,
    Integer,
    Float,
    CheckConstraint,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base import Base


class IngestedFileV2(Base):
    """
    IngestedFileV2
    -----------------
    Tracks every uploaded or externally ingested file within the
    ingestion_v2 architecture.

    Each file record acts as a parent for multiple IngestedContentV2
    chunks and participates in deduplication tracking.
    """

    __tablename__ = "ingested_file"

    # -----------------------------------------------------------
    # PRIMARY KEYS + RELATIONSHIP CONTEXT
    # -----------------------------------------------------------
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    business_id = Column(UUID(as_uuid=True), nullable=True)

    # -----------------------------------------------------------
    # FILE DETAILS
    # -----------------------------------------------------------
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)  # pdf, docx, excel, csv, etc.
    file_path = Column(String(1024), nullable=True)
    source_url = Column(String(1024), nullable=True)
    source_type = Column(String(50), nullable=True)  # e.g., 'upload', 'web', 'rss', 'api'

    # -----------------------------------------------------------
    # METADATA
    # -----------------------------------------------------------
    # <<< PATCH: use callable default to avoid shared mutable dict across instances >>>
    meta_data = Column(JSON, default=dict)
    parser_used = Column(String(128), nullable=True)
    ingestion_notes = Column(String(512), nullable=True)

    # -----------------------------------------------------------
    # INGESTION METRICS
    # -----------------------------------------------------------
    total_chunks = Column(Integer, default=0)
    unique_chunks = Column(Integer, default=0)
    duplicate_chunks = Column(Integer, default=0)
    dedup_ratio = Column(Float, default=0.0)  # (duplicate_chunks / total_chunks) * 100

    # -----------------------------------------------------------
    # STATUS LIFECYCLE
    # -----------------------------------------------------------
    status = Column(
        String(50),
        default="pending",
        nullable=False,
        doc="Ingestion state: pending, processing, processed, error, archived",
    )

    error_message = Column(String(512), nullable=True)

    # -----------------------------------------------------------
    # TIMESTAMPS
    # -----------------------------------------------------------
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'processing', 'processed', 'error', 'archived')",
            name="check_ingested_file_v2_status",
        ),
    )

    # -----------------------------------------------------------
    # RELATIONSHIPS
    # -----------------------------------------------------------
    #chunks = relationship(
    #    "IngestedContentV2",
    #    back_populates="file",
    #    cascade="all, delete-orphan",
    #    lazy="selectin",
    #)

    # -----------------------------------------------------------
    # METHODS
    # -----------------------------------------------------------
    def update_dedup_stats(self):
        """Recalculate deduplication metrics."""
        if self.total_chunks:
            self.dedup_ratio = round(
                (self.duplicate_chunks / max(1, self.total_chunks)) * 100, 2
            )
        else:
            self.dedup_ratio = 0.0

    def __repr__(self):
        return (
            f"<IngestedFileV2(id={self.id}, file_name={self.file_name}, "
            f"status={self.status}, total_chunks={self.total_chunks}, "
            f"unique={self.unique_chunks}, duplicate={self.duplicate_chunks})>"
        )
