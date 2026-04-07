import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Text,
    JSON,
    ForeignKey,
    TIMESTAMP,
    Integer,
    Float,
    Boolean,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.db.base import Base


class IngestedContentV2(Base):
    """
    IngestedContentV2
    -----------------
    Stores semantically chunked content derived from ingested files.
    Each chunk links to:
        - ingested_file (source document)
        - business (optional tenant)
        - global_content_index (for deduplication tracking)

    Enhanced for ingestion_v2:
      • Added duplicate tracking fields
      • Added semantic confidence score
      • Fully async-compatible ORM relationships
    """

    __tablename__ = "ingested_content"

    # -----------------------------------------------------------
    # PRIMARY + FOREIGN KEYS
    # -----------------------------------------------------------
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_id = Column(UUID(as_uuid=True), ForeignKey("ingested_file.id", ondelete="CASCADE"), nullable=False)
    business_id = Column(UUID(as_uuid=True), nullable=True)
    global_content_id = Column(UUID(as_uuid=True), ForeignKey("global_content_index.id", ondelete="SET NULL"), nullable=True)

    # -----------------------------------------------------------
    # CHUNK INFO
    # -----------------------------------------------------------
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=True)
    parent_chunk_id = Column(UUID(as_uuid=True), nullable=True)
    text = Column(Text, nullable=False)
    cleaned_text = Column(Text, nullable=False)
    tokens = Column(Integer, nullable=False)
    semantic_hash = Column(String(256), nullable=False, index=True)
    confidence = Column(Float, default=1.0)
    source_type = Column(String(50), nullable=True)

    # -----------------------------------------------------------
    # DUPLICATE & DEDUP INFO
    # -----------------------------------------------------------
    is_duplicate = Column(Boolean, default=False)
    duplicate_of = Column(UUID(as_uuid=True), nullable=True)
    similarity_score = Column(Float, nullable=True)
    duplicate_percentage = Column(Float, nullable=True)

    # -----------------------------------------------------------
    # METADATA + TIMESTAMPS
    # -----------------------------------------------------------
    # <<< PATCH: use callable default to avoid shared mutable dict across instances >>>
    meta_data = Column(JSON, default=dict)
    reasoning_ingestion = Column(JSONB, nullable=True)
    validation_layer = Column(JSONB, nullable=True)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)

    # -----------------------------------------------------------
    # RELATIONSHIPS
    # -----------------------------------------------------------
    
   # global_content = relationship("GlobalContentIndexV2", back_populates="ingested_chunks", lazy="selectin")

    # -----------------------------------------------------------
    # REPRESENTATION
    # -----------------------------------------------------------
    def __repr__(self):
        return (
            f"<IngestedContentV2(id={self.id}, file_id={self.file_id}, "
            f"chunk_index={self.chunk_index}, duplicate={self.is_duplicate}, "
            f"semantic_hash={self.semantic_hash[:10]}...)>"
        )
