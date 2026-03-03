import uuid
from datetime import datetime
from sqlalchemy import Column, String, JSON, Integer, Float, TIMESTAMP, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.future import select
from sqlalchemy.exc import SQLAlchemyError
from uuid import UUID as UUIDType

from app.db.base import Base
from app.utils.logger import log_info, log_warning


class GlobalContentIndexV2(Base):
    """
    GlobalContentIndexV2
    -------------------
    Master registry of globally unique semantic chunks across ingested content.
    Tracks deduplication, occurrence frequency, and embeddings confidence.
    """

    __tablename__ = "global_content_index"

    # -----------------------------------------------------------
    # PRIMARY + CORE FIELDS
    # -----------------------------------------------------------
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    semantic_hash = Column(String(256), unique=True, index=True, nullable=False)
    cleaned_text = Column(Text, nullable=False)
    raw_text = Column(Text, nullable=True)
    tokens = Column(Integer, nullable=False, default=0)

    # -----------------------------------------------------------
    # SEMANTIC TRACKING
    # -----------------------------------------------------------
    embedding_model = Column(String(128), default="BAAI/bge-large-en-v1.5")
    confidence_avg = Column(Float, default=0.0)
    occurrence_count = Column(Integer, default=1)

    # -----------------------------------------------------------
    # CONTEXT + SOURCE
    # -----------------------------------------------------------
    business_id = Column(UUID(as_uuid=True), nullable=True)
    first_seen_file_id = Column(UUID(as_uuid=True), nullable=True)
    source_type = Column(String(50), nullable=True)

    # -----------------------------------------------------------
    # METADATA + TIMESTAMPS
    # -----------------------------------------------------------
    # <<< PATCH: use callable default to avoid shared mutable dict across instances >>>
    meta_data = Column(JSON, default=dict)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)

    # -----------------------------------------------------------
    # RELATIONSHIPS
    # -----------------------------------------------------------
    #ingested_chunks = relationship(
    #    "IngestedContentV2",
    #    back_populates="global_content",
    #    lazy="selectin",
    #    cascade="all, delete-orphan",
    #)

    # -----------------------------------------------------------
    # STATIC METHODS (ASYNC UTILITIES)
    # -----------------------------------------------------------
    @staticmethod
    def _safe_uuid(value):
        """Return a UUID or None if invalid."""
        if not value:
            return None
        try:
            return UUIDType(str(value))
        except Exception:
            return None

    @staticmethod
    async def lookup_or_create(
        db_session,
        semantic_hash: str,
        cleaned_text: str,
        tokens: int,
        confidence: float,
        business_id=None,
        file_id=None,
        source_type=None,
        metadata: dict | None = None,
    ):
        """
        Lookup or create a new global semantic chunk.
        Ensures UUIDs are validated before insert.
        """

        try:
            result = await db_session.execute(
                select(GlobalContentIndexV2).where(GlobalContentIndexV2.semantic_hash == semantic_hash)
            )
            existing = result.scalar_one_or_none()

            if existing:
                existing.occurrence_count += 1
                existing.confidence_avg = round(
                    (existing.confidence_avg + confidence) / 2, 4
                )
                existing.updated_at = datetime.utcnow()
                await db_session.commit()
                log_info(f"[GlobalContentIndexV2] Updated occurrence for {semantic_hash[:10]}...")
                return existing

            safe_business_id = GlobalContentIndexV2._safe_uuid(business_id)
            safe_file_id = GlobalContentIndexV2._safe_uuid(file_id)

            new_entry = GlobalContentIndexV2(
                id=uuid.uuid4(),
                semantic_hash=semantic_hash,
                cleaned_text=cleaned_text,
                tokens=tokens,
                confidence_avg=confidence,
                occurrence_count=1,
                business_id=safe_business_id,
                first_seen_file_id=safe_file_id,
                source_type=source_type,
                # <<< PATCH: use dict() instead of literal {} to avoid shared mutable default >>>
                meta_data=metadata or dict(),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

            db_session.add(new_entry)
            await db_session.commit()
            log_info(f"[GlobalContentIndexV2] New semantic chunk registered: {semantic_hash[:12]}...")

            return new_entry

        except SQLAlchemyError as e:
            log_warning(f"[GlobalContentIndexV2] Database error: {e}")
            await db_session.rollback()
            return None
        except Exception as e:
            log_warning(f"[GlobalContentIndexV2] Unexpected error: {e}")
            return None

    @staticmethod
    async def semantic_match_preview(db_session, hash_list: list[str]):
        """Retrieve metadata summary for a list of semantic hashes."""
        if not hash_list:
            return []

        try:
            result = await db_session.execute(
                select(GlobalContentIndexV2).where(GlobalContentIndexV2.semantic_hash.in_(hash_list))
            )
            rows = result.scalars().all()

            return [
                {
                    "id": str(r.id),
                    "semantic_hash": r.semantic_hash,
                    "tokens": r.tokens,
                    "confidence_avg": r.confidence_avg,
                    "occurrence_count": r.occurrence_count,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]

        except Exception as e:
            log_warning(f"[GlobalContentIndexV2] Semantic preview error: {e}")
            return []
