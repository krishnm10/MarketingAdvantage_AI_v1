# phase1_taxonomy_models.py

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, DateTime, ForeignKey, JSON,
    UniqueConstraint, ARRAY, Float, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def generate_uuid():
    return str(uuid.uuid4())


# -------------------------
# APPROVED TAXONOMY TABLE
# -------------------------
class Taxonomy(Base):
    __tablename__ = "taxonomy"

    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    canonical_name = Column(String(255), nullable=False, unique=True)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("taxonomy.id"), nullable=True)
    level = Column(Integer, nullable=False)  # 0,1,2 levels only
    slug = Column(String(255), nullable=True)
    metadata = Column(JSONB, default=dict)
    embedding_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    parent = relationship("Taxonomy", remote_side=[id])


# -------------------------
# TAXONOMY ALIAS TABLE
# -------------------------
class TaxonomyAlias(Base):
    __tablename__ = "taxonomy_alias"

    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    alias_name = Column(String(255), nullable=False)
    canonical_taxonomy_id = Column(UUID(as_uuid=True),
                                   ForeignKey("taxonomy.id"),
                                   nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    taxonomy = relationship("Taxonomy")


# -------------------------
# PENDING TAXONOMY TABLE
# -------------------------
class PendingTaxonomy(Base):
    __tablename__ = "pending_taxonomy"

    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    raw_name = Column(String(255), nullable=False)
    canonical_name = Column(String(255), nullable=False)
    suggested_parent_id = Column(UUID(as_uuid=True), ForeignKey("taxonomy.id"))
    suggested_level = Column(Integer, nullable=True)
    similar_existing_ids = Column(ARRAY(String), nullable=True)
    confidence = Column(Float, default=0.0)
    status = Column(String(20), default="pending")  # pending/approved/rejected
    created_at = Column(DateTime, default=datetime.utcnow)

    parent = relationship("Taxonomy")
