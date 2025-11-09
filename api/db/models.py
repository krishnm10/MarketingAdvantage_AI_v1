# ==============================================================
# 🚀 Enterprise Marketing Knowledge Graph — SQLAlchemy Models
# ==============================================================
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, Float, Boolean, DateTime, ForeignKey, JSON, Integer
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from api.connection import Base
import uuid


# ==============================================================
# 🧩 Utility
# ==============================================================
def generate_uuid():
    return str(uuid.uuid4())


# ==============================================================
# 👤 User (Authentication)
# ==============================================================
class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    role = Column(String, default="user")  # admin, editor, viewer
    created_at = Column(DateTime, default=datetime.utcnow)


# ==============================================================
# 🏢 Business / Organization
# ==============================================================
class Business(Base):
    __tablename__ = "businesses"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    industry = Column(String)
    description = Column(Text)
    stage = Column(String)  # startup, growth, enterprise
    website = Column(String)
    goal = Column(Text)
    region = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relations
    contents = relationship("Content", back_populates="business", cascade="all, delete-orphan")
    strategies = relationship("Strategy", back_populates="business", cascade="all, delete-orphan")
    kpis = relationship("KPI", back_populates="business", cascade="all, delete-orphan")
    trends = relationship("Trend", back_populates="business", cascade="all, delete-orphan")


# ==============================================================
# 📝 Content (RAG + Creative)
# ==============================================================
class Content(Base):
    __tablename__ = "contents"

    id = Column(String, primary_key=True, default=generate_uuid)
    business_id = Column(String, ForeignKey("businesses.id", ondelete="CASCADE"))
    title = Column(String, nullable=False)
    content_type = Column(String)  # blog, video, post, ad, script
    text = Column(Text)
    summary = Column(Text)
    category = Column(String)
    sub_category = Column(String)
    tags = Column(JSON)
    content_metadata = Column(JSON)  # ✅ renamed from 'metadata'
    source = Column(String)
    fingerprint = Column(String, index=True, unique=True)
    confidence = Column(Float, default=0.9)
    chunk_index = Column(Integer, default=0)
    version = Column(String, default="1.0")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relations
    business = relationship("Business", back_populates="contents")


# ==============================================================
# 🎯 Strategy (Tactical Knowledge)
# ==============================================================
class Strategy(Base):
    __tablename__ = "strategies"

    id = Column(String, primary_key=True, default=generate_uuid)
    business_id = Column(String, ForeignKey("businesses.id", ondelete="CASCADE"))
    title = Column(String, nullable=False)
    description = Column(Text)
    category = Column(String)
    sub_category = Column(String)
    goal = Column(Text)
    tags = Column(JSON)
    source = Column(String)
    confidence = Column(Float, default=0.9)
    version = Column(String, default="1.0")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relations
    business = relationship("Business", back_populates="strategies")


# ==============================================================
# 📊 KPI (Key Performance Indicators)
# ==============================================================
class KPI(Base):
    __tablename__ = "kpis"

    id = Column(String, primary_key=True, default=generate_uuid)
    business_id = Column(String, ForeignKey("businesses.id", ondelete="CASCADE"))
    name = Column(String, nullable=False)
    metric_type = Column(String)  # engagement, revenue, retention
    value = Column(Float)
    target = Column(Float)
    trend_direction = Column(String)  # up/down/flat
    confidence = Column(Float, default=0.8)
    period = Column(String, default="monthly")
    source = Column(String)
    last_updated = Column(DateTime, default=datetime.utcnow)

    # Relations
    business = relationship("Business", back_populates="kpis")


# ==============================================================
# 🔥 Trends / Trending Topics
# ==============================================================
class Trend(Base):
    __tablename__ = "trends"

    id = Column(String, primary_key=True, default=generate_uuid)
    business_id = Column(String, ForeignKey("businesses.id", ondelete="CASCADE"), nullable=True)
    title = Column(String, nullable=False)
    category = Column(String)
    sub_category = Column(String)
    summary = Column(Text)
    source = Column(String)
    trend_score = Column(Float, default=0.5)
    confidence = Column(Float, default=0.9)
    tags = Column(JSON)
    sentiment = Column(String)
    region = Column(String)
    published_at = Column(DateTime)
    last_updated = Column(DateTime, default=datetime.utcnow)

    # Relations
    business = relationship("Business", back_populates="trends")


# ==============================================================
# 🧭 Taxonomy Category (Controlled Vocabulary)
# ==============================================================
class TaxonomyCategory(Base):
    __tablename__ = "taxonomy_categories"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    group = Column(String, nullable=False)  # content, strategy, trend, kpi, business
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


# ==============================================================
# 🔗 Relations (Knowledge Graph)
# ==============================================================
class Relation(Base):
    __tablename__ = "relations"

    id = Column(String, primary_key=True, default=generate_uuid)
    source_id = Column(String, nullable=False)
    target_id = Column(String, nullable=False)
    relation_type = Column(String)  # influences, supports, competes_with, inspired_by, etc.
    weight = Column(Float, default=1.0)
    context = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)



# ==============================================================
# 🧩 EntityLink — Universal Taxonomy → Business → Entity Mapper
# ==============================================================
class EntityLink(Base):
    __tablename__ = "entity_links"

    id = Column(String, primary_key=True, default=generate_uuid)
    category_id = Column(String, ForeignKey("taxonomy_categories.id", ondelete="SET NULL"))
    subcategory_id = Column(String, ForeignKey("taxonomy_categories.id", ondelete="SET NULL"))
    business_id = Column(String, ForeignKey("businesses.id", ondelete="CASCADE"))
    entity_type = Column(String, nullable=False)  # content, strategy, kpi, trend, etc.
    entity_id = Column(String, nullable=False)    # ID of the record in its table
    fingerprint = Column(String, unique=True, index=True)
    link_metadata = Column(JSON, default={})  # ✅ renamed
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships (optional but helpful)
    business = relationship("Business")
    category = relationship("TaxonomyCategory", foreign_keys=[category_id])
    subcategory = relationship("TaxonomyCategory", foreign_keys=[subcategory_id])

    def __repr__(self):
        return f"<EntityLink(entity_type={self.entity_type}, business={self.business_id})>"




# ==============================================================
# 📡 IngestSource — Feed Monitoring & Metrics
# ==============================================================
from sqlalchemy import Integer

class IngestSource(Base):
    __tablename__ = "ingest_sources"

    id = Column(String, primary_key=True, default=generate_uuid)
    category = Column(String, nullable=False)
    feed_url = Column(String, unique=True, nullable=False)
    articles_added = Column(Integer, default=0)
    partials = Column(Integer, default=0)
    failures = Column(Integer, default=0)
    last_fetched = Column(DateTime, nullable=True)
    status = Column(String, default="idle")  # idle, active, partial, failed
    error_message = Column(Text, nullable=True)
    avg_confidence = Column(Float, default=0.0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
