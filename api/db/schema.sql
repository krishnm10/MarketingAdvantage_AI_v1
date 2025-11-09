-- ==========================================================
-- 🚀 Marketing Advantage — Enterprise Knowledge Graph Schema (v2.1)
-- ==========================================================

-- Extensions for UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ==========================================================
-- 👤 Users
-- ==========================================================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    role TEXT DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==========================================================
-- 🏢 Businesses
-- ==========================================================
CREATE TABLE IF NOT EXISTS businesses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    industry TEXT,
    description TEXT,
    stage TEXT,
    website TEXT,
    goal TEXT,
    region TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ✅ Case-insensitive unique index to prevent duplicate businesses
CREATE UNIQUE INDEX IF NOT EXISTS uq_business_lower_name_region
ON businesses (LOWER(name), LOWER(region));

-- ==========================================================
-- 📝 Contents
-- ==========================================================
CREATE TABLE IF NOT EXISTS contents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    business_id UUID REFERENCES businesses(id) ON DELETE CASCADE,
    title TEXT,
    content_type TEXT,
    text TEXT,
    summary TEXT,
    category TEXT,
    sub_category TEXT,
    tags JSONB,
    metadata JSONB,
    source TEXT,
    fingerprint TEXT UNIQUE,
    confidence REAL DEFAULT 0.9,
    chunk_index INTEGER DEFAULT 0,
    version TEXT DEFAULT '1.0',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==========================================================
-- 🎯 Strategies
-- ==========================================================
CREATE TABLE IF NOT EXISTS strategies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    business_id UUID REFERENCES businesses(id) ON DELETE CASCADE,
    title TEXT,
    description TEXT,
    category TEXT,
    sub_category TEXT,
    goal TEXT,
    tags JSONB,
    confidence REAL DEFAULT 0.9,
    source TEXT,
    version TEXT DEFAULT '1.0',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==========================================================
-- 📊 KPIs
-- ==========================================================
CREATE TABLE IF NOT EXISTS kpis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    business_id UUID REFERENCES businesses(id) ON DELETE CASCADE,
    name TEXT,
    metric_type TEXT,
    value REAL,
    target REAL,
    trend_direction TEXT,
    confidence REAL DEFAULT 0.8,
    period TEXT DEFAULT 'monthly',
    source TEXT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==========================================================
-- 🔥 Trends
-- ==========================================================
CREATE TABLE IF NOT EXISTS trends (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    business_id UUID REFERENCES businesses(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    category TEXT,
    sub_category TEXT,
    summary TEXT,
    source TEXT,
    trend_score REAL DEFAULT 0.5,
    confidence REAL DEFAULT 0.9,
    tags JSONB,
    sentiment TEXT,
    region TEXT,
    published_at TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==========================================================
-- 🧭 Taxonomy Categories
-- ==========================================================
CREATE TABLE IF NOT EXISTS taxonomy_categories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    "group" TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ✅ Case-insensitive unique index to prevent duplicate taxonomy entries
CREATE UNIQUE INDEX IF NOT EXISTS uq_taxonomy_lower_name_group
ON taxonomy_categories (LOWER(name), LOWER("group"));

-- ==========================================================
-- 🔗 Relations (Knowledge Graph Relationships)
-- ==========================================================
CREATE TABLE IF NOT EXISTS relations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID,
    target_id UUID,
    relation_type TEXT,
    weight REAL DEFAULT 1.0,
    context TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==========================================================
-- 🧩 Entity Links — Universal Taxonomy → Business → Entity Mapper
-- ==========================================================
CREATE TABLE IF NOT EXISTS entity_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    category_id UUID REFERENCES taxonomy_categories(id) ON DELETE SET NULL,
    subcategory_id UUID REFERENCES taxonomy_categories(id) ON DELETE SET NULL,
    business_id UUID REFERENCES businesses(id) ON DELETE CASCADE,
    entity_type TEXT NOT NULL,     -- content, strategy, kpi, trend, etc.
    entity_id UUID NOT NULL,       -- ID from the corresponding entity table
    fingerprint TEXT UNIQUE,       -- SHA256 hash for duplicate prevention
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_entity_link UNIQUE (category_id, subcategory_id, business_id, entity_type, entity_id)
);

-- ==========================================================
-- ⚡ Indexes (Performance)
-- ==========================================================
CREATE INDEX IF NOT EXISTS idx_entity_links_business 
    ON entity_links(business_id);
CREATE INDEX IF NOT EXISTS idx_entity_links_entity_type 
    ON entity_links(entity_type);
CREATE INDEX IF NOT EXISTS idx_entity_links_category_subcategory 
    ON entity_links(category_id, subcategory_id);
CREATE INDEX IF NOT EXISTS idx_entity_links_fingerprint 
    ON entity_links(fingerprint);
CREATE INDEX IF NOT EXISTS idx_entity_links_entity_id 
    ON entity_links(entity_id);
