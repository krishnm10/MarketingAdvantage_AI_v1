-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;


-- ===========================================================
-- 1. APPROVED TAXONOMY TABLE
-- ===========================================================

CREATE TABLE taxonomy (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Human readable name
    name TEXT NOT NULL,
    
    -- Canonical deduped version
    canonical_name TEXT NOT NULL UNIQUE,
    
    -- Parent: NULL only for level 0
    parent_id UUID NULL REFERENCES taxonomy(id) ON DELETE CASCADE,
    
    -- 0 = industry, 1 = subcategory, 2 = sub-subcategory
    level INT NOT NULL CHECK (level IN (0, 1, 2)),
    
    -- SEO / UI-friendly string
    slug TEXT,
    
    -- Any flexible metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Reference ID stored in ChromaDB
    embedding_id TEXT NULL,
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for speed
CREATE INDEX idx_taxonomy_parent_id ON taxonomy(parent_id);
CREATE INDEX idx_taxonomy_level ON taxonomy(level);
CREATE INDEX idx_taxonomy_canonical_name_trgm ON taxonomy USING gin (canonical_name gin_trgm_ops);


-- ===========================================================
-- 2. TAXONOMY ALIAS TABLE (Synonym → Canonical)
-- ===========================================================

CREATE TABLE taxonomy_alias (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    alias_name TEXT NOT NULL,
    
    canonical_taxonomy_id UUID NOT NULL
        REFERENCES taxonomy(id)
        ON DELETE CASCADE, 
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- Prevent duplicates in aliases
CREATE UNIQUE INDEX idx_taxonomy_alias_unique
    ON taxonomy_alias(alias_name, canonical_taxonomy_id);


-- ===========================================================
-- 3. PENDING TAXONOMY TABLE (Needs Admin Approval)
-- ===========================================================

CREATE TABLE pending_taxonomy (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    raw_name TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    
    -- AI-suggested parent and level
    suggested_parent_id UUID NULL REFERENCES taxonomy(id),
    suggested_level INT CHECK (suggested_level IN (0, 1, 2)),
    
    -- Similar existing canonical category IDs
    similar_existing_ids TEXT[],
    
    -- Similarity confidence (0.0 - 1.0)
    confidence FLOAT DEFAULT 0.0,
    
    -- pending / approved / rejected
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'approved', 'rejected')),
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for admin review speed
CREATE INDEX idx_pending_taxonomy_status ON pending_taxonomy(status);
CREATE INDEX idx_pending_taxonomy_canonical_trgm ON pending_taxonomy USING gin (canonical_name gin_trgm_ops);


-- ===========================================================
-- 4. BUSINESS → TAXONOMY RELATION TABLE
-- ===========================================================

CREATE TABLE business_taxonomy (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    business_id UUID NOT NULL,
    taxonomy_id UUID NOT NULL REFERENCES taxonomy(id) ON DELETE CASCADE,
    
    relationship_type TEXT NOT NULL DEFAULT 'primary',
    
    -- AI/manual confidence
    confidence FLOAT DEFAULT 1.0,
    
    -- manual / ai / auto / suggested
    source TEXT NOT NULL DEFAULT 'manual',
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Prevent assigning same category twice
CREATE UNIQUE INDEX idx_business_taxonomy_unique
    ON business_taxonomy(business_id, taxonomy_id);

-- Index classification-related fields
CREATE INDEX idx_business_taxonomy_relationship_type ON business_taxonomy(relationship_type);
CREATE INDEX idx_business_taxonomy_source ON business_taxonomy(source);
