CREATE TABLE global_content_index (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    semantic_hash VARCHAR(256) UNIQUE NOT NULL,
    cleaned_text TEXT NOT NULL,
    raw_text TEXT NULL,
    tokens INT DEFAULT 0,
    embedding_model VARCHAR(128) DEFAULT 'BAAI/bge-large-en',
    confidence_avg FLOAT DEFAULT 0,
    occurrence_count INT DEFAULT 1,
    business_id UUID NULL,
    first_seen_file_id UUID NULL REFERENCES ingested_file(id),
    meta_data JSONB DEFAULT '{}'::jsonb,
    source_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_gci_semhash ON global_content_index(semantic_hash);
CREATE INDEX idx_gci_business_id ON global_content_index(business_id);
CREATE INDEX idx_gci_source_type ON global_content_index(source_type);
