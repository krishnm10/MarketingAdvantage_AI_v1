CREATE TABLE ingested_content (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    file_id UUID NOT NULL
        REFERENCES ingested_file(id)
        ON DELETE CASCADE,

    business_id UUID NULL,                           -- optional link
    chunk_index INT NOT NULL,                        -- sequential ordering per file

    text TEXT NOT NULL,                              -- raw extracted text
    cleaned_text TEXT NOT NULL,                      -- cleaned text after processing
    tokens INT NOT NULL,                             -- token count (for LLM window mgmt)

    source_type TEXT NOT NULL,                       -- pdf, text, csv_row, table_cell, etc.

    metadata JSONB DEFAULT '{}'::jsonb,

    confidence FLOAT DEFAULT 1.0,                    -- RSC merge confidence
    semantic_hash TEXT NOT NULL,                     -- SHA-256 hash for deduplication

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(file_id, chunk_index),                    -- strong integrity
    UNIQUE(semantic_hash)                            -- deduping guarantee
);

CREATE INDEX idx_ingested_content_file_id ON ingested_content(file_id);
CREATE INDEX idx_ingested_content_business_id ON ingested_content(business_id);
CREATE INDEX idx_ingested_content_source_type ON ingested_content(source_type);
CREATE INDEX idx_ingested_content_confidence ON ingested_content(confidence);
