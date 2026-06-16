CREATE TABLE business_classification (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    content_id UUID NOT NULL REFERENCES ingested_content(id) ON DELETE CASCADE,

    industry_id UUID NULL REFERENCES taxonomy(id),
    sub_industry_id UUID NULL REFERENCES taxonomy(id),
    sub_sub_industry_id UUID NULL REFERENCES taxonomy(id),

    pending_taxonomy_id UUID NULL REFERENCES pending_taxonomy(id),

    confidence FLOAT NOT NULL DEFAULT 0,
    llm_model VARCHAR(64) DEFAULT 'llama-3.1-8b-instruct',

    raw_output JSONB DEFAULT '{}'::jsonb,

    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

CREATE TABLE classification_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    content_id UUID NOT NULL REFERENCES ingested_content(id) ON DELETE CASCADE,

    taxonomy_path TEXT NULL,
    confidence FLOAT NULL,

    embed_scores JSONB NULL,
    llm_scores JSONB NULL,

    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);
