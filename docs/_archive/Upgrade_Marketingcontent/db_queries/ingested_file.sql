CREATE TABLE ingested_file (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    business_id UUID NULL,                           -- optional
    file_name TEXT NOT NULL,
    file_type TEXT NOT NULL,                         -- pdf, docx, excel, csv, api, rss, web

    file_path TEXT NULL,                             -- optional (if stored locally)
    source_url TEXT NULL,                            -- web scraped / API / RSS sources
    metadata JSONB DEFAULT '{}'::jsonb,

    total_chunks INT DEFAULT 0,
    status TEXT DEFAULT 'processed'
        CHECK (status IN ('pending', 'processing', 'processed', 'error')),

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_ingested_file_business_id ON ingested_file(business_id);
CREATE INDEX idx_ingested_file_file_type ON ingested_file(file_type);
