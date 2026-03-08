-- Table: public.ingested_content

-- DROP TABLE IF EXISTS public.ingested_content;

CREATE TABLE IF NOT EXISTS public.ingested_content
(
    id uuid NOT NULL DEFAULT uuid_generate_v4(),
    file_id uuid NOT NULL,
    business_id uuid,
    global_content_id uuid,
    chunk_index integer NOT NULL,
    text text COLLATE pg_catalog."default" NOT NULL,
    cleaned_text text COLLATE pg_catalog."default" NOT NULL,
    tokens integer NOT NULL,
    semantic_hash text COLLATE pg_catalog."default" NOT NULL,
    confidence double precision DEFAULT 1.0,
    source_type text COLLATE pg_catalog."default" NOT NULL,
    is_duplicate boolean DEFAULT false,
    duplicate_of uuid,
    similarity_score double precision,
    duplicate_percentage double precision,
    meta_data jsonb DEFAULT '{}'::jsonb,
    created_at timestamp without time zone DEFAULT now(),
    updated_at timestamp without time zone DEFAULT now(),
    reasoning_ingestion jsonb,
    validation_layer jsonb,
    CONSTRAINT ingested_content_pkey PRIMARY KEY (id),
    CONSTRAINT ingested_content_file_id_chunk_index_key UNIQUE (file_id, chunk_index),
    CONSTRAINT ingested_content_file_id_fkey FOREIGN KEY (file_id)
        REFERENCES public.ingested_file (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE CASCADE,
    CONSTRAINT ingested_content_global_content_id_fkey FOREIGN KEY (global_content_id)
        REFERENCES public.global_content_index (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT check_duplicate_consistency CHECK (is_duplicate = false AND duplicate_of IS NULL OR is_duplicate = true AND duplicate_of IS NOT NULL)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.ingested_content
    OWNER to postgres;
-- Index: idx_ingested_content_business_id

-- DROP INDEX IF EXISTS public.idx_ingested_content_business_id;

CREATE INDEX IF NOT EXISTS idx_ingested_content_business_id
    ON public.ingested_content USING btree
    (business_id ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_ingested_content_confidence

-- DROP INDEX IF EXISTS public.idx_ingested_content_confidence;

CREATE INDEX IF NOT EXISTS idx_ingested_content_confidence
    ON public.ingested_content USING btree
    (confidence ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_ingested_content_file_id

-- DROP INDEX IF EXISTS public.idx_ingested_content_file_id;

CREATE INDEX IF NOT EXISTS idx_ingested_content_file_id
    ON public.ingested_content USING btree
    (file_id ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_ingested_content_semantic_hash

-- DROP INDEX IF EXISTS public.idx_ingested_content_semantic_hash;

CREATE INDEX IF NOT EXISTS idx_ingested_content_semantic_hash
    ON public.ingested_content USING btree
    (semantic_hash COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_ingested_content_source_type

-- DROP INDEX IF EXISTS public.idx_ingested_content_source_type;

CREATE INDEX IF NOT EXISTS idx_ingested_content_source_type
    ON public.ingested_content USING btree
    (source_type COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;

-- Trigger: trg_update_ingested_content

-- DROP TRIGGER IF EXISTS trg_update_ingested_content ON public.ingested_content;

CREATE OR REPLACE TRIGGER trg_update_ingested_content
    BEFORE UPDATE 
    ON public.ingested_content
    FOR EACH ROW
    EXECUTE FUNCTION public.update_timestamp();
