-- Table: public.ingested_file

-- DROP TABLE IF EXISTS public.ingested_file;

CREATE TABLE IF NOT EXISTS public.ingested_file
(
    id uuid NOT NULL DEFAULT uuid_generate_v4(),
    business_id uuid,
    file_name text COLLATE pg_catalog."default" NOT NULL,
    file_type text COLLATE pg_catalog."default" NOT NULL,
    file_path text COLLATE pg_catalog."default",
    source_url text COLLATE pg_catalog."default",
    meta_data jsonb DEFAULT '{}'::jsonb,
    total_chunks integer DEFAULT 0,
    unique_chunks integer DEFAULT 0,
    duplicate_chunks integer DEFAULT 0,
    dedup_ratio double precision DEFAULT 0,
    status text COLLATE pg_catalog."default" DEFAULT 'pending'::text,
    created_at timestamp without time zone DEFAULT now(),
    updated_at timestamp without time zone DEFAULT now(),
    source_type character varying(50) COLLATE pg_catalog."default",
    parser_used character varying(255) COLLATE pg_catalog."default",
    ingestion_notes character varying(255) COLLATE pg_catalog."default",
    error_message character varying(255) COLLATE pg_catalog."default",
    media_hash character varying(64) COLLATE pg_catalog."default",
    last_processed_chunk_index integer NOT NULL DEFAULT 0,
    last_processed_page integer NOT NULL DEFAULT 0,
    last_processed_at timestamp with time zone,
    CONSTRAINT ingested_file_pkey PRIMARY KEY (id),
    CONSTRAINT uq_ingested_file_media_hash UNIQUE (media_hash),
    CONSTRAINT ingested_file_status_check CHECK (status = ANY (ARRAY['uploaded'::text, 'pending'::text, 'processing'::text, 'processed'::text, 'failed'::text]))
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.ingested_file
    OWNER to postgres;
-- Index: idx_ingested_file_business_id

-- DROP INDEX IF EXISTS public.idx_ingested_file_business_id;

CREATE INDEX IF NOT EXISTS idx_ingested_file_business_id
    ON public.ingested_file USING btree
    (business_id ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_ingested_file_file_type

-- DROP INDEX IF EXISTS public.idx_ingested_file_file_type;

CREATE INDEX IF NOT EXISTS idx_ingested_file_file_type
    ON public.ingested_file USING btree
    (file_type COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_ingested_file_media_hash

-- DROP INDEX IF EXISTS public.idx_ingested_file_media_hash;

CREATE INDEX IF NOT EXISTS idx_ingested_file_media_hash
    ON public.ingested_file USING btree
    (media_hash COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: ux_ingested_file_media_hash

-- DROP INDEX IF EXISTS public.ux_ingested_file_media_hash;

CREATE UNIQUE INDEX IF NOT EXISTS ux_ingested_file_media_hash
    ON public.ingested_file USING btree
    (media_hash COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default
    WHERE media_hash IS NOT NULL;

-- Trigger: trg_update_ingested_file

-- DROP TRIGGER IF EXISTS trg_update_ingested_file ON public.ingested_file;

CREATE OR REPLACE TRIGGER trg_update_ingested_file
    BEFORE UPDATE 
    ON public.ingested_file
    FOR EACH ROW
    EXECUTE FUNCTION public.update_timestamp();

ALTER TABLE ingested_file ADD CONSTRAINT check_ingested_file_v2_status
  CHECK (status IN ('pending','processing','uploaded','processed','duplicate','failed','error','archived'));