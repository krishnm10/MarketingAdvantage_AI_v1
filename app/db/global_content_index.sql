-- Table: public.global_content_index

-- DROP TABLE IF EXISTS public.global_content_index;

CREATE TABLE IF NOT EXISTS public.global_content_index
(
    id uuid NOT NULL DEFAULT uuid_generate_v4(),
    semantic_hash character varying(256) COLLATE pg_catalog."default" NOT NULL,
    cleaned_text text COLLATE pg_catalog."default" NOT NULL,
    raw_text text COLLATE pg_catalog."default",
    tokens integer DEFAULT 0,
    embedding_model character varying(128) COLLATE pg_catalog."default" DEFAULT 'BAAI/bge-large-en'::character varying,
    confidence_avg double precision DEFAULT 0,
    occurrence_count integer DEFAULT 1,
    business_id uuid,
    first_seen_file_id uuid,
    meta_data jsonb DEFAULT '{}'::jsonb,
    source_type character varying(50) COLLATE pg_catalog."default",
    created_at timestamp without time zone DEFAULT now(),
    updated_at timestamp without time zone DEFAULT now(),
    CONSTRAINT global_content_index_pkey PRIMARY KEY (id),
    CONSTRAINT global_content_index_semantic_hash_key UNIQUE (semantic_hash),
    CONSTRAINT global_content_index_first_seen_file_id_fkey FOREIGN KEY (first_seen_file_id)
        REFERENCES public.ingested_file (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT global_content_index_confidence_avg_check CHECK (confidence_avg >= 0::double precision AND confidence_avg <= 1::double precision)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.global_content_index
    OWNER to postgres;
-- Index: idx_gci_business_id

-- DROP INDEX IF EXISTS public.idx_gci_business_id;

CREATE INDEX IF NOT EXISTS idx_gci_business_id
    ON public.global_content_index USING btree
    (business_id ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_gci_semhash

-- DROP INDEX IF EXISTS public.idx_gci_semhash;

CREATE INDEX IF NOT EXISTS idx_gci_semhash
    ON public.global_content_index USING btree
    (semantic_hash COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_gci_source_type

-- DROP INDEX IF EXISTS public.idx_gci_source_type;

CREATE INDEX IF NOT EXISTS idx_gci_source_type
    ON public.global_content_index USING btree
    (source_type COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;

-- Trigger: trg_update_global_content

-- DROP TRIGGER IF EXISTS trg_update_global_content ON public.global_content_index;

CREATE OR REPLACE TRIGGER trg_update_global_content
    BEFORE UPDATE 
    ON public.global_content_index
    FOR EACH ROW
    EXECUTE FUNCTION public.update_timestamp();
