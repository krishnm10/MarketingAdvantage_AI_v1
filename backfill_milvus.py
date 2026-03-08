"""Idempotent backfill from global_content_index to Milvus."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text

from app.config.ingestion_settings import EMBEDDING_MODEL_NAME

try:
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
except ImportError as exc:
    raise ImportError("PyMilvus not installed. Run: pip install pymilvus>=2.4.0") from exc

try:
    from sentence_transformers import SentenceTransformer

    _SBERT = SentenceTransformer("BAAI/bge-large-en-v1.5")
    print("Embedder:", EMBEDDING_MODEL_NAME)
except Exception:
    _SBERT = None


ENV_PATH = Path(__file__).resolve().with_name(".env")
load_dotenv(ENV_PATH)


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:Mahadeva%40123@localhost/marketing_advantage",
)

MILVUS_URI = os.getenv("MILVUS_URI", "").strip()
MILVUS_HOST = os.getenv("MILVUS_HOST", "").strip()
MILVUS_PORT_RAW = os.getenv("MILVUS_PORT", "").strip()
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN") or None
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME", "default")
MILVUS_ALIAS = os.getenv("MILVUS_ALIAS", "backfill_milvus")

MILVUS_COLLECTION = os.getenv("MAI_COLLECTION", "ingested_content")
EMBED_DIM = int(os.getenv("MAI_EMBED_DIM", "1024"))

GCI_TABLE = "global_content_index"
DOC_ID_FIELD = "doc_id"
VECTOR_FIELD = "embedding"
TEXT_FIELD = "_text"
METADATA_FIELD = "_metadata"

BATCH = int(os.getenv("MILVUS_BACKFILL_BATCH", "16"))
SLEEP_BETWEEN_BATCHES = float(os.getenv("MILVUS_BACKFILL_SLEEP", "0.2"))

MILVUS_PORT = int(MILVUS_PORT_RAW) if MILVUS_PORT_RAW else 19530


def connect_milvus() -> None:
    if MILVUS_URI:
        kw: dict[str, Any] = {"alias": MILVUS_ALIAS, "uri": MILVUS_URI}
        if MILVUS_TOKEN:
            kw["token"] = MILVUS_TOKEN
        connections.connect(**kw)
        return

    if not MILVUS_HOST:
        raise RuntimeError(
            f"Milvus configuration missing. Set MILVUS_URI or MILVUS_HOST in {ENV_PATH}."
        )

    kw = {
        "alias": MILVUS_ALIAS,
        "host": MILVUS_HOST,
        "port": str(MILVUS_PORT),
        "db_name": MILVUS_DB_NAME,
    }
    if MILVUS_TOKEN:
        kw["token"] = MILVUS_TOKEN
    connections.connect(**kw)


def ensure_collection() -> Collection:
    if utility.has_collection(MILVUS_COLLECTION, using=MILVUS_ALIAS):
        collection = Collection(name=MILVUS_COLLECTION, using=MILVUS_ALIAS)
        collection.load()
        return collection

    schema = CollectionSchema(
        fields=[
            FieldSchema(
                name=DOC_ID_FIELD,
                dtype=DataType.VARCHAR,
                max_length=512,
                is_primary=True,
                auto_id=False,
            ),
            FieldSchema(name=VECTOR_FIELD, dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
            FieldSchema(name=TEXT_FIELD, dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name=METADATA_FIELD, dtype=DataType.JSON),
        ],
        description="MarketingAdvantage AI document chunks",
        enable_dynamic_field=True,
    )
    collection = Collection(name=MILVUS_COLLECTION, schema=schema, using=MILVUS_ALIAS)
    collection.create_index(
        field_name=VECTOR_FIELD,
        index_name="embedding_idx",
        index_params={
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 256},
        },
    )
    collection.load()
    return collection


def compute_embeddings(texts: list[str]) -> list[list[float]]:
    if _SBERT:
        embeddings = _SBERT.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [list(embedding) for embedding in embeddings]
    raise RuntimeError(
        "No embedder available. Install sentence-transformers or replace compute_embeddings()."
    )


def detect_gci_schema(engine) -> str:
    inspector = inspect(engine)
    for schema in inspector.get_schema_names():
        if GCI_TABLE in inspector.get_table_names(schema=schema):
            return schema
    raise RuntimeError("global_content_index table not found in any schema")


def fetch_rows(engine, schema: str) -> list[dict[str, Any]]:
    query = text(
        f"""
        SELECT id, semantic_hash, cleaned_text, raw_text, created_at
        FROM {schema}.{GCI_TABLE}
        WHERE semantic_hash IS NOT NULL
        ORDER BY created_at DESC
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    mapped: list[dict[str, Any]] = []
    for row in rows:
        if hasattr(row, "_mapping"):
            mapped.append(dict(row._mapping))
        else:
            mapped.append(dict(row))
    return mapped


def existing_ids(collection: Collection, ids: list[str]) -> set[str]:
    if not ids:
        return set()

    ids_expr = ", ".join(f'"{doc_id}"' for doc_id in ids)
    try:
        rows = collection.query(
            expr=f'{DOC_ID_FIELD} in [{ids_expr}]',
            output_fields=[DOC_ID_FIELD],
        )
    except Exception:
        return set()
    return {str(row[DOC_ID_FIELD]) for row in rows}


def main() -> None:
    if not DATABASE_URL:
        raise RuntimeError("Set DATABASE_URL first.")

    engine = create_engine(DATABASE_URL)
    gci_schema = detect_gci_schema(engine)
    print("Using GCI table:", f"{gci_schema}.{GCI_TABLE}")

    connect_milvus()
    collection = ensure_collection()
    print("Milvus collection ready:", MILVUS_COLLECTION, "count:", collection.num_entities)

    rows = fetch_rows(engine, gci_schema)
    hashes = [row["semantic_hash"] for row in rows if row.get("semantic_hash")]
    print("GCI total with semantic_hash:", len(hashes))

    missing_hashes: list[str] = []
    for start in range(0, len(hashes), BATCH):
        batch = hashes[start : start + BATCH]
        found = existing_ids(collection, batch)
        for semantic_hash in batch:
            if semantic_hash not in found:
                missing_hashes.append(semantic_hash)

    print("Missing hashes count:", len(missing_hashes))
    if not missing_hashes:
        print("Nothing to backfill. Exiting.")
        connections.disconnect(MILVUS_ALIAS)
        return

    gci_map = {row["semantic_hash"]: row for row in rows if row.get("semantic_hash")}

    for start in range(0, len(missing_hashes), BATCH):
        batch_hashes = missing_hashes[start : start + BATCH]
        texts: list[str] = []
        metadata_rows: list[dict[str, Any]] = []

        for semantic_hash in batch_hashes:
            entry = gci_map[semantic_hash]
            text_to_embed = (entry.get("cleaned_text") or entry.get("raw_text") or "")[:15000]
            texts.append(text_to_embed)
            metadata_rows.append(
                {
                    "global_content_id": str(entry.get("id")),
                    "semantic_hash": semantic_hash,
                }
            )

        embeddings = compute_embeddings(texts)
        rows_to_insert = [
            {
                DOC_ID_FIELD: batch_hashes[index],
                VECTOR_FIELD: embeddings[index],
                TEXT_FIELD: texts[index][:65530],
                METADATA_FIELD: metadata_rows[index],
            }
            for index in range(len(batch_hashes))
        ]

        try:
            collection.insert(rows_to_insert)
            collection.flush()
            print(f"Upserted batch {start // BATCH + 1}: {len(batch_hashes)} items")
        except Exception as exc:
            print("Milvus insert failed:", exc)

        time.sleep(SLEEP_BETWEEN_BATCHES)

    connections.disconnect(MILVUS_ALIAS)
    print("Backfill completed.")


if __name__ == "__main__":
    main()
