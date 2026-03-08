"""Idempotent backfill from global_content_index to Redis Stack."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text

from app.config.ingestion_settings import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME
from app.core.vectordb.redis_v1 import RedisVectorBackendUnavailable, RedisVectorDB

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

REDIS_URL = os.getenv("REDIS_URL", "").strip() or None
REDIS_HOST = os.getenv("REDIS_HOST", "").strip()
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_USERNAME = os.getenv("REDIS_USERNAME") or None
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
REDIS_SSL = os.getenv("REDIS_SSL", "false").lower() == "true"
REDIS_SSL_CA_CERTS = os.getenv("REDIS_SSL_CA_CERTS") or None
REDIS_PREFIX = os.getenv("REDIS_PREFIX", "vec:")

COLLECTION_NAME = os.getenv("MAI_COLLECTION", "ingested_content")
GCI_TABLE = "global_content_index"
BATCH = int(os.getenv("REDIS_BACKFILL_BATCH", "64"))
SLEEP_BETWEEN_BATCHES = float(os.getenv("REDIS_BACKFILL_SLEEP", "0.1"))


def get_vectordb() -> RedisVectorDB:
    if not REDIS_URL and not REDIS_HOST:
        raise RuntimeError(f"Set REDIS_URL or REDIS_HOST in {ENV_PATH}")
    return RedisVectorDB(
        url=REDIS_URL,
        host=REDIS_HOST or "localhost",
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        username=REDIS_USERNAME,
        db=REDIS_DB,
        ssl=REDIS_SSL,
        ssl_ca_certs=REDIS_SSL_CA_CERTS,
        prefix=REDIS_PREFIX,
    )


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


def main() -> None:
    vectordb = get_vectordb()
    try:
        vectordb.ensure_collection(
            COLLECTION_NAME,
            embedding_dim=EMBEDDING_DIMENSION,
            distance_metric="cosine",
        )
    except RedisVectorBackendUnavailable as exc:
        raise RuntimeError(
            "Redis endpoint is reachable, but vector indexing is unavailable. "
            "Use Redis Stack or enable RediSearch on the target Redis service."
        ) from exc

    engine = create_engine(DATABASE_URL)
    schema = detect_gci_schema(engine)
    print("Using GCI table:", f"{schema}.{GCI_TABLE}")

    rows = fetch_rows(engine, schema)
    doc_ids = [row["semantic_hash"] for row in rows if row.get("semantic_hash")]
    print("GCI total with semantic_hash:", len(doc_ids))

    missing_ids: list[str] = []
    for start in range(0, len(doc_ids), BATCH):
        batch = doc_ids[start : start + BATCH]
        existing = set(vectordb.exists(collection=COLLECTION_NAME, ids=batch))
        for doc_id in batch:
            if doc_id not in existing:
                missing_ids.append(doc_id)

    print("Missing hashes count:", len(missing_ids))
    if not missing_ids:
        print("Nothing to backfill. Exiting.")
        return

    gci_map = {row["semantic_hash"]: row for row in rows if row.get("semantic_hash")}

    for start in range(0, len(missing_ids), BATCH):
        batch_ids = missing_ids[start : start + BATCH]
        texts: list[str] = []
        metadata_rows: list[dict[str, Any]] = []

        for semantic_hash in batch_ids:
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
        result = vectordb.batch_upsert(
            collection=COLLECTION_NAME,
            doc_ids=batch_ids,
            embeddings=embeddings,
            texts=texts,
            metadatas=metadata_rows,
        )
        print(
            f"Upserted batch {start // BATCH + 1}: inserted={result.inserted} "
            f"updated={result.updated} failed={result.failed}"
        )
        time.sleep(SLEEP_BETWEEN_BATCHES)

    print("Backfill completed.")


if __name__ == "__main__":
    main()
