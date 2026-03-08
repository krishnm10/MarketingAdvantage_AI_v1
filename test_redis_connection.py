"""Standalone Redis vector connection test using .env configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from app.config.ingestion_settings import EMBEDDING_DIMENSION
from app.core.vectordb.redis_v1 import RedisVectorBackendUnavailable, RedisVectorDB


ENV_PATH = Path(__file__).resolve().with_name(".env")
load_dotenv(ENV_PATH)

REDIS_URL = os.getenv("REDIS_URL", "").strip() or None
REDIS_HOST = os.getenv("REDIS_HOST", "").strip()
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_USERNAME = os.getenv("REDIS_USERNAME") or None
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
REDIS_SSL = os.getenv("REDIS_SSL", "false").lower() == "true"
REDIS_SSL_CA_CERTS = os.getenv("REDIS_SSL_CA_CERTS") or None
REDIS_PREFIX = os.getenv("REDIS_PREFIX", "vec:")


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


def main() -> None:
    vectordb = get_vectordb()
    collection = "__redis_connection_test__"
    doc_id = "test-doc-001"
    embedding = [0.1] * EMBEDDING_DIMENSION
    text = "This is a Marketing Advantage AI Redis test document."
    metadata = {"source": "test", "status": "active"}

    print("Checking Redis ping...")
    if not vectordb.health_check():
        raise RuntimeError("Redis ping failed.")
    print("OK Redis ping successful")

    print("Ensuring test collection...")
    try:
        vectordb.ensure_collection(collection, embedding_dim=EMBEDDING_DIMENSION, distance_metric="cosine")
    except RedisVectorBackendUnavailable as exc:
        raise RuntimeError(
            "Redis ping works, but this endpoint does not support vector indexing. "
            "Use Redis Stack or enable RediSearch on the target Redis service."
        ) from exc
    print("OK test collection ready")

    print("Inserting test document...")
    vectordb.upsert(
        collection=collection,
        doc_id=doc_id,
        embedding=embedding,
        text=text,
        metadata=metadata,
    )
    print("OK insert complete")

    print("Running search...")
    hits = vectordb.search(collection=collection, query_embedding=embedding, top_k=1)
    print(f"OK search returned {len(hits)} hit(s)")
    if hits:
        print(f"Top hit: id={hits[0].id} score={hits[0].score:.4f}")

    print("Cleaning up test collection...")
    vectordb.delete_collection(collection)
    print("OK cleanup complete")


if __name__ == "__main__":
    main()
