"""Safe full wipe script for Redis vector collection."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from app.config.ingestion_settings import EMBEDDING_DIMENSION
from app.core.vectordb.redis_v1 import RedisVectorBackendUnavailable, RedisVectorDB
from app.utils.logger import log_info


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
COLLECTION_NAME = os.getenv("MAI_COLLECTION", "ingested_content")


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


def clear_redis_collection() -> None:
    try:
        vectordb = get_vectordb()
        log_info("[RedisCleanup] Starting Redis vector cleanup...")
        try:
            vectordb.delete_collection(COLLECTION_NAME)
        except RedisVectorBackendUnavailable as exc:
            raise RuntimeError(
                "Redis endpoint is reachable, but vector indexing is unavailable. "
                "Use Redis Stack or enable RediSearch on the target Redis service."
            ) from exc
        log_info(f"[RedisCleanup] Deleted collection '{COLLECTION_NAME}' if it existed.")
        vectordb.ensure_collection(
            COLLECTION_NAME,
            embedding_dim=EMBEDDING_DIMENSION,
            distance_metric="cosine",
        )
        log_info(
            f"[RedisCleanup] Recreated empty '{COLLECTION_NAME}' collection with dim={EMBEDDING_DIMENSION}."
        )
    except Exception as exc:
        log_info(f"[ERROR] Failed to clear Redis: {exc}")


if __name__ == "__main__":
    clear_redis_collection()
