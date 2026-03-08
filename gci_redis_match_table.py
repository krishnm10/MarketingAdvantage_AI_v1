"""Show which recent global_content_index hashes exist in Redis."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text

from app.core.vectordb.redis_v1 import RedisVectorBackendUnavailable, RedisVectorDB


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
LIMIT = int(os.getenv("GCI_REDIS_MATCH_LIMIT", "200"))
BATCH = int(os.getenv("GCI_REDIS_MATCH_BATCH", "64"))


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


def detect_gci_schema(engine) -> str:
    inspector = inspect(engine)
    for schema in inspector.get_schema_names():
        if GCI_TABLE in inspector.get_table_names(schema=schema):
            return schema
    raise RuntimeError("global_content_index table not found in any schema")


def fetch_gci(engine, schema: str) -> list[dict[str, Any]]:
    query = text(
        f"""
        SELECT semantic_hash, id, created_at
        FROM {schema}.{GCI_TABLE}
        ORDER BY created_at DESC
        LIMIT :lim
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"lim": LIMIT}).fetchall()

    result: list[dict[str, Any]] = []
    for row in rows:
        if hasattr(row, "_mapping"):
            data = dict(row._mapping)
        else:
            data = dict(row)
        result.append({"semantic_hash": data["semantic_hash"], "gci_id": data["id"]})
    return result


def main() -> None:
    engine = create_engine(DATABASE_URL)
    schema = detect_gci_schema(engine)
    vectordb = get_vectordb()

    try:
        vectordb.count(COLLECTION_NAME)
    except RedisVectorBackendUnavailable as exc:
        raise RuntimeError(
            "Redis endpoint is reachable, but vector indexing is unavailable. "
            "Use Redis Stack or enable RediSearch on the target Redis service."
        ) from exc

    print("\nFetching recent GCI rows...")
    gci_rows = fetch_gci(engine, schema)
    hashes = [row["semantic_hash"] for row in gci_rows]
    print(f"Loaded {len(hashes)} semantic hashes from GCI.")

    redis_found: set[str] = set()
    for start in range(0, len(hashes), BATCH):
        batch = hashes[start : start + BATCH]
        redis_found.update(vectordb.exists(collection=COLLECTION_NAME, ids=batch))

    print("\n=== GCI <-> Redis Matching Table ===\n")
    print(f"{'Semantic Hash':<70} | {'GCI'} | {'Redis'} | Match")
    print("-" * 114)

    for row in gci_rows:
        semantic_hash = row["semantic_hash"]
        in_redis = "YES" if semantic_hash in redis_found else "NO"
        match = "OK" if semantic_hash in redis_found else "NO"
        print(f"{semantic_hash:<70} | {'YES':^3} | {in_redis:^7} | {match}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
