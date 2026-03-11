"""Show which recent global_content_index hashes exist in Weaviate."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text

from app.core.vectordb.weaviate_v1 import WeaviateVectorDB


ENV_PATH = Path(__file__).resolve().with_name(".env")
load_dotenv(ENV_PATH)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:Mahadeva%40123@localhost/marketing_advantage",
)

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY") or None
WEAVIATE_EMBEDDED = os.getenv("WEAVIATE_EMBEDDED", "false").lower() in ("1", "true", "yes")
WEAVIATE_GRPC_HOST = os.getenv("WEAVIATE_GRPC_HOST") or None
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_SKIP_INIT_CHECKS = os.getenv("WEAVIATE_SKIP_INIT_CHECKS", "false").lower() in ("1", "true", "yes")
WEAVIATE_HEADERS_RAW = os.getenv("WEAVIATE_ADDITIONAL_HEADERS_JSON", "").strip()
WEAVIATE_HEADERS = json.loads(WEAVIATE_HEADERS_RAW) if WEAVIATE_HEADERS_RAW else {}

COLLECTION_NAME = os.getenv("MAI_COLLECTION", "ingested_content")
GCI_TABLE = "global_content_index"
LIMIT = int(os.getenv("GCI_WEAVIATE_MATCH_LIMIT", "200"))
BATCH = int(os.getenv("GCI_WEAVIATE_MATCH_BATCH", "64"))


def get_vectordb() -> WeaviateVectorDB:
    return WeaviateVectorDB(
        url=WEAVIATE_URL,
        api_key=WEAVIATE_API_KEY,
        additional_headers=WEAVIATE_HEADERS,
        embedded=WEAVIATE_EMBEDDED,
        grpc_host=WEAVIATE_GRPC_HOST,
        grpc_port=WEAVIATE_GRPC_PORT,
        skip_init_checks=WEAVIATE_SKIP_INIT_CHECKS,
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
        data = dict(row._mapping) if hasattr(row, "_mapping") else dict(row)
        result.append({"semantic_hash": data["semantic_hash"], "gci_id": data["id"]})
    return result


def main() -> None:
    engine = create_engine(DATABASE_URL)
    schema = detect_gci_schema(engine)
    vectordb = get_vectordb()

    print("\nFetching recent GCI rows...")
    gci_rows = fetch_gci(engine, schema)
    hashes = [row["semantic_hash"] for row in gci_rows if row.get("semantic_hash")]
    print(f"Loaded {len(hashes)} semantic hashes from GCI.")

    weaviate_found: set[str] = set()
    for start in range(0, len(hashes), BATCH):
        batch = hashes[start : start + BATCH]
        weaviate_found.update(vectordb.exists(collection=COLLECTION_NAME, ids=batch))

    print("\n=== GCI <-> Weaviate Matching Table ===\n")
    print(f"{'Semantic Hash':<70} | {'GCI'} | {'Weaviate'} | Match")
    print("-" * 117)

    for row in gci_rows:
        semantic_hash = row["semantic_hash"]
        in_weaviate = "YES" if semantic_hash in weaviate_found else "NO"
        match = "OK" if semantic_hash in weaviate_found else "NO"
        print(f"{semantic_hash:<70} | {'YES':^3} | {in_weaviate:^10} | {match}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
