"""Show which recent global_content_index hashes exist in Milvus."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text

try:
    from pymilvus import Collection, connections, utility
except ImportError as exc:
    raise ImportError("PyMilvus not installed. Run: pip install pymilvus>=2.4.0") from exc


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
MILVUS_ALIAS = os.getenv("MILVUS_ALIAS", "gci_milvus_match")

COLLECTION_NAME = os.getenv("MAI_COLLECTION", "ingested_content")
GCI_TABLE = "global_content_index"
DOC_ID_FIELD = "doc_id"

LIMIT = int(os.getenv("GCI_MILVUS_MATCH_LIMIT", "200"))
BATCH = int(os.getenv("GCI_MILVUS_MATCH_BATCH", "64"))

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


def milvus_lookup(collection: Collection, hashes: list[str]) -> set[str]:
    found: set[str] = set()

    for start in range(0, len(hashes), BATCH):
        batch = hashes[start : start + BATCH]
        ids_expr = ", ".join(f'"{semantic_hash}"' for semantic_hash in batch)
        try:
            rows = collection.query(
                expr=f'{DOC_ID_FIELD} in [{ids_expr}]',
                output_fields=[DOC_ID_FIELD],
            )
        except Exception:
            continue

        for row in rows:
            found.add(str(row[DOC_ID_FIELD]))

    return found


def main() -> None:
    if not DATABASE_URL:
        raise RuntimeError("Set DATABASE_URL first.")

    engine = create_engine(DATABASE_URL)
    gci_schema = detect_gci_schema(engine)

    print("\nFetching recent GCI rows...")
    gci_rows = fetch_gci(engine, gci_schema)
    hashes = [row["semantic_hash"] for row in gci_rows]
    print(f"Loaded {len(hashes)} semantic hashes from GCI.")

    connect_milvus()
    if not utility.has_collection(COLLECTION_NAME, using=MILVUS_ALIAS):
        print(f"\nMilvus collection '{COLLECTION_NAME}' does not exist.")
        connections.disconnect(MILVUS_ALIAS)
        return

    collection = Collection(name=COLLECTION_NAME, using=MILVUS_ALIAS)
    collection.load()
    milvus_found = milvus_lookup(collection, hashes)

    print("\n=== GCI <-> Milvus Matching Table ===\n")
    print(f"{'Semantic Hash':<70} | {'GCI'} | {'Milvus'} | Match")
    print("-" * 115)

    for row in gci_rows:
        semantic_hash = row["semantic_hash"]
        in_milvus = "YES" if semantic_hash in milvus_found else "NO"
        match = "OK" if semantic_hash in milvus_found else "NO"
        print(f"{semantic_hash:<70} | {'YES':^3} | {in_milvus:^7} | {match}")

    connections.disconnect(MILVUS_ALIAS)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
