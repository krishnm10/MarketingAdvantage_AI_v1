"""Standalone Milvus connection test.

Usage:
    python test_milvus_connection.py
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().with_name(".env")
load_dotenv(ENV_PATH)

URI = os.getenv("MILVUS_URI", "").strip()
HOST = os.getenv("MILVUS_HOST", "").strip()
PORT_RAW = os.getenv("MILVUS_PORT", "").strip()
TOKEN = os.getenv("MILVUS_TOKEN") or None
DB_NAME = os.getenv("MILVUS_DB_NAME", "default")

COLLECTION = os.getenv("MAI_COLLECTION", "ingested_content")
EMBED_DIM = int(os.getenv("MAI_EMBED_DIM", "1024"))
PORT = int(PORT_RAW) if PORT_RAW else 19530

STEP_SEPARATOR = "=" * 60


def build_attempts(alias: str, timeout_seconds: int) -> list[tuple[str, dict[str, Any]]]:
    attempts: list[tuple[str, dict[str, Any]]] = []
    if URI:
        params: dict[str, Any] = {"alias": alias, "uri": URI, "timeout": timeout_seconds}
        if TOKEN:
            params["token"] = TOKEN
        attempts.append(("uri", params))

    if HOST:
        host_params: dict[str, Any] = {
            "alias": alias,
            "host": HOST,
            "port": str(PORT),
            "db_name": DB_NAME,
            "timeout": timeout_seconds,
        }
        if TOKEN:
            host_params["token"] = TOKEN
        attempts.append((f"host:{PORT}", host_params))

    if HOST and PORT != 19530:
        fallback = dict(host_params)
        fallback["port"] = "19530"
        attempts.append(("host:19530", fallback))

    return attempts


def main() -> None:
    if not URI and not HOST:
        print(f"  FAIL missing Milvus config. Set MILVUS_URI or MILVUS_HOST in {ENV_PATH}")
        sys.exit(1)

    print(STEP_SEPARATOR)
    print("  Milvus Connection Test")
    print(STEP_SEPARATOR)

    print("\n[1/7] Checking pymilvus installation...")
    try:
        import pymilvus
    except ImportError:
        print("  FAIL pymilvus is not installed. Run: pip install pymilvus>=2.4.0")
        sys.exit(1)

    print(f"  OK pymilvus {pymilvus.__version__} installed")

    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

    print("\n[2/7] Connecting to Milvus...")
    alias = "_milvus_test"
    timeout_seconds = 10
    connected = False

    for label, params in build_attempts(alias, timeout_seconds):
        try:
            safe_params = {key: value for key, value in params.items() if key != "token"}
            print(f"  Trying {label}: {safe_params}")
            connections.connect(**params)
            connected = True
            print(f"  OK connected via {label}")
            break
        except Exception as exc:
            print(f"  FAIL {label}: {exc}")
            try:
                connections.disconnect(alias)
            except Exception:
                pass

    if not connected:
        print("\n  FAIL all connection attempts failed. Check host, port, token, and firewall.")
        sys.exit(1)

    print("\n[3/7] Listing existing collections...")
    try:
        collections = utility.list_collections(using=alias)
        print(f"  OK found {len(collections)} collection(s): {collections}")
    except Exception as exc:
        print(f"  FAIL list_collections failed: {exc}")
        connections.disconnect(alias)
        sys.exit(1)

    test_collection = "__mai_connection_test__"
    print(f"\n[4/7] Creating temp collection '{test_collection}'...")
    try:
        if utility.has_collection(test_collection, using=alias):
            utility.drop_collection(test_collection, using=alias)
            print("  Removed existing temp collection first")

        schema = CollectionSchema(
            fields=[
                FieldSchema("doc_id", DataType.VARCHAR, max_length=512, is_primary=True, auto_id=False),
                FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=EMBED_DIM),
                FieldSchema("_text", DataType.VARCHAR, max_length=65535),
                FieldSchema("_metadata", DataType.JSON),
            ],
            description="MAI connection test",
            enable_dynamic_field=True,
        )
        collection = Collection(name=test_collection, schema=schema, using=alias)
        collection.create_index(
            field_name="embedding",
            index_name="embedding_idx",
            index_params={
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 256},
            },
        )
        collection.load()
        print("  OK collection created, indexed, and loaded")
    except Exception as exc:
        print(f"  FAIL create collection failed: {exc}")
        connections.disconnect(alias)
        sys.exit(1)

    print("\n[5/7] Inserting test document...")
    fake_embedding = [random.uniform(-1, 1) for _ in range(EMBED_DIM)]
    try:
        collection.insert(
            [
                {
                    "doc_id": "test-doc-001",
                    "embedding": fake_embedding,
                    "_text": "This is a Marketing Advantage AI test document.",
                    "_metadata": {"source": "test", "status": "active"},
                }
            ]
        )
        collection.flush()
        print(f"  OK inserted 1 doc. Total entities: {collection.num_entities}")
    except Exception as exc:
        print(f"  FAIL insert failed: {exc}")

    print("\n[6/7] Running test search...")
    try:
        results = collection.search(
            data=[fake_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=5,
            output_fields=["_text", "_metadata"],
        )
        hits = results[0]
        print(f"  OK search returned {len(hits)} hit(s)")
        for hit in hits:
            text_preview = str(hit.entity.get("_text") or "")[:60]
            print(f"    id={hit.id} score={hit.score:.4f} text={text_preview}")
    except Exception as exc:
        print(f"  FAIL search failed: {exc}")

    print(f"\n[7/7] Cleaning up temp collection '{test_collection}'...")
    try:
        utility.drop_collection(test_collection, using=alias)
        print("  OK temp collection dropped")
    except Exception as exc:
        print(f"  WARN cleanup failed: {exc}")

    connections.disconnect(alias)

    print(f"\n{STEP_SEPARATOR}")
    print("  Milvus connection test completed")
    print("  Next: set MAI_VECTORDB=milvus and restart the app if needed")
    print(STEP_SEPARATOR)

    if COLLECTION in collections:
        print(f"\n  Info: production collection '{COLLECTION}' already exists.")
    else:
        print(f"\n  Info: production collection '{COLLECTION}' does not exist yet.")


if __name__ == "__main__":
    main()
