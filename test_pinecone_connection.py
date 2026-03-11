"""Pinecone setup smoke test.

Usage:
    python test_pinecone_connection.py

Reads Pinecone config from environment and validates that the configured
backend can be created and used for a minimal vector write/read cycle.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def main() -> int:
    load_dotenv(_project_root() / ".env")

    mode = os.getenv("PINECONE_MODE", "cloud").strip().lower()
    index_name = os.getenv("PINECONE_INDEX_NAME", "ingested-content").strip()
    namespace = os.getenv("PINECONE_NAMESPACE", "default").strip() or "default"
    metric = os.getenv("PINECONE_METRIC", "cosine").strip() or "cosine"
    embedding_dim = int(os.getenv("PINECONE_EMBEDDING_DIM", "3") or "3")
    local_path = os.getenv("PINECONE_LOCAL_PATH", "./pinecone_local_db").strip() or "./pinecone_local_db"
    api_key = os.getenv("PINECONE_API_KEY", "").strip() or None

    if mode != "local" and not api_key:
        print("PINECONE_API_KEY is required when PINECONE_MODE=cloud", file=sys.stderr)
        return 1

    from app.core.vectordb.pinecone_v1 import PineconeVectorDB

    cleanup_path: Path | None = None
    if mode == "local":
        cleanup_path = (_project_root() / ".tmp_pinecone_smoke").resolve()
        shutil.rmtree(cleanup_path, ignore_errors=True)
        local_path = str(cleanup_path)

    print(
        f"Testing Pinecone backend | mode={mode} | index={index_name} "
        f"| namespace={namespace} | metric={metric} | dim={embedding_dim}"
    )

    vdb = PineconeVectorDB(
        mode=mode,
        api_key=api_key,
        index_name=index_name,
        namespace=namespace,
        embedding_dim=embedding_dim,
        metric=metric,
        cloud=os.getenv("PINECONE_CLOUD", "aws"),
        region=os.getenv("PINECONE_REGION", "us-east-1"),
        pod_type=os.getenv("PINECONE_POD_TYPE") or None,
        local_path=local_path if mode == "local" else None,
    )

    collection = index_name
    vdb.ensure_collection(collection, embedding_dim=embedding_dim, distance_metric=metric)

    doc_id = "pinecone-smoke-doc"
    vector = [0.1] * embedding_dim
    vdb.upsert(
        collection=collection,
        doc_id=doc_id,
        embedding=vector,
        text="pinecone smoke test",
        metadata={"source": "smoke-test"},
    )

    existing = vdb.exists(collection=collection, ids=[doc_id])
    hits = vdb.search(collection=collection, query_embedding=vector, top_k=1)

    print(f"exists={existing}")
    print(f"search_hits={len(hits)}")

    if doc_id not in existing:
        print("Smoke test failed: inserted vector was not found.", file=sys.stderr)
        return 2

    vdb.delete_many(collection=collection, doc_ids=[doc_id])

    if cleanup_path is not None:
        shutil.rmtree(cleanup_path, ignore_errors=True)

    print("Pinecone setup OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
