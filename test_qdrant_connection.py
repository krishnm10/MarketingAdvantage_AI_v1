"""Qdrant setup smoke test."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> int:
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")

    from app.core.vectordb.qdrant_v1 import QdrantVectorDB

    vdb = QdrantVectorDB(
        url=os.getenv("QDRANT_URL") or None,
        host=os.getenv("QDRANT_HOST") or None,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        api_key=os.getenv("QDRANT_API_KEY") or None,
        prefer_grpc=os.getenv("QDRANT_PREFER_GRPC", "false").lower() in ("1", "true", "yes"),
        timeout=int(float(os.getenv("QDRANT_TIMEOUT", "30"))),
    )

    collection = "__qdrant_connection_test__"
    vector = [0.1, 0.2, 0.3]
    vdb.ensure_collection(collection, embedding_dim=len(vector), distance_metric="cosine")
    vdb.upsert(collection=collection, doc_id="qdrant-smoke-doc", embedding=vector, text="qdrant smoke test", metadata={"source": "smoke-test"})
    hits = vdb.search(collection=collection, query_embedding=vector, top_k=1)
    print(f"search_hits={len(hits)}")
    vdb.delete_many(collection=collection, doc_ids=["qdrant-smoke-doc"])
    vdb.delete_collection(collection)
    print("Qdrant setup OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
