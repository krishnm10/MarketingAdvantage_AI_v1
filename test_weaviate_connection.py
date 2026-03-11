"""Weaviate setup smoke test."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> int:
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")

    headers_raw = os.getenv("WEAVIATE_ADDITIONAL_HEADERS_JSON", "").strip()
    additional_headers = json.loads(headers_raw) if headers_raw else {}

    from app.core.vectordb.weaviate_v1 import WeaviateVectorDB

    vdb = WeaviateVectorDB(
        url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
        api_key=os.getenv("WEAVIATE_API_KEY") or None,
        additional_headers=additional_headers,
        embedded=os.getenv("WEAVIATE_EMBEDDED", "false").lower() in ("1", "true", "yes"),
        grpc_host=os.getenv("WEAVIATE_GRPC_HOST") or None,
        grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
        skip_init_checks=os.getenv("WEAVIATE_SKIP_INIT_CHECKS", "false").lower() in ("1", "true", "yes"),
    )

    collection = "__weaviate_connection_test__"
    vector = [0.1, 0.2, 0.3]
    vdb.ensure_collection(collection, embedding_dim=len(vector), distance_metric="cosine")
    vdb.upsert(collection=collection, doc_id="weaviate-smoke-doc", embedding=vector, text="weaviate smoke test", metadata={"source": "smoke-test"})
    hits = vdb.search(collection=collection, query_embedding=vector, top_k=1)
    print(f"search_hits={len(hits)}")
    vdb.delete_many(collection=collection, doc_ids=["weaviate-smoke-doc"])
    vdb.delete_collection(collection)
    print("Weaviate setup OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
