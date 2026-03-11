"""Chroma setup smoke test."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> int:
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")

    host = os.getenv("CHROMA_HOST", "").strip() or None
    local_path = os.getenv("CHROMA_PATH", "./chroma_db").strip() or "./chroma_db"
    cleanup_path: Path | None = None
    if not host:
        cleanup_path = (root / ".tmp_chroma_smoke").resolve()
        shutil.rmtree(cleanup_path, ignore_errors=True)
        local_path = str(cleanup_path)

    from app.core.vectordb.chroma_v1 import ChromaVectorDB

    vdb = ChromaVectorDB(
        persist_directory=local_path if not host else None,
        anonymized_telemetry=os.getenv("CHROMA_TELEMETRY", "false").lower() in ("1", "true", "yes"),
        host=host,
        port=int(os.getenv("CHROMA_PORT", "8000")),
        ssl=os.getenv("CHROMA_SSL", "false").lower() in ("1", "true", "yes"),
        api_key=os.getenv("CHROMA_API_KEY") or None,
        tenant=os.getenv("CHROMA_TENANT", "default_tenant"),
        database=os.getenv("CHROMA_DATABASE", "default_database"),
    )

    collection = "__chroma_connection_test__"
    vector = [0.1, 0.2, 0.3]
    vdb.ensure_collection(collection, embedding_dim=len(vector), distance_metric="cosine")
    vdb.upsert(collection=collection, doc_id="chroma-smoke-doc", embedding=vector, text="chroma smoke test", metadata={"source": "smoke-test"})
    hits = vdb.search(collection=collection, query_embedding=vector, top_k=1)
    print(f"search_hits={len(hits)}")
    vdb.delete_many(collection=collection, doc_ids=["chroma-smoke-doc"])
    vdb.delete_collection(collection)

    if cleanup_path is not None:
        shutil.rmtree(cleanup_path, ignore_errors=True)

    print("Chroma setup OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
