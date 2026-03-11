"""Safe full wipe script for the configured Weaviate collection."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from app.config.ingestion_settings import EMBEDDING_DIMENSION
from app.core.vectordb.weaviate_v1 import WeaviateVectorDB
from app.utils.logger import log_info


ENV_PATH = Path(__file__).resolve().with_name(".env")
load_dotenv(ENV_PATH)

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY") or None
WEAVIATE_EMBEDDED = os.getenv("WEAVIATE_EMBEDDED", "false").lower() in ("1", "true", "yes")
WEAVIATE_GRPC_HOST = os.getenv("WEAVIATE_GRPC_HOST") or None
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_SKIP_INIT_CHECKS = os.getenv("WEAVIATE_SKIP_INIT_CHECKS", "false").lower() in ("1", "true", "yes")
WEAVIATE_HEADERS_RAW = os.getenv("WEAVIATE_ADDITIONAL_HEADERS_JSON", "").strip()
WEAVIATE_HEADERS = json.loads(WEAVIATE_HEADERS_RAW) if WEAVIATE_HEADERS_RAW else {}
COLLECTION_NAME = os.getenv("MAI_COLLECTION", "ingested_content")


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


def clear_weaviate_collection() -> None:
    vectordb = get_vectordb()
    log_info("[WeaviateCleanup] Starting Weaviate collection cleanup...")
    try:
        vectordb.delete_collection(COLLECTION_NAME)
        log_info(f"[WeaviateCleanup] Deleted collection '{COLLECTION_NAME}' if it existed.")
    except Exception as exc:
        log_info(f"[WeaviateCleanup] Delete skipped for '{COLLECTION_NAME}': {exc}")

    vectordb.ensure_collection(
        COLLECTION_NAME,
        embedding_dim=EMBEDDING_DIMENSION,
        distance_metric="cosine",
    )
    log_info(
        f"[WeaviateCleanup] Recreated empty '{COLLECTION_NAME}' collection with dim={EMBEDDING_DIMENSION}."
    )


if __name__ == "__main__":
    clear_weaviate_collection()
