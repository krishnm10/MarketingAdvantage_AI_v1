"""Safe full wipe script for Qdrant."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from app.config.ingestion_settings import EMBEDDING_DIMENSION
from app.utils.logger import log_info


ENV_PATH = Path(__file__).resolve().with_name(".env")
load_dotenv(ENV_PATH)

QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_HOST = os.getenv("QDRANT_HOST", "").strip()
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_PREFER_GRPC = os.getenv("QDRANT_PREFER_GRPC", "false").strip().lower() == "true"
COLLECTION_NAME = os.getenv("MAI_COLLECTION", "ingested_content")


def get_client() -> QdrantClient:
    if QDRANT_URL:
        log_info(f"[QdrantCleanup] Connecting to Qdrant URL {QDRANT_URL}...")
        return QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            prefer_grpc=QDRANT_PREFER_GRPC,
            timeout=30,
        )

    if not QDRANT_HOST:
        raise RuntimeError(f"Set QDRANT_URL or QDRANT_HOST in {ENV_PATH}")

    log_info(f"[QdrantCleanup] Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY,
        prefer_grpc=QDRANT_PREFER_GRPC,
        timeout=30,
    )


def clear_qdrant_collection() -> None:
    try:
        client = get_client()
        collections = {col.name for col in client.get_collections().collections}

        if COLLECTION_NAME in collections:
            log_info(f"[QdrantCleanup] Found collection '{COLLECTION_NAME}'. Deleting...")
            client.delete_collection(collection_name=COLLECTION_NAME)
            log_info(f"[QdrantCleanup] Deleted collection '{COLLECTION_NAME}'.")
        else:
            log_info(f"[QdrantCleanup] No collection named '{COLLECTION_NAME}' found.")

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qdrant_models.VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=qdrant_models.Distance.COSINE,
            ),
        )
        log_info(
            f"[QdrantCleanup] Recreated empty '{COLLECTION_NAME}' collection with dim={EMBEDDING_DIMENSION}."
        )
    except Exception as exc:
        log_info(f"[ERROR] Failed to clear Qdrant: {exc}")


if __name__ == "__main__":
    log_info("[QdrantCleanup] Starting full Qdrant cleanup...")
    clear_qdrant_collection()
    log_info("[QdrantCleanup] Cleanup complete.")
