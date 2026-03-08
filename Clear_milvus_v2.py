"""Safe full wipe script for Milvus."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from app.config.ingestion_settings import EMBEDDING_DIMENSION
from app.utils.logger import log_info


ENV_PATH = Path(__file__).resolve().with_name(".env")
load_dotenv(ENV_PATH)

MILVUS_URI = os.getenv("MILVUS_URI", "").strip()
MILVUS_HOST = os.getenv("MILVUS_HOST", "").strip()
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN") or None
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME", "default")
MILVUS_ALIAS = os.getenv("MILVUS_ALIAS", "clear_milvus_v2")
COLLECTION_NAME = os.getenv("MAI_COLLECTION", "ingested_content")


def connect_milvus() -> None:
    if MILVUS_URI:
        params: dict[str, Any] = {"alias": MILVUS_ALIAS, "uri": MILVUS_URI}
        if MILVUS_TOKEN:
            params["token"] = MILVUS_TOKEN
        log_info(f"[MilvusCleanup] Connecting to Milvus URI {MILVUS_URI}...")
        connections.connect(**params)
        return

    if not MILVUS_HOST:
        raise RuntimeError(f"Set MILVUS_URI or MILVUS_HOST in {ENV_PATH}")

    params = {
        "alias": MILVUS_ALIAS,
        "host": MILVUS_HOST,
        "port": str(MILVUS_PORT),
        "db_name": MILVUS_DB_NAME,
    }
    if MILVUS_TOKEN:
        params["token"] = MILVUS_TOKEN
    log_info(f"[MilvusCleanup] Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    connections.connect(**params)


def recreate_collection() -> None:
    schema = CollectionSchema(
        fields=[
            FieldSchema("doc_id", DataType.VARCHAR, max_length=512, is_primary=True, auto_id=False),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSION),
            FieldSchema("_text", DataType.VARCHAR, max_length=65535),
            FieldSchema("_metadata", DataType.JSON),
        ],
        description="MarketingAdvantage AI document chunks",
        enable_dynamic_field=True,
    )
    collection = Collection(name=COLLECTION_NAME, schema=schema, using=MILVUS_ALIAS)
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


def clear_milvus_collection() -> None:
    try:
        connect_milvus()

        if utility.has_collection(COLLECTION_NAME, using=MILVUS_ALIAS):
            log_info(f"[MilvusCleanup] Found collection '{COLLECTION_NAME}'. Deleting...")
            utility.drop_collection(COLLECTION_NAME, using=MILVUS_ALIAS)
            log_info(f"[MilvusCleanup] Deleted collection '{COLLECTION_NAME}'.")
        else:
            log_info(f"[MilvusCleanup] No collection named '{COLLECTION_NAME}' found.")

        recreate_collection()
        log_info(
            f"[MilvusCleanup] Recreated empty '{COLLECTION_NAME}' collection with dim={EMBEDDING_DIMENSION}."
        )
    except Exception as exc:
        log_info(f"[ERROR] Failed to clear Milvus: {exc}")
    finally:
        try:
            connections.disconnect(MILVUS_ALIAS)
        except Exception:
            pass


if __name__ == "__main__":
    log_info("[MilvusCleanup] Starting full Milvus cleanup...")
    clear_milvus_collection()
    log_info("[MilvusCleanup] Cleanup complete.")
