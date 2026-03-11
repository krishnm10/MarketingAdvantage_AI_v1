"""Lightweight Weaviate connectivity check using .env configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from app.core.vectordb.weaviate_v1 import WeaviateVectorDB


ENV_PATH = Path(__file__).resolve().with_name(".env")
load_dotenv(ENV_PATH)


def _build_client() -> WeaviateVectorDB:
    headers_raw = os.getenv("WEAVIATE_ADDITIONAL_HEADERS_JSON", "").strip()
    additional_headers = json.loads(headers_raw) if headers_raw else {}

    return WeaviateVectorDB(
        url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
        api_key=os.getenv("WEAVIATE_API_KEY") or None,
        additional_headers=additional_headers,
        embedded=os.getenv("WEAVIATE_EMBEDDED", "false").lower() in ("1", "true", "yes"),
        grpc_host=os.getenv("WEAVIATE_GRPC_HOST") or None,
        grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
        skip_init_checks=os.getenv("WEAVIATE_SKIP_INIT_CHECKS", "false").lower() in ("1", "true", "yes"),
    )


def main() -> None:
    client = _build_client()
    print("Checking Weaviate...")
    if client.health_check():
        print("Weaviate connection successful.")
        print(f"READY -> {os.getenv('WEAVIATE_URL', 'http://localhost:8080')}")
        return
    raise RuntimeError("Weaviate health check returned false.")


if __name__ == "__main__":
    main()
