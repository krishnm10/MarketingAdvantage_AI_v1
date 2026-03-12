import json
import os
from pathlib import Path
from pprint import pprint

from dotenv import load_dotenv
from app.core.vectordb.weaviate_v1 import WeaviateVectorDB


ENV_PATH = Path(__file__).resolve().with_name(".env")
load_dotenv(ENV_PATH)

COLLECTION = os.getenv("MAI_COLLECTION", "ingested_content")
WEAVIATE_URL = os.getenv("WEAVIATE_URL") or None
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY") or None
WEAVIATE_EMBEDDED = os.getenv("WEAVIATE_EMBEDDED", "false").lower() in {
    "1",
    "true",
    "yes",
}
WEAVIATE_GRPC_HOST = os.getenv("WEAVIATE_GRPC_HOST") or None
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_SKIP_INIT_CHECKS = os.getenv("WEAVIATE_SKIP_INIT_CHECKS", "false").lower() in {
    "1",
    "true",
    "yes",
}
WEAVIATE_HEADERS_RAW = os.getenv("WEAVIATE_ADDITIONAL_HEADERS_JSON", "").strip()
WEAVIATE_ADDITIONAL_HEADERS = json.loads(WEAVIATE_HEADERS_RAW) if WEAVIATE_HEADERS_RAW else {}


def main() -> None:
    if not WEAVIATE_EMBEDDED and not WEAVIATE_URL:
        raise RuntimeError(f"Set WEAVIATE_URL in {ENV_PATH} or enable WEAVIATE_EMBEDDED")
    client = WeaviateVectorDB(
        url=WEAVIATE_URL or "http://invalid.local",
        api_key=WEAVIATE_API_KEY,
        additional_headers=WEAVIATE_ADDITIONAL_HEADERS,
        embedded=WEAVIATE_EMBEDDED,
        grpc_host=WEAVIATE_GRPC_HOST,
        grpc_port=WEAVIATE_GRPC_PORT,
        skip_init_checks=WEAVIATE_SKIP_INIT_CHECKS,
    )

    print(f"Collection: {COLLECTION}")
    print(f"Total records: {client.count(COLLECTION)}")

    results = client.get_all(
        collection=COLLECTION,
        include=["documents", "metadatas"],
    )
    ids = results.get("ids", [])[:5]
    documents = results.get("documents", [])[:5]
    metadatas = results.get("metadatas", [])[:5]

    for idx, vector_id in enumerate(ids, start=1):
        print(f"\n--- Document {idx} ---")
        print("ID:", vector_id)
        print("Metadata:")
        pprint(metadatas[idx - 1] if idx - 1 < len(metadatas) else {})
        text = documents[idx - 1] if idx - 1 < len(documents) else ""
        print("Text Snippet:", str(text)[:500])


if __name__ == "__main__":
    main()
