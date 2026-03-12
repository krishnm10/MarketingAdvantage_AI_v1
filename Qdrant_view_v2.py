import os
from pathlib import Path
from pprint import pprint

from dotenv import load_dotenv
from qdrant_client import QdrantClient


ENV_PATH = Path(__file__).resolve().with_name(".env")
load_dotenv(ENV_PATH)

COLLECTION = os.getenv("MAI_COLLECTION", "ingested_content")
QDRANT_URL = os.getenv("QDRANT_URL") or None
QDRANT_HOST = os.getenv("QDRANT_HOST") or None
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_PREFER_GRPC = os.getenv("QDRANT_PREFER_GRPC", "false").lower() in {
    "1",
    "true",
    "yes",
}
QDRANT_TIMEOUT = float(os.getenv("QDRANT_TIMEOUT", "30"))


def build_client() -> QdrantClient:
    if not QDRANT_URL and not QDRANT_HOST:
        raise RuntimeError(f"Set QDRANT_URL or QDRANT_HOST in {ENV_PATH}")
    if QDRANT_URL:
        return QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            prefer_grpc=QDRANT_PREFER_GRPC,
            timeout=QDRANT_TIMEOUT,
        )
    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        prefer_grpc=QDRANT_PREFER_GRPC,
        timeout=QDRANT_TIMEOUT,
    )


def main() -> None:
    client = build_client()
    total = client.count(collection_name=COLLECTION, exact=True).count
    print(f"Collection: {COLLECTION}")
    print(f"Total records: {total}")

    points, _ = client.scroll(
        collection_name=COLLECTION,
        limit=5,
        with_payload=True,
        with_vectors=False,
    )
    for idx, point in enumerate(points, start=1):
        payload = dict(point.payload or {})
        text = str(payload.pop("text", ""))
        print(f"\n--- Document {idx} ---")
        print("ID:", point.id)
        print("Metadata:")
        pprint(payload)
        print("Text Snippet:", text[:500])


if __name__ == "__main__":
    main()
