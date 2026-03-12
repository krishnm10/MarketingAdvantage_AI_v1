import os
from pathlib import Path
from pprint import pprint
from typing import Iterable, List

from dotenv import load_dotenv


ENV_PATH = Path(__file__).resolve().with_name(".env")
load_dotenv(ENV_PATH)

COLLECTION = os.getenv("MAI_COLLECTION", "ingested_content")
MODE = os.getenv("PINECONE_MODE", "cloud").strip().lower()
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ingested-content")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")
EMBEDDING_DIM = int(os.getenv("PINECONE_EMBEDDING_DIM", "1024"))
LOCAL_PATH = os.getenv("PINECONE_LOCAL_PATH") or "./pinecone_local_db"
API_KEY = os.getenv("PINECONE_API_KEY") or None


def iter_ids(index, namespace: str) -> Iterable[str]:
    if hasattr(index, "list_paginated"):
        token = None
        while True:
            response = index.list_paginated(namespace=namespace, pagination_token=token)
            vectors = getattr(response, "vectors", None) or response.get("vectors", [])
            for item in vectors:
                if isinstance(item, dict):
                    vector_id = item.get("id")
                else:
                    vector_id = getattr(item, "id", None)
                if vector_id:
                    yield str(vector_id)
            token = getattr(response, "pagination_token", None) or response.get("pagination_token")
            if not token:
                break
        return

    if hasattr(index, "list"):
        try:
            for item in index.list(namespace=namespace):
                if isinstance(item, dict):
                    vector_id = item.get("id")
                else:
                    vector_id = getattr(item, "id", None) or str(item)
                if vector_id:
                    yield str(vector_id)
        except TypeError:
            for item in index.list():
                if isinstance(item, dict):
                    vector_id = item.get("id")
                else:
                    vector_id = getattr(item, "id", None) or str(item)
                if vector_id:
                    yield str(vector_id)


def print_match(idx: int, vector_id: str, payload: dict) -> None:
    metadata = dict(payload or {})
    text = str(metadata.pop("_text", ""))
    print(f"\n--- Document {idx} ---")
    print("ID:", vector_id)
    print("Metadata:")
    pprint(metadata)
    print("Text Snippet:", text[:500])


def main() -> None:
    if MODE == "local":
        import chromadb

        if not os.getenv("PINECONE_LOCAL_PATH"):
            raise RuntimeError(f"Set PINECONE_LOCAL_PATH in {ENV_PATH} for local mode")
        client = chromadb.PersistentClient(path=LOCAL_PATH)
        collection = client.get_collection(COLLECTION)
        print(f"Collection: {COLLECTION}")
        print(f"Mode: local-emulation ({LOCAL_PATH})")
        print(f"Total records: {collection.count()}")
        results = collection.peek(limit=5)
        for idx, doc in enumerate(results.get("documents", []), start=1):
            print(f"\n--- Document {idx} ---")
            print("ID:", results["ids"][idx - 1])
            print("Metadata:")
            pprint(results["metadatas"][idx - 1])
            print("Text Snippet:", str(doc)[:500])
        return

    from pinecone import Pinecone

    if not API_KEY:
        raise RuntimeError(f"Set PINECONE_API_KEY in {ENV_PATH} for cloud mode")

    pc = Pinecone(api_key=API_KEY)
    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    namespaces = stats.get("namespaces", {}) if isinstance(stats, dict) else getattr(stats, "namespaces", {})
    namespace_stats = namespaces.get(NAMESPACE, {}) if isinstance(namespaces, dict) else {}
    total = namespace_stats.get("vector_count", 0) if isinstance(namespace_stats, dict) else 0

    print(f"Collection: {COLLECTION}")
    print(f"Index: {INDEX_NAME}")
    print(f"Namespace: {NAMESPACE}")
    print(f"Total records: {int(total)}")

    sample_ids: List[str] = []
    for vector_id in iter_ids(index, NAMESPACE):
        sample_ids.append(vector_id)
        if len(sample_ids) == 5:
            break

    if not sample_ids:
        print("\nSample view skipped: installed Pinecone client/index does not expose iterable IDs.")
        print("Count information above is still valid.")
        return

    fetched = index.fetch(ids=sample_ids, namespace=NAMESPACE)
    vectors = fetched.get("vectors", {}) if isinstance(fetched, dict) else getattr(fetched, "vectors", {})
    for idx, vector_id in enumerate(sample_ids, start=1):
        vector = vectors.get(vector_id, {}) if isinstance(vectors, dict) else {}
        payload = vector.get("metadata", {}) if isinstance(vector, dict) else getattr(vector, "metadata", {})
        print_match(idx, vector_id, payload or {})


if __name__ == "__main__":
    main()
