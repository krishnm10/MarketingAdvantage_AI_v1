import os
import uuid
from sqlalchemy import create_engine, text
from qdrant_client import QdrantClient

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:Mahadeva%40123@localhost/marketing_advantage"
)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("MAI_COLLECTION", "ingested_content")

LIMIT = 200
BATCH = 64


# ---------------------------------------------------------
# SAME deterministic ID logic used in QdrantVectorDB
# ---------------------------------------------------------
def to_qdrant_id(doc_id: str) -> str:
    try:
        uuid.UUID(doc_id)
        return doc_id
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))


# ---------------------------------------------------------
# Fetch recent GCI hashes from PostgreSQL
# ---------------------------------------------------------
def fetch_gci(engine):
    query = text("""
        SELECT semantic_hash, id, created_at
        FROM public.global_content_index
        ORDER BY created_at DESC
        LIMIT :lim
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"lim": LIMIT}).fetchall()

    return [{"semantic_hash": r[0], "gci_id": r[1]} for r in rows]


# ---------------------------------------------------------
# Qdrant lookup
# ---------------------------------------------------------
def qdrant_lookup(client, collection, hashes):
    found = set()

    for i in range(0, len(hashes), BATCH):
        batch = hashes[i:i+BATCH]

        qdrant_ids = [to_qdrant_id(h) for h in batch]

        try:
            results = client.retrieve(
                collection_name=collection,
                ids=qdrant_ids,
                with_payload=False,
                with_vectors=False,
            )
        except Exception:
            continue

        # Map back UUID → original semantic_hash
        uuid_to_hash = {
            to_qdrant_id(h): h
            for h in batch
        }

        for r in results:
            original_hash = uuid_to_hash.get(str(r.id))
            if original_hash:
                found.add(original_hash)

    return found


def main():
    if not DATABASE_URL:
        raise RuntimeError("Set DATABASE_URL first.")

    engine = create_engine(DATABASE_URL)

    print("\nFetching recent GCI rows...")
    gci_rows = fetch_gci(engine)
    hashes = [r["semantic_hash"] for r in gci_rows]

    print(f"Loaded {len(hashes)} semantic hashes from GCI.")

    # Connect to Qdrant
    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )

    qdrant_found = qdrant_lookup(client, COLLECTION_NAME, hashes)

    print("\n=== GCI ↔ Qdrant Matching Table ===\n")
    print(f"{'Semantic Hash':<70} | {'GCI'} | {'Qdrant'} | Match")
    print("-" * 115)

    for r in gci_rows:
        h = r["semantic_hash"]
        in_gci = "YES"
        in_qdrant = "YES" if h in qdrant_found else "NO"
        match = "✔" if h in qdrant_found else "✘"

        print(f"{h:<70} | {in_gci:^3} | {in_qdrant:^7} | {match}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()