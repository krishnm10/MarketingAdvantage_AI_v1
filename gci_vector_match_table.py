import os
import chromadb
from sqlalchemy import create_engine, text
from chromadb.config import Settings

DATABASE_URL = os.getenv("DATABASE_URL","postgresql://postgres:Mahadeva%40123@localhost/marketing_advantage")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "ingested_content_local")

LIMIT = 200   # how many recent GCI rows to inspect
BATCH = 32


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


def chroma_lookup(collection, hashes):
    found = set()
    for i in range(0, len(hashes), BATCH):
        batch = hashes[i:i+BATCH]
        try:
            resp = collection.get(
                ids=batch,
                include=["metadatas"]
            )
        except Exception:
            continue

        # Chroma returns dict with metadata list aligned to ids
        ids = resp.get("ids", [])
        for _id in ids:
            found.add(_id)

    return found


def main():
    if not DATABASE_URL:
        raise RuntimeError("Set DATABASE_URL first.")

    engine = create_engine(DATABASE_URL)

    # Fetch recent GCI rows
    print("\nFetching recent GCI rows...")
    gci_rows = fetch_gci(engine)
    hashes = [r["semantic_hash"] for r in gci_rows]

    print(f"Loaded {len(hashes)} semantic hashes from GCI.")

    # Connect to Chroma
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    chroma_found = chroma_lookup(collection, hashes)

    print("\n=== GCI ↔ Chroma Matching Table ===\n")
    print(f"{'Semantic Hash':<70} | {'GCI'} | {'Chroma'} | Match")
    print("-" * 110)

    for r in gci_rows:
        h = r["semantic_hash"]
        in_gci = "YES"
        in_chroma = "YES" if h in chroma_found else "NO"
        match = "✔" if h in chroma_found else "✘"
        print(f"{h:<70} | {in_gci:^3} | {in_chroma:^7} | {match}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
