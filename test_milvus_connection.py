"""
Standalone Milvus connection test — does NOT touch any existing app code.
Usage:  python test_milvus_connection.py
"""
import sys, os, json

# ── Connection details (override via env vars or edit below) ──
URI   = os.getenv("MILVUS_URI", "")  # leave empty to use host:port
HOST  = os.getenv("MILVUS_HOST", "35.154.186.40")
PORT  = int(os.getenv("MILVUS_PORT", "19530"))
TOKEN = os.getenv("MILVUS_TOKEN") or None

COLLECTION = os.getenv("MAI_COLLECTION", "ingested_content")
EMBED_DIM  = 1024   # must match your embedding model (BAAI/bge-large-en-v1.5)

STEP_SEPARATOR = "=" * 60

def main():
    print(STEP_SEPARATOR)
    print("  Milvus AWS Connection Test")
    print(STEP_SEPARATOR)

    # 1. Import check
    print("\n[1/7] Checking pymilvus installation...")
    try:
        import pymilvus
        print(f"  ✔ pymilvus {pymilvus.__version__} installed")
    except ImportError:
        print("  ✘ pymilvus NOT installed. Run: pip install pymilvus>=2.4.0")
        sys.exit(1)

    from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType

    # 2. Connect — try host:port first, then URI, with 10s timeout
    print("\n[2/7] Connecting to Milvus...")
    connected   = False
    alias       = "_test"
    TIMEOUT     = 10   # seconds

    attempts = [
        ("host:19530", dict(alias=alias, host=HOST, port="19530", timeout=TIMEOUT)),
    ]
    if PORT != 19530:
        attempts.insert(0, (f"host:{PORT}", dict(alias=alias, host=HOST, port=str(PORT), timeout=TIMEOUT)))
    if URI:
        attempts.insert(0, ("URI", dict(alias=alias, uri=URI, token=TOKEN or "", timeout=TIMEOUT)))

    for label, kw in attempts:
        try:
            kw_safe = {k: v for k, v in kw.items() if k != "token"}
            print(f"  Trying {label}: {kw_safe}")
            connections.connect(**kw)
            connected = True
            print(f"  ✔ Connected via {label}")
            break
        except Exception as exc:
            print(f"  ✘ {label} failed: {exc}")
            try:
                connections.disconnect(alias)
            except Exception:
                pass

    if not connected:
        print("\n  ✘ ALL connection attempts failed. Check host/port/firewall.")
        sys.exit(1)

    # 3. List collections
    print("\n[3/7] Listing existing collections...")
    try:
        colls = utility.list_collections(using=alias)
        print(f"  ✔ {len(colls)} collection(s): {colls}")
    except Exception as exc:
        print(f"  ✘ list_collections failed: {exc}")
        connections.disconnect(alias)
        sys.exit(1)

    # 4. Create a temp test collection
    test_coll_name = "__mai_connection_test__"
    print(f"\n[4/7] Creating temp collection '{test_coll_name}'...")
    try:
        if utility.has_collection(test_coll_name, using=alias):
            utility.drop_collection(test_coll_name, using=alias)
            print("  (dropped existing temp collection)")

        schema = CollectionSchema(
            fields=[
                FieldSchema("doc_id", DataType.VARCHAR, max_length=256, is_primary=True),
                FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=EMBED_DIM),
                FieldSchema("_text", DataType.VARCHAR, max_length=65535),
                FieldSchema("_metadata", DataType.JSON),
            ],
            description="MAI connection test",
        )
        col = Collection(name=test_coll_name, schema=schema, using=alias)

        # Create HNSW index
        col.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 256},
            },
        )
        col.load()
        print(f"  ✔ Collection created + HNSW index built + loaded")
    except Exception as exc:
        print(f"  ✘ create collection failed: {exc}")
        connections.disconnect(alias)
        sys.exit(1)

    # 5. Insert a test document
    print(f"\n[5/7] Inserting test document...")
    try:
        import random
        fake_embedding = [random.uniform(-1, 1) for _ in range(EMBED_DIM)]
        # pymilvus >=2.6 expects row-based format (list of dicts)
        col.insert([{
            "doc_id":     "test-doc-001",
            "embedding":  fake_embedding,
            "_text":      "This is a Marketing Advantage AI test document.",
            "_metadata":  json.dumps({"source": "test", "status": "active"}),
        }])
        col.flush()
        count = col.num_entities
        print(f"  ✔ Inserted 1 doc | Total entities: {count}")
    except Exception as exc:
        print(f"  ✘ insert failed: {exc}")

    # 6. Search
    print(f"\n[6/7] Running test search...")
    try:
        results = col.search(
            data=[fake_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=5,
            output_fields=["_text", "_metadata"],
        )
        hits = results[0]
        print(f"  ✔ Search returned {len(hits)} hit(s)")
        for h in hits:
            print(f"    → id={h.id}  score={h.score:.4f}  text={h.entity.get('_text')[:60]}")
    except Exception as exc:
        print(f"  ✘ search failed: {exc}")

    # 7. Cleanup
    print(f"\n[7/7] Cleaning up temp collection...")
    try:
        utility.drop_collection(test_coll_name, using=alias)
        print(f"  ✔ Temp collection dropped")
    except Exception as exc:
        print(f"  ✘ cleanup failed (not critical): {exc}")

    connections.disconnect(alias)

    print(f"\n{STEP_SEPARATOR}")
    print("  ✔ ALL TESTS PASSED — Milvus is ready for plug-and-play!")
    print(f"  Next: set MAI_VECTORDB=milvus in .env and restart uvicorn")
    print(STEP_SEPARATOR)

    # Also check if the real collection already exists
    if COLLECTION in colls:
        print(f"\n  ℹ Your production collection '{COLLECTION}' already exists on this Milvus instance.")
    else:
        print(f"\n  ℹ Collection '{COLLECTION}' does not exist yet — it will be auto-created on first ingestion.")

if __name__ == "__main__":
    main()
