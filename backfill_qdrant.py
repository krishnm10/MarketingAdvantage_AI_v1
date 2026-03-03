# backfill_qdrant.py — idempotent backfill from global_content_index -> Qdrant (LOCAL)

import os
import uuid
import time
from sqlalchemy import create_engine, text, inspect
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from app.config.ingestion_settings import EMBEDDING_MODEL_NAME

# Optional fallback embedder
try:
    from sentence_transformers import SentenceTransformer
    _SBERT = SentenceTransformer("BAAI/bge-large-en-v1.5")
    print("Embedder:", EMBEDDING_MODEL_NAME)
except Exception:
    _SBERT = None

# -------- CONFIG ----------
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:Mahadeva%40123@localhost/marketing_advantage"
)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("MAI_COLLECTION", "ingested_content")

GCI_TABLE = "global_content_index"

BATCH = 16
SLEEP_BETWEEN_BATCHES = 0.2
# --------------------------

if not DATABASE_URL:
    raise RuntimeError("Set DATABASE_URL (sync URL).")

engine = create_engine(DATABASE_URL)
insp = inspect(engine)

# detect schema
GCI_TABLE_SCHEMA = None
for schema in insp.get_schema_names():
    if GCI_TABLE in insp.get_table_names(schema=schema):
        GCI_TABLE_SCHEMA = schema
        break
if not GCI_TABLE_SCHEMA:
    raise RuntimeError("global_content_index not found")

print("Using GCI table:", GCI_TABLE_SCHEMA + "." + GCI_TABLE)

# Qdrant client (LOCAL)
client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
)
# Qdrant client (LOCAL)
client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
)

# ---- Display collection count ----
try:
    info = client.get_collection(collection_name=QDRANT_COLLECTION)
    count = info.points_count or 0
    print(f"Qdrant collection ready: {QDRANT_COLLECTION} | count: {count}")
except Exception:
    print(f"Collection '{QDRANT_COLLECTION}' does not exist yet.")
    count = 0
print("Qdrant connected:", QDRANT_COLLECTION)


# ---------- Deterministic ID (MUST match your vectordb code) ----------
def to_qdrant_id(doc_id: str) -> str:
    try:
        uuid.UUID(doc_id)
        return doc_id
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))


# ---------- Embedding ----------
def compute_embeddings(texts):
    if _SBERT:
        embs = _SBERT.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return [list(e) for e in embs]

    raise RuntimeError("No embedder available.")


# ---------- Load GCI rows ----------
with engine.connect() as conn:
    q = text(f"""
        SELECT id, semantic_hash, cleaned_text, raw_text
        FROM {GCI_TABLE_SCHEMA}.{GCI_TABLE}
        ORDER BY created_at DESC
    """)
    rows = conn.execute(q).fetchall()

    def row_map(r):
        if hasattr(r, "_mapping"):
            return dict(r._mapping)
        return dict(r)

    rows = [row_map(r) for r in rows]

hashes = [r["semantic_hash"] for r in rows if r.get("semantic_hash")]
print("GCI total with semantic_hash:", len(hashes))


# ---------- Check existing in Qdrant ----------
missing_hashes = []

for i in range(0, len(hashes), BATCH):
    batch = hashes[i:i+BATCH]
    q_ids = [to_qdrant_id(h) for h in batch]

    try:
        results = client.retrieve(
            collection_name=QDRANT_COLLECTION,
            ids=q_ids,
            with_payload=False,
            with_vectors=False,
        )
        found_ids = {str(r.id) for r in results}
    except Exception:
        found_ids = set()

    uuid_map = {to_qdrant_id(h): h for h in batch}

    for qid in q_ids:
        if qid not in found_ids:
            missing_hashes.append(uuid_map[qid])

print("Missing hashes count:", len(missing_hashes))

if not missing_hashes:
    print("Nothing to backfill. Exiting.")
    exit(0)


# ---------- Build lookup ----------
gci_map = {r["semantic_hash"]: r for r in rows if r.get("semantic_hash")}


# ---------- Backfill ----------
for i in range(0, len(missing_hashes), BATCH):
    batch_hashes = missing_hashes[i:i+BATCH]

    texts = []
    metas = []

    for h in batch_hashes:
        entry = gci_map.get(h)
        text_to_embed = (
            entry.get("cleaned_text")
            or entry.get("raw_text")
            or ""
        )[:15000]

        texts.append(text_to_embed)

        metas.append({
            "text": text_to_embed,
            "global_content_id": str(entry.get("id")),
            "semantic_hash": h,
        })

    try:
        embs = compute_embeddings(texts)
    except Exception as e:
        raise RuntimeError("Embedding failed: " + str(e))

    points = [
        qdrant_models.PointStruct(
            id=to_qdrant_id(batch_hashes[idx]),
            vector=embs[idx],
            payload=metas[idx],
        )
        for idx in range(len(batch_hashes))
    ]

    try:
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points,
        )
        print(f"Upserted batch {i//BATCH + 1}: {len(batch_hashes)} items")
    except Exception as e:
        print("Qdrant upsert failed:", e)

    time.sleep(SLEEP_BETWEEN_BATCHES)

print("Backfill completed.")