# backfill_chroma.py — idempotent backfill from global_content_index -> Chroma
# WARNING: set DATABASE_URL (sync) and CHROMA_PATH env vars before running

import os
import math
from sqlalchemy import create_engine, text, inspect
from chromadb import PersistentClient
import time
from app.config.ingestion_settings import EMBEDDING_MODEL_NAME

# Optional fallback embedder (sentence-transformers); replace with your project's embedder if available.
try:
    from sentence_transformers import SentenceTransformer
    _SBERT = SentenceTransformer("BAAI/bge-large-en-v1.5")
    print(EMBEDDING_MODEL_NAME)
except Exception:
    _SBERT = None

# -------- CONFIG ----------
DATABASE_URL = os.environ.get("DATABASE_URL","postgresql://postgres:Mahadeva%40123@localhost/marketing_advantage")  # must be sync-style: postgresql://user:pass@host/db
CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_db")
GCI_TABLE_SCHEMA = None   # auto-detected below
GCI_TABLE = "global_content_index"
CHROMA_COLLECTION = "ingested_content_local"

BATCH = 16   # how many items to embed per batch (tune)
SLEEP_BETWEEN_BATCHES = 0.2
# --------------------------

if not DATABASE_URL:
    raise RuntimeError("Set DATABASE_URL env var (sync URL, e.g. postgresql://...)")

engine = create_engine(DATABASE_URL)
insp = inspect(engine)
# detect schema for global_content_index
for schema in insp.get_schema_names():
    if GCI_TABLE in insp.get_table_names(schema=schema):
        GCI_TABLE_SCHEMA = schema
        break
if not GCI_TABLE_SCHEMA:
    raise RuntimeError("global_content_index table not found in any schema")

print("Using GCI table:", GCI_TABLE_SCHEMA + "." + GCI_TABLE)

# Chroma client
client = PersistentClient(path=CHROMA_PATH)
col = client.get_or_create_collection(CHROMA_COLLECTION)
print("Chroma collection ready:", CHROMA_COLLECTION, "count:", col.count())

# --------- helper: compute embeddings ----------
def compute_embeddings(texts):
    """
    Replace this function with your project's embedder for accurate production embeddings.
    Fallback uses sentence-transformers if available.
    """
    if _SBERT:
        embs = _SBERT.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [list(e) for e in embs]
    # fallback: very naive embeddings (not useful in prod) — raise to force replacement
    raise RuntimeError("No embedder available. Install sentence-transformers OR replace compute_embeddings() with your embedder.")

# --------- find GCI rows and missing ones ----------
with engine.connect() as conn:
    q = text(f"SELECT id, semantic_hash, cleaned_text, raw_text FROM {GCI_TABLE_SCHEMA}.{GCI_TABLE} ORDER BY created_at DESC")
    rows = conn.execute(q).fetchall()
    # robust conversion
    def row_map(r):
        if hasattr(r, "_mapping"):
            return dict(r._mapping)
        try:
            return dict(r)
        except Exception:
            cols = [c["name"] for c in insp.get_columns(GCI_TABLE, schema=GCI_TABLE_SCHEMA)]
            return dict(zip(cols, r))
    rows = [row_map(r) for r in rows]

hashes = [r["semantic_hash"] for r in rows if r.get("semantic_hash")]
print("GCI total with semantic_hash:", len(hashes))

# check which hashes already present in Chroma
missing_hashes = []
# We'll check in batches to avoid client issues
for i in range(0, len(hashes), BATCH):
    batch = hashes[i:i+BATCH]
    try:
        resp = col.get(ids=batch, include=["metadatas"])  # include allowed fields
        present_ids = set(resp.get("ids", []) or [])
    except Exception as e:
        # Some clients complain about 'ids' in include; fallback to trying get by docs or simply assume none found
        try:
            resp = col.get(limit=0)  # fallback
            present_ids = set()
        except Exception:
            present_ids = set()
    for h in batch:
        if h not in present_ids:
            missing_hashes.append(h)

print("Missing hashes count:", len(missing_hashes))

if not missing_hashes:
    print("Nothing to backfill. Exiting.")
    exit(0)

# Create lookup map for semantic_hash -> text and metadata
gci_map = {r["semantic_hash"]: r for r in rows if r.get("semantic_hash")}

# Backfill by batching embeddings and upserting to Chroma
for i in range(0, len(missing_hashes), BATCH):
    batch_hashes = missing_hashes[i:i+BATCH]
    texts = []
    metas = []
    for h in batch_hashes:
        entry = gci_map.get(h)
        text_to_embed = (entry.get("cleaned_text") or entry.get("raw_text") or "")[:15000]
        texts.append(text_to_embed)
        metas.append({
            "global_content_id": str(entry.get("id")),
            "semantic_hash": h,
        })
    try:
        embs = compute_embeddings(texts)  # list of lists
    except Exception as e:
        raise RuntimeError("Embedding failed: " + str(e))

    # upsert into Chroma (ids=semantic_hash)
    try:
        col.upsert(
            ids=batch_hashes,
            embeddings=embs,
            metadatas=metas,
            documents=texts
        )
        print(f"Upserted batch {i//BATCH + 1}: {len(batch_hashes)} items")
    except Exception as e:
        print("Chroma upsert failed:", e)
    time.sleep(SLEEP_BETWEEN_BATCHES)

print("Backfill completed.")
