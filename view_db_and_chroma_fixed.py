# view_db_and_chroma_fixed.py
# Diagnostic: view DB tables and check Global Content Index -> Chroma alignment
# Safe, read-only. Designed to replace previous problematic diagnostic script.

import os
import sys
import json
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text, inspect
import chromadb
from chromadb.config import Settings
from typing import Any, Dict, List

# ---------- CONFIG ----------
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:Mahadeva%40123@localhost/marketing_advantage")  # e.g. 'postgresql://user:pass@host/db'
CHROMA_PATH = os.getenv("CHROMA_PATH", "./pluggable_db")
CHROMA_COLLECTION = os.getenv("MAI_COLLECTION", os.getenv("CHROMA_COLLECTION", "ingested_content_dedup"))

SAMPLE_LIMIT = int(os.getenv("SAMPLE_LIMIT", "5"))
RECENT_GCI_LIMIT = int(os.getenv("RECENT_GCI_LIMIT", "100"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))


# ---------- DB helpers ----------
def get_engine():
    if not DATABASE_URL:
        raise RuntimeError("Please set DATABASE_URL environment variable (or edit script).")
    return create_engine(DATABASE_URL)


def list_tables(engine):
    insp = inspect(engine)
    out = []
    for schema in insp.get_schema_names():
        try:
            for t in insp.get_table_names(schema=schema):
                out.append((schema, t))
        except Exception:
            # some schemas may error on introspect (skip)
            pass
    return out


def find_table_schema(engine, table_name):
    insp = inspect(engine)
    for schema in insp.get_schema_names():
        try:
            if table_name in insp.get_table_names(schema=schema):
                return schema
        except Exception:
            pass
    return None


def get_table_columns(engine, schema, table):
    insp = inspect(engine)
    try:
        cols = insp.get_columns(table, schema=schema)
        return [c["name"] for c in cols]
    except Exception:
        return []


def row_to_dict(row, engine=None, schema=None, table=None):
    """
    Robust conversion of SQLAlchemy row result -> dict.
    Handles:
      - Row with _mapping (SQLAlchemy 1.4+)
      - Row convertible by dict(row)
      - tuple-like rows (zip with column names)
      - final fallback: stringified row
    """
    try:
        if hasattr(row, "_mapping"):
            return dict(row._mapping)
    except Exception:
        pass

    try:
        return dict(row)
    except Exception:
        pass

    # fallback: zip with column names if available
    if engine and schema and table:
        try:
            cols = get_table_columns(engine, schema, table)
            return dict(zip(cols, row))
        except Exception:
            pass

    # last resort
    return {"row": str(row)}


def sample_table(engine, schema, table, limit=SAMPLE_LIMIT):
    cols = get_table_columns(engine, schema, table)
    order_col = None
    for c in ("created_at", "updated_at", "id"):
        if c in cols:
            order_col = c
            break

    try:
        if order_col:
            q = text(f"SELECT * FROM {schema}.{table} ORDER BY {order_col} DESC LIMIT :lim")
        else:
            q = text(f"SELECT * FROM {schema}.{table} LIMIT :lim")
        with engine.connect() as conn:
            rows = conn.execute(q, {"lim": limit}).fetchall()
    except Exception as e:
        print(f"Error sampling {schema}.{table}: {e}")
        return []

    return [row_to_dict(r, engine, schema, table) for r in rows]


def count_table(engine, schema, table):
    try:
        q = text(f"SELECT COUNT(*) FROM {schema}.{table}")
        with engine.connect() as conn:
            (count,) = conn.execute(q).fetchone()
        return count
    except Exception as e:
        print(f"Error counting {schema}.{table}: {e}")
        return None


# ---------- GCI <-> Chroma check ----------
def check_gci(engine, schema, table, chroma_collection):
    print("\n--- Checking Global Content Index → Chroma alignment ---")

    cols = get_table_columns(engine, schema, table)
    desired = ["id", "semantic_hash", "business_id", "created_at", "updated_at"]
    available = [c for c in desired if c in cols]

    if not available:
        print(f"No usable columns in {schema}.{table}. Columns: {cols}")
        return

    select_cols = ", ".join(available)
    q = text(
        f"SELECT {select_cols} FROM {schema}.{table} "
        f"ORDER BY {available[-1]} DESC LIMIT :lim"
    )

    with engine.connect() as conn:
        try:
            rows = conn.execute(q, {"lim": RECENT_GCI_LIMIT}).fetchall()
        except Exception as e:
            print("Query failed:", e)
            return

    rows = [row_to_dict(r, engine, schema, table) for r in rows]
    hashes = [r.get("semantic_hash") for r in rows if r.get("semantic_hash")]
    print(f"Fetched {len(rows)} GCI rows | Semantic hashes: {len(hashes)}")

    found = set()

    # Batch-check Chroma using supported include parameters
    for i in range(0, len(hashes), BATCH_SIZE):
        batch = hashes[i:i + BATCH_SIZE]
        try:
            # include only valid items: 'metadatas' and/or 'documents' and/or 'embeddings'
            resp = chroma_collection.get(ids=batch, include=["metadatas", "documents"])
        except Exception as e:
            print(f"Chroma get failed batch {i}: {e}")
            continue

        # resp shape may be dict (newer client) or object-like; handle safely
        if isinstance(resp, dict):
            # try to extract metadatas -> look for semantic_hash in metadatas
            metadatas = resp.get("metadatas", [])
            # if metadatas is list aligned to ids, each metadata should have semantic_hash
            for md in metadatas:
                if isinstance(md, dict) and md.get("semantic_hash"):
                    found.add(md.get("semantic_hash"))
            # fallback: if resp has 'ids' and the ids are semantic-hashes themselves
            ids = resp.get("ids", [])
            for _id in ids:
                if _id:
                    found.add(_id)
        else:
            # Unexpected type - try attribute access defensively
            try:
                metadatas = getattr(resp, "metadatas", None)
                if metadatas:
                    for md in metadatas:
                        if isinstance(md, dict) and md.get("semantic_hash"):
                            found.add(md.get("semantic_hash"))
            except Exception:
                pass

    missing = set(hashes) - found
    print(f"Chroma matched {len(found)} / {len(hashes)} | Missing {len(missing)}")
    if missing:
        print("Missing sample:", list(missing)[:10])

    print("\nSample GCI rows:")
    for r in rows[:5]:
        print(" -", {k: r.get(k) for k in available})


# ---------- MAIN ----------
def main():
    try:
        engine = get_engine()
    except Exception as e:
        print("Fatal: ", e)
        sys.exit(1)

    print("\n=== Detected DB Tables ===")
    tables = list_tables(engine)
    for schema, table in tables:
        print(f" - {schema}.{table}")

    EXPECTED = [
        "global_content_index",
        "ingested_content",
        "ingested_file",
        "pending_taxonomy",
        "taxonomy",
        "taxonomy_alias",
        "classification_logs",
        "business_classification",
        "business_taxonomy",
    ]

    print("\n=== Checking Expected Tables ===")
    table_schemas = {}
    for t in EXPECTED:
        schema = find_table_schema(engine, t)
        if not schema:
            print(f"[MISSING] {t}")
            continue
        table_schemas[t] = schema
        cnt = count_table(engine, schema, t)
        print(f"[OK] {schema}.{t} (rows={cnt})")
        sample = sample_table(engine, schema, t)
        for s in sample:
            # print first 6 keys for compactness
            keys = list(s.keys())[:6]
            print("  sample:", {k: s.get(k) for k in keys})

    # Connect to Chroma
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)
        print("\nConnected to Chroma collection:", CHROMA_COLLECTION)
        try:
            cnt = collection.count()
            print("Chroma vector count:", cnt)
        except Exception:
            print("Chroma client: count() not available.")
    except Exception as e:
        print("Chroma connection failed:", e)
        collection = None

    # Check GCI -> Chroma if table present
    if "global_content_index" in table_schemas and collection:
        check_gci(engine, table_schemas["global_content_index"], "global_content_index", collection)
    else:
        print("Skipping GCI check — table missing or Chroma unavailable.")

    # sample ingested_content
    if "ingested_content" in table_schemas:
        print("\n=== Ingested Content (sample) ===")
        sample = sample_table(engine, table_schemas["ingested_content"], "ingested_content")
        for r in sample:
            print(" -", r)

    if "ingested_file" in table_schemas:
        print("\n=== Recent Ingested Files ===")
        sample = sample_table(engine, table_schemas["ingested_file"], "ingested_file")
        for r in sample:
            print(" -", r)


if __name__ == "__main__":
    main()
