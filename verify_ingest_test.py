# ==============================================================
# 🧪 verify_ingest_test.py — Regression & Validation Ingestion Runner
# ==============================================================
"""
Performs a controlled ingestion test for MarketingAdvantage AI.

✅ Compares Postgres + ChromaDB before & after ingestion
✅ Supports all formats (.xlsx, .csv, .pdf, .txt, .docx)
✅ Detects new taxonomy categories created dynamically
✅ Optional 'force re-ingest' mode for repeated testing
✅ Detailed summaries printed in console
"""

import os
import json
from api.services.manual_ingest import ingest_file_to_vector_and_db
from api.services.rag_service import _collection
from api.connection import get_db_connection
from api.db.models import Content, TaxonomyCategory

# ==============================================================
# ⚙️ CONFIGURATION
# ==============================================================
TEST_FILE = os.path.join(os.getcwd(), "data", "uploads", "Catageroies_and_BusinessesV1.xlsx")
FORCE_REINGEST = True  # If True → delete old file records first before re-ingesting

# ==============================================================
# 📊 DATABASE INSPECTION HELPERS
# ==============================================================
def summarize_postgres():
    """Returns summary of content & taxonomy counts."""
    with get_db_connection() as db:
        total_contents = db.query(Content).count()
        total_taxonomy = db.query(TaxonomyCategory).count()
        latest_titles = (
            db.query(Content.title)
            .order_by(Content.created_at.desc())
            .limit(5)
            .all()
        )
        titles = [t[0] for t in latest_titles]
    return {
        "total_contents": total_contents,
        "total_taxonomy": total_taxonomy,
        "recent_titles": titles,
    }


def summarize_chromadb():
    """Returns summary of ChromaDB vector entries."""
    try:
        total_vectors = _collection.count()
        vectors = _collection.get(include=["metadatas"], limit=3)
        samples = [
            {
                "file": v["file_name"] if "file_name" in v else v.get("file", "Unknown"),
                "category": v.get("category"),
                "sub_category": v.get("sub_category"),
                "source": v.get("source"),
            }
            for v in vectors.get("metadatas", [])
        ]
        return {"total_vectors": total_vectors, "samples": samples}
    except Exception as e:
        return {"error": str(e)}

# ==============================================================
# 🧼 FORCE CLEANUP (optional)
# ==============================================================
def remove_existing_entries(file_name: str):
    """Deletes any old content and vectors related to the same file for clean re-tests."""
    from sqlalchemy import delete
    with get_db_connection() as db:
        deleted = db.query(Content).filter(Content.title == file_name).delete()
        db.commit()
        print(f"🧹 Removed {deleted} existing entries for {file_name} from Postgres.")

    try:
        # Remove from ChromaDB
        all_docs = _collection.get(include=["metadatas", "ids"])
        ids_to_delete = [
            doc_id
            for doc_id, meta in zip(all_docs["ids"], all_docs["metadatas"])
            if meta and meta.get("file") == file_name
        ]
        if ids_to_delete:
            _collection.delete(ids=ids_to_delete)
            print(f"🧹 Removed {len(ids_to_delete)} vectors for {file_name} from ChromaDB.")
    except Exception as e:
        print(f"⚠️ Failed to clean ChromaDB: {e}")

# ==============================================================
# 🚀 MAIN EXECUTION
# ==============================================================
if __name__ == "__main__":
    print("🚀 Starting Controlled Ingestion Verification")
    print(f"📂 Test File: {TEST_FILE}")
    if not os.path.exists(TEST_FILE):
        print("❌ Test file not found.")
        exit(1)

    # Step 1: Before snapshot
    before_pg = summarize_postgres()
    before_vec = summarize_chromadb()

    print("\n=== BEFORE INGEST ===")
    print(json.dumps(before_pg, indent=2))
    print(json.dumps(before_vec, indent=2))

    # Step 2: Clean previous entries if FORCE_REINGEST
    if FORCE_REINGEST:
        file_name = os.path.basename(TEST_FILE)
        remove_existing_entries(file_name)

    # Step 3: Run ingestion
    print("\n⚙️ Running ingestion...")
    try:
        ingest_file_to_vector_and_db(TEST_FILE, source="regression_test")
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        exit(1)

    # Step 4: After snapshot
    after_pg = summarize_postgres()
    after_vec = summarize_chromadb()

    print("\n=== AFTER INGEST ===")
    print(json.dumps(after_pg, indent=2))
    print(json.dumps(after_vec, indent=2))

    # Step 5: Summary Diff
    print("\n📊 DELTA SUMMARY")
    print(f"🗂️ New DB Records: {after_pg['total_contents'] - before_pg['total_contents']}")
    print(f"🧠 New Vectors: {after_vec['total_vectors'] - before_vec['total_vectors']}")
    print(f"🧩 New Taxonomies: {after_pg['total_taxonomy'] - before_pg['total_taxonomy']}")

    print("\n✅ Test completed successfully.")
