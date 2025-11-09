# ==========================================================
# 🚀 unified_ingest_helper.py — Enterprise Smart Ingestion Core (v6.0)
# ==========================================================
"""
Handles ingestion for all pipelines:
- Manual uploads
- File watcher
- Excel/CSV Intelligent LLM ingestion
- RSS / Strategy / Trend ingestion

Performs:
✅ Deduplication via fingerprint
✅ Automatic taxonomy + business linking
✅ Safe transaction commits
✅ Vector embedding to ChromaDB (modern PersistentClient)
"""

import os
import hashlib
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from api.connection import get_db_connection
from api.db import models
from chromadb import PersistentClient

# ==========================================================
# ⚙️ CONFIG
# ==========================================================
logger = logging.getLogger("unified_ingest")
CHROMA_PATH = os.path.join(os.getcwd(), "chroma")  # unified persistent store
COLLECTION_NAME = "content_knowledge_base"


# ==========================================================
# 🧩 UTILITY HELPERS
# ==========================================================
def clean_text_for_postgres(raw: str) -> str:
    """Remove NUL bytes and control chars for DB safety."""
    if not raw:
        return ""
    raw = raw.replace("\x00", "")
    return "".join(ch for ch in raw if ch.isprintable() or ch in ("\n", "\r", "\t"))


def compute_fingerprint(text: str) -> str:
    """Generate SHA256 fingerprint for deduplication."""
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def get_chroma_collection(name: str = COLLECTION_NAME):
    """Initialize or fetch a ChromaDB collection using new PersistentClient."""
    client = PersistentClient(path=CHROMA_PATH)
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name=name)


def reset_chroma():
    """Utility to clear all Chroma collections — for QA use only."""
    client = PersistentClient(path=CHROMA_PATH)
    for c in client.list_collections():
        client.delete_collection(c.name)
    logger.warning("⚠️ Chroma reset executed — all vectors deleted.")
    return True


# ==========================================================
# 🚀 MAIN INGEST FUNCTION
# ==========================================================
def process_and_store_document(
    text: str,
    file_name: str,
    source: str,
    category: str,
    subcategory: str,
    confidence: float,
):
    """
    Unified ingestion logic for structured storage.
    1. Dedup by fingerprint
    2. Ensure taxonomy + business exist
    3. Store content + entity link
    4. Add to ChromaDB
    """
    text = clean_text_for_postgres(text)
    fingerprint = compute_fingerprint(text)
    logger.info(f"[Fingerprint] {file_name}: {fingerprint[:12]}...")

    with get_db_connection() as db:
        try:
            # 1️⃣ Deduplication
            existing = (
                db.query(models.Content)
                .filter(models.Content.fingerprint == fingerprint)
                .first()
            )
            if existing:
                db.rollback()
                logger.warning(f"[DuplicateSkip] {file_name} already exists.")
                return {
                    "status": "skipped",
                    "file": file_name,
                    "existing_id": str(existing.id),
                }

            # 2️⃣ Taxonomy ensure + commit
            cat_obj = (
                db.query(models.TaxonomyCategory)
                .filter(models.TaxonomyCategory.name.ilike(category))
                .first()
            )
            if not cat_obj:
                cat_obj = models.TaxonomyCategory(
                    name=category, group="content", description="Auto-generated"
                )
                db.add(cat_obj)
                db.commit()

            subcat_obj = (
                db.query(models.TaxonomyCategory)
                .filter(models.TaxonomyCategory.name.ilike(subcategory))
                .first()
            )
            if not subcat_obj:
                subcat_obj = models.TaxonomyCategory(
                    name=subcategory, group="content", description="Auto-generated subcategory"
                )
                db.add(subcat_obj)
                db.commit()

            # 3️⃣ Ensure Business
            business = (
                db.query(models.Business)
                .filter(models.Business.name.ilike("Manual Upload Business"))
                .first()
            )
            if not business:
                business = models.Business(
                    name="Manual Upload Business",
                    industry="General",
                    description="Placeholder business for manual uploads.",
                )
                db.add(business)
                db.commit()

            # 4️⃣ Create Content Record
            content = models.Content(
                business_id=business.id,
                title=file_name,
                content_type="document",
                text=text,
                summary=text[:300],
                category=category,
                sub_category=subcategory,
                tags={"auto": True},
                content_metadata={
                    "source": source,
                    "confidence": confidence,
                    "file_name": file_name,
                },
                fingerprint=fingerprint,
                confidence=confidence,
            )

            db.add(content)
            db.flush()  # ensure content.id available before linking

            # 5️⃣ Create Entity Link
            entity_link = models.EntityLink(
                category_id=cat_obj.id,
                subcategory_id=subcat_obj.id,
                business_id=business.id,
                entity_type="content",
                entity_id=content.id,
                fingerprint=fingerprint,
                link_metadata={
                    "source": source,
                    "file_name": file_name,
                    "timestamp": str(datetime.utcnow()),
                },
            )
            db.add(entity_link)
            db.commit()
            db.refresh(entity_link)

            # 6️⃣ Vector Store Sync
            try:
                chroma = get_chroma_collection()
                chroma.add(
                    ids=[fingerprint],
                    documents=[text],
                    metadatas=[{
                        "category": category,
                        "subcategory": subcategory,
                        "business": business.name,
                        "file": file_name,
                        "source": source,
                        "confidence": confidence,
                    }],
                )
                logger.info(f"[Chroma] ✅ Embedding added for {file_name}")
            except Exception as e:
                logger.warning(f"[ChromaSkip] {e}")

            return {
                "status": "success",
                "file": file_name,
                "business": business.name,
                "category": category,
                "subcategory": subcategory,
                "confidence": confidence,
                "entity_link_id": entity_link.id,
            }

        except Exception as e:
            db.rollback()
            db.expire_all()
            logger.error(f"[DBError] ❌ Rollback for {file_name}: {e}")
            raise
