# app/api/v2/ingestion_integrity_api.py
#
# Ingestion Integrity APIs
# -----------------------------------------
# Purpose:
# - Detect duplicate content
# - Detect DB ↔ Chroma mismatches
# - Provide safe, admin-only repair paths
#
# IMPORTANT:
# - No ingestion logic touched
# - No automatic destructive actions
# - Human-in-the-loop governance
#

from collections import defaultdict
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.auth.guards import require_role

from app.db.session_v2 import get_db
from app.db.models.global_content_index_v2 import GlobalContentIndexV2
from app.services.ingestion.ingestion_service_v2 import (
    get_chroma_collection,
    get_embedder,
)

router = APIRouter(
    prefix="/api/v2/ingestion-admin/integrity",
    tags=["Ingestion Integrity"],
)

# ===========================================================
# 1️⃣ DUPLICATE DETECTION — GLOBAL CONTENT INDEX (DB)
# ===========================================================
@router.get("/gci-duplicates")
async def detect_gci_duplicates(
    db: AsyncSession = Depends(get_db),
    user=Depends(require_role("admin", "viewer")),
):
    """
    Detects duplicate cleaned_text rows in GlobalContentIndex.

    Definition:
      LOWER(TRIM(cleaned_text)) identical across rows

    READ-ONLY
    """

    stmt = (
        select(
            func.lower(func.trim(GlobalContentIndexV2.cleaned_text)).label("text_key"),
            func.count(GlobalContentIndexV2.id).label("row_count"),
        )
        .where(GlobalContentIndexV2.cleaned_text.isnot(None))
        .group_by("text_key")
        .having(func.count(GlobalContentIndexV2.id) > 1)
    )

    result = await db.execute(stmt)

    return [
        {
            "normalized_text": row.text_key,
            "duplicate_count": row.row_count,
        }
        for row in result.all()
    ]


# ===========================================================
# 2️⃣ DUPLICATE DETECTION — CHROMA (TEXT-LEVEL)
# ===========================================================
@router.get("/chroma-duplicates")
async def detect_chroma_duplicates():
    """
    Detects duplicate documents in Chroma by text equality.

    Uses:
      LOWER(TRIM(document))

    READ-ONLY
    """

    _, collection = get_chroma_collection()
    data = collection.get(include=["documents"])

    text_map = defaultdict(list)

    for vector_id, doc in zip(
        data.get("ids", []),
        data.get("documents", []),
    ):
        if not doc:
            continue

        key = doc.strip().lower()
        text_map[key].append(vector_id)

    duplicates = [
        {
            "normalized_text": text,
            "vector_ids": ids,
            "count": len(ids),
        }
        for text, ids in text_map.items()
        if len(ids) > 1
    ]

    return {
        "duplicate_groups": duplicates,
        "group_count": len(duplicates),
    }


# ===========================================================
# 3️⃣ CONTENT MISMATCH — DB ↔ CHROMA
# ===========================================================
@router.get("/content-mismatch")
async def detect_content_mismatch(db: AsyncSession = Depends(get_db)):
    """
    Detects mismatches between:
      GlobalContentIndex.cleaned_text
      vs
      Chroma stored document

    Keyed by semantic_hash.

    READ-ONLY
    """

    # Load DB content
    result = await db.execute(
        select(
            GlobalContentIndexV2.semantic_hash,
            GlobalContentIndexV2.cleaned_text,
        )
    )
    db_rows = result.all()

    # Load Chroma documents
    _, collection = get_chroma_collection()
    chroma_data = collection.get(include=["documents"])

    chroma_docs = dict(
        zip(
            chroma_data.get("ids", []),
            chroma_data.get("documents", []),
        )
    )

    mismatches = []

    for semantic_hash, cleaned_text in db_rows:
        if not semantic_hash or not cleaned_text:
            continue

        chroma_text = chroma_docs.get(semantic_hash)
        if chroma_text is None:
            continue

        if cleaned_text.strip() != chroma_text.strip():
            mismatches.append(
                {
                    "semantic_hash": semantic_hash,
                    "db_text": cleaned_text,
                    "chroma_text": chroma_text,
                }
            )

    return {
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


# ===========================================================
# 4️⃣ FIX CONTENT MISMATCH — DB → CHROMA (SAFE)
# ===========================================================
@router.post("/fix/db-to-chroma")
async def fix_content_db_to_chroma(
    db: AsyncSession = Depends(get_db),
    user=Depends(require_role("admin")),
):
    """
    Repairs content mismatches by making DB the source of truth.

    Behavior:
    - Replaces Chroma document with DB cleaned_text
    - Re-embeds vector
    - No DB mutation
    - Idempotent

    ADMIN ONLY
    """
    try:
        result = await db.execute(
            select(
                GlobalContentIndexV2.semantic_hash,
                GlobalContentIndexV2.cleaned_text,
            )
        )
        rows = result.all()

        _, collection = get_chroma_collection()
        embedder = get_embedder()

        chroma_data = collection.get(include=[])
        chroma_ids = set(chroma_data.get("ids", []))

        fixed = 0

        for semantic_hash, cleaned_text in rows:
            if not semantic_hash or not cleaned_text:
                continue

            if semantic_hash not in chroma_ids:
                continue

            vector = embedder.encode(
                cleaned_text,
                normalize_embeddings=True,
            ).tolist()

            collection.upsert(
                ids=[semantic_hash],
                embeddings=[vector],
                documents=[cleaned_text],
                metadatas=[{"integrity_fix": "db_to_chroma"}],
            )

            fixed += 1

        return {
            "status": "ok",
            "fixed_vectors": fixed,
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Fix DB → Chroma failed: {e}",
        )
