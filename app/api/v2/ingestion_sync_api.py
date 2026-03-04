# app/api/v2/ingestion_sync_api.py
"""
Sync API — Detects and fixes Postgres ↔ VectorDB drift.
Backend-agnostic: works with all vector DBs (qdrant, chroma, pinecone, milvus, weaviate, redis).

Strategy:
  - Uses vectordb.exists() to check which DB hashes exist in vector store (all backends)
  - Uses vectordb.count() for total vector count comparison
  - Uses vectordb.delete_many() for orphan cleanup
  - Uses vectordb.upsert() for re-embedding missing vectors
"""

import asyncio
import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.auth.guards import require_role

from app.db.session_v2 import get_db
from app.db.models.global_content_index_v2 import GlobalContentIndexV2
from app.services.ingestion.ingestion_service_v2 import (
    get_chroma_collection,
    get_embedder,
)

router = APIRouter(
    prefix="/api/v2/ingestion-admin/sync",
    tags=["Ingestion Sync"],
)

BATCH_SIZE = 500  # Check IDs in batches to avoid oversized requests


def _get_vectordb_and_collection():
    """Get the pluggable vectordb instance and collection name."""
    _, adapter = get_chroma_collection()
    # The adapter wraps a BaseVectorDB — extract for direct API calls
    vectordb = adapter._vdb
    collection = adapter.name
    return vectordb, collection


async def _batch_exists(vectordb, collection: str, all_ids: list) -> set:
    """Check which IDs exist in vector DB, in batches (thread-safe)."""
    loop = asyncio.get_running_loop()
    existing = set()
    for i in range(0, len(all_ids), BATCH_SIZE):
        batch = all_ids[i : i + BATCH_SIZE]
        batch_existing = await loop.run_in_executor(
            None,
            lambda b=batch: vectordb.exists(collection=collection, ids=b),
        )
        existing.update(batch_existing)
    return existing


# -----------------------------------------------------------
# DETECT ORPHANS (READ-ONLY)
# -----------------------------------------------------------
@router.get("/orphans")
async def detect_orphans(
    db: AsyncSession = Depends(get_db),
    user=Depends(require_role("admin", "viewer")),
):
    """
    Detects Postgres ↔ VectorDB drift.
    Works with ALL vector backends.
    Read-only. No mutations.
    """
    try:
        vectordb, collection = _get_vectordb_and_collection()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Vector DB not available: {e}",
        )

    backend = vectordb.kind

    # 1. Get all semantic_hashes from DB
    result = await db.execute(
        select(GlobalContentIndexV2.semantic_hash)
    )
    db_hashes = {row[0] for row in result.all() if row[0]}

    # 2. Check which DB hashes exist in vector store (batch check)
    try:
        existing_in_vdb = await _batch_exists(vectordb, collection, list(db_hashes))
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Vector DB ({backend}) unreachable — cannot check existence: {str(e)[:200]}",
        )

    # 3. Get total vector count
    try:
        loop = asyncio.get_running_loop()
        vdb_total = await loop.run_in_executor(
            None, lambda: vectordb.count(collection)
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Vector DB ({backend}) unreachable — cannot get count: {str(e)[:200]}",
        )

    # 4. Compute drift
    db_without_vdb = sorted(db_hashes - existing_in_vdb)
    # Approximate vector orphans: vectors not accounted for by DB hashes
    estimated_vdb_orphans = max(0, vdb_total - len(existing_in_vdb))

    return {
        "backend": backend,
        "db_without_vectordb": db_without_vdb[:100],  # Cap response size
        "estimated_vectordb_orphans": estimated_vdb_orphans,
        "counts": {
            "db_total": len(db_hashes),
            "vectordb_total": vdb_total,
            "db_matched_in_vectordb": len(existing_in_vdb),
            "db_orphans": len(db_without_vdb),
            "estimated_vectordb_orphans": estimated_vdb_orphans,
        },
    }


# -----------------------------------------------------------
# FIX: VECTORDB → DB (DELETE ORPHAN VECTORS — Chroma only)
# -----------------------------------------------------------
@router.post("/fix/vectordb-to-db")
async def fix_vectordb_to_db(
    db: AsyncSession = Depends(get_db),
    user=Depends(require_role("admin")),
):
    """
    Delete vectors that have no matching DB record.
    NOTE: Full orphan vector detection requires bulk ID listing,
    which is efficient only for ChromaDB. For other backends,
    use the detect endpoint to see estimated orphan counts.
    """
    vectordb, collection = _get_vectordb_and_collection()

    result = await db.execute(
        select(GlobalContentIndexV2.semantic_hash)
    )
    db_hashes = {row[0] for row in result.all() if row[0]}

    # Try bulk listing (works for Chroma via adapter, fallback for others)
    _, adapter = get_chroma_collection()
    try:
        all_data = adapter.get(include=[])
        vdb_ids = set(all_data.get("ids", []))
    except Exception:
        vdb_ids = set()

    if not vdb_ids:
        return {
            "status": "skipped",
            "message": f"Bulk ID listing not available for {vectordb.kind}. Use detect endpoint instead.",
            "deleted_vectors": 0,
        }

    orphans = list(vdb_ids - db_hashes)

    if orphans:
        loop = asyncio.get_running_loop()
        deleted = await loop.run_in_executor(
            None,
            lambda: vectordb.delete_many(collection=collection, doc_ids=orphans),
        )
    else:
        deleted = 0

    return {
        "status": "ok",
        "backend": vectordb.kind,
        "deleted_vectors": deleted if isinstance(deleted, int) else len(orphans),
    }


# -----------------------------------------------------------
# FIX: DB → VECTORDB (RE-EMBED MISSING VECTORS)
# -----------------------------------------------------------
@router.post("/fix/db-to-vectordb")
async def fix_db_to_vectordb(
    db: AsyncSession = Depends(get_db),
    user=Depends(require_role("admin")),
):
    """
    Re-embeds semantic hashes that exist in DB
    but are missing from vector store.
    Works with ALL backends. SAFE & IDEMPOTENT.
    """
    vectordb, collection = _get_vectordb_and_collection()

    result = await db.execute(
        select(
            GlobalContentIndexV2.semantic_hash,
            GlobalContentIndexV2.cleaned_text,
        )
    )
    rows = result.all()

    all_hashes = [r[0] for r in rows if r[0]]
    existing = await _batch_exists(vectordb, collection, all_hashes)

    embedder = get_embedder()
    reembedded = 0
    loop = asyncio.get_running_loop()

    for semantic_hash, cleaned_text in rows:
        if not semantic_hash or not cleaned_text:
            continue
        if semantic_hash in existing:
            continue

        vector = embedder.encode(
            cleaned_text,
            normalize_embeddings=True,
        ).tolist()

        await loop.run_in_executor(
            None,
            lambda sh=semantic_hash, v=vector, t=cleaned_text: vectordb.upsert(
                collection=collection,
                doc_id=sh,
                embedding=v,
                text=t,
                metadata={"repair_source": "db_to_vectordb", "semantic_hash": sh},
            ),
        )
        reembedded += 1

    return {
        "status": "ok",
        "backend": vectordb.kind,
        "reembedded": reembedded,
    }


# -----------------------------------------------------------
# FIX: ORPHANS (DB ↔ VECTORDB FULL CLEANUP)
# -----------------------------------------------------------
@router.post("/orphans")
async def cleanup_orphans(
    db: AsyncSession = Depends(get_db),
    user=Depends(require_role("admin")),
):
    """
    Detects and re-embeds DB records missing from vector store.
    Works with ALL backends.
    """
    vectordb, collection = _get_vectordb_and_collection()

    try:
        # STEP 1 — Get all DB hashes
        result = await db.execute(select(GlobalContentIndexV2.semantic_hash))
        db_hashes = {row[0] for row in result.all() if row[0]}

        # STEP 2 — Check which exist in vector store
        existing = await _batch_exists(vectordb, collection, list(db_hashes))

        db_orphans = db_hashes - existing

        # STEP 3 — Re-embed DB orphans into vector store
        if db_orphans:
            embedder = get_embedder()
            loop = asyncio.get_running_loop()

            # Fetch texts for orphan hashes
            orphan_result = await db.execute(
                select(
                    GlobalContentIndexV2.semantic_hash,
                    GlobalContentIndexV2.cleaned_text,
                ).where(
                    GlobalContentIndexV2.semantic_hash.in_(list(db_orphans)[:500])
                )
            )
            orphan_rows = orphan_result.all()

            reembedded = 0
            for sh, text in orphan_rows:
                if not text:
                    continue
                vector = embedder.encode(text, normalize_embeddings=True).tolist()
                await loop.run_in_executor(
                    None,
                    lambda s=sh, v=vector, t=text: vectordb.upsert(
                        collection=collection,
                        doc_id=s,
                        embedding=v,
                        text=t,
                        metadata={"repair_source": "orphan_cleanup", "semantic_hash": s},
                    ),
                )
                reembedded += 1
        else:
            reembedded = 0

        return {
            "status": "ok",
            "backend": vectordb.kind,
            "db_orphans_found": len(db_orphans),
            "reembedded": reembedded,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))