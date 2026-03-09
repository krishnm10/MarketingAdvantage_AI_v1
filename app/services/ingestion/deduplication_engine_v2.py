# =============================================
# deduplication_engine_v2.py — Enterprise 3-Layer Deduplication
#
# ARCHITECTURE: Read-then-Write separation
#
#   PHASE 1 — DEDUP (pure reads, zero DB writes):
#     L1: Normalized hash, in-memory set        O(1)/chunk, zero DB calls
#     L2: Batch GCI cross-file lookup           1 SQL query total, not N
#     L3: Vector similarity (semantic)          async, executor-offloaded
#
#   PHASE 2 — COMMIT (writes, called from ingestion_service):
#     register_unique_chunks_in_gci()           INSERT/ON CONFLICT UPDATE
#
#   WHY THIS ORDER MATTERS:
#     The previous design wrote to GCI inside the segmenter (make_chunk_dict).
#     Because segmenter is called multiple times per file (one per page / chunk
#     group), boundary chunks that appear in two parsing passes got
#     occurrence_count=2 before dedup ran. Dedup saw count≥2 → false duplicate.
#     Result: 50-100% false positive rate on first ingestion of any file.
#
#     Now: GCI only contains content from COMPLETED prior ingestions.
#     Any GCI hit = definitive cross-file duplicate. No counting needed.
#
#   PERFORMANCE:
#     L1 — zero DB calls (Python set)
#     L2 — 1 batch IN query for all N chunks (was N individual queries)
#     L3 — runs only on L1+L2 survivors, embeddings cached on chunk dict
#     register_unique_chunks_in_gci — batch loop with on_conflict_do_update
#
#   CORRECTNESS CONTRACT:
#     count=1 in GCI → registered by current session (impossible at dedup time
#                       because we only write post-dedup — this case cannot occur)
#     Any GCI hit    → from a prior committed ingestion → true duplicate
#     No GCI hit     → new unique content → pass to commit phase
# =============================================

import hashlib
import os
import re
import uuid
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.models.ingested_content_v2 import IngestedContentV2
from app.db.models.global_content_index_v2 import GlobalContentIndexV2
from app.utils.logger import log_info, log_warning


# =============================================
# LAYER 1 UTILITIES — Normalized Hash
# =============================================

def normalize_for_hash(text: str) -> str:
    """
    Ultra-aggressive normalization — minor variations never create different hashes.
    Strips case, whitespace variations, and punctuation.
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = text.strip()
    text = ' '.join(text.split())
    return text


def create_normalized_hash(text: str) -> str:
    """
    SHA-256 of normalized text.
    "Hello World" == "hello world" == "HELLO  WORLD" == "Hello World!"
    """
    normalized = normalize_for_hash(text)
    if not normalized:
        return hashlib.sha256(b"").hexdigest()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


# =============================================
# LAYER 2 — BATCH GCI LOOKUP (single IN query)
# =============================================

async def batch_check_gci(
    db: AsyncSession,
    hashes: List[str],
    business_id: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Enterprise-grade cross-file duplicate check.

    PERFORMANCE: Single SQL IN query for all N hashes, not N individual queries.
    Old design: N round-trips to DB (one per chunk).
    New design: 1 round-trip regardless of chunk count.
    At 1000 chunks: 1000x fewer DB round-trips.

    CORRECTNESS: Because GCI is only written to AFTER dedup (in
    register_unique_chunks_in_gci), any hash found here definitively
    belonged to a PRIOR ingestion session. No threshold or count check needed.

    Args:
        db:          AsyncSession
        hashes:      All normalized hashes from the current batch
        business_id: Optional tenant scope

    Returns:
        Dict of semantic_hash → gci_info for every hash that matched.
        Empty dict = all chunks are new (no cross-file duplicates).
    """
    if not hashes:
        return {}

    try:
        query = (
            select(GlobalContentIndexV2)
            .where(GlobalContentIndexV2.semantic_hash.in_(hashes))
        )
        if business_id:
            query = query.where(GlobalContentIndexV2.business_id == business_id)

        result = await db.execute(query)
        rows   = result.scalars().all()

        return {
            row.semantic_hash: {
                "gci_id":             str(row.id),
                "occurrence_count":   row.occurrence_count,
                "first_seen_file_id": str(row.first_seen_file_id) if row.first_seen_file_id else None,
                "confidence_avg":     row.confidence_avg,
                "dedup_layer":        "global_content_index",
            }
            for row in rows
        }

    except Exception as e:
        log_warning(f"[Dedup L2] Batch GCI lookup failed: {e}")
        return {}


# =============================================
# LAYER 3 — VECTOR SIMILARITY (async, cached embeddings)
# =============================================

async def check_embedding_similarity(
    db: AsyncSession,
    vectordb,
    embedder,
    chunk_text: str,
    query_embedding: List[float],
    similarity_threshold: float = 0.95,
    top_k: int = 5,
    collection_name: str = None,
) -> Optional[Dict[str, Any]]:
    """
    Semantic duplicate check via vector similarity.

    PHANTOM BUG-2 FIX: vectordb.search() is synchronous I/O. Running it
    directly inside async code freezes the event loop for all concurrent
    requests. Fixed: wrapped in run_in_executor (thread pool).

    Embeddings are cached on the chunk dict (_cached_embedding) — computed
    once in deduplicate_chunks() and reused here without re-embedding.
    """
    try:
        resolved_collection = collection_name or os.getenv("MAI_COLLECTION", "ingested_content")

        loop = asyncio.get_running_loop()
        hits = await loop.run_in_executor(
            None,
            lambda: vectordb.search(
                collection=resolved_collection,
                query_embedding=query_embedding,
                top_k=top_k,
            ),
        )
        if not hits:
            return None

        for hit in hits:
            if hit.score >= similarity_threshold:
                log_info(
                    f"[Dedup L3] Semantic duplicate: "
                    f"similarity={hit.score:.4f}, chunk_id={hit.id}"
                )
                return {
                    "is_duplicate":       True,
                    "duplicate_chunk_id": hit.id,
                    "similarity_score":   float(hit.score),
                    "duplicate_text":     hit.text,
                    "dedup_layer":        "embedding_similarity",
                }
        return None

    except Exception as e:
        log_warning(f"[Dedup L3] Embedding similarity check failed: {e}")
        return None


# =============================================
# MAIN DEDUP ENGINE — 3-Layer Pipeline
# =============================================

async def deduplicate_chunks(
    db: AsyncSession,
    chunks: List[Dict[str, Any]],
    vectordb,
    embedder,
    file_id: str,
    business_id: Optional[str] = None,
    enable_embedding_dedup: bool = True,
    similarity_threshold: float = 0.95,
    collection_name: str = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    3-Layer Deduplication Pipeline. PURE READ PHASE — zero DB writes.

    LAYER 1 — Normalized Hash (in-memory, O(1)/chunk):
      Detects intra-file and intra-batch exact duplicates.
      Uses a Python set — zero DB calls regardless of batch size.

    LAYER 2 — Batch GCI Cross-File Lookup (1 SQL query total):
      Detects content already committed in prior ingestion sessions.
      Single IN query for all L1 survivors — not N individual queries.
      Any GCI hit = definitive cross-file duplicate (no count check needed,
      because GCI is only written AFTER dedup confirms uniqueness).

    LAYER 3 — Vector Similarity (async, only L1+L2 survivors):
      Detects semantic near-duplicates (paraphrases, reformatted content).
      Embeddings are cached on chunk dict to avoid double-compute.
      Runs only on chunks that passed L1 and L2 — minimum wasted work.

    GCI writes happen in register_unique_chunks_in_gci() AFTER this function
    returns — called by ingestion_service_v2._run_pipeline().

    Returns:
        (unique_chunks, dedup_stats)
    """
    if not chunks:
        return [], {"total": 0, "unique": 0, "duplicates": 0,
                    "layer1_hash_duplicates": 0, "layer2_embedding_duplicates": 0,
                    "layer3_gci_duplicates": 0, "dedup_ratio": 0.0}

    stats = {
        "total":                      len(chunks),
        "unique":                     0,
        "duplicates":                 0,
        "layer1_hash_duplicates":     0,
        "layer2_embedding_duplicates": 0,
        "layer3_gci_duplicates":      0,
        "dedup_ratio":                0.0,
    }

    log_info(f"[Dedup Engine] Starting 3-layer dedup for {len(chunks)} chunks")

    # ================================================================
    # LAYER 1 — Normalized Hash (in-memory set, zero DB calls)
    # ================================================================
    batch_hashes:  set               = set()
    l1_survivors:  List[Dict]        = []
    l1_duplicates: List[Dict]        = []

    for chunk in chunks:
        chunk_text = chunk.get("cleaned_text") or chunk.get("text", "")
        if not chunk_text or not chunk_text.strip():
            continue

        normalized_hash = create_normalized_hash(chunk_text)
        chunk["normalized_hash"] = normalized_hash

        if normalized_hash in batch_hashes:
            # Intra-batch exact duplicate
            stats["layer1_hash_duplicates"] += 1
            stats["duplicates"] += 1
            chunk.update({
                "is_duplicate":     True,
                "duplicate_source": "intra_batch",
                "dedup_layer":      "layer1_normalized_hash",
                # ── CONSTRAINT: check_duplicate_consistency requires ──────────────
                # is_duplicate=True → duplicate_of IS NOT NULL OR similarity_score IS NOT NULL
                # L1 dups have no GCI UUID (duplicate_of stays None).
                # Set similarity_score=1.0 to satisfy the OR branch.
                # This is semantically correct: it IS a 1.0 similarity duplicate —
                # it's character-for-character identical to another chunk in this batch.
                # ─────────────────────────────────────────────────────────────────
                "similarity_score": 1.0,
            })
            l1_duplicates.append(chunk)
            log_info(f"[Dedup L1] Intra-batch duplicate: hash={normalized_hash[:12]}...")
        else:
            batch_hashes.add(normalized_hash)
            l1_survivors.append(chunk)

    # ================================================================
    # LAYER 2 — Batch GCI Cross-File Lookup (1 SQL query, not N)
    # ================================================================
    l2_survivors:  List[Dict] = []
    l2_duplicates: List[Dict] = []

    if l1_survivors:
        survivor_hashes = [c["normalized_hash"] for c in l1_survivors]
        gci_matches     = await batch_check_gci(db, survivor_hashes, business_id)

        for chunk in l1_survivors:
            h = chunk["normalized_hash"]
            if h in gci_matches:
                gci_info = gci_matches[h]
                stats["layer3_gci_duplicates"] += 1
                stats["duplicates"] += 1
                chunk.update({
                    "is_duplicate":    True,
                    "duplicate_source": "global_content_index",
                    "dedup_layer":     "layer3_gci",
                    "gci_id":          gci_info["gci_id"],
                    "occurrence_count": gci_info["occurrence_count"],
                    "global_content_id": gci_info["gci_id"],
                })
                l2_duplicates.append(chunk)
                log_info(
                    f"[Dedup Layer 3] Found in GCI: "
                    f"id={gci_info['gci_id']}, occurrences={gci_info['occurrence_count']}"
                )
                log_info(
                    f"[Dedup L1+L3] Cross-file duplicate: "
                    f"hash={h[:12]}..., occurrences={gci_info['occurrence_count']}"
                )
            else:
                l2_survivors.append(chunk)

    # ================================================================
    # LAYER 3 — Vector Similarity (only L1+L2 survivors, embeddings cached)
    # ================================================================
    unique_chunks: List[Dict] = []

    if enable_embedding_dedup and l2_survivors:
        loop = asyncio.get_running_loop()

        for chunk in l2_survivors:
            chunk_text = chunk.get("cleaned_text") or chunk.get("text", "")
            try:
                # Reuse cached embedding if already computed (F4 optimization)
                if chunk.get("_cached_embedding"):
                    query_embedding: List[float] = chunk["_cached_embedding"]
                else:
                    query_embedding: List[float] = await loop.run_in_executor(
                        None, lambda t=chunk_text: embedder.embed_query(t)
                    )
                    chunk["_cached_embedding"] = query_embedding

                similarity_result = await check_embedding_similarity(
                    db, vectordb, embedder, chunk_text, query_embedding,
                    similarity_threshold, collection_name=collection_name,
                )

                if similarity_result:
                    stats["layer2_embedding_duplicates"] += 1
                    stats["duplicates"] += 1
                    chunk.update({
                        "is_duplicate":       True,
                        "duplicate_source":   "embedding_similarity",
                        "dedup_layer":        "layer2_embedding",
                        "similarity_score":   similarity_result["similarity_score"],
                        "duplicate_chunk_id": similarity_result["duplicate_chunk_id"],
                    })
                    log_info(
                        f"[Dedup L2] Semantic duplicate: "
                        f"similarity={similarity_result['similarity_score']:.4f}"
                    )
                else:
                    # Survived all 3 layers — truly unique
                    chunk["is_duplicate"] = False
                    unique_chunks.append(chunk)
                    stats["unique"] += 1

            except Exception as e:
                log_warning(f"[Dedup L3] Embedding check failed: {e}")
                # On error: treat as unique (conservative — better to store than lose)
                chunk["is_duplicate"] = False
                unique_chunks.append(chunk)
                stats["unique"] += 1
    else:
        # L3 disabled or no L2 survivors — L2 survivors are all unique
        for chunk in l2_survivors:
            chunk["is_duplicate"] = False
        unique_chunks.extend(l2_survivors)
        stats["unique"] += len(l2_survivors)

    stats["dedup_ratio"] = (
        (stats["duplicates"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
    )

    log_info(
        f"[Dedup Engine] Complete: {stats['unique']} unique, "
        f"{stats['duplicates']} duplicates ({stats['dedup_ratio']:.2f}% reduction) | "
        f"L1: {stats['layer1_hash_duplicates']}, "
        f"L2: {stats['layer2_embedding_duplicates']}, "
        f"L3: {stats['layer3_gci_duplicates']}"
    )

    return unique_chunks, stats


# =============================================
# GCI REGISTRATION — Called post-dedup, not during segmentation
# =============================================

async def register_unique_chunks_in_gci(
    db: AsyncSession,
    unique_chunks: List[Dict[str, Any]],
    file_id: str,
    business_id: Optional[str] = None,
    source_type: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> Dict[str, str]:
    """
    Register confirmed-unique chunks in GlobalContentIndexV2.

    Called by ingestion_service_v2._run_pipeline() AFTER deduplicate_chunks()
    confirms uniqueness. This is the ONLY place GCI is written to.

    DESIGN:
      • on_conflict_do_update handles concurrent ingestion of same content
        from two simultaneous jobs. The second writer increments the count —
        subsequent ingestions will then correctly see count≥1 and flag as duplicate.
      • Each chunk dict is updated with its GCI UUID so _insert_chunks()
        can populate the global_content_id FK column.
      • Batch loop with a single commit at the end — not N separate commits.

    Returns:
        Dict of semantic_hash → gci_id (UUID string) for all registered chunks.
        Used by _run_pipeline to populate chunk["global_content_id"].
    """
    if not unique_chunks:
        return {}

    now            = datetime.utcnow()
    hash_to_gci_id: Dict[str, str] = {}

    for chunk in unique_chunks:
        semantic_hash = chunk.get("semantic_hash") or chunk.get("normalized_hash")
        if not semantic_hash:
            continue

        chunk_source_type   = source_type   or chunk.get("source_type")
        chunk_embed_model   = embedding_model or chunk.get("embedding_model")

        try:
            stmt = (
                pg_insert(GlobalContentIndexV2)
                .values(
                    id=str(uuid.uuid4()),
                    semantic_hash=semantic_hash,
                    cleaned_text=chunk.get("cleaned_text", ""),
                    raw_text=chunk.get("text", ""),
                    tokens=chunk.get("tokens", 0),
                    business_id=business_id,
                    first_seen_file_id=file_id,
                    source_type=chunk_source_type,
                    embedding_model=chunk_embed_model,
                    occurrence_count=1,
                    created_at=now,
                    updated_at=now,
                )
                .on_conflict_do_update(
                    # Concurrent ingestion of the same content — increment count.
                    # Next ingestion will find count≥1 in batch_check_gci and skip.
                    index_elements=["semantic_hash"],
                    set_={
                        "occurrence_count": GlobalContentIndexV2.occurrence_count + 1,
                        "updated_at":       now,
                    },
                )
                .returning(GlobalContentIndexV2.id, GlobalContentIndexV2.semantic_hash)
            )

            result = await db.execute(stmt)
            row    = result.fetchone()
            if row:
                gci_id = str(row[0])
                hash_to_gci_id[semantic_hash] = gci_id
                chunk["global_content_id"] = gci_id  # update chunk in-place
                log_info(f"[GCI] Registered unique chunk {gci_id[:8]}...")

        except Exception as e:
            log_warning(f"[GCI] Registration failed for hash {semantic_hash[:12]}...: {e}")

    try:
        await db.commit()
    except Exception as e:
        log_warning(f"[GCI] Commit failed: {e}")
        await db.rollback()

    log_info(f"[GCI] Registered {len(hash_to_gci_id)} unique chunks in GlobalContentIndex")
    return hash_to_gci_id