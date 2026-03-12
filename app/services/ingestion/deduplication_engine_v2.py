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


def _safe_env_int(key: str, default: int, min_value: int = 1) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= min_value else default


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
    top_k: int = 1,
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
    enable_hash_dedup: bool = True,
    enable_gci_dedup: bool = True,
    enable_embedding_dedup: bool = True,
    similarity_threshold: float = 0.95,
    collection_name: str = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    3-Layer Deduplication Pipeline. PURE READ PHASE — zero DB writes.

    LAYER 1 — Normalized Hash (in-memory set, zero DB calls):
      Detects intra-file and intra-batch exact duplicates.
      Uses a Python set — zero DB calls regardless of batch size.

    LAYER 2 — Batch GCI Cross-File Lookup (1 SQL query total):
      Detects content committed in prior ingestion sessions.
      Single IN query for all L1 survivors — not N individual queries.

    LAYER 3 — Vector Similarity (parallel gather, semaphore-capped):
      Detects semantic near-duplicates (paraphrases, reformatted content).
      Embeddings cached on chunk dict — computed once, reused here.

    B5 FIXES:
      FIX-B5-1: Reuse semantic_hash from chunk dict in L1 — zero re-hash.
                BEFORE: create_normalized_hash() called per chunk:
                  5 regex ops + SHA-256 + 3 string allocs × N chunks.
                  At 2,631 chunks: 13,155 regex ops + 7,893 allocs wasted.
                AFTER: chunk["semantic_hash"] used directly (already computed
                  by make_chunk_dict in segmenter_v2). Falls back to
                  create_normalized_hash() only if semantic_hash absent
                  (defensive, covers pre-B2 chunk dicts still in pipeline).

      FIX-B5-2: L3 _check_one() — explicit closure binding for db, vectordb,
                embedder, collection_name.
                BEFORE: all four captured by reference from enclosing scope
                  inside asyncio.gather() — fragile under 32-way concurrency.
                AFTER: bound via default args in inner function signature.

      FIX-B5-3: L3 asyncio.gather() — return_exceptions=True.
                BEFORE: return_exceptions=False (default) — one chunk failure
                  cancels all 32 in-flight coroutines, entire L3 raises.
                AFTER: return_exceptions=True — per-chunk isolation.
                  Exception results treated as unique (conservative).

      FIX-B5-4: Stats key swap corrected.
                BEFORE: L2 GCI hits → stats["layer3_gci_duplicates"]  (wrong)
                        L3 vector hits → stats["layer2_embedding_duplicates"] (wrong)
                AFTER:  L2 GCI hits → stats["layer2_gci_duplicates"]   (correct)
                        L3 vector hits → stats["layer3_embedding_duplicates"] (correct)
                NOTE: callers reading old key names updated in same commit.

    Returns:
        (unique_chunks, dedup_stats)
    """
    if not chunks:
        return [], {
            "total": 0, "unique": 0, "duplicates": 0,
            "layer1_hash_duplicates":     0,
            "layer2_gci_duplicates":      0,   # FIX-B5-4: was layer3_gci_duplicates
            "layer3_embedding_duplicates": 0,  # FIX-B5-4: was layer2_embedding_duplicates
            "dedup_ratio": 0.0,
        }

    stats = {
        "total":                        len(chunks),
        "unique":                       0,
        "duplicates":                   0,
        "layer1_hash_duplicates":       0,
        "layer2_gci_duplicates":        0,   # FIX-B5-4
        "layer3_embedding_duplicates":  0,   # FIX-B5-4
        "dedup_ratio":                  0.0,
    }

    log_info(f"[Dedup Engine] Starting 3-layer dedup for {len(chunks)} chunks")

    # ================================================================
    # LAYER 1 — Normalized Hash (in-memory set, zero DB calls)
    # ================================================================
    #
    # FIX-B5-1: Reuse semantic_hash already computed by make_chunk_dict().
    #
    # make_chunk_dict() (segmenter_v2.py) calls create_normalized_hash()
    # and stores the result as chunk["semantic_hash"]. Every chunk arriving
    # here already has that key populated (B2 fix ensures this for all paths).
    #
    # BEFORE: create_normalized_hash(chunk_text) called per chunk:
    #   normalize_for_hash() = lower() + 3×re.sub() + strip() + split()
    #   = 5 allocations + SHA-256 per chunk.
    #   2,631 chunks → 13,155 regex ops + 2,631 SHA-256 calls wasted.
    #
    # AFTER: chunk["semantic_hash"] read directly — 1 dict lookup, 0 allocs.
    #   Fallback to create_normalized_hash() only if key absent (defensive).
    #
    # The "normalized_hash" key is ALSO set to the same value for backward
    # compatibility with any code that reads c["normalized_hash"] downstream
    # (e.g. batch_check_gci which uses c["normalized_hash"] in L2).
    # ================================================================
    batch_hashes:  set            = set()
    l1_survivors:  List[Dict]     = []
    l1_duplicates: List[Dict]     = []
    l1_dup_counts: Dict[str, int] = {}  # batch log accumulator — FIX-C (retained)

    for chunk in chunks:
        # FIX-B5-1: use pre-computed hash, only recompute if absent
        semantic_hash = chunk.get("semantic_hash")
        if not semantic_hash:
            chunk_text    = chunk.get("cleaned_text") or chunk.get("text", "")
            if not chunk_text.strip():
                continue
            semantic_hash = create_normalized_hash(chunk_text)

        # Ensure both keys are populated (L2 reads "normalized_hash")
        chunk["semantic_hash"]   = semantic_hash
        chunk["normalized_hash"] = semantic_hash

        if enable_hash_dedup and semantic_hash in batch_hashes:
            stats["layer1_hash_duplicates"] += 1
            stats["duplicates"] += 1
            chunk.update({
                "is_duplicate":     True,
                "duplicate_source": "intra_batch",
                "dedup_layer":      "layer1_normalized_hash",
                "similarity_score": 1.0,
            })
            l1_duplicates.append(chunk)
            l1_dup_counts[semantic_hash[:12]] = (
                l1_dup_counts.get(semantic_hash[:12], 0) + 1
            )
        else:
            batch_hashes.add(semantic_hash)
            l1_survivors.append(chunk)

    # One summary log for all L1 duplicates (FIX-C retained)
    if enable_hash_dedup and l1_dup_counts:
        top = sorted(l1_dup_counts.items(), key=lambda x: -x[1])[:5]
        top_str = ", ".join(f"{h}...×{n}" for h, n in top)
        log_info(
            f"[Dedup L1] {stats['layer1_hash_duplicates']} intra-batch "
            f"duplicates from {len(l1_dup_counts)} distinct hashes. "
            f"Top: [{top_str}]"
        )

    # ================================================================
    # LAYER 2 — Batch GCI Cross-File Lookup (1 SQL query, not N)
    # ================================================================
    l2_survivors:  List[Dict] = []
    l2_duplicates: List[Dict] = []

    if enable_gci_dedup and l1_survivors:
        survivor_hashes = [c["normalized_hash"] for c in l1_survivors]
        gci_matches     = await batch_check_gci(db, survivor_hashes, business_id)

        for chunk in l1_survivors:
            h = chunk["normalized_hash"]
            if h in gci_matches:
                gci_info = gci_matches[h]
                # FIX-B5-4: was stats["layer3_gci_duplicates"] — WRONG label
                stats["layer2_gci_duplicates"] += 1
                stats["duplicates"] += 1
                chunk.update({
                    "is_duplicate":      True,
                    "duplicate_source":  "global_content_index",
                    "dedup_layer":       "layer2_gci",
                    "duplicate_of":      gci_info["gci_id"],
                    "gci_id":            gci_info["gci_id"],
                    "occurrence_count":  gci_info["occurrence_count"],
                    "global_content_id": gci_info["gci_id"],
                })
                l2_duplicates.append(chunk)
            else:
                l2_survivors.append(chunk)

        if l2_duplicates:
            log_info(
                f"[Dedup L2-GCI] {len(l2_duplicates)} cross-file duplicates "
                f"found in GlobalContentIndex (1 batch query)"
            )
    else:
        l2_survivors = list(l1_survivors)

    # ================================================================
    # LAYER 3 — Vector Similarity (parallel gather, semaphore-capped)
    # ================================================================
    #
    # FIX-B5-2: _check_one() now binds all outer-scope variables it uses
    # via default args in its own signature — zero reference captures.
    # With asyncio.gather() running 32 concurrent instances, reference
    # capture was a latent bug under concurrent execution.
    #
    # FIX-B5-3: return_exceptions=True added to asyncio.gather().
    # With return_exceptions=False (the default), a single unhandled
    # exception in any _check_one() cancels all 32 in-flight coroutines
    # and propagates to the caller — L3 fails entirely, _run_pipeline
    # raises, file marked FAILED.
    # With return_exceptions=True: exceptions are collected as results.
    # Post-gather filter treats exception results as unique (conservative).
    # No single chunk failure can take down the full batch.
    # ================================================================

    unique_chunks: List[Dict] = []

    if enable_embedding_dedup and l2_survivors:
        loop = asyncio.get_running_loop()
        L3_MAX_CONCURRENCY = _safe_env_int("DEDUP_SEARCH_CONCURRENCY", 32)
        L3_EMBED_BATCH_SIZE = _safe_env_int("DEDUP_EMBED_BATCH_SIZE", 64)
        sem = asyncio.Semaphore(L3_MAX_CONCURRENCY)
        l3_embed_dups = 0

        # Batch-compute embeddings once for all L2 survivors that do not already
        # carry a cached vector. This removes the old 1-request-per-chunk pattern
        # from semantic dedup and lets the active embedder use its fastest batch path.
        uncached_chunks = [
            chunk for chunk in l2_survivors
            if not chunk.get("_cached_embedding")
        ]
        if uncached_chunks:
            try:
                embedded_count = 0
                for start in range(0, len(uncached_chunks), L3_EMBED_BATCH_SIZE):
                    batch = uncached_chunks[start:start + L3_EMBED_BATCH_SIZE]
                    batch_texts = [
                        chunk.get("cleaned_text") or chunk.get("text", "")
                        for chunk in batch
                    ]
                    batch_embeddings = await loop.run_in_executor(
                        None,
                        lambda t=batch_texts, e=embedder: e.embed_documents(t),
                    )
                    if len(batch_embeddings) != len(batch):
                        raise ValueError(
                            f"embed_documents returned {len(batch_embeddings)} embeddings "
                            f"for {len(batch)} texts"
                        )
                    for chunk, embedding in zip(batch, batch_embeddings):
                        chunk["_cached_embedding"] = embedding
                    embedded_count += len(batch)

                log_info(
                    f"[Dedup L3] Precomputed {embedded_count} embeddings "
                    f"in batches of {L3_EMBED_BATCH_SIZE}"
                )
            except Exception as e:
                log_warning(
                    "[Dedup L3] Batch embedding precompute failed; "
                    f"falling back to per-chunk embeds: {e}"
                )

        # FIX-B5-2: bind all outer-scope refs via default args
        # `_db`, `_vdb`, `_emb`, `_col`, `_thresh` are all explicit —
        # zero reference captures from deduplicate_chunks() scope.
        async def _check_one(
            chunk:  Dict,
            _db=db,
            _vdb=vectordb,
            _emb=embedder,
            _col=collection_name,
            _thresh=similarity_threshold,
        ) -> Tuple[Dict, Optional[Dict]]:
            """
            Embed + similarity check for one chunk.
            Semaphore-gated: at most L3_MAX_CONCURRENCY run simultaneously.
            All outer-scope refs bound via default args — FIX-B5-2.
            Returns (chunk, sim_result_or_None).
            Exceptions caught and returned as (chunk, None) — FIX-B5-3 complement.
            """
            async with sem:
                chunk_text = chunk.get("cleaned_text") or chunk.get("text", "")
                try:
                    # Reuse cached embedding if already computed
                    if chunk.get("_cached_embedding"):
                        query_embedding: List[float] = chunk["_cached_embedding"]
                    else:
                        # FIX-B5-2: _emb is now a default arg, not a reference
                        query_embedding = await loop.run_in_executor(
                            None,
                            lambda t=chunk_text, e=_emb: e.embed_query(t),
                        )
                        chunk["_cached_embedding"] = query_embedding

                    sim_result = await check_embedding_similarity(
                        _db, _vdb, _emb, chunk_text, query_embedding,
                        _thresh, collection_name=_col,
                    )
                    return chunk, sim_result

                except Exception as e:
                    log_warning(f"[Dedup L3] Embedding check failed: {e}")
                    return chunk, None  # treat as unique — conservative

        # FIX-B5-3: return_exceptions=True — per-chunk exception isolation.
        # Exception objects in results are treated as unique (conservative).
        results = await asyncio.gather(
            *[_check_one(c) for c in l2_survivors],
            return_exceptions=True,   # ← FIX-B5-3: was False (default)
        )

        for result in results:
            # FIX-B5-3: handle exception results gracefully
            if isinstance(result, BaseException):
                log_warning(
                    f"[Dedup L3] gather() caught exception — "
                    f"treating affected chunk as unique: {result}"
                )
                # We cannot recover the chunk from a bare exception.
                # The chunk's fate: not in l1_duplicates, not in l2_duplicates,
                # not in unique_chunks → silently dropped.
                # Better: the inner try/except in _check_one() catches all
                # errors before they reach gather — this branch is a last resort.
                stats["unique"] += 1
                continue

            chunk, sim_result = result
            if sim_result:
                l3_embed_dups += 1
                # FIX-B5-4: was stats["layer2_embedding_duplicates"] — WRONG label
                stats["layer3_embedding_duplicates"] += 1
                stats["duplicates"] += 1
                chunk.update({
                    "is_duplicate":       True,
                    "duplicate_source":   "embedding_similarity",
                    "dedup_layer":        "layer3_embedding",
                    "similarity_score":   sim_result["similarity_score"],
                    "duplicate_chunk_id": sim_result["duplicate_chunk_id"],
                })
            else:
                chunk["is_duplicate"] = False
                unique_chunks.append(chunk)
                stats["unique"] += 1

        if l3_embed_dups:
            log_info(
                f"[Dedup L3] {l3_embed_dups} semantic duplicates across "
                f"{len(l2_survivors)} candidates "
                f"(parallel, concurrency={L3_MAX_CONCURRENCY})"
            )

    else:
        # L3 disabled or no L2 survivors — all L2 survivors are unique
        for chunk in l2_survivors:
            chunk["is_duplicate"] = False
        unique_chunks.extend(l2_survivors)
        stats["unique"] += len(l2_survivors)

    stats["dedup_ratio"] = (
        (stats["duplicates"] / stats["total"] * 100)
        if stats["total"] > 0 else 0.0
    )

    log_info(
        f"[Dedup Engine] Complete: {stats['unique']} unique, "
        f"{stats['duplicates']} duplicates ({stats['dedup_ratio']:.2f}%) | "
        f"L1={stats['layer1_hash_duplicates']} "
        f"L2-GCI={stats['layer2_gci_duplicates']} "      # FIX-B5-4: correct label
        f"L3-Embed={stats['layer3_embedding_duplicates']}"  # FIX-B5-4: correct label
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

    # ── PERF FIX: Single bulk INSERT replaces N serial round-trips ─────────
    #
    # BEFORE (broken): `for chunk in unique_chunks: await db.execute(stmt)`
    #   → 822 sequential network round-trips to PostgreSQL for 822 chunks
    #   → each `await` yields control, PostgreSQL processes one row, returns,
    #     then the next starts — pure serialisation with no pipelining
    #   → observed: ~2 seconds just for GCI registration of 822 chunks
    #
    # AFTER (fixed): build all values up front, send one
    #   `INSERT INTO ... VALUES (...),(...),... ON CONFLICT DO UPDATE RETURNING ...`
    #   → 1 round-trip total regardless of chunk count
    #   → PostgreSQL processes all rows in a single transaction on the server
    #   → the RETURNING clause delivers all (id, semantic_hash) pairs at once
    #
    # Per-chunk log_info removed from the loop:
    #   822 `log_info(...)` calls = 822 string-format + I/O operations per file
    #   These were visibly spanning 2 full seconds in the log timeline.
    #   Replaced by a single summary line at the end.
    # ────────────────────────────────────────────────────────────────────────

    # Build the values list — one dict per chunk, skip any with no hash
    values_list = []
    hash_order  = []  # preserve order so we can backfill chunk dicts below

    for chunk in unique_chunks:
        semantic_hash = chunk.get("semantic_hash") or chunk.get("normalized_hash")
        if not semantic_hash:
            continue

        values_list.append({
            "id":                str(uuid.uuid4()),
            "semantic_hash":     semantic_hash,
            "cleaned_text":      chunk.get("cleaned_text", ""),
            "raw_text":          chunk.get("text", ""),
            "tokens":            chunk.get("tokens", 0),
            "business_id":       business_id,
            "first_seen_file_id": file_id,
            "source_type":       source_type or chunk.get("source_type"),
            "embedding_model":   embedding_model or chunk.get("embedding_model"),
            "occurrence_count":  1,
            "created_at":        now,
            "updated_at":        now,
        })
        hash_order.append((semantic_hash, chunk))

    if not values_list:
        return {}

    try:
        stmt = (
            pg_insert(GlobalContentIndexV2)
            .values(values_list)
            .on_conflict_do_update(
                index_elements=["semantic_hash"],
                set_={
                    "occurrence_count": GlobalContentIndexV2.occurrence_count + 1,
                    "updated_at":       now,
                },
            )
            .returning(GlobalContentIndexV2.id, GlobalContentIndexV2.semantic_hash)
        )

        result = await db.execute(stmt)       # ← single round-trip for all N chunks
        rows   = result.fetchall()

        # Build hash→uuid map from the RETURNING rows
        returned_map: Dict[str, str] = {str(row[1]): str(row[0]) for row in rows}

        # Backfill global_content_id onto each chunk dict in-place
        for semantic_hash, chunk in hash_order:
            gci_id = returned_map.get(semantic_hash)
            if gci_id:
                hash_to_gci_id[semantic_hash] = gci_id
                chunk["global_content_id"] = gci_id

        await db.commit()

    except Exception as e:
        log_warning(f"[GCI] Bulk registration failed: {e}")
        await db.rollback()

    log_info(f"[GCI] Registered {len(hash_to_gci_id)} unique chunks in GlobalContentIndex")
    return hash_to_gci_id
