"""
Semantic Conflict Detection Engine - Production Grade
Version: 2.0 (Async-Safe, Retrieval-Optimized, Pluggable VectorDB)

Responsibilities:
- Detect contradictory information across sources
- Compute conflict risk scores
- Provide retrieval-safe conflict modifiers
- Never block async retrieval pipeline
- ZERO direct chromadb imports — uses BaseVectorDB/BaseEmbedder only
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
import math

from app.db.session_v2 import AsyncSessionLocal
from app.db.models.ingested_content_v2 import IngestedContentV2
from app.db.models.global_content_index_v2 import GlobalContentIndexV2
from app.utils.logger import log_info, log_warning, log_debug

# ── Pluggable pipeline (ONLY way to reach VectorDB/Embedder) ──────────
# NO direct chromadb import here. chroma_v1.py is the only file
# that imports chromadb. Everything goes through BaseVectorDB.
from app.services.ingestion.ingestion_service_v2 import _get_pipeline
from app.core.vectordb.base import BaseVectorDB
from app.core.embedders.base import BaseEmbedder

# ============================================================
# VERSIONING
# ============================================================

CONFLICT_VERSION          = "conflict_analysis_v2"
CONFLICT_SCHEMA_CONTRACT  = "retrieval_v2_compatible"

# ============================================================
# CONFLICT DETECTION CONFIG
# ============================================================

class ConflictConfig:
    """Configurable conflict detection parameters."""
    SIMILARITY_THRESHOLD = 0.80   # 80% similar = potential conflict
    TOP_K_SIMILAR        = 10     # Similar docs to check per chunk
    MIN_COLLECTION_SIZE  = 2      # Skip if collection smaller than this
    SIMILARITY_WEIGHT    = 0.9    # Weight for similarity in conflict score
    POLARITY_WEIGHT      = 1.0    # Opposite polarities = stronger conflict
    CACHE_TTL            = 300    # Cache TTL seconds

# ============================================================
# POLARITY DETECTION (Domain-Specific)
# ============================================================

POSITIVE_TERMS = {
    "increase", "improve", "boost", "grow", "enhance", "strengthen",
    "optimize", "accelerate", "expand", "elevate", "maximize",
    "reduce risk", "reduce cost", "reduce time", "increase revenue",
    "increase profit", "increase efficiency", "better", "higher",
    "faster", "stronger", "successful", "effective",
}

NEGATIVE_TERMS = {
    "decrease", "reduce", "decline", "worsen", "diminish", "weaken",
    "hurt", "damage", "impair", "limit", "restrict", "constrain",
    "raise cost", "raise risk", "increase risk", "increase cost",
    "lower", "slower", "worse", "failed", "ineffective", "problematic",
}

METRIC_KEYWORDS = {
    "productivity": {
        "productivity", "efficiency", "output", "throughput",
        "utilization", "performance", "capacity",
    },
    "cost": {
        "cost", "expense", "spend", "budget", "price",
        "overhead", "opex", "capex",
    },
    "revenue": {
        "revenue", "sales", "income", "profit", "margin",
        "earnings", "return", "roi",
    },
    "risk": {
        "risk", "exposure", "liability", "threat", "vulnerability",
        "compliance", "security", "safety",
    },
    "quality": {
        "quality", "accuracy", "reliability", "consistency",
        "defect", "error", "issue", "problem",
    },
    "time": {
        "time", "duration", "speed", "latency", "cycle time",
        "lead time", "response time", "turnaround",
    },
    "customer": {
        "customer satisfaction", "nps", "churn", "retention",
        "engagement", "loyalty", "experience",
    },
}

# ============================================================
# PUBLIC ENTRY POINT (Worker)
# ============================================================

async def run_semantic_conflict_detection(batch_size: int = 50) -> Dict[str, Any]:
    """
    Main conflict detection worker.

    Multi-tenant aware:
      - Groups rows by business_id
      - Resolves correct pipeline per client (correct VectorDB + collection)
      - ZERO direct chromadb calls — uses BaseVectorDB.search() only

    Returns:
        Processing stats dict
    """
    start_time = datetime.now(timezone.utc)

    async with AsyncSessionLocal() as session:
        rows = await _fetch_candidates(session, batch_size)

        if not rows:
            log_info("[ConflictEngine] No rows pending conflict analysis.")
            return {"processed": 0, "conflicts_detected": 0, "duration_ms": 0}

        log_info(f"[ConflictEngine] Processing {len(rows)} chunks...")

        # ── Group rows by business_id (multi-tenant) ──────────────────
        # Each client has its own VectorDB path + collection.
        # Resolve pipeline ONCE per client, reuse for all their rows.
        rows_by_client: Dict[str, List[IngestedContentV2]] = defaultdict(list)
        for row in rows:
            bid = row.business_id or "default"
            rows_by_client[bid].append(row)

        processed       = 0
        conflicts_found = 0

        for business_id, client_rows in rows_by_client.items():
            # ── Resolve pipeline for this client ──────────────────────
            try:
                pipeline        = _get_pipeline(business_id)
                vectordb        = pipeline.vectordb          # BaseVectorDB
                embedder        = pipeline.embedder          # BaseEmbedder
                collection_name = pipeline.config.vectordb.collection  # per-client

                # Verify collection has minimum documents
                doc_count = vectordb.count(collection_name)
                if doc_count < ConflictConfig.MIN_COLLECTION_SIZE:
                    log_info(
                        f"[ConflictEngine] Collection '{collection_name}' "
                        f"too small ({doc_count} docs) for client '{business_id}' "
                        f"— skipping conflict detection"
                    )
                    continue

            except Exception as e:
                log_warning(
                    f"[ConflictEngine] Failed to resolve pipeline "
                    f"for client '{business_id}': {e} — skipping {len(client_rows)} rows"
                )
                continue

            # ── Process each chunk for this client ────────────────────
            for row in client_rows:
                try:
                    analysis = await analyze_conflicts_for_chunk(
                        session,
                        vectordb,
                        embedder,
                        collection_name,
                        row,
                    )

                    if analysis:
                        await append_conflict_snapshot(session, row.id, analysis)

                        if analysis.get("conflicts_detected"):
                            conflicts_found += len(analysis["conflicts_detected"])

                    processed += 1

                except Exception as e:
                    log_warning(f"[ConflictEngine] Failed for chunk {row.id}: {e}")

        await session.commit()

        duration_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        log_info(
            f"[ConflictEngine] ✅ Processed {processed} chunks, "
            f"found {conflicts_found} conflicts in {duration_ms:.2f}ms"
        )

        return {
            "processed":         processed,
            "conflicts_detected": conflicts_found,
            "duration_ms":       round(duration_ms, 2),
        }

# ============================================================
# CANDIDATE SELECTION
# ============================================================

async def _fetch_candidates(
    session: AsyncSession,
    batch_size: int,
) -> List[IngestedContentV2]:
    """
    Fetch chunks that have Step-2.1 validation but no conflict analysis yet.
    """
    stmt = (
        select(IngestedContentV2)
        .where(
            IngestedContentV2.validation_layer.is_not(None),
            ~IngestedContentV2.validation_layer.op("@>")(
                [{"method": CONFLICT_VERSION}]
            ),
        )
        .order_by(IngestedContentV2.created_at.asc())
        .limit(batch_size)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())

# ============================================================
# CORE CONFLICT ANALYSIS
# ============================================================

async def analyze_conflicts_for_chunk(
    session:         AsyncSession,
    vectordb:        BaseVectorDB,   # ← pluggable, NOT chromadb.Collection
    embedder:        BaseEmbedder,   # ← pluggable, NOT SentenceTransformer
    collection_name: str,            # ← per-client, NOT hardcoded
    chunk:           IngestedContentV2,
) -> Optional[Dict[str, Any]]:
    """
    Analyze semantic conflicts for a single chunk.

    Process:
      1. Validate chunk has GCI + canonical text
      2. Re-embed canonical text via BaseEmbedder (pluggable — works for
         Chroma, Qdrant, Pinecone, any backend)
      3. Find semantically similar chunks via BaseVectorDB.search()
      4. Check for polarity conflicts (opposing claims on same metric)
      5. Compute + return conflict risk snapshot

    Why we re-embed instead of reading stored vectors:
      Reading stored vectors requires Chroma-specific API (not in BaseVectorDB).
      Re-embedding is backend-agnostic and always produces the correct
      dimension for the current embedder. Cost: one embed_query() call per chunk.
    """

    # ── 1. Validate input ─────────────────────────────────────────────
    if not chunk.global_content_id:
        log_debug(f"[ConflictAnalysis] Chunk {chunk.id} missing global_content_id")
        return None

    gci = await session.get(GlobalContentIndexV2, chunk.global_content_id)
    if not gci:
        log_debug(f"[ConflictAnalysis] GCI not found for chunk {chunk.id}")
        return None

    if not gci.cleaned_text:
        log_debug(f"[ConflictAnalysis] Chunk {chunk.id} has no cleaned text")
        return None

    canonical_text = gci.cleaned_text
    semantic_hash  = gci.semantic_hash

    # ── 2. Embed canonical text via pluggable BaseEmbedder ────────────
    # This is backend-agnostic. Works for Chroma, Qdrant, Pinecone, all.
    # No direct chromadb call needed — no "different settings" crash possible.
    try:
        query_embedding: List[float] = embedder.embed_query(canonical_text)

        if not query_embedding or len(query_embedding) == 0:
            log_warning(f"[ConflictAnalysis] Zero-length embedding for chunk {chunk.id}")
            return None

        log_debug(
            f"[ConflictAnalysis] ✅ Embedded {len(query_embedding)}-dim vector "
            f"for chunk {chunk.id}"
        )

    except Exception as e:
        log_warning(
            f"[ConflictAnalysis] Embedding failed for chunk {chunk.id}: {e}"
        )
        return None

    # ── 3. Find similar chunks via pluggable BaseVectorDB.search() ────
    # Works identically for Chroma, Qdrant, Pinecone, Milvus, Weaviate.
    # Excludes self by filtering on semantic_hash != own hash.
    try:
        hits = vectordb.search(
            collection=collection_name,            # ← per-client, never hardcoded
            query_embedding=query_embedding,
            top_k=ConflictConfig.TOP_K_SIMILAR + 1,  # +1 because self may appear
            filters={"semantic_hash": {"$ne": semantic_hash}},
        )
    except Exception as e:
        log_warning(
            f"[ConflictAnalysis] VectorDB search failed for chunk {chunk.id}: {e}"
        )
        return None

    if not hits:
        log_debug(f"[ConflictAnalysis] No similar chunks found for {chunk.id}")
        return {
            "method":               CONFLICT_VERSION,
            "schema_contract":      CONFLICT_SCHEMA_CONTRACT,
            "evaluated_at":         utc_now_iso(),
            "chunk_id":             str(chunk.id),
            "similarity_threshold": ConflictConfig.SIMILARITY_THRESHOLD,
            "conflicts_detected":   [],
            "conflict_risk":        0.0,
            "confidence":           1.0,
        }

    # ── 4. Detect conflicts ───────────────────────────────────────────
    conflicts: List[Dict[str, Any]] = []

    for hit in hits:
        # hit.score is already cosine similarity [0,1] (converted in chroma_v1.py)
        similarity = hit.score if hasattr(hit, "score") else hit.get("score", 0.0)

        if similarity < ConflictConfig.SIMILARITY_THRESHOLD:
            continue

        other_hash = (
            hit.metadata.get("semantic_hash")
            if hasattr(hit, "metadata")
            else hit.get("metadata", {}).get("semantic_hash")
        )
        if not other_hash or other_hash == semantic_hash:
            continue

        # Fetch the other chunk's canonical text from GCI
        other_gci = await _get_gci_by_hash(session, other_hash)
        if not other_gci or not other_gci.cleaned_text:
            continue

        # Polarity conflict check
        metric, polarity_self, polarity_other = _detect_polarity_conflict(
            canonical_text,
            other_gci.cleaned_text,
        )

        if not metric:
            continue  # Same direction claims — no conflict

        pair_score = _compute_pair_conflict_score(
            similarity, polarity_self, polarity_other
        )

        conflicts.append({
            "semantic_hash":      other_hash,
            "similarity":         round(similarity, 4),
            "metric":             metric,
            "polarity_self":      polarity_self,
            "polarity_other":     polarity_other,
            "pair_conflict_score": round(pair_score, 4),
        })

    # ── 5. Aggregate conflict risk ────────────────────────────────────
    if not conflicts:
        return {
            "method":               CONFLICT_VERSION,
            "schema_contract":      CONFLICT_SCHEMA_CONTRACT,
            "evaluated_at":         utc_now_iso(),
            "chunk_id":             str(chunk.id),
            "similarity_threshold": ConflictConfig.SIMILARITY_THRESHOLD,
            "conflicts_detected":   [],
            "conflict_risk":        0.0,
            "confidence":           1.0,
        }

    conflict_risk = _aggregate_conflict_scores(
        [c["pair_conflict_score"] for c in conflicts]
    )
    confidence = min(
        1.0, len(conflicts) / ConflictConfig.TOP_K_SIMILAR + 0.5
    )

    return {
        "method":               CONFLICT_VERSION,
        "schema_contract":      CONFLICT_SCHEMA_CONTRACT,
        "evaluated_at":         utc_now_iso(),
        "chunk_id":             str(chunk.id),
        "similarity_threshold": ConflictConfig.SIMILARITY_THRESHOLD,
        "conflicts_detected":   conflicts,
        "conflict_risk":        round(conflict_risk, 4),
        "confidence":           round(confidence, 2),
        "metadata": {
            "top_k_checked":      ConflictConfig.TOP_K_SIMILAR,
            "conflicts_found":    len(conflicts),
            "embedding_dimension": len(query_embedding),
            "embedding_source":   "pluggable_embedder",   # not "chromadb"
            "collection":         collection_name,
        },
    }

# ============================================================
# POLARITY CONFLICT DETECTION  (unchanged — logic is correct)
# ============================================================

def _detect_polarity_conflict(
    text_a: str,
    text_b: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Detect if two texts make opposing claims about the same metric.
    Returns: (metric, polarity_a, polarity_b) or (None, None, None)
    """
    text_a_lower = text_a.lower()
    text_b_lower = text_b.lower()

    for metric_name, metric_keywords in METRIC_KEYWORDS.items():
        a_has_metric = any(kw in text_a_lower for kw in metric_keywords)
        b_has_metric = any(kw in text_b_lower for kw in metric_keywords)

        if not (a_has_metric and b_has_metric):
            continue

        a_positive = any(t in text_a_lower for t in POSITIVE_TERMS)
        a_negative = any(t in text_a_lower for t in NEGATIVE_TERMS)
        b_positive = any(t in text_b_lower for t in POSITIVE_TERMS)
        b_negative = any(t in text_b_lower for t in NEGATIVE_TERMS)

        polarity_a = (
            "positive" if a_positive and not a_negative
            else "negative" if a_negative and not a_positive
            else "neutral"
        )
        polarity_b = (
            "positive" if b_positive and not b_negative
            else "negative" if b_negative and not b_positive
            else "neutral"
        )

        if polarity_a != polarity_b and "neutral" not in (polarity_a, polarity_b):
            return metric_name, polarity_a, polarity_b

    return None, None, None

# ============================================================
# CONFLICT SCORING  (unchanged — logic is correct)
# ============================================================

def _compute_pair_conflict_score(
    similarity: float,
    polarity_a: str,
    polarity_b: str,
) -> float:
    base_score = similarity * ConflictConfig.SIMILARITY_WEIGHT
    polarity_factor = (
        ConflictConfig.POLARITY_WEIGHT
        if polarity_a != polarity_b and "neutral" not in (polarity_a, polarity_b)
        else 0.0
    )
    return base_score * polarity_factor


def _aggregate_conflict_scores(scores: List[float]) -> float:
    """
    Probabilistic OR aggregation: 1 - ∏(1 - score_i)
    Multiple weak conflicts compound; single strong conflict dominates.
    """
    if not scores:
        return 0.0
    risk = 1.0
    for score in scores:
        risk *= (1.0 - score)
    return max(0.0, min(1.0, 1.0 - risk))

# ============================================================
# DATABASE HELPERS
# ============================================================

async def _get_gci_by_hash(
    session: AsyncSession,
    semantic_hash: str,
) -> Optional[GlobalContentIndexV2]:
    stmt = (
        select(GlobalContentIndexV2)
        .where(GlobalContentIndexV2.semantic_hash == semantic_hash)
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalars().first()


async def append_conflict_snapshot(
    session:     AsyncSession,
    ingested_id: Any,
    snapshot:    Dict[str, Any],
) -> None:
    """
    Append conflict analysis snapshot to validation_layer using ORM.

    WHY ORM instead of raw SQL:
      Raw SQL "UPDATE ingested_content ..." had the wrong table name.
      IngestedContentV2 ORM always resolves to the correct table name
      from the model definition — no hardcoding.
    """
    result = await session.execute(
        select(IngestedContentV2).where(IngestedContentV2.id == ingested_id)
    )
    row = result.scalar_one_or_none()
    if not row:
        log_warning(f"[ConflictEngine] append_conflict_snapshot: id {ingested_id} not found")
        return

    current_layer = list(row.validation_layer or [])
    current_layer.append(snapshot)

    await session.execute(
        update(IngestedContentV2)
        .where(IngestedContentV2.id == ingested_id)
        .values(
            validation_layer=current_layer,
            updated_at=datetime.utcnow(),
        )
    )

# ============================================================
# RETRIEVAL ADAPTER (Async-Safe, unchanged — logic is correct)
# ============================================================

async def get_conflict_modifier_async(
    session:    AsyncSession,
    content_id: str,
) -> float:
    """
    Fetch conflict modifier for retrieval scoring.

    Returns float [0.0, 1.0]:
      1.0 = no conflict (full trust)
      0.0 = max conflict (zero trust)

    Guarantees: never raises, always returns valid float (fail-open = 1.0).
    """
    try:
        result = await session.execute(
            select(IngestedContentV2).where(IngestedContentV2.id == content_id)
        )
        content = result.scalar_one_or_none()

        if not content:
            log_debug(f"[ConflictModifier] {content_id} not found → 1.0")
            return 1.0

        if not content.validation_layer:
            log_debug(f"[ConflictModifier] {content_id} no validation_layer → 1.0")
            return 1.0

        for snapshot in reversed(content.validation_layer):
            if not isinstance(snapshot, dict):
                continue
            if snapshot.get("method") == CONFLICT_VERSION:
                conflict_risk = float(snapshot.get("conflict_risk", 0.0))
                modifier      = max(0.0, min(1.0, 1.0 - conflict_risk))
                log_debug(
                    f"[ConflictModifier] {content_id}: "
                    f"risk={conflict_risk:.3f} → modifier={modifier:.3f}"
                )
                return modifier

        log_debug(f"[ConflictModifier] {content_id} no conflict snapshot → 1.0")
        return 1.0

    except Exception as e:
        log_warning(
            f"[ConflictModifier] EXCEPTION for {content_id}: {e} "
            f"(fail-open → 1.0)"
        )
        return 1.0


def get_conflict_modifier_sync(content_id: str) -> float:
    """
    DEPRECATED: Synchronous wrapper (backward compatibility only).
    Use get_conflict_modifier_async() in all async contexts.
    """
    import asyncio
    async def _fetch():
        async with AsyncSessionLocal() as session:
            return await get_conflict_modifier_async(session, content_id)

    try:
        loop = asyncio.get_running_loop()
        log_warning(
            "[ConflictModifier] Sync wrapper called inside async context! "
            "Use get_conflict_modifier_async() instead. Returning 1.0."
        )
        return 1.0
    except RuntimeError:
        pass  # No running loop — safe to create one

    try:
        return asyncio.run(_fetch())
    except Exception as e:
        log_warning(f"[ConflictModifier] Sync wrapper failed: {e} → 1.0")
        return 1.0


# Backward compatibility alias
get_conflict_modifier = get_conflict_modifier_async

# ============================================================
# UTILITIES
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
