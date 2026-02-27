
# =============================================
# deduplication_engine_v2.py — 3-Layer Deduplication System
# =============================================
import hashlib
import re
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np

from app.db.models.ingested_content_v2 import IngestedContentV2
from app.db.models.global_content_index_v2 import GlobalContentIndexV2
from app.utils.logger import log_info, log_warning


# =============================================
# LAYER 1: NORMALIZED HASH (EXACT DEDUP)
# =============================================

def normalize_for_hash(text: str) -> str:
    """
    Ultra-aggressive normalization for exact duplicate detection.
    This ensures minor variations don't create different hashes.
    """
    if not text or not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove ALL whitespace variations (spaces, tabs, newlines, etc.)
    text = re.sub(r'\s+', ' ', text)

    # Remove ALL punctuation except alphanumeric
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    # Remove duplicate spaces
    text = ' '.join(text.split())

    return text


def create_normalized_hash(text: str) -> str:
    """
    Creates a normalized semantic hash that ignores:
    - Case differences
    - Whitespace variations
    - Punctuation differences
    - Unicode variations
    """
    normalized = normalize_for_hash(text)
    if not normalized:
        return hashlib.sha256(b"").hexdigest()

    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


# =============================================
# LAYER 2: EMBEDDING SIMILARITY (FUZZY DEDUP)
# =============================================

async def check_embedding_similarity(
    db: AsyncSession,
    vectordb: "BaseVectorDB",     # pluggable
    embedder: "BaseEmbedder",     # pluggable
    chunk_text: str,
    query_embedding: List[float], # already a List[float] — no numpy needed
    similarity_threshold: float = 0.95,
    top_k: int = 5
) -> Optional[Dict[str, Any]]:
    """
    Check if semantically similar chunks exist using vector similarity.

    Args:
        db: Database session
        chroma_collection: ChromaDB collection
        embedder: Sentence transformer model
        chunk_text: Text to check
        chunk_embedding: Pre-computed embedding vector
        similarity_threshold: Cosine similarity threshold (0.95 = 95% similar)
        top_k: Number of similar chunks to retrieve

    Returns:
        Dict with duplicate info if found, None otherwise
    """
    try:
        # Query ChromaDB for similar vectors
        hits = vectordb.search(
            collection="ingested_content",
            query_embedding=query_embedding,   # already List[float]
            top_k=top_k,
        )
        if not hits:
            return None
        for hit in hits:
            if hit.score >= similarity_threshold:
                log_info(f"[Dedup Layer 2] Found semantic duplicate "
                         f"similarity={hit.score:.4f}, chunk_id={hit.id}")
                return {
                    "is_duplicate":       True,
                    "duplicate_chunk_id": hit.id,
                    "similarity_score":   float(hit.score),
                    "duplicate_text":     hit.text,
                    "dedup_layer":        "embedding_similarity",
                }
        return None

    except Exception as e:
        log_warning(f"[Dedup Layer 2] Embedding similarity check failed: {e}")
        return None


# =============================================
# LAYER 3: GLOBAL CONTENT INDEX (CROSS-FILE DEDUP)
# =============================================

async def check_global_content_index(
    db: AsyncSession,
    normalized_hash: str,
    business_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Check if chunk exists in GlobalContentIndex (cross-file deduplication).

    Args:
        db: Database session
        normalized_hash: Normalized semantic hash
        business_id: Optional business scope for tenant isolation

    Returns:
        Dict with GCI info if exists, None otherwise
    """
    try:
        query = select(GlobalContentIndexV2).where(
            GlobalContentIndexV2.semantic_hash == normalized_hash
        )

        # Optional: Add business-level scoping
        if business_id:
            query = query.where(GlobalContentIndexV2.business_id == business_id)

        result = await db.execute(query)
        gci_entry = result.scalar_one_or_none()

        if gci_entry:
            log_info(
                f"[Dedup Layer 3] Found in GCI: "
                f"id={gci_entry.id}, occurrences={gci_entry.occurrence_count}"
            )

            return {
                'exists_in_gci': True,
                'gci_id': str(gci_entry.id),
                'occurrence_count': gci_entry.occurrence_count,
                'first_seen_file_id': str(gci_entry.first_seen_file_id) if gci_entry.first_seen_file_id else None,
                'confidence_avg': gci_entry.confidence_avg,
                'dedup_layer': 'global_content_index'
            }

        return None

    except Exception as e:
        log_warning(f"[Dedup Layer 3] GCI lookup failed: {e}")
        return None


# =============================================
# UNIFIED DEDUPLICATION ENGINE
# =============================================

async def deduplicate_chunks(
    db: AsyncSession,
    chunks: List[Dict[str, Any]],
    vectordb: "BaseVectorDB",      # replaces chroma_collection
    embedder: "BaseEmbedder",      # replaces raw SentenceTransformer
    file_id: str,
    business_id: Optional[str] = None,
    enable_embedding_dedup: bool = True,
    similarity_threshold: float = 0.95,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    3-Layer Deduplication Pipeline (Production-Grade).

    Layer 1: Normalized Hash (catches exact duplicates with minor variations)
    Layer 2: Embedding Similarity (catches semantic duplicates)
    Layer 3: Global Content Index (cross-file deduplication)

    Args:
        db: Database session
        chunks: List of chunk dictionaries
        chroma_collection: ChromaDB collection
        embedder: Sentence transformer model
        file_id: Current file ID
        business_id: Optional business ID for tenant isolation
        enable_embedding_dedup: Enable Layer 2 (embedding similarity)
        similarity_threshold: Minimum similarity for Layer 2 (0.0-1.0)

    Returns:
        Tuple of (unique_chunks, dedup_stats)
    """
    if not chunks:
        return [], {'total': 0, 'unique': 0, 'duplicates': 0, 'dedup_ratio': 0.0}

    unique_chunks = []
    duplicate_chunks = []

    # Track deduplication stats by layer
    stats = {
        'total': len(chunks),
        'unique': 0,
        'duplicates': 0,
        'layer1_hash_duplicates': 0,
        'layer2_embedding_duplicates': 0,
        'layer3_gci_duplicates': 0,
        'dedup_ratio': 0.0
    }

    # Track hashes in current batch to avoid intra-batch duplicates
    batch_hashes = set()

    log_info(f"[Dedup Engine] Starting 3-layer dedup for {len(chunks)} chunks")

    for idx, chunk in enumerate(chunks):
        chunk_text = chunk.get('cleaned_text') or chunk.get('text', '')

        if not chunk_text or not chunk_text.strip():
            continue

        # ====================================================
        # LAYER 1: NORMALIZED HASH DEDUPLICATION
        # ====================================================
        normalized_hash = create_normalized_hash(chunk_text)

        # Check within current batch (intra-file duplicates)
        if normalized_hash in batch_hashes:
            stats['layer1_hash_duplicates'] += 1
            stats['duplicates'] += 1

            chunk['is_duplicate'] = True
            chunk['duplicate_source'] = 'intra_batch'
            chunk['dedup_layer'] = 'layer1_normalized_hash'
            duplicate_chunks.append(chunk)

            log_info(f"[Dedup L1] Intra-batch duplicate: hash={normalized_hash[:12]}...")
            continue

        # Check in database (cross-file duplicates via GCI)
        gci_result = await check_global_content_index(
            db, 
            normalized_hash, 
            business_id
        )

        if gci_result:
            stats['layer3_gci_duplicates'] += 1
            stats['duplicates'] += 1

            chunk['is_duplicate'] = True
            chunk['duplicate_source'] = 'global_content_index'
            chunk['dedup_layer'] = 'layer1_normalized_hash'
            chunk['gci_id'] = gci_result['gci_id']
            chunk['occurrence_count'] = gci_result['occurrence_count']
            duplicate_chunks.append(chunk)

            log_info(
                f"[Dedup L1+L3] Cross-file duplicate: "
                f"hash={normalized_hash[:12]}..., occurrences={gci_result['occurrence_count']}"
            )
            continue

        # ====================================================
        # LAYER 2: EMBEDDING SIMILARITY DEDUPLICATION
        # ====================================================
        if enable_embedding_dedup:
            # Generate embedding for this chunk
            try:
                query_embedding: List[float] = embedder.embed_query(chunk_text)  # BaseEmbedder, returns List[float]
                embedding_result = await check_embedding_similarity(
                    db, vectordb, embedder, chunk_text, query_embedding, similarity_threshold
                )

                if embedding_result:
                    stats['layer2_embedding_duplicates'] += 1
                    stats['duplicates'] += 1

                    chunk['is_duplicate'] = True
                    chunk['duplicate_source'] = 'embedding_similarity'
                    chunk['dedup_layer'] = 'layer2_embedding'
                    chunk['similarity_score'] = embedding_result['similarity_score']
                    chunk['duplicate_chunk_id'] = embedding_result['duplicate_chunk_id']
                    duplicate_chunks.append(chunk)

                    log_info(
                        f"[Dedup L2] Semantic duplicate: "
                        f"similarity={embedding_result['similarity_score']:.4f}"
                    )
                    continue

            except Exception as e:
                log_warning(f"[Dedup L2] Embedding generation failed for chunk {idx}: {e}")

        # ====================================================
        # CHUNK IS UNIQUE - ADD TO RESULTS
        # ====================================================
        batch_hashes.add(normalized_hash)
        chunk['normalized_hash'] = normalized_hash
        chunk['is_duplicate'] = False
        unique_chunks.append(chunk)
        stats['unique'] += 1

    # Calculate deduplication ratio
    stats['dedup_ratio'] = (stats['duplicates'] / stats['total'] * 100) if stats['total'] > 0 else 0.0

    log_info(
        f"[Dedup Engine] Complete: {stats['unique']} unique, "
        f"{stats['duplicates']} duplicates ({stats['dedup_ratio']:.2f}% reduction) | "
        f"L1: {stats['layer1_hash_duplicates']}, "
        f"L2: {stats['layer2_embedding_duplicates']}, "
        f"L3: {stats['layer3_gci_duplicates']}"
    )

    return unique_chunks, stats
