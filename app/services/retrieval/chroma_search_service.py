"""
ChromaDB Search Service
Extracted from retrieve_cli.py for reusability across CLI and API

Supports BOTH remote (HttpClient) and local (PersistentClient) modes:
  - Remote: set CHROMA_HOST / CHROMA_PORT in .env
  - Local:  leave CHROMA_HOST empty → uses CHROMA_PATH (default ./chroma_db)
"""

import os
import asyncio
import chromadb
from chromadb.config import Settings
from typing import List, Tuple
from app.utils.logger import log_debug, log_info, log_warning


# =========================================================
# CONFIGURATION
# =========================================================

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("MAI_COLLECTION", "ingested_content")

_CHROMA_CLIENT = None
_COLLECTION = None


# =========================================================
# CHROMADB CONNECTION (SINGLETON)
# =========================================================

def get_chroma_collection():
    """
    Get ChromaDB collection (singleton pattern).
    Remote mode when CHROMA_HOST is set, local mode otherwise.
    
    Returns:
        chromadb.Collection instance
    """
    global _CHROMA_CLIENT, _COLLECTION
    
    if _CHROMA_CLIENT is None:
        chroma_host = os.getenv("CHROMA_HOST") or None
        if chroma_host:
            chroma_port = int(os.getenv("CHROMA_PORT") or "8000")
            use_ssl = os.getenv("CHROMA_SSL", "").lower() in ("1", "true", "yes")
            api_key = os.getenv("CHROMA_API_KEY") or None
            log_info(f"[ChromaSearch] Connecting to remote ChromaDB at {chroma_host}:{chroma_port}")
            _CHROMA_CLIENT = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                ssl=use_ssl,
                headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            log_info(f"[ChromaSearch] Initializing local ChromaDB at {CHROMA_PATH}")
            _CHROMA_CLIENT = chromadb.PersistentClient(
                path=CHROMA_PATH,
                settings=Settings(anonymized_telemetry=False),
            )
        _COLLECTION = _CHROMA_CLIENT.get_collection(COLLECTION_NAME)
        log_info(f"[ChromaSearch] ✅ Connected to collection '{COLLECTION_NAME}'")
    
    return _COLLECTION


def reset_chroma_collection():
    """Reset ChromaDB connection (for testing)"""
    global _CHROMA_CLIENT, _COLLECTION
    _CHROMA_CLIENT = None
    _COLLECTION = None


# =========================================================
# SEMANTIC SEARCH (PRODUCTION-GRADE, DEFENSIVE)
# =========================================================

async def semantic_search(
    query_embedding: List[float],
    limit: int = 200,
    where: dict = None,
    where_document: dict = None
) -> List[Tuple[str, float]]:
    """
    Search ChromaDB for semantically similar content.
    
    Args:
        query_embedding: Query vector (1024-dim for BAAI/bge-large-en)
        limit: Maximum results to return (default: 200)
        where: Metadata filters (optional)
        where_document: Document filters (optional)
    
    Returns:
        List of (semantic_hash, similarity_score) tuples
        Sorted by similarity (highest first)
    
    Note:
        - Version-safe: handles both old and new ChromaDB response formats
        - Defensive: validates all array structures before indexing
        - Async-safe: runs ChromaDB query in thread pool
    """
    
    # Validate input
    if not query_embedding or len(query_embedding) == 0:
        log_warning("[ChromaSearch] Empty query embedding provided")
        return []
    
    collection = get_chroma_collection()
    
    # Log collection size (for debugging)
    try:
        count = collection.count()
        log_debug(f"[ChromaSearch] Searching collection with {count:,} vectors")
    except Exception as e:
        log_debug(f"[ChromaSearch] Could not get count: {e}")
    
    # -------------------------------------------------
    # Execute query in thread pool (ChromaDB is sync)
    # -------------------------------------------------
    loop = asyncio.get_running_loop()
    
    def _query():
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": limit,
        }
        
        # Add filters if provided
        if where is not None:
            query_params["where"] = where
        if where_document is not None:
            query_params["where_document"] = where_document
        
        return collection.query(**query_params)
    
    try:
        result = await loop.run_in_executor(None, _query)
    except Exception as e:
        log_warning(f"[ChromaSearch] Query failed: {e}")
        return []
    
    # -------------------------------------------------
    # Parse results (defensive, version-safe)
    # -------------------------------------------------
    log_debug(f"[ChromaSearch] Raw result keys: {list(result.keys())}")
    
    # Get IDs (semantic hashes)
    ids = result.get("ids") or []
    distances = result.get("distances")
    
    # Handle nested list format (ChromaDB v0.4.x+)
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    
    if distances and isinstance(distances, list) and isinstance(distances[0], list):
        distances = distances[0]
    
    log_debug(f"[ChromaSearch] Found {len(ids)} results")
    
    # -------------------------------------------------
    # Convert to (hash, score) tuples
    # -------------------------------------------------
    hits: List[Tuple[str, float]] = []
    
    for idx, semantic_hash in enumerate(ids):
        # Convert L2 distance to similarity score
        if distances and idx < len(distances):
            # ChromaDB returns L2 distance (lower = more similar)
            # Convert to similarity: 1.0 - distance (clamped to [0, 1])
            distance = float(distances[idx])
            similarity_score = max(0.0, min(1.0, 1.0 - distance))
        else:
            # Fallback if distance missing
            similarity_score = 1.0
        
        hits.append((semantic_hash, similarity_score))
    
    # Log top result for debugging
    if hits:
        log_debug(
            f"[ChromaSearch] Top result: hash={hits[0][0][:16]}..., "
            f"score={hits[0][1]:.4f}"
        )
    
    return hits


# =========================================================
# HEALTH CHECK
# =========================================================

def health_check() -> dict:
    """
    Check ChromaDB health and return stats.
    
    Returns:
        Dict with health status and stats
    """
    try:
        collection = get_chroma_collection()
        count = collection.count()
        
        return {
            "status": "healthy",
            "collection_name": COLLECTION_NAME,
            "total_vectors": count,
            "storage_path": CHROMA_PATH
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
