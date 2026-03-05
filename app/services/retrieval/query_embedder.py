"""
Query Embedding Service
Pluggable — uses MAI_EMBEDDER from .env (ollama, openai, huggingface, etc.)
Falls back to ingestion_service_v2.get_embedder() for consistency.
"""

from typing import List
from app.utils.logger import log_debug, log_info


# =========================================================
# PLUGGABLE EMBEDDER (matches ingestion exactly)
# =========================================================

def _get_pluggable_embedder():
    """Get the pluggable embedder configured via MAI_EMBEDDER in .env."""
    from app.services.ingestion.ingestion_service_v2 import get_embedder
    return get_embedder()


def get_query_embedder():
    """
    Get embedder instance (pluggable — ollama/openai/huggingface).
    Returns the _EmbedderAdapter from ingestion_service_v2.
    """
    return _get_pluggable_embedder()


def reset_embedder():
    """Reset embedder (for testing) — no-op for pluggable embedder."""
    pass


# =========================================================
# EMBEDDING FUNCTION
# =========================================================

def embed_query(query: str) -> List[float]:
    """
    Generate embedding for query text using pluggable embedder.
    
    Args:
        query: Query text (must not be empty)
    
    Returns:
        List of floats (embedding vector)
    
    Raises:
        ValueError: If query is empty
    """
    
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    embedder = get_query_embedder()
    
    log_debug(f"[QueryEmbedder] Embedding query: '{query[:50]}...'")
    
    result = embedder.encode(query, normalize_embeddings=True)
    embedding_list = result.tolist()
    
    log_debug(f"[QueryEmbedder] Generated {len(embedding_list)}-dim embedding")
    
    return embedding_list


def embed_batch(queries: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple queries (batch).
    
    Args:
        queries: List of query texts
    
    Returns:
        List of embedding vectors
    """
    
    if not queries:
        return []
    
    embedder = get_query_embedder()
    
    log_debug(f"[QueryEmbedder] Embedding batch of {len(queries)} queries")
    
    results = [embedder.encode(q, normalize_embeddings=True).tolist() for q in queries]
    return results
