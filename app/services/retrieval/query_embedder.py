"""
Query Embedding Service
Matches ingestion embedding exactly (BAAI/bge-large-en)
"""

from sentence_transformers import SentenceTransformer
from typing import List
from app.utils.logger import log_debug, log_info
from app.config.ingestion_settings import EMBEDDING_MODEL_NAME

# =========================================================
# CONFIGURATION
# =========================================================

EMBED_MODEL_NAME = ""
EMBEDDING_DIMENSION = 1024

_EMBEDDER = None


# =========================================================
# EMBEDDER (SINGLETON)
# =========================================================

def get_query_embedder() -> SentenceTransformer:
    """
    Get sentence transformer model (singleton).
    
    Returns:
        SentenceTransformer instance
    """
    global _EMBEDDER
    
    if _EMBEDDER is None:
        log_info(f"[QueryEmbedder] Loading model: {EMBED_MODEL_NAME}")
        _EMBEDDER = SentenceTransformer(EMBED_MODEL_NAME)
        log_info(f"[QueryEmbedder] ✅ Model loaded (dimension: {EMBEDDING_DIMENSION})")
    
    return _EMBEDDER


def reset_embedder():
    """Reset embedder (for testing)"""
    global _EMBEDDER
    _EMBEDDER = None


# =========================================================
# EMBEDDING FUNCTION
# =========================================================

def embed_query(query: str) -> List[float]:
    """
    Generate embedding for query text.
    
    Args:
        query: Query text (must not be empty)
    
    Returns:
        List of floats (1024-dimensional vector)
    
    Raises:
        ValueError: If query is empty
    
    Note:
        - Uses BAAI/bge-large-en (matches ingestion)
        - Normalizes embeddings for cosine similarity
        - Returns 1024-dimensional vector
    """
    
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    embedder = get_query_embedder()
    
    log_debug(f"[QueryEmbedder] Embedding query: '{query[:50]}...'")
    
    # Generate embedding with normalization
    embedding = embedder.encode(
        [query],
        normalize_embeddings=True
    )
    
    # Convert to list
    try:
        embedding_list = embedding[0].tolist()
    except Exception:
        embedding_list = list(embedding[0])
    
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
    
    embeddings = embedder.encode(
        queries,
        normalize_embeddings=True
    )
    
    # Convert to list of lists
    return [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings]
