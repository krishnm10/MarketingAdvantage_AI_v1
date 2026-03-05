"""
ChromaDB Search Service
Compatible with RetrievalRepository

Supports BOTH remote (HttpClient) and local (PersistentClient) modes:
  - Remote: set CHROMA_HOST / CHROMA_PORT in .env
  - Local:  leave CHROMA_HOST empty → uses chroma_path (default ./chroma_db)
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


# =========================================================
# CHROMADB SEARCH CLASS
# =========================================================

class ChromaSearch:
    """ChromaDB search service — supports remote and local ChromaDB."""
    
    def __init__(self, chroma_path: str = CHROMA_PATH, collection_name: str = COLLECTION_NAME):
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB client and collection (remote or local)."""
        try:
            chroma_host = os.getenv("CHROMA_HOST") or None
            if chroma_host:
                chroma_port = int(os.getenv("CHROMA_PORT") or "8000")
                use_ssl = os.getenv("CHROMA_SSL", "").lower() in ("1", "true", "yes")
                api_key = os.getenv("CHROMA_API_KEY") or None
                log_info(f"[ChromaSearch] Connecting to remote ChromaDB at {chroma_host}:{chroma_port}")
                self._client = chromadb.HttpClient(
                    host=chroma_host,
                    port=chroma_port,
                    ssl=use_ssl,
                    headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
                    settings=Settings(anonymized_telemetry=False),
                )
            else:
                log_info(f"[ChromaSearch] Initializing local ChromaDB at {self.chroma_path}")
                self._client = chromadb.PersistentClient(
                    path=self.chroma_path,
                    settings=Settings(anonymized_telemetry=False),
                )
            self._collection = self._client.get_collection(name=self.collection_name)
            log_info(f"[ChromaSearch] ✅ Connected to collection '{self.collection_name}'")
        except Exception as e:
            log_warning(f"[ChromaSearch] Failed to initialize: {e}")
            raise
    
    async def search(
        self,
        query_embedding: List[float],
        limit: int = 200
    ) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query vector
            limit: Max results
        
        Returns:
            List of (semantic_hash, similarity_score) tuples
        """
        
        if not query_embedding or len(query_embedding) == 0:
            log_warning("[ChromaSearch] Empty query embedding")
            return []
        
        log_debug(f"[ChromaSearch] Searching with limit={limit}, dim={len(query_embedding)}")
        
        # Run in thread pool (ChromaDB is sync)
        loop = asyncio.get_running_loop()
        
        def _query():
            return self._collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
            )
        
        try:
            result = await loop.run_in_executor(None, _query)
        except Exception as e:
            log_warning(f"[ChromaSearch] Query failed: {e}")
            return []
        
        # Extract IDs and distances
        ids = result.get("ids") or []
        distances = result.get("distances")
        
        # Handle nested lists (ChromaDB format)
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        
        if distances and isinstance(distances, list) and isinstance(distances[0], list):
            distances = distances[0]
        
        log_debug(f"[ChromaSearch] Found {len(ids)} results")
        
        # Convert to (hash, score) tuples
        hits = []
        for idx, semantic_hash in enumerate(ids):
            if distances and idx < len(distances):
                # Convert L2 distance to similarity
                score = max(0.0, min(1.0, 1.0 - distances[idx]))
            else:
                score = 1.0
            hits.append((semantic_hash, score))
        
        return hits


# =========================================================
# SINGLETON
# =========================================================

_chroma_search_instance = None


def get_chroma_search() -> ChromaSearch:
    """Get global ChromaSearch instance"""
    global _chroma_search_instance
    
    if _chroma_search_instance is None:
        _chroma_search_instance = ChromaSearch()
    
    return _chroma_search_instance
