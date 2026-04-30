"""
Compatibility adapters for gradual cache migration.

These adapters bridge the old per-layer cache implementations to the
UnifiedCacheManager without removing or modifying the old systems.

Migration path:
  Phase 1 (current): Adapters wrap old caches. ENABLE_SHARED_CACHE=false.
                      Old systems remain authoritative. Zero behavior change.
  Phase 2:           ENABLE_SHARED_CACHE=true. Adapters write-through to both
                      old and new. Reads prefer UnifiedCacheManager, fall back
                      to old cache on miss (warming).
  Phase 3:           Old caches removed. UnifiedCacheManager is sole authority.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.core.runtime.runtime_flags import ENABLE_SHARED_CACHE

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Cache Adapter
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingCacheAdapter:
    """
    Bridges app/core/embedders/embedding_cache.py → UnifiedCacheManager.

    When ENABLE_SHARED_CACHE is off:
        Delegates entirely to the old embedding_cache module (unchanged).

    When ENABLE_SHARED_CACHE is on:
        Write-through: writes to both old + new.
        Read: tries UnifiedCacheManager first, falls back to old on miss.
    """

    def __init__(
        self,
        *,
        tenant_id: str,
        pipeline_id: str,
        embedder_fingerprint: str,
    ) -> None:
        self._tenant_id = tenant_id
        self._pipeline_id = pipeline_id
        self._embedder_fingerprint = embedder_fingerprint

    def get_cached_query_embedding(
        self, embedder: Any, query_text: str
    ) -> List[float]:
        """
        Drop-in replacement for embedding_cache.get_cached_query_embedding.

        Signature matches the old function so callers can swap with minimal
        changes when ready.
        """
        from app.core.embedders.embedding_cache import (
            get_cached_query_embedding as _old_get,
        )

        if not ENABLE_SHARED_CACHE:
            return _old_get(embedder, query_text)

        from app.core.cache.cache_gate import get_cache_manager

        cache = get_cache_manager()

        cached = cache.get_embedding(
            tenant_id=self._tenant_id,
            pipeline_id=self._pipeline_id,
            embedder_fingerprint=self._embedder_fingerprint,
            text=query_text,
        )
        if cached is not None:
            return cached

        vec = _old_get(embedder, query_text)

        cache.put_embedding(
            tenant_id=self._tenant_id,
            pipeline_id=self._pipeline_id,
            embedder_fingerprint=self._embedder_fingerprint,
            text=query_text,
            embedding=vec,
        )
        return vec


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Cache Adapter
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalCacheAdapter:
    """
    Adapter for retrieval result caching.

    There is no existing standalone retrieval cache module — the RAG pipeline
    currently re-runs vector search on every request. This adapter provides
    a uniform interface that does nothing when the flag is off (preserving
    current behavior) and enables caching when the flag is on.
    """

    def __init__(
        self,
        *,
        tenant_id: str,
        pipeline_id: str,
        embedder_fingerprint: str,
    ) -> None:
        self._tenant_id = tenant_id
        self._pipeline_id = pipeline_id
        self._embedder_fingerprint = embedder_fingerprint

    def get(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Return cached retrieval results, or None on miss."""
        if not ENABLE_SHARED_CACHE:
            return None

        from app.core.cache.cache_gate import get_cache_manager

        return get_cache_manager().get_retrieval(
            tenant_id=self._tenant_id,
            pipeline_id=self._pipeline_id,
            embedder_fingerprint=self._embedder_fingerprint,
            query=query,
            top_k=top_k,
            filters=filters,
        )

    def put(
        self,
        query: str,
        top_k: int,
        results: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store retrieval results in cache."""
        if not ENABLE_SHARED_CACHE:
            return

        from app.core.cache.cache_gate import get_cache_manager

        get_cache_manager().put_retrieval(
            tenant_id=self._tenant_id,
            pipeline_id=self._pipeline_id,
            embedder_fingerprint=self._embedder_fingerprint,
            query=query,
            top_k=top_k,
            results=results,
            filters=filters,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Reranker Cache Adapter
# ─────────────────────────────────────────────────────────────────────────────

class RerankerCacheAdapter:
    """
    Adapter for reranker score caching.

    No existing reranker cache exists. This adapter is a no-op when the flag
    is off, and delegates to UnifiedCacheManager when on.
    """

    def __init__(
        self,
        *,
        tenant_id: str,
        pipeline_id: str,
        embedder_fingerprint: str,
    ) -> None:
        self._tenant_id = tenant_id
        self._pipeline_id = pipeline_id
        self._embedder_fingerprint = embedder_fingerprint

    def get(
        self,
        query: str,
        doc_ids: List[str],
    ) -> Optional[List[Dict[str, Any]]]:
        """Return cached reranker scores, or None on miss."""
        if not ENABLE_SHARED_CACHE:
            return None

        from app.core.cache.cache_gate import get_cache_manager

        return get_cache_manager().get_rerank(
            tenant_id=self._tenant_id,
            pipeline_id=self._pipeline_id,
            embedder_fingerprint=self._embedder_fingerprint,
            query=query,
            doc_ids=doc_ids,
        )

    def put(
        self,
        query: str,
        doc_ids: List[str],
        scores: List[Dict[str, Any]],
    ) -> None:
        """Store reranker scores in cache."""
        if not ENABLE_SHARED_CACHE:
            return

        from app.core.cache.cache_gate import get_cache_manager

        get_cache_manager().put_rerank(
            tenant_id=self._tenant_id,
            pipeline_id=self._pipeline_id,
            embedder_fingerprint=self._embedder_fingerprint,
            query=query,
            doc_ids=doc_ids,
            scores=scores,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt / LLM Response Cache Adapter
# ─────────────────────────────────────────────────────────────────────────────

class PromptCacheAdapter:
    """
    Adapter for LLM prompt/response caching.

    No existing prompt cache exists. This is a pure pass-through to the
    UnifiedCacheManager when enabled.
    """

    def __init__(
        self,
        *,
        tenant_id: str,
        pipeline_id: str,
        embedder_fingerprint: str,
    ) -> None:
        self._tenant_id = tenant_id
        self._pipeline_id = pipeline_id
        self._embedder_fingerprint = embedder_fingerprint

    def get(self, prompt: str) -> Optional[str]:
        """Return cached LLM response for a deterministic prompt."""
        if not ENABLE_SHARED_CACHE:
            return None

        from app.core.cache.cache_gate import get_cache_manager

        cm = get_cache_manager()
        return cm.get_prompt(
            tenant_id=self._tenant_id,
            pipeline_id=self._pipeline_id,
            embedder_fingerprint=self._embedder_fingerprint,
            prompt_hash=cm.hash_prompt(prompt),
        )

    def put(self, prompt: str, response: str) -> None:
        """Cache an LLM response keyed by prompt content."""
        if not ENABLE_SHARED_CACHE:
            return

        from app.core.cache.cache_gate import get_cache_manager

        cm = get_cache_manager()
        cm.put_prompt(
            tenant_id=self._tenant_id,
            pipeline_id=self._pipeline_id,
            embedder_fingerprint=self._embedder_fingerprint,
            prompt_hash=cm.hash_prompt(prompt),
            response=response,
        )
