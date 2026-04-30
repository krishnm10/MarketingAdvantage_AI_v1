"""
Unified cache manager for the RAG pipeline execution path.

Provides a single abstraction over four independent cache layers:

  1. Embedding cache   — avoids re-embedding identical queries/chunks
  2. Retrieval cache   — caches vector search results for repeated queries
  3. Reranker cache    — caches reranking scores for (query, doc_set) pairs
  4. Prompt cache      — caches assembled prompts or LLM responses for
                         deterministic queries

Each layer is independently configurable (enabled/disabled, TTL, max size)
and operates with tenant-scoped keys to prevent cross-tenant leakage.

This module is abstraction-only — no integration with pipeline stages yet.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Cache layer enum
# ─────────────────────────────────────────────────────────────────────────────

class CacheLayer(str, Enum):
    EMBEDDING = "embedding"
    RETRIEVAL = "retrieval"
    RERANKER = "reranker"
    PROMPT = "prompt"


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CacheLayerConfig:
    """Configuration for a single cache layer."""

    enabled: bool = True
    ttl_seconds: int = 300
    max_entries: int = 1024


# ─────────────────────────────────────────────────────────────────────────────
# Cache entry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    """Internal representation of a cached value with metadata."""

    value: Any
    created_at: float = field(default_factory=time.time)
    hits: int = 0
    tenant_id: str = ""
    layer: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Cache statistics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CacheStats:
    """Aggregated stats for observability."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Unified Cache Manager
# ─────────────────────────────────────────────────────────────────────────────

class UnifiedCacheManager:
    """
    Tenant-aware, multi-layer cache manager for RAG pipeline stages.

    Each cache layer is isolated and independently configurable. Keys are
    always scoped by tenant_id to prevent cross-tenant data leakage.

    Usage (future integration):
        cache = UnifiedCacheManager(config={...})
        cached = cache.get_embedding(tenant_id, query_text)
        if cached is None:
            embedding = embedder.embed_query(query_text)
            cache.put_embedding(tenant_id, query_text, embedding)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        raw = config or {}
        self._layer_configs: Dict[CacheLayer, CacheLayerConfig] = {
            layer: CacheLayerConfig(
                **raw.get(layer.value, {}),
            )
            for layer in CacheLayer
        }
        self._stores: Dict[CacheLayer, Dict[str, CacheEntry]] = {
            layer: {} for layer in CacheLayer
        }
        self._stats: Dict[CacheLayer, CacheStats] = {
            layer: CacheStats() for layer in CacheLayer
        }

    # ─── Mandatory cache key builder ────────────────────────────────────

    @staticmethod
    def build_cache_key(
        *,
        tenant_id: str,
        pipeline_id: str,
        embedder_fingerprint: str,
        query_hash: str,
    ) -> str:
        """
        Build a fully-qualified, tenant-scoped cache key.

        ALL cache operations MUST route through this builder. The composite
        key ensures isolation across tenants, pipelines, and model versions.

        Raises:
            ValueError: If tenant_id is missing or empty.
        """
        if not tenant_id or not tenant_id.strip():
            raise ValueError(
                "Cache key construction requires a non-empty tenant_id. "
                "Refusing to build a cache key without tenant isolation."
            )
        return (
            f"{tenant_id}::{pipeline_id}::"
            f"{embedder_fingerprint}::{query_hash}"
        )

    # ─── Embedding cache ─────────────────────────────────────────────────

    def get_embedding(
        self,
        *,
        tenant_id: str,
        pipeline_id: str,
        embedder_fingerprint: str,
        text: str,
    ) -> Optional[List[float]]:
        """Retrieve a cached embedding vector for a query/chunk text."""
        key = self.build_cache_key(
            tenant_id=tenant_id,
            pipeline_id=pipeline_id,
            embedder_fingerprint=embedder_fingerprint,
            query_hash=self._hash(text),
        )
        return self._get(CacheLayer.EMBEDDING, tenant_id, key)

    def put_embedding(
        self,
        *,
        tenant_id: str,
        pipeline_id: str,
        embedder_fingerprint: str,
        text: str,
        embedding: List[float],
    ) -> None:
        """Store an embedding vector keyed by text."""
        key = self.build_cache_key(
            tenant_id=tenant_id,
            pipeline_id=pipeline_id,
            embedder_fingerprint=embedder_fingerprint,
            query_hash=self._hash(text),
        )
        self._put(CacheLayer.EMBEDDING, tenant_id, key, embedding)

    # ─── Retrieval cache ─────────────────────────────────────────────────

    def get_retrieval(
        self,
        *,
        tenant_id: str,
        pipeline_id: str,
        embedder_fingerprint: str,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached vector search results."""
        key = self.build_cache_key(
            tenant_id=tenant_id,
            pipeline_id=pipeline_id,
            embedder_fingerprint=embedder_fingerprint,
            query_hash=self._retrieval_hash(query, top_k, filters),
        )
        return self._get(CacheLayer.RETRIEVAL, tenant_id, key)

    def put_retrieval(
        self,
        *,
        tenant_id: str,
        pipeline_id: str,
        embedder_fingerprint: str,
        query: str,
        top_k: int,
        results: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Cache vector search results."""
        key = self.build_cache_key(
            tenant_id=tenant_id,
            pipeline_id=pipeline_id,
            embedder_fingerprint=embedder_fingerprint,
            query_hash=self._retrieval_hash(query, top_k, filters),
        )
        self._put(CacheLayer.RETRIEVAL, tenant_id, key, results)

    # ─── Reranker cache ──────────────────────────────────────────────────

    def get_rerank(
        self,
        *,
        tenant_id: str,
        pipeline_id: str,
        embedder_fingerprint: str,
        query: str,
        doc_ids: List[str],
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached reranking scores for a query + document set."""
        key = self.build_cache_key(
            tenant_id=tenant_id,
            pipeline_id=pipeline_id,
            embedder_fingerprint=embedder_fingerprint,
            query_hash=self._rerank_hash(query, doc_ids),
        )
        return self._get(CacheLayer.RERANKER, tenant_id, key)

    def put_rerank(
        self,
        *,
        tenant_id: str,
        pipeline_id: str,
        embedder_fingerprint: str,
        query: str,
        doc_ids: List[str],
        scores: List[Dict[str, Any]],
    ) -> None:
        """Cache reranking scores."""
        key = self.build_cache_key(
            tenant_id=tenant_id,
            pipeline_id=pipeline_id,
            embedder_fingerprint=embedder_fingerprint,
            query_hash=self._rerank_hash(query, doc_ids),
        )
        self._put(CacheLayer.RERANKER, tenant_id, key, scores)

    # ─── Prompt / LLM response cache ────────────────────────────────────

    def get_prompt(
        self,
        *,
        tenant_id: str,
        pipeline_id: str,
        embedder_fingerprint: str,
        prompt_hash: str,
    ) -> Optional[str]:
        """Retrieve a cached LLM response for a deterministic prompt."""
        key = self.build_cache_key(
            tenant_id=tenant_id,
            pipeline_id=pipeline_id,
            embedder_fingerprint=embedder_fingerprint,
            query_hash=prompt_hash,
        )
        return self._get(CacheLayer.PROMPT, tenant_id, key)

    def put_prompt(
        self,
        *,
        tenant_id: str,
        pipeline_id: str,
        embedder_fingerprint: str,
        prompt_hash: str,
        response: str,
    ) -> None:
        """Cache an LLM response keyed by prompt hash."""
        key = self.build_cache_key(
            tenant_id=tenant_id,
            pipeline_id=pipeline_id,
            embedder_fingerprint=embedder_fingerprint,
            query_hash=prompt_hash,
        )
        self._put(CacheLayer.PROMPT, tenant_id, key, response)

    # ─── Observability ───────────────────────────────────────────────────

    def stats(self, layer: Optional[CacheLayer] = None) -> Dict[str, Any]:
        """Return cache statistics, optionally filtered by layer."""
        if layer is not None:
            s = self._stats[layer]
            return {
                "layer": layer.value,
                "hits": s.hits,
                "misses": s.misses,
                "evictions": s.evictions,
                "size": s.size,
                "hit_rate": round(s.hit_rate, 4),
            }
        return {
            lay.value: self.stats(lay) for lay in CacheLayer
        }

    def invalidate(
        self,
        tenant_id: str,
        layer: Optional[CacheLayer] = None,
    ) -> int:
        """
        Invalidate all cache entries for a tenant, optionally scoped to a
        single layer. Returns the number of entries removed.
        """
        removed = 0
        layers = [layer] if layer else list(CacheLayer)
        for lay in layers:
            store = self._stores[lay]
            keys_to_remove = [
                k for k, v in store.items() if v.tenant_id == tenant_id
            ]
            for k in keys_to_remove:
                del store[k]
                removed += 1
            self._stats[lay].evictions += len(keys_to_remove)
            self._stats[lay].size = len(store)
        return removed

    def clear(self, layer: Optional[CacheLayer] = None) -> None:
        """Clear all entries, optionally scoped to a single layer."""
        layers = [layer] if layer else list(CacheLayer)
        for lay in layers:
            size = len(self._stores[lay])
            self._stores[lay].clear()
            self._stats[lay].evictions += size
            self._stats[lay].size = 0

    # ─── Internal ────────────────────────────────────────────────────────

    def _get(
        self, layer: CacheLayer, tenant_id: str, composite_key: str
    ) -> Optional[Any]:
        cfg = self._layer_configs[layer]
        if not cfg.enabled:
            return None

        store = self._stores[layer]
        entry = store.get(composite_key)

        if entry is None:
            self._stats[layer].misses += 1
            return None

        if (time.time() - entry.created_at) > cfg.ttl_seconds:
            del store[composite_key]
            self._stats[layer].evictions += 1
            self._stats[layer].misses += 1
            self._stats[layer].size = len(store)
            return None

        entry.hits += 1
        self._stats[layer].hits += 1
        return entry.value

    def _put(
        self, layer: CacheLayer, tenant_id: str, composite_key: str, value: Any
    ) -> None:
        cfg = self._layer_configs[layer]
        if not cfg.enabled:
            return

        store = self._stores[layer]

        if len(store) >= cfg.max_entries:
            self._evict_oldest(layer)

        store[composite_key] = CacheEntry(
            value=value,
            tenant_id=tenant_id,
            layer=layer.value,
        )
        self._stats[layer].size = len(store)

    def _evict_oldest(self, layer: CacheLayer) -> None:
        """Evict the oldest entry from the layer store (simple LRU proxy)."""
        store = self._stores[layer]
        if not store:
            return
        oldest_key = min(store, key=lambda k: store[k].created_at)
        del store[oldest_key]
        self._stats[layer].evictions += 1
        self._stats[layer].size = len(store)

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def _retrieval_hash(
        query: str, top_k: int, filters: Optional[Dict[str, Any]]
    ) -> str:
        parts = f"{query}|k={top_k}|f={sorted(filters.items()) if filters else ''}"
        return hashlib.sha256(parts.encode()).hexdigest()

    @staticmethod
    def _rerank_hash(query: str, doc_ids: List[str]) -> str:
        parts = f"{query}|docs={'|'.join(sorted(doc_ids))}"
        return hashlib.sha256(parts.encode()).hexdigest()

    @staticmethod
    def hash_prompt(prompt: str) -> str:
        """Utility for callers to generate a stable prompt hash."""
        return hashlib.sha256(prompt.encode()).hexdigest()
