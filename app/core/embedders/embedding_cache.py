"""
================================================================================
Marketing Advantage AI — Query Embedding Cache
File: app/core/embedders/embedding_cache.py

Optional Redis-backed LRU cache for query embeddings.
Prevents identical queries from re-computing embeddings (e.g. 10 concurrent
users asking the same question = 1 embedding call + 9 cache hits).

Usage:
    from app.core.embedders.embedding_cache import get_cached_query_embedding

    vec = get_cached_query_embedding(embedder, query_text)

When Redis is unavailable the cache degrades to a small in-memory LRU
(thread-safe, zero external dependencies).  The embedder interface is
never modified — this is a transparent wrapper.
================================================================================
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
_CACHE_TTL = int(os.getenv("EMBED_CACHE_TTL", "600"))            # seconds
_CACHE_MAX_MEM = int(os.getenv("EMBED_CACHE_MAX_MEMORY", "256"))  # in-memory LRU size
_CACHE_PREFIX = "mai:embed_cache:"
_REDIS_ENABLED = os.getenv("EMBED_CACHE_REDIS", "false").lower() == "true"


# ── In-memory fallback LRU ────────────────────────────────────────────────────
class _MemoryLRU:
    """Thread-safe bounded LRU dict."""

    def __init__(self, maxsize: int = 256):
        self._data: OrderedDict[str, List[float]] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[List[float]]:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
        return None

    def put(self, key: str, value: List[float]) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            else:
                if len(self._data) >= self._maxsize:
                    self._data.popitem(last=False)
            self._data[key] = value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_mem_cache = _MemoryLRU(maxsize=_CACHE_MAX_MEM)


# ── Cache key ─────────────────────────────────────────────────────────────────
def _cache_key(query: str, model_name: str) -> str:
    raw = f"{query}||{model_name}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ── Redis helpers (lazy connect, never crash) ─────────────────────────────────
_redis_client = None
_redis_init_done = False
_redis_lock = threading.Lock()


def _get_redis():
    global _redis_client, _redis_init_done
    if _redis_init_done:
        return _redis_client
    with _redis_lock:
        if _redis_init_done:
            return _redis_client
        _redis_init_done = True
        if not _REDIS_ENABLED:
            return None
        try:
            import redis
            url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            _redis_client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=2)
            _redis_client.ping()
            logger.info("[EmbedCache] Redis cache connected at %s", url)
        except Exception as exc:
            logger.warning("[EmbedCache] Redis unavailable (%s) — using in-memory LRU.", exc)
            _redis_client = None
    return _redis_client


def _redis_get(key: str) -> Optional[List[float]]:
    r = _get_redis()
    if r is None:
        return None
    try:
        val = r.get(f"{_CACHE_PREFIX}{key}")
        if val:
            return json.loads(val)
    except Exception:
        pass
    return None


def _redis_put(key: str, vec: List[float]) -> None:
    r = _get_redis()
    if r is None:
        return
    try:
        r.setex(f"{_CACHE_PREFIX}{key}", _CACHE_TTL, json.dumps(vec))
    except Exception:
        pass


# ── Public API ────────────────────────────────────────────────────────────────
def get_cached_query_embedding(embedder, query_text: str) -> List[float]:
    """
    Return the embedding for *query_text*, using cache when available.

    Transparent wrapper — never changes the embedder's interface.
    Safe to call with any BaseEmbedder implementation.
    """
    model_name = getattr(embedder, "_model", "") or getattr(
        getattr(embedder, "info", None), "model", "unknown"
    )
    key = _cache_key(query_text, str(model_name))

    # 1. Try in-memory
    cached = _mem_cache.get(key)
    if cached is not None:
        return cached

    # 2. Try Redis
    cached = _redis_get(key)
    if cached is not None:
        _mem_cache.put(key, cached)
        return cached

    # 3. Compute
    vec = embedder.embed_query(query_text)

    # 4. Store
    _mem_cache.put(key, vec)
    _redis_put(key, vec)
    return vec


def clear_embedding_cache() -> None:
    """Flush both in-memory and Redis embedding caches."""
    _mem_cache.clear()
    r = _get_redis()
    if r:
        try:
            cursor = 0
            while True:
                cursor, keys = r.scan(cursor, match=f"{_CACHE_PREFIX}*", count=100)
                if keys:
                    r.delete(*keys)
                if cursor == 0:
                    break
        except Exception:
            pass
