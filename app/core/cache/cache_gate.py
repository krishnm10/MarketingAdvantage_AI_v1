"""
Feature-gated cache access.

Returns UnifiedCacheManager when ENABLE_SHARED_CACHE is active,
otherwise returns a NoOpCacheManager that always reports cache misses
and silently discards writes. This ensures existing cache logic
(e.g. embedding_cache.py) remains unchanged when the flag is off.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from app.core.runtime.runtime_flags import ENABLE_SHARED_CACHE


class NoOpCacheManager:
    """
    Drop-in replacement that performs no caching.

    All gets return None (miss), all puts are discarded. Stats report zeros.
    This preserves the UnifiedCacheManager interface contract so callers
    don't need conditional logic beyond the initial acquisition.
    """

    # ─── Key builder (still validates tenant_id for contract safety) ─────

    @staticmethod
    def build_cache_key(
        *,
        tenant_id: str,
        pipeline_id: str,
        embedder_fingerprint: str,
        query_hash: str,
    ) -> str:
        if not tenant_id or not tenant_id.strip():
            raise ValueError(
                "Cache key construction requires a non-empty tenant_id."
            )
        return ""

    # ─── Embedding ───────────────────────────────────────────────────────

    def get_embedding(self, **kwargs: Any) -> None:
        return None

    def put_embedding(self, **kwargs: Any) -> None:
        return None

    # ─── Retrieval ───────────────────────────────────────────────────────

    def get_retrieval(self, **kwargs: Any) -> None:
        return None

    def put_retrieval(self, **kwargs: Any) -> None:
        return None

    # ─── Reranker ────────────────────────────────────────────────────────

    def get_rerank(self, **kwargs: Any) -> None:
        return None

    def put_rerank(self, **kwargs: Any) -> None:
        return None

    # ─── Prompt ──────────────────────────────────────────────────────────

    def get_prompt(self, **kwargs: Any) -> None:
        return None

    def put_prompt(self, **kwargs: Any) -> None:
        return None

    # ─── Observability ───────────────────────────────────────────────────

    def stats(self, layer: Any = None) -> Dict[str, Any]:
        return {}

    def invalidate(self, tenant_id: str, layer: Any = None) -> int:
        return 0

    def clear(self, layer: Any = None) -> None:
        return None

    @staticmethod
    def hash_prompt(prompt: str) -> str:
        import hashlib
        return hashlib.sha256(prompt.encode()).hexdigest()


_INSTANCE: Optional[Union["NoOpCacheManager", Any]] = None


def get_cache_manager(
    config: Optional[Dict[str, Any]] = None,
) -> Union["NoOpCacheManager", Any]:
    """
    Return the appropriate cache manager based on ENABLE_SHARED_CACHE.

    - Flag ON  → UnifiedCacheManager (full caching)
    - Flag OFF → NoOpCacheManager (no-op, existing logic unchanged)

    The instance is lazily created and reused for the process lifetime.
    """
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE

    if ENABLE_SHARED_CACHE:
        from app.core.cache.unified_cache_manager import UnifiedCacheManager
        _INSTANCE = UnifiedCacheManager(config=config)
    else:
        _INSTANCE = NoOpCacheManager()

    return _INSTANCE
