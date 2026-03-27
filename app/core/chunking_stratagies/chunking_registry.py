from __future__ import annotations

import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Dict, List, Optional, Type

from app.core.chunking_stratagies.segmenter_v2 import recursive_semantic_chunk
from app.utils.logger import log_warning


class Chunker(ABC):
    """Common interface for all ingestion chunking strategies."""

    @abstractmethod
    async def chunk(
        self,
        text: str,
        *,
        db_session=None,
        file_id=None,
        business_id=None,
        source_type: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


_REGISTRY: Dict[str, Type[Chunker]] = {}
_DEFAULTS_REGISTERED = False


def register_chunker(name: str):
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Chunker name cannot be empty")

    def _decorator(cls: Type[Chunker]) -> Type[Chunker]:
        existing = _REGISTRY.get(key)
        if existing and existing is not cls:
            log_warning(
                f"[ChunkingRegistry] overriding '{key}' from "
                f"{existing.__name__} to {cls.__name__}"
            )
        _REGISTRY[key] = cls
        return cls

    return _decorator


@register_chunker("semantic")
@register_chunker("recursive")
class SemanticChunker(Chunker):
    async def chunk(
        self,
        text: str,
        *,
        db_session=None,
        file_id=None,
        business_id=None,
        source_type: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return await recursive_semantic_chunk(
            text,
            db_session=db_session,
            file_id=file_id,
            business_id=business_id,
            source_type=source_type,
            embedding_model=embedding_model,
        )


def _ensure_default_chunkers() -> None:
    global _DEFAULTS_REGISTERED
    if _DEFAULTS_REGISTERED:
        return
    # Import side effects register optional/alternative chunkers.
    from app.core.chunking_stratagies import segmenter_overlap  # noqa: F401
    from app.core.chunking_stratagies import segmenter_recursive_overlap  # noqa: F401
    from app.core.chunking_stratagies import segmenter_rust  # noqa: F401
    from app.core.chunking_stratagies import segmenter_smart  # noqa: F401
    from app.core.chunking_stratagies import segmenter_structure_aware  # noqa: F401

    _DEFAULTS_REGISTERED = True


def list_chunking_strategies() -> List[str]:
    _ensure_default_chunkers()
    return sorted(_REGISTRY.keys())


def _normalize_strategy(strategy: Optional[str]) -> str:
    raw = strategy if strategy is not None else os.getenv("CHUNKING_STRATEGY", "semantic")
    return str(raw).strip().lower() or "semantic"


@lru_cache(maxsize=16)
def get_chunker(strategy: Optional[str] = None) -> Chunker:
    _ensure_default_chunkers()
    key = _normalize_strategy(strategy)
    cls = _REGISTRY.get(key)
    if cls is None:
        valid = ", ".join(list_chunking_strategies())
        raise ValueError(
            f"Unknown chunking strategy '{key}'. "
            f"Valid values: {valid}"
        )
    return cls()


def clear_chunker_cache() -> None:
    get_chunker.cache_clear()
