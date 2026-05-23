"""
Thin facade — tenant-scoped Chroma search lives in chroma_search_service.

Legacy ChromaSearch singleton and env defaults are removed (Gate 4).
"""

from app.services.retrieval.chroma_search_service import (
    get_chroma_collection,
    health_check,
    reset_chroma_collection,
    semantic_search,
)

__all__ = [
    "get_chroma_collection",
    "semantic_search",
    "health_check",
    "reset_chroma_collection",
]
