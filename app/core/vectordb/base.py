"""
VectorDB connector contracts
File: app/core/vectordb/base.py

A minimal, enterprise-friendly interface so your retrieval/ingestion logic
does not care which VectorDB is plugged in.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class VectorHit:
    """Normalized search hit across all VectorDB backends."""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class BaseVectorDB(abc.ABC):
    """All VectorDB connectors must implement this interface."""

    @property
    @abc.abstractmethod
    def kind(self) -> str:
        """Short identifier like 'chroma' or 'qdrant'."""
        raise NotImplementedError

    @abc.abstractmethod
    def health_check(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def ensure_collection(self, collection: str, *, embedding_dim: int) -> None:
        """Create collection if missing (no-op if exists)."""
        raise NotImplementedError

    @abc.abstractmethod
    def upsert(
        self,
        *,
        collection: str,
        doc_id: str,
        embedding: List[float],
        text: str,
        metadata: Dict[str, Any],
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def search(
        self,
        *,
        collection: str,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorHit]:
        raise NotImplementedError

    # app/core/vectordb/base.py  ← ADD these lines after search(), before delete()

    @abc.abstractmethod
    def exists(self, *, collection: str, ids: List[str]) -> List[str]:
        """
        Return the subset of given IDs that already exist in the collection.
        Used by _embed_and_store() to skip re-embedding already-indexed chunks.
        """
        raise NotImplementedError

    


    @abc.abstractmethod
    def delete(self, *, collection: str, doc_id: str) -> None:
        raise NotImplementedError
