# ============================================================
# app/core/vectordb/base.py
#
# WHAT THIS FILE IS:
#   The single source of truth for EVERY vector database operation.
#   Chroma, Qdrant, Pinecone, Milvus, Weaviate — ALL implement this.
#   No other file in the system imports chromadb/qdrant/pinecone directly.
#
# WHY EACH METHOD EXISTS:
#   upsert()       → insert or update ONE document vector
#   batch_upsert() → insert/update MANY at once (critical for ingestion speed)
#   search()       → find semantically similar documents by vector
#   exists()       → check which IDs already exist (prevents re-embedding)
#   get_by_ids()   → fetch documents by exact ID (for dedup lookup)
#   delete()       → remove one document
#   delete_many()  → remove many documents (for re-ingestion workflows)
#   count()        → total vectors stored (monitoring, health checks)
#   ensure_collection() → create collection/index if missing
#   delete_collection() → wipe a collection (admin, test cleanup)
#   health_check() → ping the backend
# ============================================================

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class VectorHit:
    """
    One search result, normalized across all backends.

    id       → The document ID (same as doc_id passed to upsert)
    text     → The original text of the chunk
    score    → Similarity score 0.0-1.0 (higher = more similar)
    metadata → Any metadata stored alongside the chunk
    """
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchUpsertResult:
    """
    Result from a batch_upsert() call.
    tells you exactly what happened — useful for logging and monitoring.
    """
    inserted: int = 0       # new documents added
    updated: int = 0        # existing documents updated
    failed: int = 0         # documents that failed (errors logged separately)
    skipped: int = 0        # documents skipped (already identical hash)


class BaseVectorDB(abc.ABC):
    """
    ALL vector databases implement this interface.

    RULE: If it's not in this file, it doesn't exist in the system.
    No connector may expose chromadb/qdrant/pinecone APIs directly.
    """

    # ── Identity ──────────────────────────────────────────────────────────

    @property
    @abc.abstractmethod
    def kind(self) -> str:
        """
        Short string identifier.
        Examples: 'chroma', 'qdrant', 'pinecone', 'milvus', 'weaviate'
        Used in logs, health checks, and metrics.
        """
        raise NotImplementedError

    # ── Lifecycle ─────────────────────────────────────────────────────────

    @abc.abstractmethod
    def health_check(self) -> bool:
        """
        Ping the backend. Return True if alive.
        Should complete within 2 seconds or raise TimeoutError.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def ensure_collection(
        self,
        collection: str,
        *,
        embedding_dim: int,
        distance_metric: str = "cosine",
    ) -> None:
        """
        Create the collection/index if it does not exist.
        If it already exists, do nothing (no error).

        Args:
            collection:      Collection name (e.g. "ingested_content")
            embedding_dim:   Vector dimension — MUST match your embedder
            distance_metric: "cosine" | "dotproduct" | "euclidean"
        """
        raise NotImplementedError

    @abc.abstractmethod
    def delete_collection(self, collection: str) -> None:
        """
        Permanently delete a collection and ALL its data.
        USE WITH EXTREME CAUTION. Mainly for tests and admin resets.
        """
        raise NotImplementedError

    # ── Write ─────────────────────────────────────────────────────────────

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
        """
        Insert or update a SINGLE document.
        If doc_id already exists, update it.
        For large ingestions, prefer batch_upsert() for performance.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batch_upsert(
        self,
        *,
        collection: str,
        doc_ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> BatchUpsertResult:
        """
        Insert or update MANY documents in one call.
        ALWAYS prefer this over calling upsert() in a loop.

        Performance note:
            Chroma:   1 API call for entire batch
            Qdrant:   Uses PointStruct batch upload
            Pinecone: Uses upsert() with vectors[] array
            Milvus:   Uses collection.insert() batch

        Args:
            doc_ids:    List of unique IDs (same length as embeddings)
            embeddings: List of embedding vectors
            texts:      List of original text strings
            metadatas:  List of metadata dicts

        Returns:
            BatchUpsertResult with counts of inserted/updated/failed
        """
        raise NotImplementedError

    # ── Read ──────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def search(
        self,
        *,
        collection: str,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorHit]:
        """
        Dense vector similarity search.

        Args:
            collection:      Collection to search
            query_embedding: The embedded user query
            top_k:           Max results to return
            filters:         Metadata filters (e.g. {"business_id": "acme"})

        Returns:
            List of VectorHit sorted by score DESC (best first)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def exists(
        self,
        *,
        collection: str,
        ids: List[str],
    ) -> List[str]:
        """
        Return the SUBSET of given IDs that already exist in the collection.

        Used by ingestion to skip re-embedding already-indexed chunks.
        Critical for deduplication — call this BEFORE embedding.

        Example:
            exists(collection="ingested_content", ids=["hash1", "hash2", "hash3"])
            → ["hash1", "hash3"]   (hash2 is new)

        Args:
            ids: List of document IDs to check

        Returns:
            List of IDs from input that already exist (may be empty)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_by_ids(
        self,
        *,
        collection: str,
        ids: List[str],
    ) -> List[VectorHit]:
        """
        Fetch full documents by their IDs (not similarity search).

        Used for:
        - Dedup engine: check if a hash's content matches
        - Admin tooling: inspect stored vectors
        - Backfill: verify what's in the DB

        Returns:
            List of VectorHit for IDs that exist (missing IDs are silently skipped)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def count(self, collection: str) -> int:
        """
        Total number of vectors in a collection.
        Used for: monitoring, health checks, progress bars during ingestion.
        """
        raise NotImplementedError

    # ── Delete ────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def delete(self, *, collection: str, doc_id: str) -> None:
        """Delete a single document by ID."""
        raise NotImplementedError

    @abc.abstractmethod
    def delete_many(self, *, collection: str, doc_ids: List[str]) -> int:
        """
        Delete multiple documents by ID.
        Returns count of documents actually deleted.
        More efficient than calling delete() in a loop.
        """
        raise NotImplementedError
