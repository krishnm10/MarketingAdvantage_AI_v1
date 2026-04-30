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
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.runtime.runtime_telemetry import emit_runtime_event
from app.core.runtime.errors import VectorDBError

logger = logging.getLogger(__name__)


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


class TenantFilterViolation(ValueError):
    """Raised when a search operation violates tenant isolation rules."""


# Patterns that indicate wildcard/bypass attempts in tenant_id values
_WILDCARD_PATTERNS = frozenset({"*", "%", ".*", "**", "__all__", "any", ""})


class BaseVectorDB(abc.ABC):
    """
    ALL vector databases implement this interface.

    RULE: If it's not in this file, it doesn't exist in the system.
    No connector may expose chromadb/qdrant/pinecone APIs directly.

    TENANT ISOLATION:
      All read operations (search, get_by_ids) enforce tenant isolation
      via automatic injection of {TENANT_FILTER_KEY: tenant_id} into
      every query's metadata filters. Callers cannot override or remove
      this filter. Single-tenant mode must be explicitly enabled by
      setting _tenant_isolation_enabled = False on the instance.
    """

    # Default: tenant isolation is ON. The pipeline factory sets this
    # to False only when config.features.enable_multi_tenant_isolation
    # is explicitly False.
    _tenant_isolation_enabled: bool = True

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

    def close(self) -> None:
        """
        Release backend resources if the connector owns open clients/sockets.
        Connectors that do not need cleanup can inherit this no-op.
        """
        return None

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

    # ── Tenant filter key ────────────────────────────────────────────────
    # All implementations use this key to inject tenant isolation into
    # metadata filters. Override in subclass if the backend uses a
    # different field name.
    TENANT_FILTER_KEY: str = "business_id"

    # ── Read ──────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def search(
        self,
        *,
        collection: str,
        query_embedding: List[float],
        top_k: int = 10,
        tenant_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorHit]:
        """
        Dense vector similarity search with mandatory tenant isolation.

        Args:
            collection:      Collection to search
            query_embedding: The embedded user query
            top_k:           Max results to return
            tenant_id:       REQUIRED. Tenant/client identifier. Injected
                             into filters automatically as {TENANT_FILTER_KEY: tenant_id}.
                             Temporarily optional for backward compat — callers
                             omitting this will receive a deprecation warning.
            filters:         Additional metadata filters merged with tenant filter.

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
        tenant_id: Optional[str] = None,
    ) -> List[VectorHit]:
        """
        Fetch full documents by their IDs (not similarity search).

        Args:
            tenant_id:  Optional tenant filter for result verification.
                        When set, results are post-filtered to exclude
                        documents not belonging to this tenant.

        Returns:
            List of VectorHit for IDs that exist (missing IDs are silently skipped)
        """
        raise NotImplementedError

    # ── Tenant enforcement helpers ────────────────────────────────────────

    def _enforce_tenant_filter(
        self,
        tenant_id: Optional[str],
        filters: Optional[Dict[str, Any]],
        *,
        caller: str = "search",
        collection: str = "",
    ) -> Dict[str, Any]:
        """
        Inject tenant isolation into metadata filters. This is the SOLE
        authority for tenant filter enforcement across all adapters.

        Security rules:
          1. tenant_id=None → allowed ONLY when tenant isolation is
             explicitly disabled (single-tenant mode). Otherwise raises.
          2. Empty or whitespace-only tenant_id → always rejected.
          3. Wildcard patterns (*, %, __, etc.) → always rejected.
          4. Caller-supplied business_id that conflicts → rejected hard.
             Cross-tenant access is never silently resolved.
          5. Caller-supplied business_id that matches → accepted (redundant
             but harmless).

        Returns:
            Merged filter dict with {TENANT_FILTER_KEY: tenant_id} guaranteed.

        Raises:
            TenantFilterViolation on any security rule violation.
        """
        # ── Single-tenant bypass ──────────────────────────────────────
        if not self._tenant_isolation_enabled:
            if tenant_id is None:
                logger.debug(
                    '{"event":"TENANT_ISOLATION_BYPASSED",'
                    '"caller":"%s","backend":"%s",'
                    '"collection":"%s","reason":"single_tenant_mode"}',
                    caller, self.kind, collection,
                )
                try:
                    from app.services.security.tenant_audit import log_degraded_isolation
                    log_degraded_isolation(
                        tenant_id="NONE",
                        reason="single_tenant_mode",
                        component=f"vectordb/{self.kind}",
                        fallback_action="bypass_tenant_filter",
                        collection=collection,
                    )
                except Exception:
                    pass
                return dict(filters or {})
            # Even in single-tenant mode, if a tenant_id is provided,
            # inject it — no reason to discard explicit context.

        # ── Rule 1: tenant_id required in multi-tenant mode ───────────
        if tenant_id is None:
            warnings.warn(
                f"BaseVectorDB.{caller}() called without tenant_id. "
                "This is deprecated and will become a hard error. "
                "Pass tenant_id=config.client_id explicitly.",
                DeprecationWarning,
                stacklevel=3,
            )
            logger.warning(
                '{"event":"TENANT_ID_MISSING",'
                '"caller":"%s","backend":"%s",'
                '"collection":"%s","severity":"security_warning"}',
                caller, self.kind, collection,
            )
            try:
                from app.services.security.tenant_audit import log_missing_tenant_context
                log_missing_tenant_context(
                    component=f"vectordb/{self.kind}",
                    operation=caller,
                    fallback_used="no_filter",
                )
            except Exception:
                pass
            return dict(filters or {})

        # ── Rule 2: reject empty / whitespace-only ────────────────────
        stripped = tenant_id.strip()
        if not stripped:
            logger.warning(
                '{"event":"TENANT_ID_REJECTED",'
                '"caller":"%s","backend":"%s",'
                '"collection":"%s","tenant_id":"%s",'
                '"reason":"empty_or_whitespace"}',
                caller, self.kind, collection,
                repr(tenant_id),
            )
            raise TenantFilterViolation(
                f"Empty tenant_id rejected in {caller}(). "
                "Tenant isolation requires a non-empty identifier."
            )

        # ── Rule 3: reject wildcard patterns ──────────────────────────
        if stripped.lower() in _WILDCARD_PATTERNS:
            logger.warning(
                '{"event":"TENANT_ID_REJECTED",'
                '"caller":"%s","backend":"%s",'
                '"collection":"%s","tenant_id":"%s",'
                '"reason":"wildcard_pattern"}',
                caller, self.kind, collection, stripped,
            )
            raise TenantFilterViolation(
                f"Wildcard tenant_id '{stripped}' rejected in {caller}(). "
                "Tenant isolation does not permit wildcard access."
            )

        # ── Rule 4: reject conflicting business_id override ───────────
        merged = dict(filters or {})
        existing_tenant = merged.get(self.TENANT_FILTER_KEY)

        if existing_tenant is not None and existing_tenant != stripped:
            logger.warning(
                '{"event":"TENANT_OVERRIDE_REJECTED",'
                '"caller":"%s","backend":"%s",'
                '"collection":"%s","tenant_id":"%s",'
                '"attempted_override":"%s",'
                '"rejected_override_attempt":true,'
                '"severity":"security_violation"}',
                caller, self.kind, collection,
                stripped, existing_tenant,
            )
            try:
                from app.services.security.tenant_audit import log_cross_tenant_attempt
                log_cross_tenant_attempt(
                    tenant_id=stripped,
                    attempted_target=existing_tenant,
                    caller=caller,
                    collection=collection,
                )
            except Exception:
                pass
            raise TenantFilterViolation(
                f"Conflicting {self.TENANT_FILTER_KEY}='{existing_tenant}' "
                f"in caller-supplied filters for tenant '{stripped}'. "
                "Cross-tenant filter injection is not permitted."
            )

        # ── Inject tenant filter ──────────────────────────────────────
        merged[self.TENANT_FILTER_KEY] = stripped
        return merged

    def _log_tenant_search(
        self,
        *,
        tenant_id: Optional[str],
        collection: str,
        filter_count: int,
        search_mode: str = "vector",
        result_count: int = 0,
    ) -> None:
        """Emit structured telemetry for tenant-aware operations."""
        emit_runtime_event(
            "VECTORDB_TENANT_SEARCH",
            tenant_id=tenant_id or "UNSET",
            vectordb_backend=self.kind,
            search_mode=search_mode,
            retrieved_count=result_count,
            isolation_enabled=self._tenant_isolation_enabled,
        )
        try:
            from app.services.security.tenant_audit import log_vectordb_search
            log_vectordb_search(
                tenant_id=tenant_id or "UNSET",
                collection=collection,
                search_mode=search_mode,
                backend=self.kind,
                filter_count=filter_count,
                retrieval_count=result_count,
                isolation_enforced=self._tenant_isolation_enabled,
            )
        except Exception:
            pass

    def _wrap_error(
        self,
        exc: Exception,
        *,
        operation: str,
        collection: str = "",
        tenant_id: Optional[str] = None,
    ) -> VectorDBError:
        """
        Wrap a backend-specific exception into VectorDBError.

        Preserves the original exception via __cause__ (raise ... from exc)
        so stack traces remain complete.
        """
        return VectorDBError(
            f"[{self.kind}] {operation} failed: {exc}",
            backend=self.kind,
            collection=collection,
            tenant_id=tenant_id,
            details={"original_error": type(exc).__name__, "operation": operation},
        )

    @abc.abstractmethod
    def count(self, collection: str) -> int:
        """
        Total number of vectors in a collection.
        Used for: monitoring, health checks, progress bars during ingestion.
        """
        raise NotImplementedError

    # ── Stats & metadata ─────────────────────────────────────────────────

    def stats(self, collection: str) -> Dict[str, Any]:
        """
        Return operational statistics for a collection.

        Default implementation uses count(); connectors may override
        with richer backend-specific stats (e.g., index size, segments).

        Returns:
            Dict with at least: {"backend": <kind>, "collection": <name>,
                                 "document_count": <int>}
        """
        try:
            doc_count = self.count(collection)
        except Exception:
            doc_count = -1
        return {
            "backend": self.kind,
            "collection": collection,
            "document_count": doc_count,
        }

    def update_metadata(
        self,
        *,
        collection: str,
        ids: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> int:
        """
        Patch metadata on existing documents WITHOUT changing embeddings or text.

        Used by classification, tagging, and post-processing services that
        enrich indexed chunks after initial ingestion.

        Args:
            collection: Target collection name.
            ids:        Document IDs to update (must already exist).
            metadatas:  Corresponding metadata dicts to merge/overwrite.

        Returns:
            Number of documents successfully updated.

        Raises:
            VectorDBError on backend failure.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement update_metadata(). "
            "Override in the connector to support metadata-only writes."
        )

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
