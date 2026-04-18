# ============================================================
# app/core/vectordb/chroma_v1.py
#
# ChromaDB implementation of BaseVectorDB.
# This is the ONLY file that imports chromadb.
# All other files use BaseVectorDB only.
#
# Supports two connection modes:
#   LOCAL  — PersistentClient (persist_directory)
#   REMOTE — HttpClient       (host + port)
#
# Config comes via ClientConfig.vectordb.chroma — ZERO hardcoding.
# ============================================================

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from app.core.vectordb.base import BaseVectorDB, BatchUpsertResult, VectorHit

logger = logging.getLogger(__name__)


class ChromaVectorDB(BaseVectorDB):
    """
    ChromaDB connector implementing BaseVectorDB.

    Connection modes:
    - HttpClient:       connect to a remote ChromaDB server (host + port)
    - PersistentClient: data saved to local disk (persist_directory)
    - EphemeralClient:  in-memory only (testing — neither host nor path)

    Constructor args:
        host:                 Remote ChromaDB server host. If set, uses HttpClient.
        port:                 Remote server port (default 8000).
        ssl:                  Use HTTPS for remote connection (default False).
        api_key:              API key for auth-enabled ChromaDB servers.
        tenant:               ChromaDB tenant (default 'default_tenant').
        database:             ChromaDB database (default 'default_database').
        persist_directory:    Local disk path. Used when host is not set.
        anonymized_telemetry: Set False to disable ChromaDB telemetry (default False).
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        anonymized_telemetry: bool = False,
        *,
        host: Optional[str] = None,
        port: int = 8000,
        ssl: bool = False,
        api_key: Optional[str] = None,
        tenant: str = "default_tenant",
        database: str = "default_database",
    ):
        # ── Determine connection mode ──────────────────────────────
        if host:
            self._mode = "remote"
        elif persist_directory:
            self._mode = "local"
        else:
            self._mode = "ephemeral"

        self._path = persist_directory
        self._host = host
        self._port = port
        self._anonymized_telemetry = anonymized_telemetry

        self._client = self._make_client(
            host=host,
            port=port,
            ssl=ssl,
            api_key=api_key,
            tenant=tenant,
            database=database,
            persist_directory=persist_directory,
            anonymized_telemetry=anonymized_telemetry,
        )

        # Cache open collections: {collection_name: chromadb.Collection}
        self._collections: Dict[str, chromadb.Collection] = {}

        if self._mode == "remote":
            logger.info(
                "[ChromaVectorDB] Initialized REMOTE | host=%s | port=%s | ssl=%s",
                host, port, ssl,
            )
        else:
            logger.info(
                "[ChromaVectorDB] Initialized %s | path=%s | telemetry=%s",
                self._mode.upper(), persist_directory, anonymized_telemetry,
            )

    # ── Identity ──────────────────────────────────────────────────────

    @property
    def kind(self) -> str:
        return "chroma"

    # ── Lifecycle ─────────────────────────────────────────────────────

    def health_check(self) -> bool:
        try:
            self._client.heartbeat()
            return True
        except Exception as exc:
            logger.warning("[ChromaVectorDB] health_check failed: %s", exc)
            return False

    def ensure_collection(
        self,
        collection: str,
        *,
        embedding_dim: int,
        distance_metric: str = "cosine",
    ) -> None:
        """
        Create the ChromaDB collection if it does not exist.
        If it exists, validate that the stored dimension matches embedding_dim.

        DIMENSION GUARD:
          ChromaDB stores embedding_dim in collection metadata on first creation.
          On subsequent calls, we compare stored dim vs requested dim.
          If they differ → raise immediately with a clear action message.
          This prevents silent corruption (wrong vectors stored in collection).

        Args:
            collection:      Collection name (unique per client).
            embedding_dim:   Dimension of vectors this embedder produces.
            distance_metric: cosine | dotproduct | euclidean
        """
        _metric_map = {
            "cosine":      "cosine",
            "dotproduct":  "ip",   # inner product
            "euclidean":   "l2",
        }
        hnsw_space = _metric_map.get(distance_metric, "cosine")

        # ── Check if collection already exists ────────────────────────
        existing_names = [c.name for c in self._client.list_collections()]

        if collection in existing_names:
            existing_col = self._client.get_collection(name=collection)
            stored_dim = (existing_col.metadata or {}).get("embedding_dim")

            # ── DIMENSION GUARD ───────────────────────────────────────
            if stored_dim is not None and stored_dim != embedding_dim:
                raise ValueError(
                    f"[ChromaVectorDB] DIMENSION MISMATCH on collection '{collection}':\n"
                    f"  Stored dimension : {stored_dim}\n"
                    f"  Current embedder : {embedding_dim}\n"
                    f"ACTION REQUIRED:\n"
                    f"  1. Stop the server.\n"
                    f"  2. Delete the collection: client.delete_collection('{collection}')\n"
                    f"     OR wipe the folder: rmdir /s /q {self._path}\n"
                    f"  3. Change embedder model back to dim={stored_dim},\n"
                    f"     OR re-ingest all documents with the new embedder.\n"
                    f"  NEVER mix two different embedding dimensions in one collection."
                )

            # Dimension matches (or metadata missing on old collection) — cache and return
            self._collections[collection] = existing_col
            logger.info(
                "[ChromaVectorDB] Collection '%s' found | dim=%s | metric=%s",
                collection,
                stored_dim or "unknown",
                hnsw_space,
            )

        else:
            # ── Create new collection with dimension stored in metadata ─
            new_col = self._client.create_collection(
                name=collection,
                metadata={
                    "hnsw:space":    hnsw_space,
                    "embedding_dim": embedding_dim,   # ← locked on creation
                },
            )
            self._collections[collection] = new_col
            logger.info(
                "[ChromaVectorDB] Collection '%s' created | dim=%d | metric=%s",
                collection, embedding_dim, hnsw_space,
            )

    def delete_collection(self, collection: str) -> None:
        try:
            self._client.delete_collection(name=collection)
            self._collections.pop(collection, None)
            logger.info("[ChromaVectorDB] Collection '%s' deleted.", collection)
        except Exception as exc:
            logger.warning(
                "[ChromaVectorDB] delete_collection '%s' failed: %s", collection, exc
            )

    def get_raw_collection(self, collection: str) -> chromadb.Collection:
        """
        Return the raw chromadb.Collection object.

        USE CASE:
          Background engines (conflict_detection, etc.) that need the
          shared collection must call this instead of creating their own
          chromadb.PersistentClient — which causes the
          'instance already exists with different settings' error.
        """
        return self._get_collection(collection)

    # ── Write ─────────────────────────────────────────────────────────

    def upsert(
        self,
        *,
        collection: str,
        doc_id: str,
        embedding: List[float],
        text: str,
        metadata: Dict[str, Any],
    ) -> None:
        col = self._get_collection(collection)
        safe_meta = _sanitize_metadata(metadata)
        col.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[safe_meta],
        )

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
        Single Chroma API call for the entire batch.

        FAILURE POLICY:
          Exceptions are RE-RAISED after logging.
          The caller (ingestion_service_v2) must catch and mark
          the file as failed — never silently log success.
        """
        if not doc_ids:
            return BatchUpsertResult()

        col = self._get_collection(collection)
        safe_metas = [_sanitize_metadata(m) for m in metadatas]

        # Check which IDs already exist → inserted vs updated count
        existing  = set(self.exists(collection=collection, ids=doc_ids))
        inserted  = sum(1 for i in doc_ids if i not in existing)
        updated   = len(doc_ids) - inserted

        try:
            col.upsert(
                ids=doc_ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=safe_metas,
            )

            # ── POST-UPSERT VERIFICATION ────────────────────────────
            # Spot-check a random sample of IDs to confirm the write
            # actually persisted.  Previous approach compared col.count()
            # before/after, but ChromaDB's SQLite-WAL count is NOT
            # guaranteed to be immediately consistent — producing
            # false-positive warnings.  A targeted ID lookup is reliable.
            import random
            sample_size = min(5, len(doc_ids))
            sample_ids  = random.sample(doc_ids, sample_size)
            try:
                check = col.get(ids=sample_ids, include=[])
                found = set(check["ids"]) if check and "ids" in check else set()
                missing = [sid for sid in sample_ids if sid not in found]
                if missing:
                    logger.warning(
                        "[ChromaVectorDB] batch_upsert VERIFICATION WARNING on '%s': "
                        "%d/%d sampled IDs not found after upsert: %s. "
                        "Possible silent write failure.",
                        collection, len(missing), sample_size, missing,
                    )
            except Exception as verify_exc:
                logger.warning(
                    "[ChromaVectorDB] Post-upsert verification query failed on '%s': %s",
                    collection, verify_exc,
                )

            logger.info(
                "[ChromaVectorDB] batch_upsert '%s': +%d new, ~%d updated, "
                "%d total vectors submitted",
                collection, inserted, updated, len(doc_ids),
            )
            return BatchUpsertResult(inserted=inserted, updated=updated)

        except Exception as exc:
            logger.error(
                "[ChromaVectorDB] batch_upsert FAILED on collection '%s': %s\n"
                "  Attempted: %d vectors | First ID: %s",
                collection, exc, len(doc_ids), doc_ids[0] if doc_ids else "n/a",
            )
            raise   # ← CRITICAL: re-raise so ingestion marks file as FAILED

    # ── Read ──────────────────────────────────────────────────────────

    def search(
        self,
        *,
        collection: str,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorHit]:
        col = self._get_collection(collection)
        n   = min(top_k, col.count() or 1)

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results":        n,
            "include":          ["documents", "metadatas", "distances"],
        }
        if filters:
            kwargs["where"] = filters

        try:
            res = col.query(**kwargs)
        except Exception as exc:
            logger.error("[ChromaVectorDB] search failed: %s", exc)
            return []

        ids       = res.get("ids",       [[]])[0]
        docs      = res.get("documents", [[]])[0]
        metas     = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]

        hits: List[VectorHit] = []
        for i, doc_id in enumerate(ids):
            # Chroma cosine distance: 0=identical, 2=opposite
            # Convert → similarity: 1 - (distance / 2)
            distance = distances[i] if i < len(distances) else 1.0
            score    = max(0.0, 1.0 - (distance / 2.0))
            hits.append(VectorHit(
                id       = doc_id,
                text     = docs[i]  if i < len(docs)  else "",
                score    = round(score, 6),
                metadata = metas[i] if i < len(metas) else {},
            ))
        return hits

    def exists(
        self,
        *,
        collection: str,
        ids: List[str],
    ) -> List[str]:
        """
        Returns subset of IDs that already exist in ChromaDB.
        Uses include=[] for fastest possible check (IDs only, no payload).
        """
        if not ids:
            return []
        col = self._get_collection(collection)
        try:
            result = col.get(ids=ids, include=[])
            return result.get("ids", [])
        except Exception as exc:
            logger.warning("[ChromaVectorDB] exists() check failed: %s", exc)
            return []

    def get_by_ids(
        self,
        *,
        collection: str,
        ids: List[str],
    ) -> List[VectorHit]:
        if not ids:
            return []
        col = self._get_collection(collection)
        try:
            result = col.get(ids=ids, include=["documents", "metadatas"])
        except Exception as exc:
            logger.warning("[ChromaVectorDB] get_by_ids() failed: %s", exc)
            return []

        hits  = []
        r_ids = result.get("ids", [])
        docs  = result.get("documents") or []
        metas = result.get("metadatas") or []

        for i, doc_id in enumerate(r_ids):
            hits.append(VectorHit(
                id       = doc_id,
                text     = docs[i]  if i < len(docs)  else "",
                score    = 1.0,     # No score for direct lookup
                metadata = metas[i] if i < len(metas) else {},
            ))
        return hits

    def count(self, collection: str) -> int:
        try:
            return self._get_collection(collection).count()
        except Exception as exc:
            logger.warning("[ChromaVectorDB] count() failed: %s", exc)
            return 0

    def get_all(
        self,
        *,
        collection: str,
        include: Optional[List[str]] = None,
    ) -> dict:
        """
        Return ALL documents in a collection (used by admin integrity ops).

        Delegates to chromadb's native Collection.get() which, when called
        without an ids filter, returns every stored record.

        Args:
            collection: Collection name.
            include:    Chroma include list, e.g. ["documents", "metadatas"].
                        Empty list → IDs only (fastest).

        Returns:
            Dict with keys: ids, documents, metadatas, embeddings (per include).
        """
        col = self._get_collection(collection)
        return col.get(include=include or [])

    # ── Delete ────────────────────────────────────────────────────────

    def delete(self, *, collection: str, doc_id: str) -> None:
        col = self._get_collection(collection)
        try:
            col.delete(ids=[doc_id])
        except Exception as exc:
            logger.warning("[ChromaVectorDB] delete(%s) failed: %s", doc_id, exc)

    def delete_many(self, *, collection: str, doc_ids: List[str]) -> int:
        if not doc_ids:
            return 0
        col = self._get_collection(collection)
        try:
            existing = self.exists(collection=collection, ids=doc_ids)
            if existing:
                col.delete(ids=existing)
            return len(existing)
        except Exception as exc:
            logger.warning("[ChromaVectorDB] delete_many failed: %s", exc)
            return 0

    # ── Internal helpers ──────────────────────────────────────────────

    def _get_collection(self, name: str) -> chromadb.Collection:
        """
        Return cached collection object.
        If not cached (e.g. after app restart), fetch from Chroma.
        If it doesn't exist at all, raise clearly — caller must call
        ensure_collection() first.
        """
        if name not in self._collections:
            try:
                self._collections[name] = self._client.get_collection(name=name)
            except Exception:
                raise RuntimeError(
                    f"[ChromaVectorDB] Collection '{name}' does not exist. "
                    f"Call ensure_collection('{name}', embedding_dim=N) first."
                )
        return self._collections[name]

    @staticmethod
    def _make_client(
        *,
        host: Optional[str],
        port: int,
        ssl: bool,
        api_key: Optional[str],
        tenant: str,
        database: str,
        persist_directory: Optional[str],
        anonymized_telemetry: bool,
    ) -> chromadb.ClientAPI:
        settings = Settings(
            anonymized_telemetry=anonymized_telemetry,
            chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider" if api_key else None,
            chroma_client_auth_credentials=api_key or None,
        ) if api_key else Settings(anonymized_telemetry=anonymized_telemetry)

        # ── Remote mode: HttpClient ────────────────────────────────
        if host:
            return chromadb.HttpClient(
                host=host,
                port=port,
                ssl=ssl,
                tenant=tenant,
                database=database,
                settings=settings,
            )

        # ── Local mode: PersistentClient ───────────────────────────
        if persist_directory:
            return chromadb.PersistentClient(
                path=persist_directory,
                settings=settings,
            )

        # ── Ephemeral mode: in-memory (testing) ───────────────────
        return chromadb.EphemeralClient(settings=settings)


# ── Module-level helpers ───────────────────────────────────────────────────────

def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    ChromaDB only accepts: str | int | float | bool.
    Converts None → "", list/dict → str(v).
    Prevents silent data loss from unsupported types.
    """
    safe: Dict[str, Any] = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            safe[k] = v
        elif v is None:
            safe[k] = ""
        else:
            safe[k] = str(v)
    return safe
