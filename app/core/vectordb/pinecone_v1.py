# ============================================================
# app/core/vectordb/pinecone_v1.py
# ============================================================
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from app.core.vectordb.base import BaseVectorDB, BatchUpsertResult, VectorHit
from app.core.runtime.errors import VectorDBError

logger = logging.getLogger(__name__)


class PineconeVectorDB(BaseVectorDB):
    """
    Pinecone Serverless / Pod connector.
    Install: pip install pinecone-client>=3.0.0
    """

    def __init__(
        self,
        *,
        mode: str = "cloud",
        api_key: Optional[str] = None,
        index_name: str,
        namespace: str = "default",
        embedding_dim: int,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        pod_type: Optional[str] = None,
        local_path: Optional[str] = None,
    ):
        self._mode = (mode or "cloud").strip().lower()
        try:
            from pinecone import Pinecone
        except ImportError:
            if self._mode != "local":
                raise ImportError(
                    "Pinecone client not installed. Run: pip install pinecone-client>=3.0.0"
                )

        self._index_name   = index_name
        self._namespace    = namespace
        self._embedding_dim = embedding_dim
        self._metric       = metric
        self._cloud        = cloud
        self._region       = region
        self._pod_type     = pod_type
        self._local_path   = local_path or "./pinecone_local_db"
        self._local_vdb    = None
        self._pc           = None
        self._index        = None

        if self._mode == "local":
            from app.core.vectordb.chroma_v1 import ChromaVectorDB
            self._local_vdb = ChromaVectorDB(
                persist_directory=self._local_path,
                anonymized_telemetry=False,
            )
            logger.info(
                "[PineconeVectorDB] Initialized LOCAL-EMULATION | path=%s | namespace=%s | dim=%d",
                self._local_path, namespace, embedding_dim,
            )
        else:
            if not api_key:
                raise ValueError(
                    "Pinecone cloud mode requires PINECONE_API_KEY."
                )
            self._pc = Pinecone(api_key=api_key)
            logger.info(
                "[PineconeVectorDB] Initialized CLOUD | index=%s | namespace=%s | dim=%d",
                index_name, namespace, embedding_dim,
            )

    # ── Identity ──────────────────────────────────────────────────────

    @property
    def kind(self) -> str:
        return "pinecone"

    # ── Lifecycle ─────────────────────────────────────────────────────

    def health_check(self) -> bool:
        if self._local_vdb is not None:
            return self._local_vdb.health_check()
        try:
            self._get_index().describe_index_stats()
            return True
        except Exception as exc:
            logger.warning("[PineconeVectorDB] health_check failed: %s", exc)
            return False

    def ensure_collection(
        self,
        collection: str,
        *,
        embedding_dim: int,
        distance_metric: str = "cosine",
    ) -> None:
        """
        In Pinecone, each 'collection' maps to an index.
        Creates the index if it doesn't exist.
        Serverless creation takes ~30-60s — we poll until ready.
        """
        if self._local_vdb is not None:
            self._local_vdb.ensure_collection(
                collection=collection,
                embedding_dim=embedding_dim,
                distance_metric=distance_metric,
            )
            return

        existing = {idx.name for idx in self._pc.list_indexes()}
        if self._index_name in existing:
            logger.debug(
                "[PineconeVectorDB] Index '%s' already exists — skipping.",
                self._index_name,
            )
            return

        # Map our standard metric names to Pinecone names
        _metric_map = {
            "cosine":     "cosine",
            "dotproduct": "dotproduct",
            "euclidean":  "euclidean",
        }
        metric = _metric_map.get(distance_metric.lower(), "cosine")

        spec = (
            self._build_pod_spec()
            if self._pod_type
            else self._build_serverless_spec()
        )
        self._pc.create_index(
            name=self._index_name,
            dimension=int(embedding_dim),
            metric=metric,
            spec=spec,
        )
        logger.info(
            "[PineconeVectorDB] Creating index '%s' | dim=%d | metric=%s...",
            self._index_name, embedding_dim, metric,
        )

        # Poll until ready (Serverless: ~30-60s on first create)
        deadline = time.time() + 120
        while time.time() < deadline:
            status = self._pc.describe_index(self._index_name).status
            if status.get("ready"):
                logger.info("[PineconeVectorDB] Index '%s' is ready.", self._index_name)
                return
            time.sleep(5)
        raise TimeoutError(
            f"[PineconeVectorDB] Index '{self._index_name}' not ready after 120s."
        )

    def delete_collection(self, collection: str) -> None:
        if self._local_vdb is not None:
            self._local_vdb.delete_collection(collection)
            return
        try:
            self._pc.delete_index(self._index_name)
            self._index = None
            logger.info("[PineconeVectorDB] Index '%s' deleted.", self._index_name)
        except Exception as exc:
            raise self._wrap_error(
                exc, operation="delete_collection", collection=collection,
            ) from exc

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
        if self._local_vdb is not None:
            self._local_vdb.upsert(
                collection=collection,
                doc_id=doc_id,
                embedding=embedding,
                text=text,
                metadata=metadata,
            )
            return
        payload = dict(metadata or {})
        payload["_text"] = text
        self._get_index().upsert(
            vectors=[(doc_id, embedding, payload)],
            namespace=self._namespace,
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
        Pinecone native upsert is idempotent — same ID = overwrite.
        We chunk into batches of 100 (Pinecone's recommended max per call).
        """
        if not doc_ids:
            return BatchUpsertResult()
        if self._local_vdb is not None:
            return self._local_vdb.batch_upsert(
                collection=collection,
                doc_ids=doc_ids,
                embeddings=embeddings,
                texts=texts,
                metadatas=metadatas,
            )

        PINECONE_BATCH = 100
        total_upserted = 0
        try:
            for i in range(0, len(doc_ids), PINECONE_BATCH):
                batch_ids   = doc_ids[i:i + PINECONE_BATCH]
                batch_emb   = embeddings[i:i + PINECONE_BATCH]
                batch_texts = texts[i:i + PINECONE_BATCH]
                batch_metas = metadatas[i:i + PINECONE_BATCH]

                vectors = []
                for j in range(len(batch_ids)):
                    payload = dict(batch_metas[j] or {})
                    payload["_text"] = batch_texts[j]
                    vectors.append((batch_ids[j], batch_emb[j], payload))

                self._get_index().upsert(
                    vectors=vectors,
                    namespace=self._namespace,
                )
                total_upserted += len(batch_ids)

            logger.info(
                "[PineconeVectorDB] batch_upsert: %d vectors upserted to '%s'",
                total_upserted, self._index_name,
            )
            # Pinecone upsert is always "upsert" — can't distinguish insert vs update
            return BatchUpsertResult(inserted=total_upserted)
        except Exception as exc:
            raise self._wrap_error(
                exc, operation="batch_upsert", collection=collection,
            ) from exc

    # ── Read ──────────────────────────────────────────────────────────

    def search(
        self,
        *,
        collection: str,
        query_embedding: List[float],
        top_k: int = 10,
        tenant_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorHit]:
        effective_filters = self._enforce_tenant_filter(
            tenant_id, filters, caller="search", collection=collection,
        )
        if self._local_vdb is not None:
            return self._local_vdb.search(
                collection=collection,
                query_embedding=query_embedding,
                top_k=top_k,
                tenant_id=tenant_id,
                filters=effective_filters,
            )
        pinecone_filter = None
        if effective_filters:
            pinecone_filter = {
                k: (v if isinstance(v, dict) else {"$eq": v})
                for k, v in effective_filters.items()
            }
        res = self._get_index().query(
            vector=query_embedding,
            top_k=int(top_k),
            namespace=self._namespace,
            filter=pinecone_filter,
            include_metadata=True,
        )
        hits: List[VectorHit] = []
        for match in res.get("matches", []):
            meta = dict(match.get("metadata") or {})
            text = str(meta.pop("_text", ""))
            hits.append(VectorHit(
                id=str(match["id"]),
                text=text,
                score=round(float(match.get("score", 0.0)), 6),
                metadata=meta,
            ))
        self._log_tenant_search(
            tenant_id=tenant_id,
            collection=collection,
            filter_count=len(effective_filters),
            search_mode="vector",
            result_count=len(hits),
        )
        return hits

    def exists(self, *, collection: str, ids: List[str]) -> List[str]:
        """
        Pinecone fetch() returns only the IDs that exist.
        We fetch with include_values=False for maximum speed.
        """
        if not ids:
            return []
        if self._local_vdb is not None:
            return self._local_vdb.exists(collection=collection, ids=ids)
        try:
            # Pinecone fetch is limited to 1000 IDs per call
            FETCH_BATCH = 1000
            found = []
            for i in range(0, len(ids), FETCH_BATCH):
                batch = ids[i:i + FETCH_BATCH]
                res = self._get_index().fetch(
                    ids=batch,
                    namespace=self._namespace,
                )
                found.extend(res.get("vectors", {}).keys())
            return found
        except Exception as exc:
            raise self._wrap_error(
                exc, operation="exists", collection=collection,
            ) from exc

    def get_by_ids(self, *, collection: str, ids: List[str], tenant_id: Optional[str] = None) -> List[VectorHit]:
        if not ids:
            return []
        if self._local_vdb is not None:
            return self._local_vdb.get_by_ids(collection=collection, ids=ids, tenant_id=tenant_id)
        try:
            res = self._get_index().fetch(
                ids=ids,
                namespace=self._namespace,
            )
            hits = []
            for doc_id, vec in res.get("vectors", {}).items():
                meta = dict(vec.get("metadata") or {})
                text = str(meta.pop("_text", ""))
                hits.append(VectorHit(
                    id=str(doc_id),
                    text=text,
                    score=1.0,
                    metadata=meta,
                ))
            return hits
        except Exception as exc:
            raise self._wrap_error(
                exc, operation="get_by_ids", collection=collection,
            ) from exc

    def count(self, collection: str) -> int:
        if self._local_vdb is not None:
            return self._local_vdb.count(collection)
        try:
            stats = self._get_index().describe_index_stats()
            ns_stats = stats.get("namespaces", {}).get(self._namespace, {})
            return int(ns_stats.get("vector_count", 0))
        except Exception as exc:
            raise self._wrap_error(
                exc, operation="count", collection=collection,
            ) from exc

    def stats(self, collection: str) -> Dict[str, Any]:
        if self._local_vdb is not None:
            return self._local_vdb.stats(collection)
        try:
            raw = self._get_index().describe_index_stats()
            ns_stats = raw.get("namespaces", {}).get(self._namespace, {})
            return {
                "backend": self.kind,
                "collection": collection,
                "document_count": int(ns_stats.get("vector_count", 0)),
                "namespace": self._namespace,
                "index_name": self._index_name,
                "dimension": raw.get("dimension"),
            }
        except Exception:
            return {
                "backend": self.kind,
                "collection": collection,
                "document_count": -1,
            }

    def update_metadata(
        self,
        *,
        collection: str,
        ids: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> int:
        if not ids:
            return 0
        if self._local_vdb is not None:
            return self._local_vdb.update_metadata(
                collection=collection, ids=ids, metadatas=metadatas,
            )
        try:
            index = self._get_index()
            for doc_id, meta in zip(ids, metadatas):
                index.update(id=doc_id, set_metadata=meta, namespace=self._namespace)
            logger.debug(
                "[PineconeVectorDB] update_metadata '%s': %d doc(s)",
                collection, len(ids),
            )
            return len(ids)
        except Exception as exc:
            raise self._wrap_error(
                exc, operation="update_metadata", collection=collection,
            ) from exc

    # ── Delete ────────────────────────────────────────────────────────

    def delete(self, *, collection: str, doc_id: str) -> None:
        if self._local_vdb is not None:
            self._local_vdb.delete(collection=collection, doc_id=doc_id)
            return
        self._get_index().delete(ids=[doc_id], namespace=self._namespace)

    def delete_many(self, *, collection: str, doc_ids: List[str]) -> int:
        if not doc_ids:
            return 0
        if self._local_vdb is not None:
            return self._local_vdb.delete_many(collection=collection, doc_ids=doc_ids)
        try:
            self._get_index().delete(ids=doc_ids, namespace=self._namespace)
            return len(doc_ids)
        except Exception as exc:
            raise self._wrap_error(
                exc, operation="delete_many", collection=collection,
            ) from exc

    # ── Internal helpers ──────────────────────────────────────────────

    def _get_index(self):
        if self._pc is None:
            raise RuntimeError("[PineconeVectorDB] Pinecone client is not initialized.")
        if self._index is None:
            self._index = self._pc.Index(self._index_name)
        return self._index

    def _build_serverless_spec(self):
        from pinecone import ServerlessSpec
        return ServerlessSpec(cloud=self._cloud, region=self._region)

    def _build_pod_spec(self):
        from pinecone import PodSpec
        return PodSpec(environment=self._region, pod_type=self._pod_type)
