# ============================================================
# app/core/vectordb/weaviate_v1.py
# ============================================================
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.core.vectordb.base import BaseVectorDB, BatchUpsertResult, VectorHit

logger = logging.getLogger(__name__)


class WeaviateVectorDB(BaseVectorDB):
    """
    Weaviate v4 connector (BYO embeddings — no built-in vectoriser).
    Supports: WCS Cloud, local Docker, embedded (zero-infra dev mode).
    Install: pip install weaviate-client>=4.0.0
    """

    def __init__(
        self,
        *,
        url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        additional_headers: Optional[Dict[str, str]] = None,
        embedded: bool = False,
    ):
        try:
            import weaviate
        except ImportError:
            raise ImportError(
                "Weaviate client not installed. Run: pip install weaviate-client>=4.0.0"
            )

        if embedded:
            import weaviate.embedded as _emb
            self._client = weaviate.WeaviateClient(
                embedded_options=_emb.EmbeddedOptions()
            )
            self._client.connect()
        else:
            from weaviate.auth import AuthApiKey
            auth = AuthApiKey(api_key) if api_key else None
            self._client = weaviate.connect_to_custom(
                http_host=url,
                http_port=80 if url.startswith("http://") else 443,
                http_secure=url.startswith("https://"),
                grpc_host=url,
                grpc_port=50051,
                grpc_secure=url.startswith("https://"),
                auth_credentials=auth,
                headers=additional_headers or {},
            )

        logger.info(
            "[WeaviateVectorDB] Connected | url=%s | embedded=%s", url, embedded
        )

    # ── Identity ──────────────────────────────────────────────────────

    @property
    def kind(self) -> str:
        return "weaviate"

    # ── Lifecycle ─────────────────────────────────────────────────────

    def health_check(self) -> bool:
        try:
            return self._client.is_ready()
        except Exception as exc:
            logger.warning("[WeaviateVectorDB] health_check failed: %s", exc)
            return False

    def ensure_collection(
        self,
        collection: str,
        *,
        embedding_dim: int,
        distance_metric: str = "cosine",
    ) -> None:
        """
        Create a Weaviate class if it doesn't exist.
        We always use BYO embeddings (vectorizer=none).
        embedding_dim is stored in metadata — Weaviate infers dim from first insert.
        """
        import weaviate.classes.config as wc

        _metric_map = {
            "cosine":     wc.VectorDistances.COSINE,
            "dotproduct": wc.VectorDistances.DOT,
            "euclidean":  wc.VectorDistances.L2_SQUARED,
        }
        distance = _metric_map.get(distance_metric.lower(), wc.VectorDistances.COSINE)
        cls_name = self._class_name(collection)

        existing = {cls.name for cls in self._client.collections.list_all()}
        if cls_name in existing:
            logger.debug(
                "[WeaviateVectorDB] Collection '%s' exists — skipping.", cls_name
            )
            return

        self._client.collections.create(
            name=cls_name,
            vectorizer_config=wc.Configure.Vectorizer.none(),
            vector_index_config=wc.Configure.VectorIndex.hnsw(
                distance_metric=distance,
            ),
            properties=[
                wc.Property(name="_text", data_type=wc.DataType.TEXT),
            ],
        )
        logger.info(
            "[WeaviateVectorDB] Created collection '%s' | metric=%s",
            cls_name, distance_metric,
        )

    def delete_collection(self, collection: str) -> None:
        cls_name = self._class_name(collection)
        try:
            self._client.collections.delete(cls_name)
            logger.info("[WeaviateVectorDB] Collection '%s' deleted.", cls_name)
        except Exception as exc:
            logger.warning("[WeaviateVectorDB] delete_collection failed: %s", exc)

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
        cls_name = self._class_name(collection)
        col = self._client.collections.get(cls_name)
        properties = dict(metadata or {})
        properties["_text"] = text
        try:
            col.data.delete_by_id(doc_id)
        except Exception:
            pass
        col.data.insert(properties=properties, vector=embedding, uuid=doc_id)

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
        Weaviate v4 batch insert using insert_many().
        For upsert semantics: check exists → delete existing → batch insert.
        insert_many() is ~10x faster than individual inserts.
        """
        if not doc_ids:
            return BatchUpsertResult()

        cls_name = self._class_name(collection)
        col = self._client.collections.get(cls_name)

        try:
            # Check which already exist
            existing = set(self.exists(collection=collection, ids=doc_ids))
            updated  = len(existing)
            inserted = len(doc_ids) - updated

            # Delete existing ones first
            for eid in existing:
                try:
                    col.data.delete_by_id(eid)
                except Exception:
                    pass

            # Build objects list for insert_many
            from weaviate.classes.data import DataObject
            objects = [
                DataObject(
                    properties={**(metadatas[i] or {}), "_text": texts[i]},
                    vector=embeddings[i],
                    uuid=doc_ids[i],
                )
                for i in range(len(doc_ids))
            ]

            result = col.data.insert_many(objects)

            if result.has_errors:
                failed = len(result.errors)
                logger.warning(
                    "[WeaviateVectorDB] batch_upsert: %d errors in batch", failed
                )
                return BatchUpsertResult(
                    inserted=inserted - failed,
                    updated=updated,
                    failed=failed,
                )

            logger.info(
                "[WeaviateVectorDB] batch_upsert '%s': +%d new, ~%d updated",
                cls_name, inserted, updated,
            )
            return BatchUpsertResult(inserted=inserted, updated=updated)

        except Exception as exc:
            logger.error("[WeaviateVectorDB] batch_upsert failed: %s", exc)
            return BatchUpsertResult(failed=len(doc_ids))

    # ── Read ──────────────────────────────────────────────────────────

    def search(
        self,
        *,
        collection: str,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorHit]:
        import weaviate.classes.query as wq
        import weaviate.classes.filters as wf

        cls_name = self._class_name(collection)
        col = self._client.collections.get(cls_name)

        where_filter = None
        if filters:
            conditions = [wf.Filter.by_property(k).equal(v) for k, v in filters.items()]
            where_filter = (
                conditions[0] if len(conditions) == 1
                else wf.Filter.all_of(conditions)
            )

        res = col.query.near_vector(
            near_vector=query_embedding,
            limit=int(top_k),
            filters=where_filter,
            return_metadata=wq.MetadataQuery(certainty=True),
        )

        hits: List[VectorHit] = []
        for obj in res.objects:
            props = dict(obj.properties or {})
            text  = str(props.pop("_text", ""))
            score = float(obj.metadata.certainty or 0.0) if obj.metadata else 0.0
            hits.append(VectorHit(
                id=str(obj.uuid),
                text=text,
                score=round(score, 6),
                metadata=props,
            ))
        return hits

    def exists(self, *, collection: str, ids: List[str]) -> List[str]:
        """
        Check existence using Weaviate's fetch_object_by_id.
        Returns only the IDs that actually exist.
        """
        if not ids:
            return []
        cls_name = self._class_name(collection)
        col = self._client.collections.get(cls_name)
        found = []
        for doc_id in ids:
            try:
                obj = col.query.fetch_object_by_id(doc_id, include_vector=False)
                if obj is not None:
                    found.append(doc_id)
            except Exception:
                pass
        return found

    def get_by_ids(self, *, collection: str, ids: List[str]) -> List[VectorHit]:
        if not ids:
            return []
        cls_name = self._class_name(collection)
        col = self._client.collections.get(cls_name)
        hits = []
        for doc_id in ids:
            try:
                obj = col.query.fetch_object_by_id(doc_id, include_vector=False)
                if obj:
                    props = dict(obj.properties or {})
                    text  = str(props.pop("_text", ""))
                    hits.append(VectorHit(
                        id=str(obj.uuid),
                        text=text,
                        score=1.0,
                        metadata=props,
                    ))
            except Exception:
                pass
        return hits

    def count(self, collection: str) -> int:
        try:
            cls_name = self._class_name(collection)
            col = self._client.collections.get(cls_name)
            result = col.aggregate.over_all(total_count=True)
            return int(result.total_count or 0)
        except Exception as exc:
            logger.warning("[WeaviateVectorDB] count() failed: %s", exc)
            return 0

    # ── Delete ────────────────────────────────────────────────────────

    def delete(self, *, collection: str, doc_id: str) -> None:
        cls_name = self._class_name(collection)
        col = self._client.collections.get(cls_name)
        col.data.delete_by_id(doc_id)

    def delete_many(self, *, collection: str, doc_ids: List[str]) -> int:
        if not doc_ids:
            return 0
        cls_name = self._class_name(collection)
        col = self._client.collections.get(cls_name)
        deleted = 0
        for doc_id in doc_ids:
            try:
                col.data.delete_by_id(doc_id)
                deleted += 1
            except Exception:
                pass
        return deleted

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _class_name(collection: str) -> str:
        """
        Weaviate requires class names starting with uppercase.
        'ingested_content' → 'IngestedContent'
        """
        return "".join(
            word.capitalize()
            for word in collection.replace("-", "_").split("_")
        )

    def __del__(self):
        try:
            self._client.close()
        except Exception:
            pass
