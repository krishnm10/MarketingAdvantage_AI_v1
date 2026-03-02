# ============================================================
# app/core/vectordb/milvus_v1.py
# ============================================================
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from app.core.vectordb.base import BaseVectorDB, BatchUpsertResult, VectorHit

logger = logging.getLogger(__name__)

_FIELD_ID       = "doc_id"
_FIELD_VECTOR   = "embedding"
_FIELD_TEXT     = "_text"
_FIELD_METADATA = "_metadata"
_INDEX_NAME     = "embedding_idx"
_METRIC_TYPE    = "COSINE"


class MilvusVectorDB(BaseVectorDB):
    """
    Milvus / Zilliz Cloud connector.
    Supports: local Docker, on-premise cluster, Zilliz Cloud.
    Best for: billion-scale vectors, data-residency (India DPDP).
    Install: pip install pymilvus>=2.4.0
    """

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 19530,
        uri: Optional[str] = None,
        token: Optional[str] = None,
        db_name: str = "default",
        alias: str = "default",
    ):
        try:
            from pymilvus import connections
        except ImportError:
            raise ImportError("PyMilvus not installed. Run: pip install pymilvus>=2.4.0")

        from pymilvus import connections
        self._alias   = alias
        self._db_name = db_name

        if uri:
            connections.connect(alias=alias, uri=uri, token=token or "")
        else:
            kw: Dict[str, Any] = {
                "alias": alias, "host": host,
                "port": str(port), "db_name": db_name,
            }
            if token:
                kw["token"] = token
            connections.connect(**kw)

        logger.info(
            "[MilvusVectorDB] Connected | uri=%s | host=%s:%d | alias=%s",
            uri or "N/A", host, port, alias,
        )

    # ── Identity ──────────────────────────────────────────────────────

    @property
    def kind(self) -> str:
        return "milvus"

    # ── Lifecycle ─────────────────────────────────────────────────────

    def health_check(self) -> bool:
        try:
            from pymilvus import utility
            utility.list_collections(using=self._alias)
            return True
        except Exception as exc:
            logger.warning("[MilvusVectorDB] health_check failed: %s", exc)
            return False

    def ensure_collection(
        self,
        collection: str,
        *,
        embedding_dim: int,
        distance_metric: str = "cosine",
    ) -> None:
        """
        Create collection + HNSW index if it doesn't exist.
        distance_metric is mapped to Milvus metric type.
        Loads the collection into memory (required before search).
        """
        from pymilvus import Collection, utility

        _metric_map = {
            "cosine":     "COSINE",
            "dotproduct": "IP",
            "euclidean":  "L2",
        }
        metric = _metric_map.get(distance_metric.lower(), "COSINE")

        if utility.has_collection(collection, using=self._alias):
            self._get_collection(collection).load()
            logger.debug("[MilvusVectorDB] Collection '%s' exists — loaded.", collection)
            return

        schema = self._schema(embedding_dim)
        col = Collection(name=collection, schema=schema, using=self._alias)
        self._create_index(col, metric)
        col.load()
        logger.info(
            "[MilvusVectorDB] Created + loaded '%s' | dim=%d | metric=%s",
            collection, embedding_dim, metric,
        )

    def delete_collection(self, collection: str) -> None:
        from pymilvus import utility
        try:
            utility.drop_collection(collection, using=self._alias)
            logger.info("[MilvusVectorDB] Collection '%s' deleted.", collection)
        except Exception as exc:
            logger.warning("[MilvusVectorDB] delete_collection failed: %s", exc)

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
        col.delete(expr=f'{_FIELD_ID} == "{doc_id}"')
        col.insert(data={
            _FIELD_ID:       [doc_id],
            _FIELD_VECTOR:   [embedding],
            _FIELD_TEXT:     [str(text)[:65_530] if text else ""],
            _FIELD_METADATA: [json.dumps(metadata or {}, ensure_ascii=False)],
        })
        col.flush()

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
        Batch upsert for Milvus.
        Milvus does not have native upsert for VARCHAR PKs,
        so we: bulk delete existing IDs → bulk insert all.
        This is the most efficient pattern for Milvus.
        """
        if not doc_ids:
            return BatchUpsertResult()
        col = self._get_collection(collection)
        try:
            # Check which already exist for inserted vs updated count
            existing = set(self.exists(collection=collection, ids=doc_ids))
            updated  = len(existing)
            inserted = len(doc_ids) - updated

            # Bulk delete existing (single expression is faster than N deletes)
            if existing:
                ids_expr = ", ".join(f'"{i}"' for i in existing)
                col.delete(expr=f'{_FIELD_ID} in [{ids_expr}]')

            col.insert(data={
                _FIELD_ID:       doc_ids,
                _FIELD_VECTOR:   embeddings,
                _FIELD_TEXT:     [str(t)[:65_530] if t else "" for t in texts],
                _FIELD_METADATA: [
                    json.dumps(m or {}, ensure_ascii=False) for m in metadatas
                ],
            })
            col.flush()
            logger.info(
                "[MilvusVectorDB] batch_upsert '%s': +%d new, ~%d updated",
                collection, inserted, updated,
            )
            return BatchUpsertResult(inserted=inserted, updated=updated)
        except Exception as exc:
            logger.error("[MilvusVectorDB] batch_upsert failed: %s", exc)
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
        col = self._get_collection(collection)

        expr = None
        if filters:
            clauses = []
            for k, v in filters.items():
                val_str = f'"{v}"' if isinstance(v, str) else str(v).lower()
                clauses.append(f'{_FIELD_METADATA}["{k}"] == {val_str}')
            expr = " && ".join(clauses)

        search_params = {
            "metric_type": _METRIC_TYPE,
            "params": {"ef": min(top_k * 4, 512)},
        }
        res = col.search(
            data=[query_embedding],
            anns_field=_FIELD_VECTOR,
            param=search_params,
            limit=int(top_k),
            expr=expr,
            output_fields=[_FIELD_TEXT, _FIELD_METADATA],
        )

        hits: List[VectorHit] = []
        for result in res[0]:
            raw_meta = result.entity.get(_FIELD_METADATA) or "{}"
            meta = json.loads(raw_meta) if isinstance(raw_meta, str) else (raw_meta or {})
            hits.append(VectorHit(
                id=str(result.id),
                text=str(result.entity.get(_FIELD_TEXT) or ""),
                score=float(result.score),
                metadata=meta,
            ))
        return hits

    def exists(self, *, collection: str, ids: List[str]) -> List[str]:
        """
        Check which IDs already exist by querying the primary key field.
        Returns only the IDs that were found.
        """
        if not ids:
            return []
        col = self._get_collection(collection)
        try:
            ids_expr = ", ".join(f'"{i}"' for i in ids)
            results = col.query(
                expr=f'{_FIELD_ID} in [{ids_expr}]',
                output_fields=[_FIELD_ID],
            )
            return [r[_FIELD_ID] for r in results]
        except Exception as exc:
            logger.warning("[MilvusVectorDB] exists() failed: %s", exc)
            return []

    def get_by_ids(self, *, collection: str, ids: List[str]) -> List[VectorHit]:
        if not ids:
            return []
        col = self._get_collection(collection)
        try:
            ids_expr = ", ".join(f'"{i}"' for i in ids)
            results = col.query(
                expr=f'{_FIELD_ID} in [{ids_expr}]',
                output_fields=[_FIELD_ID, _FIELD_TEXT, _FIELD_METADATA],
            )
            hits = []
            for r in results:
                raw_meta = r.get(_FIELD_METADATA) or "{}"
                meta = json.loads(raw_meta) if isinstance(raw_meta, str) else {}
                hits.append(VectorHit(
                    id=str(r[_FIELD_ID]),
                    text=str(r.get(_FIELD_TEXT) or ""),
                    score=1.0,
                    metadata=meta,
                ))
            return hits
        except Exception as exc:
            logger.warning("[MilvusVectorDB] get_by_ids() failed: %s", exc)
            return []

    def count(self, collection: str) -> int:
        try:
            col = self._get_collection(collection)
            return col.num_entities
        except Exception as exc:
            logger.warning("[MilvusVectorDB] count() failed: %s", exc)
            return 0

    # ── Delete ────────────────────────────────────────────────────────

    def delete(self, *, collection: str, doc_id: str) -> None:
        col = self._get_collection(collection)
        col.delete(expr=f'{_FIELD_ID} == "{doc_id}"')
        col.flush()

    def delete_many(self, *, collection: str, doc_ids: List[str]) -> int:
        if not doc_ids:
            return 0
        col = self._get_collection(collection)
        try:
            existing = self.exists(collection=collection, ids=doc_ids)
            if not existing:
                return 0
            ids_expr = ", ".join(f'"{i}"' for i in existing)
            col.delete(expr=f'{_FIELD_ID} in [{ids_expr}]')
            col.flush()
            return len(existing)
        except Exception as exc:
            logger.warning("[MilvusVectorDB] delete_many failed: %s", exc)
            return 0

    # ── Internal helpers ──────────────────────────────────────────────

    def _get_collection(self, collection: str):
        from pymilvus import Collection
        return Collection(name=collection, using=self._alias)

    def _schema(self, embedding_dim: int):
        from pymilvus import CollectionSchema, FieldSchema, DataType
        fields = [
            FieldSchema(name=_FIELD_ID, dtype=DataType.VARCHAR,
                        max_length=512, is_primary=True, auto_id=False),
            FieldSchema(name=_FIELD_VECTOR, dtype=DataType.FLOAT_VECTOR,
                        dim=int(embedding_dim)),
            FieldSchema(name=_FIELD_TEXT, dtype=DataType.VARCHAR,
                        max_length=65_535),
            FieldSchema(name=_FIELD_METADATA, dtype=DataType.JSON),
        ]
        return CollectionSchema(
            fields=fields,
            description="MarketingAdvantage AI — document chunks",
            enable_dynamic_field=True,
        )

    def _create_index(self, collection, metric: str = "COSINE") -> None:
        collection.create_index(
            field_name=_FIELD_VECTOR,
            index_params={
                "metric_type": metric,
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 256},
            },
            index_name=_INDEX_NAME,
        )
        logger.info("[MilvusVectorDB] HNSW index created on '%s'.", collection.name)

    def __del__(self):
        try:
            from pymilvus import connections
            connections.disconnect(self._alias)
        except Exception:
            pass
