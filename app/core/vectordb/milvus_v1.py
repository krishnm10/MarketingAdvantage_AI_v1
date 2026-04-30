# ============================================================
# app/core/vectordb/milvus_v1.py
# ============================================================
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from app.core.vectordb.base import BaseVectorDB, BatchUpsertResult, VectorHit
from app.core.runtime.errors import VectorDBError

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
        col = self._get_collection(collection)
        col.delete(expr=f'{_FIELD_ID} == "{doc_id}"')
        # pymilvus >=2.6 expects row-based format (list of dicts)
        col.insert([{
            _FIELD_ID:       doc_id,
            _FIELD_VECTOR:   embedding,
            _FIELD_TEXT:     str(text)[:65_530] if text else "",
            _FIELD_METADATA: _sanitize_milvus_metadata(metadata),
        }])
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

            # pymilvus >=2.6 expects row-based format (list of dicts)
            rows = [
                {
                    _FIELD_ID:       doc_ids[i],
                    _FIELD_VECTOR:   embeddings[i],
                    _FIELD_TEXT:     str(texts[i])[:65_530] if texts[i] else "",
                    _FIELD_METADATA: _sanitize_milvus_metadata(metadatas[i]),
                }
                for i in range(len(doc_ids))
            ]
            col.insert(rows)
            col.flush()
            logger.info(
                "[MilvusVectorDB] batch_upsert '%s': +%d new, ~%d updated",
                collection, inserted, updated,
            )
            return BatchUpsertResult(inserted=inserted, updated=updated)
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
        col = self._get_collection(collection)

        expr = _build_milvus_filter_expr(effective_filters)

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
            meta = _normalize_milvus_metadata(result.entity.get(_FIELD_METADATA))
            hits.append(VectorHit(
                id=str(result.id),
                text=str(result.entity.get(_FIELD_TEXT) or ""),
                score=float(result.score),
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
            raise self._wrap_error(
                exc, operation="exists", collection=collection,
            ) from exc

    def get_by_ids(self, *, collection: str, ids: List[str], tenant_id: Optional[str] = None) -> List[VectorHit]:
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
                meta = _normalize_milvus_metadata(r.get(_FIELD_METADATA))
                hits.append(VectorHit(
                    id=str(r[_FIELD_ID]),
                    text=str(r.get(_FIELD_TEXT) or ""),
                    score=1.0,
                    metadata=meta,
                ))
            return hits
        except Exception as exc:
            raise self._wrap_error(
                exc, operation="get_by_ids", collection=collection,
            ) from exc

    def count(self, collection: str) -> int:
        try:
            col = self._get_collection(collection)
            return col.num_entities
        except Exception as exc:
            raise self._wrap_error(
                exc, operation="count", collection=collection,
            ) from exc

    def stats(self, collection: str) -> Dict[str, Any]:
        try:
            col = self._get_collection(collection)
            return {
                "backend": self.kind,
                "collection": collection,
                "document_count": col.num_entities,
                "alias": self._alias,
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
        """
        Milvus has no native partial-metadata update. Strategy:
        query existing rows → merge metadata → delete → re-insert.
        """
        if not ids:
            return 0
        col = self._get_collection(collection)
        try:
            ids_expr = ", ".join(f'"{i}"' for i in ids)
            existing_rows = col.query(
                expr=f'{_FIELD_ID} in [{ids_expr}]',
                output_fields=[_FIELD_ID, _FIELD_VECTOR, _FIELD_TEXT, _FIELD_METADATA],
            )
            row_map = {r[_FIELD_ID]: r for r in existing_rows}

            updated = 0
            reinsert_rows = []
            for doc_id, new_meta in zip(ids, metadatas):
                row = row_map.get(doc_id)
                if row is None:
                    continue
                old_meta = json.loads(row[_FIELD_METADATA]) if row[_FIELD_METADATA] else {}
                old_meta.update(new_meta)
                reinsert_rows.append({
                    _FIELD_ID:       doc_id,
                    _FIELD_VECTOR:   row[_FIELD_VECTOR],
                    _FIELD_TEXT:     row[_FIELD_TEXT],
                    _FIELD_METADATA: json.dumps(old_meta),
                })
                updated += 1

            if reinsert_rows:
                col.delete(expr=f'{_FIELD_ID} in [{ids_expr}]')
                col.insert(reinsert_rows)
                col.flush()

            logger.debug(
                "[MilvusVectorDB] update_metadata '%s': %d doc(s)",
                collection, updated,
            )
            return updated
        except Exception as exc:
            raise self._wrap_error(
                exc, operation="update_metadata", collection=collection,
            ) from exc

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
            raise self._wrap_error(
                exc, operation="delete_many", collection=collection,
            ) from exc

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


def _build_milvus_filter_expr(filters: Optional[Dict[str, Any]]) -> Optional[str]:
    if not filters:
        return None

    clauses: List[str] = []
    for key, raw in filters.items():
        field = f'{_FIELD_METADATA}["{key}"]'

        if not isinstance(raw, dict):
            clauses.append(f"{field} == {_milvus_literal(raw)}")
            continue

        for op, value in raw.items():
            if op == "$eq":
                clauses.append(f"{field} == {_milvus_literal(value)}")
            elif op == "$ne":
                clauses.append(f"{field} != {_milvus_literal(value)}")
            elif op == "$in":
                clauses.append(f"{field} in {_milvus_list_literal(value)}")
            elif op == "$nin":
                clauses.append(f"{field} not in {_milvus_list_literal(value)}")
            elif op == "$gt":
                clauses.append(f"{field} > {_milvus_literal(value)}")
            elif op == "$gte":
                clauses.append(f"{field} >= {_milvus_literal(value)}")
            elif op == "$lt":
                clauses.append(f"{field} < {_milvus_literal(value)}")
            elif op == "$lte":
                clauses.append(f"{field} <= {_milvus_literal(value)}")
            else:
                logger.warning("[MilvusVectorDB] Unsupported filter operator: %s", op)

    return " && ".join(clauses) if clauses else None


def _milvus_literal(value: Any) -> str:
    if isinstance(value, dict):
        raise ValueError("Milvus metadata filters do not support dict values as direct literals.")
    if isinstance(value, (list, tuple, set)):
        raise ValueError("Milvus metadata filters require scalar values for direct comparison.")
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if value is None:
        return '""'
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _milvus_list_literal(value: Any) -> str:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return "[" + ", ".join(_milvus_literal(item) for item in values) + "]"


def _sanitize_milvus_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if value is None:
            safe[key] = ""
        elif isinstance(value, (str, int, float, bool, list, dict)):
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


def _normalize_milvus_metadata(raw_meta: Any) -> Dict[str, Any]:
    if isinstance(raw_meta, dict):
        return raw_meta
    if isinstance(raw_meta, str):
        raw_meta = raw_meta.strip()
        if not raw_meta:
            return {}
        try:
            parsed = json.loads(raw_meta)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}
