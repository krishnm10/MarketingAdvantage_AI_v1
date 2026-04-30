# ============================================================
# app/core/vectordb/qdrant_v1.py
#
# WHAT THIS FILE IS:
#   Qdrant implementation of BaseVectorDB.
#   Supports both local (docker) and Qdrant Cloud.
#
# HOW IT'S REGISTERED:
#   vectordb_registry.register("qdrant", QdrantVectorDB)
# ============================================================

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from app.core.vectordb.base import BaseVectorDB, BatchUpsertResult, VectorHit
from app.core.runtime.errors import VectorDBError

logger = logging.getLogger(__name__)

# Qdrant client is lazily imported so the app starts even if qdrant-client
# is not installed (you only pay the install cost if you use Qdrant).
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False


class QdrantVectorDB(BaseVectorDB):
    """
    Qdrant connector implementing BaseVectorDB.

    Constructor args:
        url:         Full Qdrant URL (e.g. "http://localhost:6333" or cloud URL)
        api_key:     Optional Qdrant API key (required only for secured clusters)
        prefer_grpc: Use gRPC instead of HTTP for ~3x faster throughput
        timeout:     Request timeout in seconds
    """

    def __init__(
        self,
        url: Optional[str] = None,
        host: Optional[str] = None,
        port: int = 6333,
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
        timeout: int = 30,
    ):
        if not _QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is not installed. "
                "Install it: pip install qdrant-client"
            )
    
        # ---- LOCAL MODE (default) ----
        if not url:
            host = host or "localhost"
    
            self._client = QdrantClient(
                host=host,
                port=port,
                prefer_grpc=prefer_grpc,
                timeout=timeout,
            )
    
            logger.info(
                "[QdrantVectorDB] Initialized LOCAL | host=%s | port=%s | grpc=%s",
                host, port, prefer_grpc,
            )
    
        # ---- URL MODE (cloud or self-hosted endpoint) ----
        else:
            self._client = QdrantClient(
                url=url,
                api_key=api_key,
                prefer_grpc=prefer_grpc,
                timeout=timeout,
            )
    
            logger.info(
                "[QdrantVectorDB] Initialized URL | url=%s | grpc=%s | api_key=%s",
                url, prefer_grpc, bool(api_key),
            )

    @property
    def kind(self) -> str:
        return "qdrant"

    def health_check(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception as exc:
            logger.warning("[QdrantVectorDB] health_check failed: %s", exc)
            return False

    def ensure_collection(
        self,
        collection: str,
        *,
        embedding_dim: int,
        distance_metric: str = "cosine",
    ) -> None:
        _metric_map = {
            "cosine":      qdrant_models.Distance.COSINE,
            "dotproduct":  qdrant_models.Distance.DOT,
            "euclidean":   qdrant_models.Distance.EUCLID,
        }
        distance = _metric_map.get(distance_metric, qdrant_models.Distance.COSINE)

        existing = {c.name for c in self._client.get_collections().collections}
        if collection not in existing:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=qdrant_models.VectorParams(
                    size=embedding_dim,
                    distance=distance,
                ),
            )
            logger.info(
                "[QdrantVectorDB] Created collection '%s' | dim=%d | metric=%s",
                collection, embedding_dim, distance_metric,
            )
        else:
            logger.info("[QdrantVectorDB] Collection '%s' already exists.", collection)

    def delete_collection(self, collection: str) -> None:
        self._client.delete_collection(collection_name=collection)
        logger.info("[QdrantVectorDB] Collection '%s' deleted.", collection)

    def upsert(
        self,
        *,
        collection: str,
        doc_id: str,
        embedding: List[float],
        text: str,
        metadata: Dict[str, Any],
    ) -> None:
        payload = {"text": text, **{k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                                    for k, v in metadata.items()}}
        self._client.upsert(
            collection_name=collection,
            points=[qdrant_models.PointStruct(
                id=_to_qdrant_id(doc_id),
                vector=embedding,
                payload=payload,
            )],
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
        if not doc_ids:
            return BatchUpsertResult()
        try:
            points = [
                qdrant_models.PointStruct(
                    id=_to_qdrant_id(doc_ids[i]),
                    vector=embeddings[i],
                    payload={"text": texts[i], **_sanitize_qdrant_payload(metadatas[i])},
                )
                for i in range(len(doc_ids))
            ]
            self._client.upsert(collection_name=collection, points=points)
            return BatchUpsertResult(inserted=len(doc_ids))
        except Exception as exc:
            raise self._wrap_error(
                exc, operation="batch_upsert", collection=collection,
            ) from exc

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
        qdrant_filter = _build_qdrant_filter(effective_filters) if effective_filters else None
        try:
            response = self._client.query_points(
                collection_name=collection,
                query=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            results = response.points
        except Exception as exc:
            raise self._wrap_error(
                exc, operation="search", collection=collection, tenant_id=tenant_id,
            ) from exc

        hits = []
        for r in results:
            payload = dict(r.payload) if r.payload else {}
            hits.append(VectorHit(
                id=str(r.id),
                text=payload.pop("text", ""),
                score=round(r.score, 6),
                metadata=payload,
            ))
        self._log_tenant_search(
            tenant_id=tenant_id,
            collection=collection,
            filter_count=len(effective_filters),
            search_mode="vector",
            result_count=len(hits),
        )
        return hits

    def exists(
        self,
        *,
        collection: str,
        ids: List[str],
    ) -> List[str]:
        if not ids:
            return []
        try:
            qdrant_ids = [_to_qdrant_id(i) for i in ids]
            id_str_map = {_to_qdrant_id(i): i for i in ids}
            results = self._client.retrieve(
                collection_name=collection,
                ids=qdrant_ids,
                with_payload=False,
                with_vectors=False,
            )
            return [id_str_map[str(r.id)] for r in results if str(r.id) in id_str_map]
        except Exception as exc:
            raise self._wrap_error(
                exc, operation="exists", collection=collection,
            ) from exc

    def get_by_ids(
        self,
        *,
        collection: str,
        ids: List[str],
        tenant_id: Optional[str] = None,
    ) -> List[VectorHit]:
        if not ids:
            return []
        try:
            results = self._client.retrieve(
                collection_name=collection,
                ids=[_to_qdrant_id(i) for i in ids],
                with_payload=True,
                with_vectors=False,
            )
            hits = []
            for r in results:
                payload = r.payload or {}
                hits.append(VectorHit(
                    id=str(r.id),
                    text=payload.pop("text", ""),
                    score=1.0,
                    metadata=payload,
                ))
            return hits
        except Exception as exc:
            raise self._wrap_error(
                exc, operation="get_by_ids", collection=collection, tenant_id=tenant_id,
            ) from exc

    def count(self, collection: str) -> int:
        try:
            info = self._client.get_collection(collection_name=collection)
            return info.points_count or 0
        except Exception as exc:
            raise self._wrap_error(
                exc, operation="count", collection=collection,
            ) from exc

    def stats(self, collection: str) -> Dict[str, Any]:
        try:
            info = self._client.get_collection(collection_name=collection)
            return {
                "backend": self.kind,
                "collection": collection,
                "document_count": info.points_count or 0,
                "segments_count": info.segments_count,
                "status": str(info.status),
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
        try:
            for doc_id, meta in zip(ids, metadatas):
                self._client.set_payload(
                    collection_name=collection,
                    payload=meta,
                    points=[_to_qdrant_id(doc_id)],
                )
            logger.debug(
                "[QdrantVectorDB] update_metadata '%s': %d doc(s)",
                collection, len(ids),
            )
            return len(ids)
        except Exception as exc:
            raise self._wrap_error(
                exc, operation="update_metadata", collection=collection,
            ) from exc

    def delete(self, *, collection: str, doc_id: str) -> None:
        self._client.delete(
            collection_name=collection,
            points_selector=qdrant_models.PointIdsList(
                points=[_to_qdrant_id(doc_id)]
            ),
        )

    def delete_many(self, *, collection: str, doc_ids: List[str]) -> int:
        if not doc_ids:
            return 0
        try:
            self._client.delete(
                collection_name=collection,
                points_selector=qdrant_models.PointIdsList(
                    points=[_to_qdrant_id(i) for i in doc_ids]
                ),
            )
            return len(doc_ids)
        except Exception as exc:
            raise self._wrap_error(
                exc, operation="delete_many", collection=collection,
            ) from exc


def _to_qdrant_id(doc_id: str) -> str:
    """
    Qdrant requires IDs to be either unsigned integers or valid UUIDs.
    Since our semantic hashes are SHA-256 hex strings (not UUID format),
    we generate a deterministic UUID v5 from the string.
    This ensures the same string always maps to the same Qdrant ID.
    """
    try:
        uuid.UUID(doc_id)   # already a valid UUID
        return doc_id
    except ValueError:
        # Generate deterministic UUID from the string
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))


def _sanitize_qdrant_payload(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Qdrant accepts any JSON-serializable values in payload."""
    safe = {}
    for k, v in metadata.items():
        if v is None:
            safe[k] = ""
        elif isinstance(v, (str, int, float, bool, list, dict)):
            safe[k] = v
        else:
            safe[k] = str(v)
    return safe


def _build_qdrant_filter(filters: Dict[str, Any]) -> qdrant_models.Filter:
    """
    Converts generic metadata filters into Qdrant Filter.

    Supported forms:
      - {"field": "value"}                    -> equality
      - {"field": {"$eq": "value"}}          -> equality
      - {"field": {"$ne": "value"}}          -> inequality
      - {"field": {"$in": ["a", "b"]}}       -> any-of
      - {"field": {"$nin": ["a", "b"]}}      -> not-in
      - {"field": {"$gt": 1, "$lte": 10}}    -> numeric range
    """
    must: List[Any] = []
    must_not: List[Any] = []

    for key, raw in filters.items():
        # Simple equality: {"field": value}
        if not isinstance(raw, dict):
            must.append(
                qdrant_models.FieldCondition(
                    key=key,
                    match=qdrant_models.MatchValue(value=raw),
                )
            )
            continue

        # Operator form: {"field": {"$op": value}}
        range_kwargs: Dict[str, Any] = {}

        for op, val in raw.items():
            if op == "$eq":
                must.append(
                    qdrant_models.FieldCondition(
                        key=key,
                        match=qdrant_models.MatchValue(value=val),
                    )
                )
            elif op == "$ne":
                must_not.append(
                    qdrant_models.FieldCondition(
                        key=key,
                        match=qdrant_models.MatchValue(value=val),
                    )
                )
            elif op == "$in":
                values = list(val) if isinstance(val, (list, tuple, set)) else [val]
                must.append(
                    qdrant_models.FieldCondition(
                        key=key,
                        match=qdrant_models.MatchAny(any=values),
                    )
                )
            elif op == "$nin":
                values = list(val) if isinstance(val, (list, tuple, set)) else [val]
                must_not.append(
                    qdrant_models.FieldCondition(
                        key=key,
                        match=qdrant_models.MatchAny(any=values),
                    )
                )
            elif op in {"$gt", "$gte", "$lt", "$lte"}:
                if op == "$gt":
                    range_kwargs["gt"] = val
                elif op == "$gte":
                    range_kwargs["gte"] = val
                elif op == "$lt":
                    range_kwargs["lt"] = val
                elif op == "$lte":
                    range_kwargs["lte"] = val
            else:
                logger.warning("[QdrantVectorDB] Unsupported filter operator: %s", op)

        if range_kwargs:
            must.append(
                qdrant_models.FieldCondition(
                    key=key,
                    range=qdrant_models.Range(**range_kwargs),
                )
            )

    return qdrant_models.Filter(
        must=must or None,
        must_not=must_not or None,
    )
