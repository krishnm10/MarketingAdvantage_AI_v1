# ============================================================
# app/core/vectordb/weaviate_v1.py
# ============================================================
from __future__ import annotations

import logging
import uuid
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional

import httpx

from app.core.vectordb.base import BaseVectorDB, BatchUpsertResult, VectorHit

logger = logging.getLogger(__name__)

_REST_ID_BATCH_SIZE = 50
_REST_SCAN_PAGE_SIZE = 200


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
        prefer_grpc: bool = True,
        grpc_host: Optional[str] = None,
        grpc_port: int = 50051,
        skip_init_checks: bool = False,
    ):
        try:
            import weaviate
        except ImportError:
            raise ImportError(
                "Weaviate client not installed. Run: pip install weaviate-client>=4.0.0"
            )

        self._url = url.rstrip("/")
        self._api_key = api_key
        self._additional_headers = additional_headers or {}
        self._prefer_grpc = bool(prefer_grpc)
        # If startup checks are skipped, assume gRPC may be unavailable and prefer REST reads.
        self._prefer_rest_reads = bool(skip_init_checks) or not self._prefer_grpc

        if embedded:
            import weaviate.embedded as _emb
            self._client = weaviate.WeaviateClient(
                embedded_options=_emb.EmbeddedOptions()
            )
            self._client.connect()
        else:
            from weaviate.auth import AuthApiKey
            parsed = urlparse(url)
            http_host = parsed.hostname or url
            http_secure = parsed.scheme == "https"
            http_port = parsed.port or (443 if http_secure else 80)
            resolved_grpc_host = grpc_host or http_host
            auth = AuthApiKey(api_key) if api_key else None
            self._client = weaviate.connect_to_custom(
                http_host=http_host,
                http_port=http_port,
                http_secure=http_secure,
                grpc_host=resolved_grpc_host,
                grpc_port=grpc_port,
                grpc_secure=http_secure,
                auth_credentials=auth,
                headers=additional_headers or {},
                skip_init_checks=skip_init_checks,
            )

        logger.info(
            "[WeaviateVectorDB] Connected | url=%s | grpc=%s:%s | prefer_grpc=%s | embedded=%s | skip_init_checks=%s",
            url,
            grpc_host or (urlparse(url).hostname or url),
            grpc_port,
            self._prefer_grpc,
            embedded,
            skip_init_checks,
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

        existing = self._list_collection_names()
        if cls_name in existing:
            logger.debug(
                "[WeaviateVectorDB] Collection '%s' exists — skipping.", cls_name
            )
            return

        self._client.collections.create(
            name=cls_name,
            vector_config=wc.Configure.Vectors.self_provided(
                vector_index_config=wc.Configure.VectorIndex.hnsw(
                    distance_metric=distance,
                ),
            ),
            properties=[
                wc.Property(name="_text", data_type=wc.DataType.TEXT),
                wc.Property(name="semantic_hash", data_type=wc.DataType.TEXT),
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
        properties.setdefault("semantic_hash", doc_id)
        properties["_text"] = text
        object_uuid = self._uuid_for_doc_id(doc_id)
        try:
            if col.data.exists(object_uuid):
                col.data.replace(
                    uuid=object_uuid,
                    properties=properties,
                    vector=embedding,
                )
            else:
                col.data.insert(
                    properties=properties,
                    vector=embedding,
                    uuid=object_uuid,
                )
        except Exception:
            # Fallback for client/server versions where exists() may be unavailable or flaky.
            try:
                col.data.replace(
                    uuid=object_uuid,
                    properties=properties,
                    vector=embedding,
                )
            except Exception:
                col.data.insert(
                    properties=properties,
                    vector=embedding,
                    uuid=object_uuid,
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
        Weaviate v4 batch insert using insert_many().
        For upsert semantics: check exists → delete existing → batch insert.
        insert_many() is ~10x faster than individual inserts.
        """
        if not doc_ids:
            return BatchUpsertResult()

        if self._prefer_rest_reads:
            return self._batch_upsert_via_upsert(
                collection=collection,
                doc_ids=doc_ids,
                embeddings=embeddings,
                texts=texts,
                metadatas=metadatas,
            )

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
                    col.data.delete_by_id(self._uuid_for_doc_id(eid))
                except Exception:
                    pass

            # Build objects list for insert_many
            from weaviate.classes.data import DataObject
            objects = [
                DataObject(
                    properties={
                        **(metadatas[i] or {}),
                        "semantic_hash": doc_ids[i],
                        "_text": texts[i],
                    },
                    vector=embeddings[i],
                    uuid=self._uuid_for_doc_id(doc_ids[i]),
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
            logger.warning(
                "[WeaviateVectorDB] batch_upsert gRPC path failed, retrying item-by-item: %s",
                exc,
            )
            return self._batch_upsert_via_upsert(
                collection=collection,
                doc_ids=doc_ids,
                embeddings=embeddings,
                texts=texts,
                metadatas=metadatas,
            )

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
        try:
            from weaviate.collections.classes.filters import Filter
        except ImportError:
            from weaviate.classes.query import Filter

        cls_name = self._class_name(collection)
        col = self._client.collections.get(cls_name)

        where_filter = None
        if filters:
            conditions = self._build_weaviate_filters(filters, Filter)
            if conditions:
                where_filter = (
                    conditions[0] if len(conditions) == 1
                    else Filter.all_of(conditions)
                )

        if self._prefer_rest_reads:
            logger.debug(
                "[WeaviateVectorDB] search using REST fallback because skip_init_checks is enabled."
            )
            return self._search_rest(
                collection=collection,
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
            )

        try:
            res = col.query.near_vector(
                near_vector=query_embedding,
                limit=int(top_k),
                filters=where_filter,
                return_metadata=wq.MetadataQuery(certainty=True),
            )

            hits: List[VectorHit] = []
            for obj in res.objects:
                props = dict(obj.properties or {})
                text = str(props.pop("_text", ""))
                score = float(obj.metadata.certainty or 0.0) if obj.metadata else 0.0
                hits.append(VectorHit(
                    id=str(props.get("semantic_hash") or obj.uuid),
                    text=text,
                    score=round(score, 6),
                    metadata=props,
                ))
            return hits
        except Exception as exc:
            logger.warning("[WeaviateVectorDB] search gRPC path failed, using REST fallback: %s", exc)
            return self._search_rest(collection=collection, query_embedding=query_embedding, top_k=top_k, filters=filters)

    def exists(self, *, collection: str, ids: List[str]) -> List[str]:
        """
        Check existence using Weaviate's fetch_object_by_id.
        Returns only the IDs that actually exist.
        """
        if not ids:
            return []
        cls_name = self._class_name(collection)
        if self._prefer_rest_reads:
            return self._exists_rest(collection, ids)
        col = self._client.collections.get(cls_name)
        found = []
        try:
            for doc_id in ids:
                try:
                    obj = col.query.fetch_object_by_id(
                        self._uuid_for_doc_id(doc_id),
                        include_vector=False,
                    )
                    if obj is not None:
                        found.append(doc_id)
                except Exception:
                    pass
            return found
        except Exception as exc:
            logger.warning("[WeaviateVectorDB] exists() gRPC path failed, using REST fallback: %s", exc)
            return self._exists_rest(collection, ids)

    def get_by_ids(self, *, collection: str, ids: List[str]) -> List[VectorHit]:
        if not ids:
            return []
        if self._prefer_rest_reads:
            return self._get_by_ids_rest(collection, ids)
        cls_name = self._class_name(collection)
        col = self._client.collections.get(cls_name)
        hits = []
        try:
            for doc_id in ids:
                try:
                    obj = col.query.fetch_object_by_id(
                        self._uuid_for_doc_id(doc_id),
                        include_vector=False,
                    )
                    if obj:
                        props = dict(obj.properties or {})
                        text = str(props.pop("_text", ""))
                        hits.append(VectorHit(
                            id=str(props.get("semantic_hash") or doc_id),
                            text=text,
                            score=1.0,
                            metadata=props,
                        ))
                except Exception:
                    pass
            return hits
        except Exception as exc:
            logger.warning("[WeaviateVectorDB] get_by_ids() gRPC path failed, using REST fallback: %s", exc)
            return self._get_by_ids_rest(collection, ids)

    def count(self, collection: str) -> int:
        try:
            if self._prefer_rest_reads:
                return self._count_rest(collection)
            cls_name = self._class_name(collection)
            col = self._client.collections.get(cls_name)
            result = col.aggregate.over_all(total_count=True)
            return int(result.total_count or 0)
        except Exception as exc:
            logger.warning("[WeaviateVectorDB] count() gRPC path failed, using REST fallback: %s", exc)
            try:
                return self._count_rest(collection)
            except Exception as rest_exc:
                logger.warning("[WeaviateVectorDB] count() REST fallback failed: %s", rest_exc)
                return 0

    def get_all(
        self,
        *,
        collection: str,
        include: Optional[List[str]] = None,
    ) -> Dict[str, List[Any]]:
        """
        Return all objects in a collection in Chroma-style shape.

        This is used by admin integrity tooling to compare vector IDs against
        PostgreSQL. We prefer a REST GraphQL scan here because it works even
        when gRPC is unavailable and skip_init_checks is enabled.
        """
        requested = set(include or [])
        fetch_documents = "documents" in requested
        fetch_metadatas = "metadatas" in requested
        fetch_embeddings = "embeddings" in requested

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        embeddings: List[List[float]] = []

        cls_name = self._class_name(collection)
        offset = 0

        while True:
            fields = self._graphql_get_fields(
                include_documents=fetch_documents,
                include_semantic_hash=True,
                include_additional_id=not fetch_metadatas,
                include_vectors=fetch_embeddings,
            )
            query = (
                "{ Get { "
                f"{cls_name}(limit:{_REST_SCAN_PAGE_SIZE}, offset:{offset}) "
                f"{{ {fields} }} "
                "} }"
            )
            payload = self._graphql(query)
            rows = payload["data"]["Get"].get(cls_name, [])
            if not rows:
                break

            for row in rows:
                additional = row.get("_additional") or {}
                semantic_hash = row.get("semantic_hash")
                resolved_id = str(semantic_hash or additional.get("id") or "")
                if not resolved_id:
                    continue
                ids.append(resolved_id)
                if fetch_documents:
                    documents.append(str(row.get("_text", "")))
                if fetch_metadatas:
                    metadatas.append(
                        {
                            k: v
                            for k, v in row.items()
                            if k not in {"_text", "_additional"}
                        }
                    )
                if fetch_embeddings:
                    embeddings.append(self._extract_vector_from_graphql_row(row))

            if len(rows) < _REST_SCAN_PAGE_SIZE:
                break
            offset += _REST_SCAN_PAGE_SIZE

        result: Dict[str, List[Any]] = {"ids": ids}
        if fetch_documents:
            result["documents"] = documents
        if fetch_metadatas:
            result["metadatas"] = metadatas
        if fetch_embeddings:
            result["embeddings"] = embeddings
        return result

    # ── Delete ────────────────────────────────────────────────────────

    def delete(self, *, collection: str, doc_id: str) -> None:
        cls_name = self._class_name(collection)
        col = self._client.collections.get(cls_name)
        col.data.delete_by_id(self._uuid_for_doc_id(doc_id))

    def delete_many(self, *, collection: str, doc_ids: List[str]) -> int:
        if not doc_ids:
            return 0
        cls_name = self._class_name(collection)
        col = self._client.collections.get(cls_name)
        deleted = 0
        for doc_id in doc_ids:
            try:
                col.data.delete_by_id(self._uuid_for_doc_id(doc_id))
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

    def _request_headers(self) -> Dict[str, str]:
        headers = dict(self._additional_headers)
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _graphql(self, query: str) -> Dict[str, Any]:
        resp = httpx.post(
            f"{self._url}/v1/graphql",
            headers=self._request_headers(),
            json={"query": query},
            timeout=20.0,
            trust_env=False,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("errors"):
            raise RuntimeError(str(payload["errors"]))
        return payload

    def _count_rest(self, collection: str) -> int:
        cls_name = self._class_name(collection)
        payload = self._graphql(
            f"{{ Aggregate {{ {cls_name} {{ meta {{ count }} }} }} }}"
        )
        return int(payload["data"]["Aggregate"][cls_name][0]["meta"]["count"])

    def _exists_rest(self, collection: str, ids: List[str]) -> List[str]:
        found: List[str] = []
        cls_name = self._class_name(collection)
        for batch in self._chunked(ids, _REST_ID_BATCH_SIZE):
            try:
                query = (
                    "{ Get { "
                    f"{cls_name}(where:{self._semantic_hash_where_any(batch)}, limit:{len(batch)}) "
                    "{ semantic_hash } "
                    "} }"
                )
                payload = self._graphql(query)
                rows = payload["data"]["Get"].get(cls_name, [])
                found.extend(
                    str(row.get("semantic_hash"))
                    for row in rows
                    if row.get("semantic_hash")
                )
            except Exception:
                pass
        return list(dict.fromkeys(found))

    def _get_by_ids_rest(self, collection: str, ids: List[str]) -> List[VectorHit]:
        hits: List[VectorHit] = []
        cls_name = self._class_name(collection)
        for batch in self._chunked(ids, _REST_ID_BATCH_SIZE):
            try:
                query = (
                    "{ Get { "
                    f"{cls_name}(where:{self._semantic_hash_where_any(batch)}, limit:{len(batch)}) "
                    "{ _text semantic_hash _additional { id } } "
                    "} }"
                )
                payload = self._graphql(query)
                rows = payload["data"]["Get"].get(cls_name, [])
                for obj in rows:
                    props = {k: v for k, v in obj.items() if k not in {"_text", "_additional"}}
                    text = str(obj.get("_text", ""))
                    hits.append(VectorHit(
                        id=str(props.get("semantic_hash") or ""),
                        text=text,
                        score=1.0,
                        metadata=props,
                    ))
            except Exception:
                pass
        return hits

    def _search_rest(
        self,
        *,
        collection: str,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorHit]:
        cls_name = self._class_name(collection)
        filter_clause = ""
        if filters:
            clauses = self._graphql_filter_clauses(filters)
            if clauses:
                where = clauses[0] if len(clauses) == 1 else "{operator:And,operands:[%s]}" % ",".join(clauses)
                filter_clause = f", where:{where}"

        vector_str = ",".join(str(float(x)) for x in query_embedding)
        fields = self._graphql_get_fields(
            include_semantic_hash=True,
            include_documents=True,
            include_additional_id=True,
            include_certainty=True,
        )
        query = (
            "{ Get { "
            f"{cls_name}(nearVector:{{vector:[{vector_str}]}}, limit:{int(top_k)}{filter_clause}) "
            f"{{ {fields} }} "
            "} }"
        )
        payload = self._graphql(query)
        rows = payload["data"]["Get"].get(cls_name, [])
        hits: List[VectorHit] = []
        for row in rows:
            additional = row.get("_additional") or {}
            hits.append(VectorHit(
                id=str(row.get("semantic_hash") or additional.get("id", "")),
                text=str(row.get("_text", "")),
                score=round(float(additional.get("certainty") or 0.0), 6),
                metadata={k: v for k, v in row.items() if k not in {"_text", "_additional"}},
            ))
        return hits

    @staticmethod
    def _graphql_quote(value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    def _list_collection_names(self) -> set[str]:
        """
        Normalize list_all() responses across Weaviate client versions.
        """
        raw = self._client.collections.list_all()

        if isinstance(raw, dict):
            return {str(name) for name in raw.keys()}

        names: set[str] = set()
        for item in raw or []:
            if isinstance(item, str):
                names.add(item)
                continue

            name = getattr(item, "name", None)
            if name:
                names.add(str(name))
                continue

            if isinstance(item, dict):
                candidate = item.get("name") or item.get("class")
                if candidate:
                    names.add(str(candidate))

        return names

    @staticmethod
    def _uuid_for_doc_id(doc_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))

    @staticmethod
    def _chunked(values: List[str], size: int) -> List[List[str]]:
        return [values[i:i + size] for i in range(0, len(values), size)]

    def _semantic_hash_where_any(self, ids: List[str]) -> str:
        if len(ids) == 1:
            return (
                "{path:[\"semantic_hash\"],operator:Equal,"
                + f"valueText:{self._graphql_quote(ids[0])}"
                + "}"
            )
        operands = ",".join(
            (
                "{path:[\"semantic_hash\"],operator:Equal,"
                + f"valueText:{self._graphql_quote(doc_id)}"
                + "}"
            )
            for doc_id in ids
        )
        return f"{{operator:Or,operands:[{operands}]}}"

    @staticmethod
    def _build_weaviate_filters(filters: Dict[str, Any], Filter: Any) -> List[Any]:
        conditions: List[Any] = []
        for key, value in filters.items():
            builder = Filter.by_property(key)
            if isinstance(value, dict):
                if "$ne" in value:
                    conditions.append(builder.not_equal(value["$ne"]))
                    continue
                if "$eq" in value:
                    conditions.append(builder.equal(value["$eq"]))
                    continue
            conditions.append(builder.equal(value))
        return conditions

    def _graphql_filter_clauses(self, filters: Dict[str, Any]) -> List[str]:
        clauses: List[str] = []
        for key, value in filters.items():
            if isinstance(value, dict):
                if "$ne" in value:
                    clauses.append(
                        '{path:["%s"],operator:NotEqual,valueText:%s}' % (
                            key,
                            self._graphql_quote(str(value["$ne"])),
                        )
                    )
                    continue
                if "$eq" in value:
                    clauses.append(
                        '{path:["%s"],operator:Equal,valueText:%s}' % (
                            key,
                            self._graphql_quote(str(value["$eq"])),
                        )
                    )
                    continue
            clauses.append(
                '{path:["%s"],operator:Equal,valueText:%s}' % (
                    key,
                    self._graphql_quote(str(value)),
                )
            )
        return clauses

    @staticmethod
    def _graphql_get_fields(
        *,
        include_semantic_hash: bool = False,
        include_documents: bool = False,
        include_metadatas: bool = False,
        include_additional_id: bool = False,
        include_certainty: bool = False,
        include_vectors: bool = False,
    ) -> str:
        fields: List[str] = []
        if include_semantic_hash:
            fields.append("semantic_hash")
        if include_documents:
            fields.append("_text")
        additional_fields: List[str] = []
        if include_additional_id:
            additional_fields.append("id")
        if include_certainty:
            additional_fields.append("certainty")
        if include_vectors:
            additional_fields.append("vector")
        if additional_fields:
            fields.append("_additional { %s }" % " ".join(additional_fields))
        return " ".join(fields) or "_additional { id }"

    @staticmethod
    def _extract_vector_from_graphql_row(row: Dict[str, Any]) -> List[float]:
        additional = row.get("_additional") or {}
        vector = additional.get("vector")
        if isinstance(vector, list):
            return [float(x) for x in vector]
        return []

    def _batch_upsert_via_upsert(
        self,
        *,
        collection: str,
        doc_ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> BatchUpsertResult:
        existing = set(self.exists(collection=collection, ids=doc_ids))
        inserted = 0
        updated = 0
        failed = 0

        for i, doc_id in enumerate(doc_ids):
            try:
                self.upsert(
                    collection=collection,
                    doc_id=doc_id,
                    embedding=embeddings[i],
                    text=texts[i],
                    metadata=metadatas[i] or {},
                )
                if doc_id in existing:
                    updated += 1
                else:
                    inserted += 1
            except Exception as exc:
                failed += 1
                logger.warning(
                    "[WeaviateVectorDB] upsert fallback failed for '%s': %s",
                    doc_id,
                    exc,
                )

        return BatchUpsertResult(inserted=inserted, updated=updated, failed=failed)
