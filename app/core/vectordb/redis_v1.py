# ============================================================
# app/core/vectordb/redis_v1.py
#
# Redis Vector DB Connector - Local, Remote, and Cloud
#
# Supports:
#   - Local:  localhost:6379 (default Docker/Redis Stack)
#   - Remote: any host:port with optional auth
#   - Cloud:  Redis Cloud URL with username + password
#
# Backend: Redis Stack (RediSearch + RedisJSON)
#   pip install redis>=5.0.0
#
# Architecture:
#   - Vectors stored as HASH keys with FLOAT32 blobs
#   - FT.CREATE index with HNSW for ANN search
#   - Metadata fields stored as TAG/TEXT in the hash
#   - Full BaseVectorDB interface compliance
#
# ============================================================

from __future__ import annotations

import json
import logging
import struct
from typing import Any, Dict, List, Optional

from app.core.vectordb.base import BaseVectorDB, BatchUpsertResult, VectorHit
from app.core.runtime.errors import VectorDBError

logger = logging.getLogger(__name__)


class RedisVectorBackendUnavailable(RuntimeError):
    """Raised when Redis is reachable but RediSearch/vector commands are unavailable."""


def _float_list_to_bytes(vec: List[float]) -> bytes:
    """Convert a list of floats to a compact FLOAT32 binary blob."""
    return struct.pack(f"<{len(vec)}f", *vec)


def _bytes_to_float_list(blob: bytes) -> List[float]:
    """Convert a FLOAT32 binary blob back to a list of floats."""
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


class RedisVectorDB(BaseVectorDB):
    """
    Redis Stack vector database connector.

    Supports three deployment modes:
      1. Local:  host=localhost, port=6379
      2. Remote: host=<ip>, port=<port>, password=<secret>
      3. Cloud:  url=redis://<user>:<pass>@<host>:<port>

    Requires Redis Stack (redis/redis-stack Docker image) or
    Redis Cloud with RediSearch module enabled.

    pip install redis>=5.0.0
    """

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        username: Optional[str] = None,
        db: int = 0,
        ssl: bool = False,
        ssl_ca_certs: Optional[str] = None,
        prefix: str = "vec:",
    ):
        """
        Args:
            url:          Full Redis URL (redis://... or rediss://...).
                          Takes precedence over host/port if set.
            host:         Redis host (local or remote).
            port:         Redis port.
            password:     Redis password (AUTH).
            username:     Redis username (ACL, Redis 6+).
            db:           Redis database number.
            ssl:          Enable TLS (required for most cloud providers).
            ssl_ca_certs: Path to CA cert file for TLS verification.
            prefix:       Key prefix for vector hashes (default: "vec:").
        """
        try:
            import redis as redis_lib
        except ImportError:
            raise ImportError(
                "redis-py required: pip install redis>=5.0.0\n"
                "Also requires Redis Stack server with RediSearch module."
            )

        self._prefix = prefix

        if url:
            self._client = redis_lib.Redis.from_url(
                url,
                decode_responses=False,
            )
            logger.info("[RedisVectorDB] Connected via URL: %s", url[:30] + "...")
        else:
            kwargs: Dict[str, Any] = {
                "host": host,
                "port": port,
                "db": db,
                "decode_responses": False,
            }
            if password:
                kwargs["password"] = password
            if username:
                kwargs["username"] = username
            if ssl:
                kwargs["ssl"] = True
                if ssl_ca_certs:
                    kwargs["ssl_ca_certs"] = ssl_ca_certs

            self._client = redis_lib.Redis(**kwargs)
            logger.info(
                "[RedisVectorDB] Connected to %s:%d (db=%d, ssl=%s)",
                host, port, db, ssl,
            )

        self._created_indices: set[str] = set()
        self._vector_capability_checked = False

    @property
    def kind(self) -> str:
        return "redis"

    def _key(self, collection: str, doc_id: str) -> str:
        """Redis key for a single vector document."""
        return f"{self._prefix}{collection}:{doc_id}"

    def _index_name(self, collection: str) -> str:
        """RediSearch index name for a collection."""
        return f"idx:{self._prefix}{collection}"

    def _raise_if_redisearch_missing(self, exc: Exception) -> None:
        err_msg = str(exc).lower()
        if "unknown command 'ft." in err_msg or 'unknown command "ft.' in err_msg:
            raise RedisVectorBackendUnavailable(
                "Redis endpoint is reachable but RediSearch/Redis Stack is not enabled. "
                "This vector backend requires FT.CREATE and FT.SEARCH support. "
                "Use Redis Stack or a Redis service with RediSearch enabled."
            ) from exc

    def _assert_vector_backend_ready(self) -> None:
        if self._vector_capability_checked:
            return
        try:
            self._client.execute_command("FT._LIST")
        except Exception as exc:
            self._raise_if_redisearch_missing(exc)
            raise
        self._vector_capability_checked = True

    def _index_exists(self, collection: str) -> bool:
        """Check if a RediSearch index already exists."""
        idx = self._index_name(collection)
        try:
            self._client.execute_command("FT.INFO", idx)
            return True
        except Exception as exc:
            self._raise_if_redisearch_missing(exc)
            return False

    def health_check(self) -> bool:
        try:
            return self._client.ping()
        except Exception as exc:
            logger.warning("[RedisVectorDB] health_check failed: %s", exc)
            return False

    def ensure_collection(
        self,
        collection: str,
        *,
        embedding_dim: int,
        distance_metric: str = "cosine",
    ) -> None:
        self._assert_vector_backend_ready()
        idx = self._index_name(collection)

        if collection in self._created_indices:
            return
        if self._index_exists(collection):
            self._created_indices.add(collection)
            return

        metric_map = {
            "cosine": "COSINE",
            "dotproduct": "IP",
            "euclidean": "L2",
            "ip": "IP",
            "l2": "L2",
        }
        redis_metric = metric_map.get(distance_metric.lower(), "COSINE")
        prefix = f"{self._prefix}{collection}:"

        try:
            self._client.execute_command(
                "FT.CREATE", idx,
                "ON", "HASH",
                "PREFIX", "1", prefix,
                "SCHEMA",
                "embedding", "VECTOR", "HNSW", "6",
                    "TYPE", "FLOAT32",
                    "DIM", str(embedding_dim),
                    "DISTANCE_METRIC", redis_metric,
                "_text", "TEXT", "NOINDEX",
                "_metadata", "TEXT", "NOINDEX",
                "doc_id", "TAG", "SEPARATOR", "|",
            )
            self._created_indices.add(collection)
            logger.info(
                "[RedisVectorDB] Created index '%s' | dim=%d | metric=%s",
                idx, embedding_dim, redis_metric,
            )
        except Exception as exc:
            self._raise_if_redisearch_missing(exc)
            err_msg = str(exc).lower()
            if "index already exists" in err_msg:
                self._created_indices.add(collection)
                logger.debug("[RedisVectorDB] Index '%s' already exists", idx)
            else:
                raise

    def delete_collection(self, collection: str) -> None:
        self._assert_vector_backend_ready()
        idx = self._index_name(collection)
        try:
            self._client.execute_command("FT.DROPINDEX", idx, "DD")
            self._created_indices.discard(collection)
            logger.info("[RedisVectorDB] Dropped index '%s' with documents", idx)
        except Exception as exc:
            self._raise_if_redisearch_missing(exc)
            err_msg = str(exc).lower()
            if "unknown index" in err_msg:
                logger.debug("[RedisVectorDB] Index '%s' does not exist", idx)
            else:
                raise

    def upsert(
        self,
        *,
        collection: str,
        doc_id: str,
        embedding: List[float],
        text: str,
        metadata: Dict[str, Any],
    ) -> None:
        key = self._key(collection, doc_id)
        mapping = {
            "doc_id": doc_id.encode("utf-8"),
            "embedding": _float_list_to_bytes(embedding),
            "_text": text.encode("utf-8"),
            "_metadata": json.dumps(metadata, default=str).encode("utf-8"),
        }
        self._client.hset(key, mapping=mapping)

    def batch_upsert(
        self,
        *,
        collection: str,
        doc_ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> BatchUpsertResult:
        inserted = 0
        updated = 0
        failed = 0

        pipe = self._client.pipeline(transaction=False)
        batch_size = 500

        for i, (did, emb, txt, meta) in enumerate(
            zip(doc_ids, embeddings, texts, metadatas)
        ):
            try:
                key = self._key(collection, did)
                mapping = {
                    "doc_id": did.encode("utf-8"),
                    "embedding": _float_list_to_bytes(emb),
                    "_text": txt.encode("utf-8"),
                    "_metadata": json.dumps(meta, default=str).encode("utf-8"),
                }
                pipe.hset(key, mapping=mapping)
                inserted += 1

                if (i + 1) % batch_size == 0:
                    pipe.execute()
                    pipe = self._client.pipeline(transaction=False)

            except Exception as exc:
                logger.warning(
                    "[RedisVectorDB] batch_upsert failed for doc_id=%s: %s",
                    did, exc,
                )
                failed += 1

        try:
            pipe.execute()
        except Exception as exc:
            raise self._wrap_error(
                exc, operation="batch_upsert", collection=collection,
            ) from exc

        return BatchUpsertResult(inserted=inserted, updated=updated, failed=failed)

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
        self._assert_vector_backend_ready()
        idx = self._index_name(collection)
        blob = _float_list_to_bytes(query_embedding)

        server_filter_expr = _build_redis_server_filter(effective_filters)
        fetch_k = _redis_fetch_limit(top_k, effective_filters)
        query_str = f"({server_filter_expr})=>[KNN {fetch_k} @embedding $BLOB AS _score]"

        try:
            raw = self._client.execute_command(
                "FT.SEARCH", idx,
                query_str,
                "PARAMS", "2", "BLOB", blob,
                "SORTBY", "_score",
                "LIMIT", "0", str(fetch_k),
                "DIALECT", "2",
            )
        except Exception as exc:
            self._raise_if_redisearch_missing(exc)
            raise self._wrap_error(
                exc, operation="search", collection=collection, tenant_id=tenant_id,
            ) from exc

        hits = self._parse_search_results(raw, collection)
        if effective_filters:
            hits = [hit for hit in hits if _matches_redis_filters(hit, effective_filters)]
        self._log_tenant_search(
            tenant_id=tenant_id,
            collection=collection,
            filter_count=len(effective_filters),
            search_mode="vector",
            result_count=min(len(hits), top_k),
        )
        return hits[:top_k]

    def _parse_search_results(self, raw: Any, collection: str) -> List[VectorHit]:
        if not raw or raw[0] == 0:
            return []

        results: List[VectorHit] = []
        i = 1

        while i < len(raw):
            key = raw[i]
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            i += 1

            if i >= len(raw):
                break

            fields = raw[i]
            i += 1

            field_dict: Dict[str, Any] = {}
            if isinstance(fields, list):
                for j in range(0, len(fields), 2):
                    fname = fields[j]
                    fval = fields[j + 1]
                    if isinstance(fname, bytes):
                        fname = fname.decode("utf-8")
                    field_dict[fname] = fval

            doc_id = field_dict.get("doc_id", b"")
            if isinstance(doc_id, bytes):
                doc_id = doc_id.decode("utf-8")
            if not doc_id:
                prefix = f"{self._prefix}{collection}:"
                doc_id = key[len(prefix):] if key.startswith(prefix) else key

            text = field_dict.get("_text", b"")
            if isinstance(text, bytes):
                text = text.decode("utf-8")

            score_raw = field_dict.get("_score", b"1.0")
            if isinstance(score_raw, bytes):
                score_raw = score_raw.decode("utf-8")
            try:
                raw_score = float(score_raw)
                score = max(0.0, 1.0 - raw_score)
            except (ValueError, TypeError):
                score = 0.0

            meta_raw = field_dict.get("_metadata", b"{}")
            if isinstance(meta_raw, bytes):
                meta_raw = meta_raw.decode("utf-8")
            try:
                metadata = json.loads(meta_raw)
            except (json.JSONDecodeError, TypeError):
                metadata = {}

            results.append(VectorHit(
                id=doc_id,
                text=text,
                score=score,
                metadata=metadata,
            ))

        results.sort(key=lambda h: h.score, reverse=True)
        return results

    def exists(
        self,
        *,
        collection: str,
        ids: List[str],
    ) -> List[str]:
        if not ids:
            return []

        pipe = self._client.pipeline(transaction=False)
        for doc_id in ids:
            key = self._key(collection, doc_id)
            pipe.exists(key)

        results = pipe.execute()
        return [
            doc_id
            for doc_id, exists_flag in zip(ids, results)
            if exists_flag
        ]

    def get_by_ids(
        self,
        *,
        collection: str,
        ids: List[str],
        tenant_id: Optional[str] = None,
    ) -> List[VectorHit]:
        if not ids:
            return []

        hits: List[VectorHit] = []
        pipe = self._client.pipeline(transaction=False)
        for doc_id in ids:
            key = self._key(collection, doc_id)
            pipe.hgetall(key)

        results = pipe.execute()

        for doc_id, data in zip(ids, results):
            if not data:
                continue

            text = data.get(b"_text", b"")
            if isinstance(text, bytes):
                text = text.decode("utf-8")

            meta_raw = data.get(b"_metadata", b"{}")
            if isinstance(meta_raw, bytes):
                meta_raw = meta_raw.decode("utf-8")
            try:
                metadata = json.loads(meta_raw)
            except (json.JSONDecodeError, TypeError):
                metadata = {}

            hits.append(VectorHit(
                id=doc_id,
                text=text,
                score=1.0,
                metadata=metadata,
            ))

        return hits

    def count(self, collection: str) -> int:
        self._assert_vector_backend_ready()
        idx = self._index_name(collection)
        try:
            info = self._client.execute_command("FT.INFO", idx)
            for j in range(0, len(info), 2):
                key = info[j]
                if isinstance(key, bytes):
                    key = key.decode("utf-8")
                if key == "num_docs":
                    val = info[j + 1]
                    if isinstance(val, bytes):
                        val = val.decode("utf-8")
                    return int(val)
            return 0
        except Exception as exc:
            self._raise_if_redisearch_missing(exc)
            raise self._wrap_error(
                exc, operation="count", collection=collection,
            ) from exc

    def stats(self, collection: str) -> Dict[str, Any]:
        try:
            doc_count = self.count(collection)
        except Exception:
            doc_count = -1
        return {
            "backend": self.kind,
            "collection": collection,
            "document_count": doc_count,
            "prefix": self._prefix,
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
        updated = 0
        try:
            for doc_id, new_meta in zip(ids, metadatas):
                key = self._key(collection, doc_id)
                raw = self._client.hget(key, "_metadata")
                if raw is None:
                    continue
                existing = json.loads(raw)
                existing.update(new_meta)
                self._client.hset(
                    key, "_metadata",
                    json.dumps(existing, default=str).encode("utf-8"),
                )
                updated += 1
            logger.debug(
                "[RedisVectorDB] update_metadata '%s': %d doc(s)",
                collection, updated,
            )
            return updated
        except Exception as exc:
            raise self._wrap_error(
                exc, operation="update_metadata", collection=collection,
            ) from exc

    def delete(self, *, collection: str, doc_id: str) -> None:
        key = self._key(collection, doc_id)
        self._client.delete(key)

    def delete_many(self, *, collection: str, doc_ids: List[str]) -> int:
        if not doc_ids:
            return 0
        keys = [self._key(collection, did) for did in doc_ids]
        return self._client.delete(*keys)


def _build_redis_server_filter(filters: Optional[Dict[str, Any]]) -> str:
    if not filters:
        return "*"

    parts: List[str] = []
    doc_id_filter = filters.get("doc_id")
    if doc_id_filter is None:
        return "*"

    if isinstance(doc_id_filter, dict):
        values = doc_id_filter.get("$in")
        if values is not None:
            vals = values if isinstance(values, (list, tuple, set)) else [values]
            encoded = "|".join(_escape_redis_tag_value(str(v)) for v in vals)
            parts.append(f"@doc_id:{{{encoded}}}")
        elif "$eq" in doc_id_filter:
            parts.append(f"@doc_id:{{{_escape_redis_tag_value(str(doc_id_filter['$eq']))}}}")
    else:
        parts.append(f"@doc_id:{{{_escape_redis_tag_value(str(doc_id_filter))}}}")

    return " ".join(parts) if parts else "*"


def _redis_fetch_limit(top_k: int, filters: Optional[Dict[str, Any]]) -> int:
    if not filters:
        return top_k
    return min(max(top_k * 5, top_k + 10), 200)


def _matches_redis_filters(hit: VectorHit, filters: Dict[str, Any]) -> bool:
    for key, raw in filters.items():
        candidate = hit.id if key == "doc_id" else hit.metadata.get(key)
        if not _matches_redis_filter_value(candidate, raw):
            return False
    return True


def _matches_redis_filter_value(candidate: Any, raw: Any) -> bool:
    if not isinstance(raw, dict):
        return candidate == raw

    for op, value in raw.items():
        if op == "$eq" and candidate != value:
            return False
        if op == "$ne" and candidate == value:
            return False
        if op == "$in":
            values = value if isinstance(value, (list, tuple, set)) else [value]
            if candidate not in values:
                return False
        if op == "$nin":
            values = value if isinstance(value, (list, tuple, set)) else [value]
            if candidate in values:
                return False
        if op == "$gt" and not (candidate is not None and candidate > value):
            return False
        if op == "$gte" and not (candidate is not None and candidate >= value):
            return False
        if op == "$lt" and not (candidate is not None and candidate < value):
            return False
        if op == "$lte" and not (candidate is not None and candidate <= value):
            return False
    return True


def _escape_redis_tag_value(value: str) -> str:
    escaped = value
    for ch in ("-", "{", "}", "|", " ", ":", "@"):
        escaped = escaped.replace(ch, f"\\{ch}")
    return escaped
