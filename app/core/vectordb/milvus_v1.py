"""
================================================================================
Marketing Advantage AI — Milvus VectorDB Connector
File: app/core/vectordb/milvus_v1.py

Supports:
  - Milvus Standalone (local Docker)          → host + port
  - Milvus Cluster   (on-premise enterprise)  → host + port
  - Zilliz Cloud     (managed Milvus)         → uri + token

Milvus concepts mapped to our interface:
  - collection   → Milvus Collection
  - doc_id       → stored as primary key field "doc_id" (VARCHAR)
  - embedding    → Milvus FLOAT_VECTOR field "embedding"
  - text         → VARCHAR field "_text"
  - metadata     → JSON field "_metadata" (Milvus supports JSON natively)

Milvus is the strongest choice for:
  - Billion-scale vectors on-premise
  - Strict data-residency requirements (India DPDP compliance)
  - Hybrid search (vector + scalar filtering)

Installation:
  pip install pymilvus>=2.4.0

IMPORTANT: This file does NOT touch any existing Chroma ingestion/retrieval
code in app/services/ingestion or app/services/retrieval.
================================================================================
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from app.core.vectordb.base import BaseVectorDB, VectorHit

logger = logging.getLogger(__name__)


# Milvus field name constants (single source of truth)
_FIELD_ID        = "doc_id"
_FIELD_VECTOR    = "embedding"
_FIELD_TEXT      = "_text"
_FIELD_METADATA  = "_metadata"
_INDEX_NAME      = "embedding_idx"
_METRIC_TYPE     = "COSINE"


class MilvusVectorDB(BaseVectorDB):
    """
    Milvus / Zilliz connector.

    Args:
        host:           Milvus host (used for local/standalone/cluster).
        port:           Milvus gRPC port (default 19530).
        uri:            Zilliz Cloud URI (takes precedence over host/port).
        token:          Zilliz Cloud API token or Milvus username:password.
        db_name:        Milvus database (Milvus 2.4+ supports multi-DB).
        alias:          Connection alias — useful when connecting to multiple
                        Milvus instances in the same process.
    """

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 19530,
        uri: Optional[str] = None,         # Zilliz Cloud → provide URI
        token: Optional[str] = None,       # Zilliz token or "user:pass"
        db_name: str = "default",
        alias: str = "default",
    ):
        try:
            from pymilvus import connections
        except ImportError:
            raise ImportError(
                "PyMilvus not installed. Run: pip install pymilvus>=2.4.0"
            )

        from pymilvus import connections

        self._alias = alias
        self._db_name = db_name

        if uri:
            # Zilliz Cloud or remote Milvus with URI
            connections.connect(alias=alias, uri=uri, token=token or "")
        else:
            # Local or cluster Milvus
            connect_kwargs: Dict[str, Any] = {
                "alias": alias,
                "host": host,
                "port": str(port),
                "db_name": db_name,
            }
            if token:
                connect_kwargs["token"] = token
            connections.connect(**connect_kwargs)

        logger.info(
            "[MilvusVectorDB] Connected | uri=%s | host=%s:%d | alias=%s",
            uri or "N/A", host, port, alias,
        )

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _get_collection(self, collection: str):
        from pymilvus import Collection
        return Collection(name=collection, using=self._alias)

    def _schema(self, embedding_dim: int):
        """
        Build Milvus collection schema.
        Fields:
          doc_id       VARCHAR(512) — primary key (string type for UUID support)
          embedding    FLOAT_VECTOR(dim)
          _text        VARCHAR(65535) — raw document text
          _metadata    JSON           — arbitrary metadata
        """
        from pymilvus import CollectionSchema, FieldSchema, DataType

        fields = [
            FieldSchema(
                name=_FIELD_ID,
                dtype=DataType.VARCHAR,
                max_length=512,
                is_primary=True,
                auto_id=False,
                description="Document ID (UUID or hash)",
            ),
            FieldSchema(
                name=_FIELD_VECTOR,
                dtype=DataType.FLOAT_VECTOR,
                dim=int(embedding_dim),
                description="Dense embedding vector",
            ),
            FieldSchema(
                name=_FIELD_TEXT,
                dtype=DataType.VARCHAR,
                max_length=65_535,
                description="Raw document chunk text",
            ),
            FieldSchema(
                name=_FIELD_METADATA,
                dtype=DataType.JSON,
                description="Document metadata as JSON",
            ),
        ]
        return CollectionSchema(
            fields=fields,
            description="Marketing Advantage AI — document chunks",
            enable_dynamic_field=True,     # allows extra fields later
        )

    def _create_index(self, collection) -> None:
        """
        Create HNSW index on the embedding field after collection creation.
        HNSW is the best choice for in-memory ANN on Milvus.
        """
        index_params = {
            "metric_type": _METRIC_TYPE,
            "index_type":  "HNSW",
            "params": {"M": 16, "efConstruction": 256},
        }
        collection.create_index(
            field_name=_FIELD_VECTOR,
            index_params=index_params,
            index_name=_INDEX_NAME,
        )
        logger.info("[MilvusVectorDB] HNSW index created on '%s'.", collection.name)

    # -------------------------------------------------------------------
    # BaseVectorDB implementation
    # -------------------------------------------------------------------

    @property
    def kind(self) -> str:
        return "milvus"

    def health_check(self) -> bool:
        try:
            from pymilvus import utility
            utility.list_collections(using=self._alias)
            return True
        except Exception as exc:
            logger.warning("[MilvusVectorDB] health_check failed: %s", exc)
            return False

    def ensure_collection(self, collection: str, *, embedding_dim: int) -> None:
        """
        Create a Milvus collection + HNSW index if it doesn't exist.
        Load the collection into memory (required before search in Milvus).
        """
        from pymilvus import Collection, utility

        if utility.has_collection(collection, using=self._alias):
            # Already exists — just make sure it's loaded
            self._get_collection(collection).load()
            logger.debug(
                "[MilvusVectorDB] Collection '%s' exists — loaded.", collection
            )
            return

        schema = self._schema(embedding_dim)
        col = Collection(
            name=collection,
            schema=schema,
            using=self._alias,
        )
        self._create_index(col)
        col.load()
        logger.info(
            "[MilvusVectorDB] Created + loaded collection '%s' (dim=%d).",
            collection, embedding_dim,
        )

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
        Upsert a document vector.

        Milvus doesn't have native upsert for primary-key string fields
        in all versions, so we: delete if exists → insert.
        """
        col = self._get_collection(collection)

        # Delete existing record to simulate upsert
        col.delete(expr=f'{_FIELD_ID} == "{doc_id}"')

        # Serialize metadata to JSON string for storage
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

        # Clamp text to Milvus VARCHAR max length
        safe_text = str(text)[:65_530] if text else ""

        col.insert(
            data={
                _FIELD_ID:       [doc_id],
                _FIELD_VECTOR:   [embedding],
                _FIELD_TEXT:     [safe_text],
                _FIELD_METADATA: [metadata_json],
            }
        )
        col.flush()                         # persist to disk

    def search(
        self,
        *,
        collection: str,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorHit]:
        """
        Vector ANN search in Milvus.

        Milvus supports hybrid search via an expression string.
        We convert a flat metadata dict into a Milvus boolean expr:
          {"client_id": "abc", "lang": "en"}
          → '_metadata["client_id"] == "abc" && _metadata["lang"] == "en"'
        """
        col = self._get_collection(collection)

        # Build Milvus JSON-field expression
        expr = None
        if filters:
            clauses = []
            for k, v in filters.items():
                val_str = f'"{v}"' if isinstance(v, str) else str(v).lower()
                clauses.append(f'{_FIELD_METADATA}["{k}"] == {val_str}')
            expr = " && ".join(clauses)

        search_params = {
            "metric_type": _METRIC_TYPE,
            "params": {"ef": min(top_k * 4, 512)},     # ef >= top_k for accuracy
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
            hits.append(
                VectorHit(
                    id=str(result.id),
                    text=str(result.entity.get(_FIELD_TEXT) or ""),
                    score=float(result.score),
                    metadata=meta,
                )
            )
        return hits

    def exists(self, *, collection: str, ids: List[str]) -> List[str]:
        return []   # TODO: implement per-backend


    def delete(self, *, collection: str, doc_id: str) -> None:
        col = self._get_collection(collection)
        col.delete(expr=f'{_FIELD_ID} == "{doc_id}"')
        col.flush()

    def __del__(self):
        """Cleanly disconnect Milvus connection on teardown."""
        try:
            from pymilvus import connections
            connections.disconnect(self._alias)
        except Exception:
            pass
