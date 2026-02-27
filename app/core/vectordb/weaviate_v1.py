"""
================================================================================
Marketing Advantage AI — Weaviate VectorDB Connector
File: app/core/vectordb/weaviate_v1.py

Supports:
  - Weaviate Cloud Services (WCS)      → pass url + api_key
  - Local Docker                        → pass url only (http://localhost:8080)
  - Local Embedded Weaviate             → pass embedded=True

Weaviate concepts mapped to our interface:
  - collection   → Weaviate "Class" (must start uppercase)
  - doc_id       → Weaviate object UUID
  - metadata     → Weaviate properties
  - text         → stored as property "_text" inside the object

Installation:
  pip install weaviate-client>=4.0.0

IMPORTANT: This file does NOT touch or alter any existing Chroma
ingestion/retrieval code in app/services/ingestion or app/services/retrieval.
================================================================================
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.core.vectordb.base import BaseVectorDB, VectorHit

logger = logging.getLogger(__name__)


class WeaviateVectorDB(BaseVectorDB):
    """
    Weaviate v4 connector.

    Weaviate stores vectors alongside structured properties.
    Each "class" in Weaviate is our "collection".

    Args:
        url:           Weaviate instance URL (e.g., https://xyz.weaviate.network)
        api_key:       Weaviate Cloud API key (leave None for local)
        openai_key:    Optional OpenAI key if using Weaviate's built-in vectoriser
                       (leave None — we bring our own embeddings)
        additional_headers: Any extra HTTP headers e.g. cohere/huggingface keys
        embedded:      Set True to spin up an in-process embedded Weaviate
                       (useful for local dev/testing without Docker)
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
            from weaviate.auth import AuthApiKey
        except ImportError:
            raise ImportError(
                "Weaviate client not installed. Run: pip install weaviate-client>=4.0.0"
            )

        if embedded:
            # Embedded Weaviate: zero infra, great for dev
            import weaviate.embedded as _emb
            self._client = weaviate.WeaviateClient(
                embedded_options=_emb.EmbeddedOptions()
            )
            self._client.connect()
        else:
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

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------
    @staticmethod
    def _class_name(collection: str) -> str:
        """
        Weaviate requires class names to start with an uppercase letter.
        e.g. 'marketing_docs' → 'MarketingDocs'
        """
        return "".join(word.capitalize() for word in collection.replace("-", "_").split("_"))

    # -------------------------------------------------------------------
    # BaseVectorDB implementation
    # -------------------------------------------------------------------

    @property
    def kind(self) -> str:
        return "weaviate"

    def health_check(self) -> bool:
        try:
            return self._client.is_ready()
        except Exception as exc:
            logger.warning("[WeaviateVectorDB] health_check failed: %s", exc)
            return False

    def ensure_collection(self, collection: str, *, embedding_dim: int) -> None:
        """
        Create a Weaviate class (collection) only if it doesn't exist yet.
        We disable Weaviate's built-in vectoriser (none) because we always
        supply our own pre-computed embeddings.
        """
        import weaviate.classes.config as wc

        cls_name = self._class_name(collection)

        # Skip creation if class already exists
        existing_classes = {
            cls.name for cls in self._client.collections.list_all()
        }
        if cls_name in existing_classes:
            logger.debug(
                "[WeaviateVectorDB] Collection '%s' already exists — skipping.",
                cls_name,
            )
            return

        # Create class with a single text property for "_text"
        self._client.collections.create(
            name=cls_name,
            vectorizer_config=wc.Configure.Vectorizer.none(),  # BYO embeddings
            vector_index_config=wc.Configure.VectorIndex.hnsw(
                distance_metric=wc.VectorDistances.COSINE,
            ),
            properties=[
                wc.Property(name="_text", data_type=wc.DataType.TEXT),
            ],
        )
        logger.info("[WeaviateVectorDB] Created collection '%s'", cls_name)

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
        Insert or overwrite a document.
        Weaviate v4 upsert: insert_many with override UUID.
        """
        cls_name = self._class_name(collection)
        col = self._client.collections.get(cls_name)

        # Build flat properties dict (all metadata + _text)
        properties = dict(metadata or {})
        properties["_text"] = text

        # Upsert = delete if exists then re-insert
        try:
            col.data.delete_by_id(doc_id)
        except Exception:
            pass  # Object not found — that's fine

        col.data.insert(
            properties=properties,
            vector=embedding,
            uuid=doc_id,
        )

    def search(
        self,
        *,
        collection: str,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorHit]:
        """
        Vector similarity search using pre-computed query embedding.
        Applies metadata filters if provided (Weaviate 'where' filter).
        """
        import weaviate.classes.query as wq
        import weaviate.classes.filters as wf

        cls_name = self._class_name(collection)
        col = self._client.collections.get(cls_name)

        # Build where filter from flat key:value dict
        where_filter = None
        if filters:
            conditions = [
                wf.Filter.by_property(k).equal(v)
                for k, v in filters.items()
            ]
            # Combine with AND if multiple filters
            where_filter = (
                conditions[0]
                if len(conditions) == 1
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
            text = str(props.pop("_text", ""))
            score = float(obj.metadata.certainty or 0.0) if obj.metadata else 0.0
            hits.append(
                VectorHit(
                    id=str(obj.uuid),
                    text=text,
                    score=score,
                    metadata=props,
                )
            )
        return hits
        
    def exists(self, *, collection: str, ids: List[str]) -> List[str]:
        return []   # TODO: implement per-backend
    
    

    def delete(self, *, collection: str, doc_id: str) -> None:
        cls_name = self._class_name(collection)
        col = self._client.collections.get(cls_name)
        col.data.delete_by_id(doc_id)

    def __del__(self):
        """Gracefully close client connection on teardown."""
        try:
            self._client.close()
        except Exception:
            pass
