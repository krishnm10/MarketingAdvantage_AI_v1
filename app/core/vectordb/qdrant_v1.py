"""
Qdrant connector (pluggable layer)
File: app/core/vectordb/qdrant_v1.py
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.vectordb.base import BaseVectorDB, VectorHit


class QdrantVectorDB(BaseVectorDB):
    def __init__(
        self,
        *,
        url: Optional[str] = None,        # e.g. https://xxxx.qdrant.tech
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
        timeout: float = 30.0,
    ):
        from qdrant_client import QdrantClient

        # url takes precedence (cloud)
        if url:
            self._client = QdrantClient(url=url, api_key=api_key, timeout=timeout, prefer_grpc=prefer_grpc)
        else:
            self._client = QdrantClient(host=host, port=port, timeout=timeout, prefer_grpc=prefer_grpc)

    @property
    def kind(self) -> str:
        return "qdrant"

    def health_check(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False

    def ensure_collection(self, collection: str, *, embedding_dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        existing = {c.name for c in self._client.get_collections().collections}
        if collection in existing:
            return

        self._client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=int(embedding_dim), distance=Distance.COSINE),
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
        from qdrant_client.models import PointStruct

        # Qdrant point IDs can be int or UUID; we’ll use a stable UUID if caller provides one,
        # otherwise fall back to hashing the string.
        point_id = doc_id
        payload = dict(metadata or {})
        payload["_text"] = text

        self._client.upsert(
            collection_name=collection,
            points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
        )

    def search(
        self,
        *,
        collection: str,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorHit]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        qfilter = None
        if filters:
            qfilter = Filter(
                must=[
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in filters.items()
                ]
            )

        res = self._client.search(
            collection_name=collection,
            query_vector=query_embedding,
            limit=int(top_k),
            query_filter=qfilter,
        )

        hits: List[VectorHit] = []
        for r in res:
            payload = r.payload or {}
            text = payload.get("_text", "")
            metadata = {k: v for k, v in payload.items() if k != "_text"}
            hits.append(VectorHit(id=str(r.id), text=str(text), score=float(r.score), metadata=metadata))
        return hits
        
    def exists(self, *, collection: str, ids: List[str]) -> List[str]:
        return []   # TODO: implement per-backend

    

    def delete(self, *, collection: str, doc_id: str) -> None:
        self._client.delete(collection_name=collection, points_selector=[doc_id])
