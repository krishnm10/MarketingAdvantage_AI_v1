"""
Chroma connector (pluggable layer)
File: app/core/vectordb/chroma_v1.py

This is additive. Your existing Chroma ingestion/retrieval modules remain unchanged. [cite:8]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.vectordb.base import BaseVectorDB, VectorHit


class ChromaVectorDB(BaseVectorDB):
    def __init__(
        self,
        *,
        persist_directory: str,
        anonymized_telemetry: bool = False,
    ):
        import chromadb
        from chromadb.config import Settings

        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=anonymized_telemetry),
        )

    @property
    def kind(self) -> str:
        return "chroma"

    def health_check(self) -> bool:
        try:
            self._client.heartbeat()
            return True
        except Exception:
            return False

    def ensure_collection(self, collection: str, *, embedding_dim: int) -> None:
        # Chroma doesn't require dim at creation time
        self._client.get_or_create_collection(collection)

    def upsert(
        self,
        *,
        collection: str,
        doc_id: str,
        embedding: List[float],
        text: str,
        metadata: Dict[str, Any],
    ) -> None:
        col = self._client.get_or_create_collection(collection)
        col.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )

    def search(
        self,
        *,
        collection: str,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorHit]:
        col = self._client.get_or_create_collection(collection)
        res = col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters or None,
            include=["documents", "metadatas", "distances"],
        )

        hits: List[VectorHit] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        for i in range(len(ids)):
            # distance -> similarity (common convention)
            score = 1.0 - float(dists[i]) if dists and dists[i] is not None else 0.0
            hits.append(VectorHit(id=str(ids[i]), text=str(docs[i]), score=score, metadata=metas[i] or {}))
        return hits
    
    
    def exists(self, *, collection: str, ids: List[str]) -> List[str]:
        """Return which of the given IDs are already stored in Chroma."""
        if not ids:
            return []
        col = self._client.get_or_create_collection(collection)   # ← _client not self
        try:
            result = col.get(ids=ids)
            return result.get("ids", []) if isinstance(result, dict) else []
        except Exception:
            return []





    def delete(self, *, collection: str, doc_id: str) -> None:
        col = self._client.get_or_create_collection(collection)
        col.delete(ids=[doc_id])
