"""
================================================================================
Marketing Advantage AI — Pinecone VectorDB Connector
File: app/core/vectordb/pinecone_v1.py

Supports:
  - Pinecone Serverless  (recommended for enterprise SaaS)
  - Pinecone Pod-based   (for dedicated throughput)

Pinecone concepts mapped to our interface:
  - collection   → Pinecone "Index"  (one index per collection / client)
  - doc_id       → Pinecone vector ID (string)
  - metadata     → Pinecone vector metadata dict (max 40KB/record)
  - namespace    → optional Pinecone namespace for multi-tenancy within 1 index
  - text         → stored as metadata["_text"]

Installation:
  pip install pinecone-client>=3.0.0

IMPORTANT: This file does NOT touch any existing Chroma ingestion/retrieval
code in app/services/ingestion or app/services/retrieval.
================================================================================
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.core.vectordb.base import BaseVectorDB, VectorHit

logger = logging.getLogger(__name__)


class PineconeVectorDB(BaseVectorDB):
    """
    Pinecone Serverless/Pod connector.

    Args:
        api_key:        Pinecone API key (required).
        index_name:     Name of the Pinecone index (required, unique per client).
        namespace:      Pinecone namespace for multi-tenancy within one index.
                        Useful when one Pinecone index holds multiple clients.
        embedding_dim:  Vector dimension (must match your transformer output).
        metric:         Similarity metric — "cosine" | "dotproduct" | "euclidean"
        cloud:          Cloud provider for serverless — "aws" | "gcp" | "azure"
        region:         Cloud region e.g. "us-east-1", "us-central1"
        pod_type:       Set to use Pod-based instead of Serverless
                        e.g. "p1.x1", "s1.x1" (leave None for Serverless)
    """

    def __init__(
        self,
        *,
        api_key: str,
        index_name: str,
        namespace: str = "default",
        embedding_dim: int,                     # REQUIRED — no guessing dim
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        pod_type: Optional[str] = None,         # None → Serverless
    ):
        try:
            from pinecone import Pinecone, ServerlessSpec, PodSpec
        except ImportError:
            raise ImportError(
                "Pinecone client not installed. Run: pip install pinecone-client>=3.0.0"
            )

        self._index_name = index_name
        self._namespace = namespace
        self._embedding_dim = embedding_dim
        self._metric = metric
        self._cloud = cloud
        self._region = region
        self._pod_type = pod_type
        self._pc = Pinecone(api_key=api_key)
        self._index = None                      # lazy init — see _get_index()

        logger.info(
            "[PineconeVectorDB] Initialized | index=%s | namespace=%s | dim=%d",
            index_name, namespace, embedding_dim,
        )

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _get_index(self):
        """Lazy-load the Pinecone Index object (avoids network call at init)."""
        if self._index is None:
            self._index = self._pc.Index(self._index_name)
        return self._index

    def _build_serverless_spec(self):
        from pinecone import ServerlessSpec
        return ServerlessSpec(cloud=self._cloud, region=self._region)

    def _build_pod_spec(self):
        from pinecone import PodSpec
        return PodSpec(environment=self._region, pod_type=self._pod_type)

    # -------------------------------------------------------------------
    # BaseVectorDB implementation
    # -------------------------------------------------------------------

    @property
    def kind(self) -> str:
        return "pinecone"

    def health_check(self) -> bool:
        try:
            self._get_index().describe_index_stats()
            return True
        except Exception as exc:
            logger.warning("[PineconeVectorDB] health_check failed: %s", exc)
            return False

    def ensure_collection(self, collection: str, *, embedding_dim: int) -> None:
        """
        In Pinecone, each 'collection' maps to an index.
        This creates the index if it doesn't exist.

        Note: Pinecone index creation is async (~60s for Serverless).
        We poll until it's ready (up to 120s).
        """
        import time

        existing = {idx.name for idx in self._pc.list_indexes()}
        if self._index_name in existing:
            logger.debug(
                "[PineconeVectorDB] Index '%s' already exists — skipping.",
                self._index_name,
            )
            return

        spec = (
            self._build_pod_spec()
            if self._pod_type
            else self._build_serverless_spec()
        )

        self._pc.create_index(
            name=self._index_name,
            dimension=int(embedding_dim),
            metric=self._metric,
            spec=spec,
        )
        logger.info(
            "[PineconeVectorDB] Creating index '%s' (dim=%d, metric=%s)...",
            self._index_name, embedding_dim, self._metric,
        )

        # Wait until index is ready (Serverless takes ~30-60s on first create)
        deadline = time.time() + 120
        while time.time() < deadline:
            status = self._pc.describe_index(self._index_name).status
            if status.get("ready"):
                logger.info(
                    "[PineconeVectorDB] Index '%s' is ready.", self._index_name
                )
                return
            time.sleep(5)

        raise TimeoutError(
            f"[PineconeVectorDB] Index '{self._index_name}' not ready after 120s."
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
        Upsert a vector into Pinecone.
        Pinecone upserts are atomic — same ID = overwrite.
        """
        payload = dict(metadata or {})
        payload["_text"] = text             # store text inside metadata

        self._get_index().upsert(
            vectors=[(doc_id, embedding, payload)],
            namespace=self._namespace,
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
        Semantic search in Pinecone.
        Pinecone metadata filters use MongoDB-style syntax:
          {"field": {"$eq": value}} or just {"field": value}
        We auto-wrap plain dict values to {"$eq": value}.
        """
        # Build Pinecone-style filter from flat dict
        pinecone_filter = None
        if filters:
            pinecone_filter = {
                k: (v if isinstance(v, dict) else {"$eq": v})
                for k, v in filters.items()
            }

        res = self._get_index().query(
            vector=query_embedding,
            top_k=int(top_k),
            namespace=self._namespace,
            filter=pinecone_filter,
            include_metadata=True,
        )

        hits: List[VectorHit] = []
        for match in res.get("matches", []):
            meta = dict(match.get("metadata") or {})
            text = str(meta.pop("_text", ""))
            hits.append(
                VectorHit(
                    id=str(match["id"]),
                    text=text,
                    score=float(match.get("score", 0.0)),
                    metadata=meta,
                )
            )
        return hits
        
    
    def exists(self, *, collection: str, ids: List[str]) -> List[str]:
        return []   # TODO: implement per-backend

    
    def delete(self, *, collection: str, doc_id: str) -> None:
        self._get_index().delete(ids=[doc_id], namespace=self._namespace)
