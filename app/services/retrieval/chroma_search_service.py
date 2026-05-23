"""
ChromaDB Search Service — tenant-scoped via ClientConfig + ingestion pipeline.

All entrypoints require client_id (tenant slug). No process-wide env singletons.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional, Tuple

from app.core.config.client_config_resolver import get_client_config
from app.core.config.client_config_schema import VectorDBType
from app.core.vectordb.base import VectorHit
from app.services.ingestion.ingestion_service_v2 import _get_ingestion_pipeline_for_client
from app.utils.logger import log_debug, log_info, log_warning
from app.utils.tenant_storage_uuid import storage_uuid_str_for_vectordb_metadata


def _require_client_id(client_id: Optional[str]) -> str:
    if not client_id or not str(client_id).strip():
        raise ValueError(
            "client_id is required for chroma_search_service operations. "
            "Pass the tenant slug from ClientConfig."
        )
    return str(client_id).strip()


def _resolve_tenant_chroma(client_id: str):
    cfg = get_client_config(client_id)
    if cfg.vectordb.type != VectorDBType.CHROMA:
        raise ValueError(
            f"chroma_search_service: client_id={client_id!r} uses vectordb "
            f"{cfg.vectordb.type.value}, not chroma"
        )
    coll = (cfg.vectordb.collection or "").strip()
    if not coll:
        raise ValueError(
            f"vectordb.collection is required for client_id={client_id!r}"
        )
    pipeline = _get_ingestion_pipeline_for_client(client_id)
    ch = cfg.vectordb.chroma
    path = (ch.persist_directory if ch else None) or ""
    return pipeline.vectordb, coll, path


def get_chroma_collection(client_id: str):
    """
    Return raw chromadb.Collection for the tenant (Chroma backends only).

    Prefer BaseVectorDB.search() via semantic_search() for tenant-filtered reads.
    """
    slug = _require_client_id(client_id)
    vdb, coll_name, _path = _resolve_tenant_chroma(slug)
    return vdb._get_collection(coll_name)  # type: ignore[attr-defined]


def reset_chroma_collection() -> None:
    """Test-only: clear ingestion pipeline cache for chroma clients."""
    if not os.getenv("PYTEST_CURRENT_TEST"):
        raise RuntimeError(
            "reset_chroma_collection() is only available during pytest runs."
        )
    from app.services.ingestion.ingestion_service_v2 import clear_ingestion_pipeline_cache

    clear_ingestion_pipeline_cache()


async def semantic_search(
    query_embedding: List[float],
    client_id: str,
    limit: int = 200,
    where: Optional[dict] = None,
    where_document: Optional[dict] = None,
) -> List[Tuple[str, float]]:
    """
    Tenant-isolated semantic search via BaseVectorDB.search().
    """
    slug = _require_client_id(client_id)
    vdb, coll_name, _path = _resolve_tenant_chroma(slug)
    tenant_key = storage_uuid_str_for_vectordb_metadata(slug)
    filters: Dict[str, Any] = dict(where or {})
    if where_document is not None:
        log_warning(
            "[ChromaSearch] where_document is not supported via BaseVectorDB; "
            "ignored for tenant-scoped search."
        )

    loop = asyncio.get_running_loop()

    def _search() -> List[VectorHit]:
        return vdb.search(
            collection=coll_name,
            query_embedding=query_embedding,
            top_k=limit,
            tenant_id=tenant_key,
            filters=filters or None,
        )

    try:
        hits = await loop.run_in_executor(None, _search)
    except Exception as e:
        log_warning(f"[ChromaSearch] Search failed for client_id={slug}: {e}")
        return []

    log_debug(f"[ChromaSearch] Found {len(hits)} results for tenant={slug}")

    results: List[Tuple[str, float]] = []
    for hit in hits:
        doc_id = str(hit.id)
        score = max(0.0, min(1.0, float(hit.score)))
        results.append((doc_id, score))

    if results:
        log_debug(
            f"[ChromaSearch] Top result: id={results[0][0][:16]}..., "
            f"score={results[0][1]:.4f}"
        )

    return results


def health_check(client_id: str) -> dict:
    """Check ChromaDB health for a specific tenant."""
    slug = _require_client_id(client_id)
    try:
        vdb, coll_name, path = _resolve_tenant_chroma(slug)
        healthy = vdb.health_check()
        if not healthy:
            return {
                "status": "unhealthy",
                "client_id": slug,
                "error": "vectordb.health_check() returned False",
            }
        collection = get_chroma_collection(slug)
        count = collection.count()
        return {
            "status": "healthy",
            "collection_name": coll_name,
            "total_vectors": count,
            "storage_path": path,
            "client_id": slug,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "client_id": slug,
            "error": str(e),
        }
