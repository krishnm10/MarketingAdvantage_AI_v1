"""
Resolved pipeline identity from ClientConfig (JSON merges only).

Pipeline semantics (embedder, vectordb, collection, retrieval mode, LLM, reranker)
must not be driven by deprecated MAI_* environment variables — use
`get_client_config(client_id)` and this helper for dashboards and APIs.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Env vars that used to overlay JSON — forbidden for pipeline semantics going forward.
DEPRECATED_PIPELINE_ENV_VARS: tuple[str, ...] = (
    "MAI_EMBEDDER",
    "MAI_VECTORDB",
    "MAI_COLLECTION",
    "MAI_SEARCH_MODE",
    "MAI_LLM",
    "MAI_RERANKER",
    "MAI_RERANKER_MODEL",
)


def warn_if_deprecated_pipeline_env_set() -> None:
    """Log once-worth warning if obsolete pipeline env vars are present (ignored by resolver)."""
    found = [k for k in DEPRECATED_PIPELINE_ENV_VARS if os.getenv(k)]
    if not found:
        return
    logger.warning(
        "[pipeline_runtime] Deprecated pipeline env vars are set but ignored "
        "(configure embedder/vectordb/retrieval in Client JSON): %s",
        ", ".join(found),
    )


def get_pipeline_identity(client_id: str) -> Dict[str, Any]:
    """Return a JSON-serializable summary from the merged ClientConfig (no env overlay for semantics)."""
    from app.core.config.client_config_resolver import get_client_config

    cfg = get_client_config(client_id, apply_env=True)
    llm_provider = ""
    llm_model = ""
    if cfg.llm and cfg.llm.single:
        llm_provider = cfg.llm.single.type.value
        llm_model = cfg.llm.single.model
    rerank = ""
    rerank_model = ""
    if cfg.reranker:
        rerank = cfg.reranker.type.value
        rerank_model = cfg.reranker.model or ""
    embedder_model = ""
    if getattr(cfg, "embedder", None) is not None:
        emb = cfg.embedder
        sub = getattr(emb, emb.type.value, None)
        embedder_model = getattr(sub, "model", "") if sub else ""
    chroma_pd = ""
    vectordb_subconfig: Dict[str, Any] = {}
    try:
        vtype = cfg.vectordb.type.value
        sub_key = vtype
        sub = getattr(cfg.vectordb, sub_key, None)
        if sub is not None:
            vectordb_subconfig = sub.model_dump(mode="json")
        if vtype == "chroma" and cfg.vectordb.chroma:
            chroma_pd = (cfg.vectordb.chroma.persist_directory or "") or ""
    except Exception:
        chroma_pd = ""
        vectordb_subconfig = {}
    ing = getattr(cfg, "ingestion", None)
    tok = getattr(cfg, "tokenization", None)
    cdy = getattr(cfg, "celery_dispatch", None)
    return {
        "client_id": client_id,
        "client_name": getattr(cfg, "client_name", "") or "",
        "vectordb": cfg.vectordb.type.value,
        "collection": cfg.vectordb.collection,
        "chroma_persist_directory": chroma_pd,
        "vectordb_subconfig": vectordb_subconfig,
        "embedder": cfg.embedder.type.value,
        "embedder_model": embedder_model,
        "llm": llm_provider,
        "llm_model": llm_model,
        "search_mode": cfg.retrieval.search_mode.value,
        "reranker": rerank or "none",
        "reranker_model": rerank_model,
        "transport": os.getenv("MAI_VECTOR_TRANSPORT", "auto"),
        "ingestion": ing.model_dump(mode="json") if ing is not None else {},
        "tokenization": tok.model_dump(mode="json") if tok is not None else {},
        "celery_dispatch": cdy.model_dump(mode="json") if cdy is not None else {},
    }


def list_deprecated_pipeline_env_set() -> List[str]:
    return [k for k in DEPRECATED_PIPELINE_ENV_VARS if os.getenv(k)]
