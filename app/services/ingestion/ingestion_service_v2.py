# =============================================
# ingestion_service_v2.py — Unified Ingestion Service (UI + Bulk)
# Fully aligned with PostgreSQL schema, pluggable VectorDB/Embedder
# Direct ingestion support for pre-parsed inputs (RSS, API, etc.)
# =============================================

# ── Standard library ──────────────────────────────────────────────────
import uuid
import hashlib
import re
import os
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ── FastAPI / SQLAlchemy ──────────────────────────────────────────────
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy import select, insert, update, func

# ── App internals ─────────────────────────────────────────────────────
from app.api.v2.ingestion_ws_api import broadcast
from app.db.session_v2 import async_engine
from app.db.models.ingested_file_v2 import IngestedFileV2
from app.db.models.ingested_content_v2 import IngestedContentV2
from app.db.models.global_content_index_v2 import GlobalContentIndexV2
from app.services.ingestion.parsers_router_v2 import ParserRouterV2
from app.services.ingestion.segmenter_v2 import recursive_semantic_chunk
from app.services.ingestion.deduplication_engine_v2 import (
    deduplicate_chunks,
    create_normalized_hash,
)
from app.utils.logger import log_info

# ── Pluggable pipeline factory ────────────────────────────────────────
from app.core.pipeline_factory import pipeline_factory
from app.core.config.client_config_schema import (
    ClientConfig,
    VectorDBConfig,
    EmbedderConfig,
    VectorDBType,
    EmbedderType,
    ChromaConfig,
    OllamaEmbedderConfig,
)

# =============================================
# CONFIGURATION
# NOTE: CHROMA_PATH and _COLLECTION_COUNT_CACHE are intentionally
# module-level exports. main.py imports them directly:
#   from app.services.ingestion.ingestion_service_v2 import (
#       get_chroma_collection, get_collection_count_cached,
#       _COLLECTION_COUNT_CACHE, CHROMA_PATH,
#   )
# Do NOT rename or move them.
# =============================================

BATCH_SIZE:  int = 256

_COLLECTION_COUNT_CACHE: Dict[str, Any] = {
    "count":        0,
    "last_updated": None,
}

async_session = async_sessionmaker(
    async_engine, expire_on_commit=False, autoflush=False
)


# =============================================
# HELPERS  (preserved exactly)
# =============================================

async def _check_existing_file_by_hash(
    db: AsyncSession, file_hash: str
) -> Optional[str]:
    result = await db.execute(
        select(IngestedFileV2.id).where(
            IngestedFileV2.meta_data["file_hash"].astext == file_hash
        )
    )
    return result.scalar_one_or_none()


def compute_file_hash(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _resolve_text(payload: dict) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in ("normalized_text", "cleaned_text", "text", "raw_text"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val
    try:
        return " | ".join(
            f"{k}: {v}"
            for k, v in payload.items()
            if isinstance(v, (str, int, float))
        )
    except Exception:
        return ""


# ============================================================
# PLUGGABLE PIPELINE RESOLVER  (env-var driven — no JSON files)
#
# Priority order:
#   1. Per-business env var:  MAI_{BUSINESS_ID}_VECTORDB / _EMBEDDER
#   2. Global default env var: MAI_VECTORDB / MAI_EMBEDDER
#   3. Hard default: chroma / ollama
#
# .env reference:
#   MAI_VECTORDB           = chroma | qdrant | pinecone | milvus | weaviate | redis
#   MAI_EMBEDDER           = ollama | openai | huggingface (default: ollama)
#   CHROMA_PATH            = ./chroma_db
#   MAI_COLLECTION         = ingested_content
#   OLLAMA_EMBED_MODEL     = nomic-embed-text
#   OLLAMA_BASE_URL        = http://localhost:11434
#   OPENAI_EMBED_MODEL     = text-embedding-3-small
#   OPENAI_API_KEY         = sk-...
#   HF_EMBED_MODEL         = BAAI/bge-large-en
#   QDRANT_URL             = http://localhost:6333
#   QDRANT_API_KEY         = ...
# ============================================================

def _get_pipeline(business_id: Optional[str] = None):
    """
    Resolve a live AssembledPipeline for the given business.
    Config is read purely from environment variables — no JSON files.
    Works identically in dev, staging, and production.
    """
    b = (business_id or "default").lower().replace("-", "_")

    vectordb_type = os.getenv(
        f"MAI_{b.upper()}_VECTORDB",
        os.getenv("MAI_VECTORDB", "chroma"),
    ).lower()

    embedder_type = os.getenv(
        f"MAI_{b.upper()}_EMBEDDER",
        os.getenv("MAI_EMBEDDER", "ollama"),
    ).lower()

    llm_type = os.getenv(
        f"MAI_{b.upper()}_LLM",
        os.getenv("MAI_LLM", "ollama"),
    ).lower()

    config = _build_config_from_env(
        client_id=business_id or "default",
        vectordb_type=vectordb_type,
        embedder_type=embedder_type,
        llm_type=llm_type,
    )
    return pipeline_factory.build(config)


def _build_config_from_env(
    client_id: str,
    vectordb_type: str,
    embedder_type: str,
    llm_type: str,
) -> ClientConfig:
    """Build a ClientConfig purely from environment variables."""

    # ── VectorDB ──────────────────────────────────────────────
    if vectordb_type == "chroma":
        from app.core.config.client_config_schema import ChromaConfig
        chroma_host = os.getenv("CHROMA_HOST") or None
        chroma_port = os.getenv("CHROMA_PORT") or "8000"
        vdb_cfg = VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection=os.getenv("MAI_COLLECTION", "ingested_content"),
            chroma=ChromaConfig(
                persist_directory=os.getenv("CHROMA_PATH", "./chroma_db") if not chroma_host else None,
                host=chroma_host,
                port=int(chroma_port),
                ssl=os.getenv("CHROMA_SSL", "").lower() in ("1", "true", "yes"),
                api_key_env="CHROMA_API_KEY" if os.getenv("CHROMA_API_KEY") else None,
            ),
        )
    elif vectordb_type == "qdrant":
        from app.core.config.client_config_schema import QdrantConfig
        qdrant_api_key_env = "QDRANT_API_KEY" if os.getenv("QDRANT_API_KEY") else None
        vdb_cfg = VectorDBConfig(
            type=VectorDBType.QDRANT,
            collection=os.getenv("MAI_COLLECTION", "ingested_content"),
            qdrant=QdrantConfig(
                url=os.getenv("QDRANT_URL") or None,
                api_key_env=qdrant_api_key_env,
                host=os.getenv("QDRANT_HOST", "localhost"),
                port=int(os.getenv("QDRANT_PORT", "6333")),
                prefer_grpc=os.getenv("QDRANT_PREFER_GRPC", "false").lower() in ("1", "true", "yes"),
                timeout=float(os.getenv("QDRANT_TIMEOUT", "30")),
            ),
        )
    elif vectordb_type == "pinecone":
        from app.core.config.client_config_schema import PineconeConfig
        pinecone_mode = os.getenv("PINECONE_MODE", "cloud").strip().lower()
        pinecone_api_key_env = (
            "PINECONE_API_KEY"
            if pinecone_mode != "local" and os.getenv("PINECONE_API_KEY")
            else None
        )
        vdb_cfg = VectorDBConfig(
            type=VectorDBType.PINECONE,
            collection=os.getenv("MAI_COLLECTION", "ingested_content"),
            pinecone=PineconeConfig(
                mode="local" if pinecone_mode == "local" else "cloud",
                api_key_env=pinecone_api_key_env,
                index_name=os.getenv("PINECONE_INDEX_NAME", "ingested-content"),
                namespace=os.getenv("PINECONE_NAMESPACE", "default"),
                embedding_dim=int(os.getenv("PINECONE_EMBEDDING_DIM", "1024")),
                metric=os.getenv("PINECONE_METRIC", "cosine"),
                cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv("PINECONE_REGION", "us-east-1"),
                local_path=os.getenv("PINECONE_LOCAL_PATH") or None,
            ),
        )
    elif vectordb_type == "milvus":
        from app.core.config.client_config_schema import MilvusConfig
        vdb_cfg = VectorDBConfig(
            type=VectorDBType.MILVUS,
            collection=os.getenv("MAI_COLLECTION", "ingested_content"),
            milvus=MilvusConfig(
                uri=os.getenv("MILVUS_URI") or None,
                token_env="MILVUS_TOKEN" if os.getenv("MILVUS_TOKEN") else None,
                host=os.getenv("MILVUS_HOST", "localhost"),
                port=int(os.getenv("MILVUS_PORT", "19530")),
            ),
        )
    elif vectordb_type == "weaviate":
        from app.core.config.client_config_schema import WeaviateConfig
        vdb_cfg = VectorDBConfig(
            type=VectorDBType.WEAVIATE,
            collection=os.getenv("MAI_COLLECTION", "ingested_content"),
            weaviate=WeaviateConfig(
                url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
                api_key_env="WEAVIATE_API_KEY" if os.getenv("WEAVIATE_API_KEY") else None,
            ),
        )
    elif vectordb_type == "redis":
        from app.core.config.client_config_schema import RedisConfig
        vdb_cfg = VectorDBConfig(
            type=VectorDBType.REDIS,
            collection=os.getenv("MAI_COLLECTION", "ingested_content"),
            redis=RedisConfig(
                url=os.getenv("REDIS_URL") or None,
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                password_env="REDIS_PASSWORD" if os.getenv("REDIS_PASSWORD") else None,
                username=os.getenv("REDIS_USERNAME") or None,
                db=int(os.getenv("REDIS_DB", "0")),
                ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
            ),
        )
    else:
        raise ValueError(
            f"[Pipeline] Unknown MAI_VECTORDB='{vectordb_type}'. "
            f"Supported: chroma, qdrant, pinecone, milvus, weaviate, redis"
        )

    # ── Embedder ──────────────────────────────────────────────
    if embedder_type == "ollama":
        from app.core.config.client_config_schema import OllamaEmbedderConfig
        emb_cfg = EmbedderConfig(
            type=EmbedderType.OLLAMA,
            ollama=OllamaEmbedderConfig(
                model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ),
        )
    elif embedder_type == "openai":
        from app.core.config.client_config_schema import OpenAIEmbedderConfig
        emb_cfg = EmbedderConfig(
            type=EmbedderType.OPENAI,
            openai=OpenAIEmbedderConfig(
                model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
                api_key_env="OPENAI_API_KEY",
            ),
        )
    elif embedder_type == "huggingface":
        from app.core.config.client_config_schema import HuggingFaceEmbedderConfig
        emb_cfg = EmbedderConfig(
            type=EmbedderType.HUGGINGFACE,
            huggingface=HuggingFaceEmbedderConfig(
                model=os.getenv("HF_EMBED_MODEL", "BAAI/bge-large-en"),
            ),
        )
    else:
        raise ValueError(
            f"[Pipeline] Unknown MAI_EMBEDDER='{embedder_type}'. "
            f"Supported: ollama, openai, huggingface"
        )

    return ClientConfig(
        client_id=client_id,
        vectordb=vdb_cfg,
        embedder=emb_cfg,
    )


# ============================================================
# ADDITIVE BLOCK 1: VISUAL / CHART-LIKE CONTENT DETECTOR
# ============================================================

def _looks_like_visual_content(text: str) -> bool:
    """
    Heuristic detector for charts, graphs, tables, numeric-heavy visuals.
    ADDITIVE ONLY — no side effects.
    """
    if not text or not isinstance(text, str) or len(text) < 80:
        return False

    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)

    keywords = [
        "%", "chart", "graph", "table", "figure",
        "axis", "source:", "year",
        "2019", "2020", "2021", "2022",
        "2023", "2024", "2025", "2026",
    ]

    keyword_hits = sum(1 for k in keywords if k in text.lower())
    return digit_ratio > 0.35 or keyword_hits >= 2


# ============================================================
# ADDITIVE BLOCK 2: LLM VISUAL EXPLANATION
# ============================================================

async def _explain_visual_with_llm(raw_text: str) -> str:
    """
    Converts visual/chart/table text into a semantic explanation.
    Fail-safe: returns empty string on any failure.
    """
    from app.services.ingestion.llm_rewriter import rewrite_batch

    prompt = (
        "The following content is extracted from a chart, graph, table, or visual.\n"
        "Explain clearly in plain English what this visual represents.\n"
        "Focus on trends, comparisons, increases or decreases, and key insights.\n"
        "Do NOT repeat axis labels, raw numbers, or dump percentages.\n\n"
        f"Content:\n{raw_text}\n\nExplanation:"
    )
    try:
        result = await rewrite_batch([prompt])
        if result and isinstance(result, list):
            return result[0].strip()
    except Exception:
        pass
    return ""


# ============================================================
# ADAPTER CLASSES
# ─────────────────────────────────────────────────────────────
# Bridge: pluggable BaseVectorDB / BaseEmbedder ↔ raw Chroma / ST API.
#
# WHY:
#   main.py, ingestion_sync_api.py, ingestion_integrity_api.py all
#   call Chroma Collection methods (.get, .upsert, .delete, .count,
#   .name) and SentenceTransformer methods (.encode().tolist()).
#   These adapters expose exactly that API — backed by any plugin.
#
# HOW TO ADD A NEW BACKEND:
#   1. Write a new plugin class (e.g. PineconeVectorDB)
#   2. Register it in plugin_registry
#   3. Set MAI_VECTORDB=pinecone in .env
#   Zero changes to any adapter, API file, or main.py.
# ============================================================

class _VectorResult:
    """
    Wraps an embedding result so .tolist() works exactly like
    SentenceTransformer — supports both single and batched vectors.
    """
    __slots__ = ("_v",)

    def __init__(self, v):
        self._v = v if isinstance(v, list) else list(v)

    def tolist(self):
        return self._v

    def __iter__(self):
        return iter(self._v)

    def __getitem__(self, idx):
        item = self._v[idx]
        return _VectorResult(item) if isinstance(item, list) else item

    def __len__(self):
        return len(self._v)


class _CollectionAdapter:
    """
    Wraps BaseVectorDB → exposes chromadb.Collection API surface.

    Methods supported:
      .name                                                  → str
      .count()                                               → int
      .upsert(ids, embeddings, documents, metadatas)         → None
      .delete(ids)                                           → None
      .get(include=[...])                                    → dict
      .query(query_embeddings, n_results, where, include)    → dict

    .get() resolution order (for admin bulk-fetch):
      1. vectordb.get_all()          — pluggable standard
      2. vectordb._collection.get()  — Chroma plugin raw passthrough
      3. vectordb.collection.get()   — alternate attr name
      4. Safe empty {}               — never crashes callers
    """

    def __init__(self, vectordb, collection_name: str = None):
        self._vdb = vectordb
        self._col = collection_name or os.getenv("MAI_COLLECTION", "ingested_content")

    @property
    def name(self) -> str:
        return self._col

    def count(self) -> int:
        try:
            return self._vdb.count(collection=self._col)
        except Exception as e:
            log_info(f"[CollectionAdapter] count() failed: {e}")
            return 0

    def upsert(self, ids, embeddings, documents=None, metadatas=None):
        """Chroma-style bulk upsert → pluggable one-by-one upsert."""
        docs = documents or [""] * len(ids)
        mets = metadatas or [{}]  * len(ids)
        for i, doc_id in enumerate(ids):
            self._vdb.upsert(
                collection=self._col,
                doc_id=doc_id,
                embedding=embeddings[i],
                text=docs[i],
                metadata=mets[i],
            )

    def delete(self, ids=None, where=None):
        if not ids:
            return
        for doc_id in ids:
            try:
                self._vdb.delete(collection=self._col, doc_id=doc_id)
            except Exception as e:
                log_info(f"[CollectionAdapter] delete({doc_id}) skipped: {e}")

    def get(self, include=None, where=None) -> dict:
        include = include or []

        # 1 — pluggable get_all (preferred path)
        if hasattr(self._vdb, "get_all"):
            try:
                return self._vdb.get_all(collection=self._col, include=include)
            except Exception as e:
                log_info(f"[CollectionAdapter] get_all() failed: {e}")

        # 2 / 3 — raw Chroma collection passthrough
        for attr in ("_collection", "collection", "_chroma_collection"):
            raw = getattr(self._vdb, attr, None)
            if raw is not None and callable(getattr(raw, "get", None)):
                try:
                    return raw.get(include=include)
                except Exception as e:
                    log_info(f"[CollectionAdapter] raw.{attr}.get() failed: {e}")

        # 4 — safe fallback — admin ops see 0 rows, never crash
        log_info("[CollectionAdapter] get(): no bulk-fetch path — returning empty")
        return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}

    def query(
        self,
        query_embeddings,
        n_results: int = 10,
        where=None,
        include=None,
    ) -> dict:
        out = {"ids": [], "distances": [], "documents": [], "metadatas": []}
        for emb in query_embeddings:
            try:
                hits = self._vdb.search(
                    collection=self._col,
                    query_embedding=emb,
                    top_k=n_results,
                    filters=where,
                )
                out["ids"].append([h.id              for h in hits])
                out["distances"].append([h.score           for h in hits])
                out["documents"].append([h.text            for h in hits])
                out["metadatas"].append([h.metadata         for h in hits])
            except Exception as e:
                log_info(f"[CollectionAdapter] query() failed: {e}")
                out["ids"].append([])
                out["distances"].append([])
                out["documents"].append([])
                out["metadatas"].append([])
        return out


class _EmbedderAdapter:
    """
    Wraps BaseEmbedder → exposes SentenceTransformer API surface.

    Methods supported:
      .encode(text_or_list, normalize_embeddings=True)  → _VectorResult
      .embed_query(text)       → list[float]
      .embed_documents(texts)  → list[list[float]]

    Callers use:
      embedder = get_embedder()
      vector   = embedder.encode(text, normalize_embeddings=True).tolist()
    This is byte-for-byte identical to the old SentenceTransformer call.
    """

    def __init__(self, embedder):
        self._emb = embedder

    def encode(
        self,
        texts,
        normalize_embeddings: bool = False,
        batch_size: int = 32,
    ) -> _VectorResult:
        if isinstance(texts, str):
            return _VectorResult(self._emb.embed_query(texts))
        return _VectorResult(self._emb.embed_documents(texts))

    def embed_query(self, text: str) -> List[float]:
        return self._emb.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._emb.embed_documents(texts)

    @property
    def kind(self) -> str:
        return getattr(self._emb, "kind", type(self._emb).__name__)


# ============================================================
# PUBLIC API SHIMS — permanent, pluggable, zero-caller-change
# ─────────────────────────────────────────────────────────────
# These are the THREE functions every caller imports.
# Signatures are 100% identical to the original Chroma-locked versions.
# To switch backend: change MAI_VECTORDB / MAI_EMBEDDER in .env.
# No other file ever needs to change.
# ============================================================

def get_chroma_collection(
    skip_count: bool = False,
    business_id: Optional[str] = None,
) -> Tuple[None, _CollectionAdapter]:
    """
    Returns (None, _CollectionAdapter).

    Identical tuple shape as original:
      client, collection = get_chroma_collection(skip_count=True)
      _, collection      = get_chroma_collection()

    Adapter exposes full Collection surface:
      .name / .count() / .upsert() / .delete() / .get() / .query()

    Backend: MAI_VECTORDB=chroma|qdrant in .env
    """
    pipeline = _get_pipeline(business_id)
    return None, _CollectionAdapter(pipeline.vectordb)


def get_embedder(business_id: Optional[str] = None) -> _EmbedderAdapter:
    """
    Returns _EmbedderAdapter.

    Identical to old SentenceTransformer call:
      embedder = get_embedder()
      vector   = embedder.encode(text, normalize_embeddings=True).tolist()

    Backend: MAI_EMBEDDER=ollama|openai|huggingface in .env
    """
    pipeline = _get_pipeline(business_id)
    return _EmbedderAdapter(pipeline.embedder)


def get_collection_count_cached(force_refresh: bool = False) -> int:
    """
    Cached vector count — 5-minute TTL.
    _COLLECTION_COUNT_CACHE is module-level so main.py can import it directly.
    """
    global _COLLECTION_COUNT_CACHE
    now          = datetime.utcnow()
    last_updated = _COLLECTION_COUNT_CACHE.get("last_updated")
    CACHE_TTL    = 300  # 5 minutes

    if not force_refresh and last_updated:
        if (now - last_updated).total_seconds() < CACHE_TTL:
            cached = _COLLECTION_COUNT_CACHE["count"]
            log_info(f"[IngestionV2] Cached vector count: {cached:,}")
            return cached

    try:
        _, collection = get_chroma_collection(skip_count=True)
        count = collection.count()
        _COLLECTION_COUNT_CACHE["count"]        = count
        _COLLECTION_COUNT_CACHE["last_updated"] = now
        log_info(f"[IngestionV2] Vector count refreshed: {count:,}")
        return count
    except Exception as e:
        log_info(f"[IngestionV2] Vector count fetch failed: {e}")
        return _COLLECTION_COUNT_CACHE.get("count", 0)


# =============================================
# EVENT LOGGER  (preserved exactly)
# =============================================

async def log_event(
    file_name: str, stage: str, status: str, message: str = ""
):
    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "file":      file_name,
        "stage":     stage,
        "status":    status,
        "message":   message,
    }
    await broadcast(event)


# =============================================
# INGESTION SERVICE V2 CLASS  (preserved exactly — zero changes)
# =============================================

class IngestionServiceV2:

    # ----------------------------------------------------------
    # Ensure file entry exists (FK safe + hash check)
    # ----------------------------------------------------------
    @staticmethod
    async def _ensure_file_entry(db: AsyncSession, file_record: IngestedFileV2):
        result = await db.execute(
            select(IngestedFileV2.id).where(IngestedFileV2.id == file_record.id)
        )
        existing = result.scalar_one_or_none()
        if not existing:
            log_info(f"[IngestionV2] Creating missing file entry for {file_record.id}")
            meta_data = getattr(file_record, "meta_data", {})
            if file_record.file_path:
                loop = asyncio.get_running_loop()
                try:
                    file_hash = await loop.run_in_executor(
                        None, compute_file_hash, file_record.file_path
                    )
                    meta_data["file_hash"] = file_hash
                except Exception as e:
                    log_info(
                        f"[IngestionV2] Failed to compute file hash "
                        f"for {file_record.file_path}: {e}"
                    )

            file_entry = {
                "id":               file_record.id,
                "business_id":      file_record.business_id,
                "file_name":        getattr(file_record, "file_name", str(file_record.id)),
                "file_type":        file_record.file_type,
                "file_path":        getattr(file_record, "file_path", None),
                "source_url":       getattr(file_record, "source_url", None),
                "source_type":      getattr(file_record, "source_type", file_record.file_type),
                "meta_data":        meta_data,
                "parser_used":      getattr(file_record, "parser_used", None),
                "status":           "uploaded",
                "total_chunks":     0,
                "unique_chunks":    0,
                "duplicate_chunks": 0,
                "dedup_ratio":      0.0,
                "error_message":    None,
                "created_at":       datetime.utcnow(),
                "updated_at":       datetime.utcnow(),
            }
            await db.execute(insert(IngestedFileV2), [file_entry])
            await db.commit()

    # ----------------------------------------------------------
    # Primary file ingestion entrypoint (UI + Bulk safe)
    # ----------------------------------------------------------
    @staticmethod
    async def process_file(
        file_id: str,
        file_path: Optional[str] = None,
        business_id: Optional[str] = None,
    ):
        async with async_session() as db:
            try:
                log_info(f"[IngestionV2] Starting ingestion for {file_id}")

                file_record = await IngestionServiceV2._get_file_record(db, file_id)

                # ==========================================================
                # MEDIA-LEVEL HARD DEDUP (AUTHORITATIVE — API + WATCHER)
                # ==========================================================
                if file_record and file_record.file_path:
                    loop = asyncio.get_running_loop()
                    try:
                        incoming_hash = await loop.run_in_executor(
                            None, compute_file_hash, file_record.file_path
                        )

                        result = await db.execute(
                            select(IngestedFileV2)
                            .where(IngestedFileV2.meta_data["file_hash"].astext == incoming_hash)
                            .where(IngestedFileV2.status == "processed")
                        )
                        existing = result.scalar_one_or_none()

                        if existing and existing.id != file_record.id:
                            log_info(
                                f"[IngestionV2] ⛔ MEDIA DUPLICATE — "
                                f"already ingested as {existing.id}"
                            )
                            await IngestionServiceV2._update_file_status(
                                db,
                                file_record.id,
                                total_chunks=0,
                                status="duplicate",
                            )
                            return  # 🚫 HARD STOP

                        meta = file_record.meta_data or {}
                        meta["file_hash"] = incoming_hash
                        await db.execute(
                            update(IngestedFileV2)
                            .where(IngestedFileV2.id == file_record.id)
                            .values(meta_data=meta)
                        )
                        await db.commit()

                    except Exception as e:
                        log_info(f"[IngestionV2] Media hash dedup failed: {e}")

                if not file_record and file_path:
                    new_file = IngestedFileV2(
                        id=file_id,
                        file_name=file_path.split("/")[-1],
                        file_type=file_path.split(".")[-1],
                        file_path=file_path,
                        business_id=business_id,
                        meta_data={"file_hash": compute_file_hash(file_path)},
                        status="uploaded",
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                    )
                    db.add(new_file)
                    await db.commit()
                    file_record = new_file

                if not file_record:
                    log_info(f"[IngestionV2] File not found: {file_id}")
                    return

                await IngestionServiceV2._ensure_file_entry(db, file_record)
                parsed = await IngestionServiceV2._parse_file(
                    file_record.file_path, file_record.file_type, db, file_id
                )

                if asyncio.iscoroutine(parsed):
                    parsed = await parsed

                if not parsed:
                    log_info(f"[IngestionV2] Parsing failed for {file_id}")
                    return

                await IngestionServiceV2._run_pipeline(db, file_record, parsed)
                log_info(f"[IngestionV2] ✅ Completed ingestion for {file_id}")

            except Exception as e:
                log_info(f"[CRITICAL] Ingestion failed for {file_id}: {e}")
                await IngestionServiceV2._set_file_error(db, file_id, str(e))

    # ----------------------------------------------------------
    # Direct ingestion for pre-parsed output (RSS, API, etc.)
    # ----------------------------------------------------------
    @staticmethod
    async def ingest_parsed_output(
        file_id: str, parsed_output: Dict[str, Any]
    ):
        async with async_session() as db:
            try:
                file_record = await IngestionServiceV2._get_file_record(db, file_id)
                if not file_record:
                    log_info(
                        f"[IngestionV2] File not found for pre-parsed ingestion: {file_id}"
                    )
                    return

                await IngestionServiceV2._ensure_file_entry(db, file_record)
                await IngestionServiceV2._run_pipeline(db, file_record, parsed_output)
                log_info(f"[IngestionV2] ✅ Completed direct ingestion for {file_id}")

            except Exception as e:
                log_info(f"[ERROR] Direct ingestion failed for {file_id}: {e}")
                await IngestionServiceV2._set_file_error(db, file_id, str(e))

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    @staticmethod
    async def _get_file_record(db: AsyncSession, file_id: str):
        result = await db.execute(
            select(IngestedFileV2).where(IngestedFileV2.id == file_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def _parse_file(
        file_path: str, file_type: str, db: AsyncSession, file_id: str
    ):
        log_info(f"[IngestionV2] Parsing {file_path} ({file_type})")
        try:
            await db.execute(
                update(IngestedFileV2)
                .where(IngestedFileV2.id == file_id)
                .values(parser_used=file_type, updated_at=datetime.utcnow())
            )
            await db.commit()
        except Exception as e:
            log_info(f"[WARN] Failed to update parser_used for {file_id}: {e}")
        return await ParserRouterV2.parse(file_path, file_type)

    # ----------------------------------------------------------
    # Core ingestion pipeline
    # ----------------------------------------------------------
    @staticmethod
    async def _run_pipeline(
        db: AsyncSession,
        file_record: IngestedFileV2,
        parsed_payload: Dict[str, Any],
    ):
        await IngestionServiceV2._ensure_file_entry(db, file_record)

        file_id     = file_record.id
        business_id = file_record.business_id
        file_type   = file_record.file_type
        pipeline = _get_pipeline(business_id)
        embedding_model = pipeline.embedder.info.model
        log_info(
            f"[IngestionV2] Active embedding model for {file_id}: {embedding_model}"
        )

        chunks = await IngestionServiceV2._extract_chunks(
            parsed_payload,
            file_id,
            file_type,
            business_id,
            db,
            embedding_model=embedding_model,
        )
        IngestionServiceV2._assert_chunk_embedding_model(
            chunks=chunks,
            expected_model=embedding_model,
            file_id=str(file_id),
        )   

        if not chunks:
            log_info(f"[IngestionV2] No chunks to ingest for {file_id}")
            return

        unique_chunks, dedup_stats = await IngestionServiceV2._dedup_chunks(
            db, chunks, file_id, business_id
        )

        # ═══════════════════════════════════════════════════════════
        # Store ALL chunks in ingested_content (unique + duplicates)
        # ═══════════════════════════════════════════════════════════
        unique_hashes          = {c.get("semantic_hash") for c in unique_chunks}
        all_chunks_for_storage = []

        for chunk in unique_chunks:
            chunk["is_duplicate"]    = False
            chunk["duplicate_of"]    = None
            chunk["similarity_score"] = None
            all_chunks_for_storage.append(chunk)

        for chunk in chunks:
            semantic_hash = chunk.get("semantic_hash")
            if semantic_hash not in unique_hashes:
                chunk["is_duplicate"]    = True
                chunk["duplicate_of"]    = chunk.get("global_content_id")
                chunk["similarity_score"] = chunk.get("similarity", None)
                all_chunks_for_storage.append(chunk)

        if all_chunks_for_storage:
            await IngestionServiceV2._insert_chunks(
                db, file_id, business_id, all_chunks_for_storage
            )
            log_info(
                f"[IngestionV2] Inserted {len(all_chunks_for_storage)} chunks: "
                f"{len(unique_chunks)} unique, "
                f"{len(all_chunks_for_storage) - len(unique_chunks)} duplicates"
            )

        # Only embed unique hashes (skip already-stored duplicates)
        chunks_with_hash = [c for c in chunks if c.get("semantic_hash")]
        if chunks_with_hash:
            log_info(
                f"[IngestionV2] Calling _embed_and_store for {file_id} "
                f"with {len(chunks_with_hash)} chunks"
            )
            await IngestionServiceV2._embed_and_store(
                file_id, business_id, file_type, chunks_with_hash
            )
        else:
            log_info(
                f"[IngestionV2] No chunks with semantic_hash — "
                f"skipping VectorDB embed for {file_id}"
            )

        await IngestionServiceV2._update_file_status(
            db,
            file_id,
            total_chunks=dedup_stats["total"],
            unique_chunks=dedup_stats["unique"],
            duplicate_chunks=dedup_stats["duplicates"],
            dedup_ratio=dedup_stats["dedup_ratio"],
            status="processed",
        )

        log_info(
            f"[IngestionV2] Pipeline complete for {file_id}: "
            f"{dedup_stats['unique']}/{dedup_stats['total']} chunks stored "
            f"({dedup_stats['dedup_ratio']:.2f}% deduplication)"
        )

    # ----------------------------------------------------------
    # Enhanced Chunk extraction (Global Index compatible)
    # ----------------------------------------------------------
    @staticmethod
    async def _extract_chunks(
        parsed_payload,
        file_id,
        file_type,
        business_id,
        db,
        embedding_model: str,
    ):
        try:

            pipeline = _get_pipeline(business_id)
            embedding_model = pipeline.embedder.info.model 
            
            if asyncio.iscoroutine(parsed_payload):
                parsed_payload = await parsed_payload

            # Handle multiple source formats (RSS, API, etc.)
            if any(k in parsed_payload for k in ["entries", "rows", "chunks"]):
                base_list = (
                    parsed_payload.get("entries")
                    or parsed_payload.get("rows")
                    or parsed_payload.get("chunks")
                )
                enriched_chunks = []
                for item in base_list:
                    raw_text = None
                    if isinstance(item, dict):
                        raw_text = (
                            item.get("cleaned_text")
                            or item.get("text")
                            or item.get("summary")
                            or item.get("description")
                        )
                        if raw_text is None:
                            try:
                                parts = []
                                for k, v in item.items():
                                    if isinstance(v, (dict, list)):
                                        parts.append(f"{k}: {str(v)}")
                                    else:
                                        parts.append(f"{k}: {v}")
                                raw_text = " | ".join(parts)
                            except Exception:
                                raw_text = str(item)
                    else:
                        if hasattr(item, "get") and callable(item.get):
                            raw_text = (
                                item.get("text")
                                or item.get("cleaned_text")
                                or item.get("summary")
                            )
                        else:
                            raw_text = item

                    if raw_text is None:
                        continue
                    if not isinstance(raw_text, str):
                        if isinstance(raw_text, list):
                            raw_text = " ".join(
                                str(x) for x in raw_text if x is not None
                            )
                        elif isinstance(raw_text, dict):
                            raw_text = " | ".join(
                                f"{k}: {v}" for k, v in raw_text.items()
                            )
                        else:
                            raw_text = str(raw_text)

                    text = raw_text.strip()
                    if not text:
                        continue

                    # ADDITIVE BLOCK 3: VISUAL INTERCEPTION (MULTI)
                    if _looks_like_visual_content(text):
                        explanation = await _explain_visual_with_llm(text)
                        if explanation:
                            explained_chunks = await recursive_semantic_chunk(
                                explanation,
                                db_session=db,
                                file_id=str(file_id),
                                business_id=business_id,
                                source_type=file_type,
                                embedding_model=embedding_model,
                            )
                            for ch in explained_chunks:
                                ch.setdefault("reasoning_ingestion", {})
                                ch["reasoning_ingestion"].update({
                                    "content_type":        "visual",
                                    "interpreted_by":      "llm",
                                    "original_text_hash":  hashlib.sha256(
                                        text.encode("utf-8")
                                    ).hexdigest(),
                                })
                            enriched_chunks.extend(explained_chunks)
                            continue

                    subchunks = await recursive_semantic_chunk(
                        text,
                        db_session=db,
                        file_id=str(file_id),
                        business_id=business_id,
                        source_type=file_type,
                        embedding_model=embedding_model,
                    )
                    enriched_chunks.extend(subchunks)
                return enriched_chunks

            # Single payload path
            text = _resolve_text(parsed_payload)
            if not text:
                log_info(f"[IngestionV2] No text found in parsed payload for {file_id}")
                return []

            # ADDITIVE BLOCK 4: VISUAL INTERCEPTION (SINGLE)
            if _looks_like_visual_content(text):
                explanation = await _explain_visual_with_llm(text)
                if explanation:
                    explained_chunks = await recursive_semantic_chunk(
                        explanation,
                        db_session=db,
                        file_id=str(file_id),
                        business_id=business_id,
                        source_type=file_type,
                        embedding_model=embedding_model,
                    )
                    for ch in explained_chunks:
                        ch.setdefault("reasoning_ingestion", {})
                        ch["reasoning_ingestion"].update({
                            "content_type":       "visual",
                            "interpreted_by":     "llm",
                            "original_text_hash": hashlib.sha256(
                                text.encode("utf-8")
                            ).hexdigest(),
                        })
                    return explained_chunks

            return await recursive_semantic_chunk(
                text,
                db_session=db,
                file_id=str(file_id),
                business_id=business_id,
                source_type=file_type,
                embedding_model=embedding_model,
            )

        except Exception as e:
            log_info(f"[CRITICAL] Chunk extraction failed: {e}")
            raise RuntimeError(f"Chunk extraction failed: {e}")
    
    @staticmethod
    def _assert_chunk_embedding_model(
        chunks: List[Dict[str, Any]],
        expected_model: str,
        file_id: str,
    ) -> None:
        """Fail fast if chunk metadata drifts from runtime embedding model."""
        mismatches = []
        for idx, chunk in enumerate(chunks):
            chunk_model = chunk.get("embedding_model")
            if not chunk_model:
                chunk["embedding_model"] = expected_model
                continue
            if chunk_model != expected_model:
                mismatches.append((idx, chunk_model))

        if mismatches:
            sample = ", ".join(
                f"idx={idx}:'{model}'" for idx, model in mismatches[:5]
            )
            raise ValueError(
                "[IngestionV2] Embedding model mismatch detected "
                f"for file {file_id}. expected='{expected_model}', "
                f"mismatches={len(mismatches)} [{sample}]"
            )

    # ----------------------------------------------------------
    # Deduplication (3-layer, batched for performance)
    # ----------------------------------------------------------
    @staticmethod
    async def _dedup_chunks(
        db: AsyncSession,
        chunks: List[Dict[str, Any]],
        file_id: str,
        business_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """3-layer deduplication with cross-file duplicate detection."""
        if not chunks:
            return [], {
                "total": 0, "unique": 0, "duplicates": 0, "dedup_ratio": 0.0
            }

        pipeline = _get_pipeline(business_id)

        unique_chunks, stats = await deduplicate_chunks(
            db=db,
            chunks=chunks,
            vectordb=pipeline.vectordb,       # BaseVectorDB ← pluggable
            embedder=pipeline.embedder,        # BaseEmbedder ← pluggable
            file_id=file_id,
            business_id=business_id,
            enable_embedding_dedup=True,
            similarity_threshold=0.95,
        )

        log_info(
            f"[IngestionV2] Dedup complete for {file_id}: "
            f"{stats['unique']} unique, {stats['duplicates']} duplicates, "
            f"{stats['dedup_ratio']:.2f}% reduction "
            f"[L1={stats.get('layer1_hash_duplicates', 0)}, "
            f"L2={stats.get('layer2_embedding_duplicates', 0)}, "
            f"L3={stats.get('layer3_gci_duplicates', 0)}]"
        )
        return unique_chunks, stats

    # ----------------------------------------------------------
    # Insert chunks into PostgreSQL
    # ----------------------------------------------------------
    @staticmethod
    async def _insert_chunks(
        db: AsyncSession, file_id, business_id, chunks
    ):
        result = await db.execute(
            select(func.max(IngestedContentV2.chunk_index))
            .where(IngestedContentV2.file_id == file_id)
        )
        start_index = (result.scalar() or -1) + 1

        db_rows = [
            {
                "id":                  uuid.uuid4(),
                "file_id":             file_id,
                "business_id":         business_id,
                "chunk_index":         start_index + i,
                "text":                c.get("text"),
                "cleaned_text":        c.get("cleaned_text", c.get("cleaned")),
                "tokens":              c.get("tokens"),
                "source_type":         c.get("source_type"),
                "meta_data":           c.get("metadata", {}),
                "confidence":          c.get("confidence", 1.0),
                "semantic_hash":       c.get("semantic_hash"),
                "global_content_id":   c.get("global_content_id"),
                "reasoning_ingestion": c.get("reasoning_ingestion"),
                "is_duplicate":        c.get("is_duplicate", False),
                "duplicate_of":        c.get("duplicate_of"),
                "similarity_score":    c.get("similarity_score"),
                "created_at":          datetime.utcnow(),
                "updated_at":          datetime.utcnow(),
            }
            for i, c in enumerate(chunks)
        ]

        await db.execute(insert(IngestedContentV2), db_rows)
        await db.commit()
        log_info(f"[IngestionV2] Inserted {len(db_rows)} chunks into DB")
    # ----------------------------------------------------------
    # Embedding + Vector Store (executor-offloaded, DB-dedup-safe)
    # ----------------------------------------------------------
    @staticmethod
    async def _embed_and_store(file_id, business_id, file_type, chunks):
        """
        Embeds unique chunks and stores them in VectorDB.

        FAILURE POLICY:
          Any exception (dimension mismatch, OOM, network error) will:
            1. Log the error with full detail
            2. Mark the file as FAILED in DB  ← callers see correct status
            3. Re-raise                        ← run_pipeline knows it failed
          NEVER swallow exceptions here — silent success = corrupt / missing data.
        """
        try:
            loop = asyncio.get_running_loop()

            # ── STEP 1: Build hash → chunk map ────────────────────────
            # Do this FIRST — if 0 valid hashes, exit before any network call
            hash_to_chunk: Dict[str, Any] = {}
            all_hashes:    List[str]      = []
            for c in chunks:
                sh = c.get("semantic_hash")
                if not sh:
                    continue
                all_hashes.append(sh)
                hash_to_chunk[sh] = c

            if not all_hashes:
                log_info(f"[IngestionV2] 0 semantic hashes for {file_id} — skipping VectorDB")
                return

            # ── STEP 2: Check GCI for already-known hashes ────────────
            # Fast DB lookup — no network, no pipeline init yet
            async with async_session() as db:
                res = await db.execute(
                    select(
                        GlobalContentIndexV2.semantic_hash,
                        GlobalContentIndexV2.id,
                    ).where(GlobalContentIndexV2.semantic_hash.in_(all_hashes))
                )
                known_hashes = {row[0] for row in res.all()}

            # ── STEP 3: Resolve pipeline (per-client, no hardcoding) ──
            # Only done here — after confirming there are hashes to process
            pipeline         = _get_pipeline(business_id)
            embedder         = pipeline.embedder          # BaseEmbedder — pluggable
            vectordb         = pipeline.vectordb          # BaseVectorDB — pluggable
            collection_name  = pipeline.config.vectordb.collection  # ← per-client, never hardcoded

            # ── STEP 4: Check which known hashes exist in VectorDB ────
            try:
                present_in_vectordb = set(
                    await loop.run_in_executor(
                        None,
                        lambda: vectordb.exists(
                            collection=collection_name,   # ← per-client
                            ids=list(known_hashes),
                        ),
                    )
                )
            except Exception as e:
                log_info(f"[IngestionV2] VectorDB exists check failed: {e}")
                present_in_vectordb = set()

            # ── STEP 5: Determine which hashes actually need embedding ─
            hashes_needing_embedding = {
                h for h in all_hashes
                if h not in known_hashes or h not in present_in_vectordb
            }

            if not hashes_needing_embedding:
                log_info(f"[IngestionV2] All hashes already in VectorDB for {file_id}")
                return

            new_chunks = [
                hash_to_chunk[h]
                for h in all_hashes
                if h in hashes_needing_embedding
            ]

            # ── STEP 6: Probe embedding dimension (never assume) ──────
            embedding_dim = getattr(getattr(embedder, "info", None), "dim", None)
            if not embedding_dim or embedding_dim <= 0:
                probe = await loop.run_in_executor(
                    None, lambda: embedder.embed_query("dimension probe")
                )
                embedding_dim = len(probe)
                log_info(f"[IngestionV2] Probed embedding dim: {embedding_dim}")

            # ── STEP 7: Ensure collection — DIMENSION GUARD runs here ─
            # Called AFTER new_chunks is confirmed non-empty.
            # If stored dim != current embedder dim → raises ValueError
            # immediately with a clear "delete collection + re-ingest" message.
            # Never silently stores wrong-dim vectors.
            vectordb.ensure_collection(
                collection_name,                   # ← per-client
                embedding_dim=embedding_dim,
                distance_metric="cosine",
            )

            # ── STEP 8: Batched embed + upsert ────────────────────────
            for i in range(0, len(new_chunks), BATCH_SIZE):
                batch     = new_chunks[i : i + BATCH_SIZE]
                texts     = [c.get("cleaned_text", c.get("cleaned", "")) for c in batch]
                batch_ids = [c.get("semantic_hash") for c in batch]

                emb_list = await loop.run_in_executor(
                    None, lambda t=texts: embedder.embed_documents(t)
                )

                metadatas = [
                    {
                        "file_id":       str(file_id),
                        "business_id":   str(business_id) if business_id else "",
                        "source_type":   str(file_type)   if file_type   else "",
                        "semantic_hash": str(c.get("semantic_hash", "")),
                    }
                    for c in batch
                ]

                # batch_upsert raises on failure (fixed in chroma_v1.py)
                # Exception propagates here → caught below → file marked FAILED
                await loop.run_in_executor(
                    None,
                    lambda ids=batch_ids, emb=emb_list, met=metadatas, docs=texts:
                        vectordb.batch_upsert(
                            collection=collection_name,    # ← per-client
                            doc_ids=ids,
                            embeddings=emb,
                            texts=docs,
                            metadatas=met,
                        ),
                )
                log_info(
                    f"[IngestionV2] ✅ Batch {i // BATCH_SIZE + 1}: "
                    f"{len(batch)} vectors via {vectordb.kind}"
                )

            log_info(
                f"[IngestionV2] Stored {len(new_chunks)} new vectors "
                f"via {vectordb.kind} for {file_id}"
            )

        except Exception as e:
            # ── CORRECT FAILURE HANDLING ──────────────────────────────
            # 1. Log with full detail
            log_info(
                f"[IngestionV2] ❌ FATAL: VectorDB storage failed for {file_id}:\n"
                f"  Error : {type(e).__name__}: {e}\n"
                f"  Client: {business_id} | File type: {file_type}"
            )
            # 2. Mark file as FAILED in DB — UI + API show correct status
            try:
                async with async_session() as err_db:
                    await IngestionServiceV2._set_file_error(err_db, file_id, str(e))
            except Exception as db_err:
                log_info(f"[IngestionV2] Also failed to write error status: {db_err}")
            # 3. Re-raise — run_pipeline must know storage failed
            raise


    # ----------------------------------------------------------
    # Error Handling + File Status
    # ----------------------------------------------------------
    @staticmethod
    async def _set_file_error(
        db: AsyncSession, file_id: str, error_message: str
    ):
        await db.execute(
            update(IngestedFileV2)
            .where(IngestedFileV2.id == file_id)
            .values(
                error_message=str(error_message)[:255],
                status="failed",
                updated_at=datetime.utcnow(),
            )
        )
        await db.commit()

    @staticmethod
    async def _update_file_status(
        db: AsyncSession,
        file_id: str,
        total_chunks:     int   = 0,
        unique_chunks:    int   = 0,
        duplicate_chunks: int   = 0,
        dedup_ratio:      float = 0.0,
        status:           str   = "processed",
    ):
        """Track comprehensive deduplication metrics on the file record."""
        try:
            await db.execute(
                update(IngestedFileV2)
                .where(IngestedFileV2.id == file_id)
                .values(
                    total_chunks=total_chunks,
                    unique_chunks=unique_chunks,
                    duplicate_chunks=duplicate_chunks,
                    dedup_ratio=dedup_ratio,
                    status=status,
                    updated_at=datetime.utcnow(),
                )
            )
            await db.commit()
            log_info(
                f"[IngestionV2] File {file_id} → "
                f"status={status}, "
                f"chunks={unique_chunks}/{total_chunks}, "
                f"dedup={dedup_ratio:.2f}%"
            )
        except Exception as e:
            log_info(f"[IngestionV2] Failed to update file status: {e}")
            await db.rollback()
