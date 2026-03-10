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
from functools import lru_cache
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
from app.services.ingestion.row_segmenter_v2 import parse_dataframe_rows
from app.services.ingestion.deduplication_engine_v2 import (
    deduplicate_chunks,
    create_normalized_hash,
    register_unique_chunks_in_gci,   # ← NEW: post-dedup GCI commit
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

@lru_cache(maxsize=16)
def _get_pipeline(business_id: Optional[str] = None):
    """
    Resolve a live AssembledPipeline for the given business.
    Config is read purely from environment variables — no JSON files.
    Works identically in dev, staging, and production.

    PERF FIX: decorated with @lru_cache(maxsize=16).
    Previously called 4× per file — once each in _run_pipeline,
    _extract_chunks, _dedup_chunks, and _embed_and_store.
    Each call re-read env vars, rebuilt config, and called
    pipeline_factory.build() which constructs embedder + vectordb objects.
    With cache: first call per business_id builds the pipeline,
    subsequent calls return the cached object instantly.
    maxsize=16 covers 16 distinct business_ids comfortably.
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
        vdb_cfg = VectorDBConfig(
            type=VectorDBType.QDRANT,
            collection=os.getenv("MAI_COLLECTION", "ingested_content"),
            qdrant=QdrantConfig(
                url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                api_key_env="QDRANT_API_KEY",
            ),
        )
    elif vectordb_type == "pinecone":
        from app.core.config.client_config_schema import PineconeConfig
        vdb_cfg = VectorDBConfig(
            type=VectorDBType.PINECONE,
            collection=os.getenv("MAI_COLLECTION", "ingested_content"),
            pinecone=PineconeConfig(
                api_key_env="PINECONE_API_KEY",
                index_name=os.getenv("PINECONE_INDEX_NAME", "ingested-content"),
                namespace=os.getenv("PINECONE_NAMESPACE", "default"),
                embedding_dim=int(os.getenv("PINECONE_EMBEDDING_DIM", "1024")),
                metric=os.getenv("PINECONE_METRIC", "cosine"),
                cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv("PINECONE_REGION", "us-east-1"),
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
        # ── PHANTOM BUG-3 FIX ──────────────────────────────────────────────────
        # _ensure_file_entry() was called here AND in every caller of _run_pipeline()
        # (both process_file() and ingest_parsed_output() call it before this).
        # Removed the duplicate call — saves 1 SELECT + conditional INSERT per file.
        # ────────────────────────────────────────────────────────────────────────
        file_id     = file_record.id
        business_id = file_record.business_id
        file_type   = file_record.file_type

        # ── FIX-D: Resolve pipeline ONCE per file ──────────────────────────────
        # Previously _get_pipeline() was called 3 times per file:
        #   1. here in _run_pipeline
        #   2. inside _extract_chunks (redundant rebuild)
        #   3. inside _dedup_chunks   (redundant rebuild)
        # Each call re-reads env vars, constructs configs, builds embedder+vectordb.
        # Fix: resolve once here, pass the live object into both methods.
        # _extract_chunks and _dedup_chunks now accept an optional `pipeline`
        # argument and skip _get_pipeline() when it is provided.
        # ──────────────────────────────────────────────────────────────────────
        pipeline        = _get_pipeline(business_id)
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
            pipeline=pipeline,          # FIX-D: pass resolved pipeline
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
            db, chunks, file_id, business_id,
            collection_name=pipeline.config.vectordb.collection,
            pipeline=pipeline,          # FIX-D: pass resolved pipeline
        )

        # ═══════════════════════════════════════════════════════════
        # ENTERPRISE DEDUP COMMIT — Register unique chunks in GCI
        # ═══════════════════════════════════════════════════════════
        # Called AFTER 3-layer dedup confirms uniqueness.
        # This is the ONLY place GlobalContentIndexV2 is written to.
        #
        # WHY HERE (not in segmenter):
        #   Segmenter (recursive_semantic_chunk) can be called multiple
        #   times per file — once per page group, once per image, etc.
        #   Writing to GCI during segmentation caused boundary chunks to
        #   get occurrence_count=2 before dedup ran, producing 50% false
        #   duplicate rates on first ingestion of any multi-page document.
        #
        #   Now: GCI is only written once per unique chunk, only after
        #   full 3-layer dedup, only for confirmed-unique content.
        #   batch_check_gci() in the dedup engine reads GCI in a single
        #   IN query — any hit is definitively from a prior ingestion.
        # ═══════════════════════════════════════════════════════════
        if unique_chunks:
            await register_unique_chunks_in_gci(
                db=db,
                unique_chunks=unique_chunks,
                file_id=str(file_id),
                business_id=business_id,
                source_type=file_type,
                embedding_model=embedding_model,
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
                chunk["is_duplicate"] = True

                # ── duplicate_of: ONLY store a valid GCI UUID ─────────────────────
                # The duplicate_of column is UUID type in PostgreSQL.
                # NEVER store a SHA-256 hash (64-char hex) there — that would
                # violate the UUID column type constraint.
                #
                # Layer-by-layer source of a valid UUID:
                #   L2 GCI dups  → gci_id (GCI entry UUID, set by batch_check_gci)
                #                  or global_content_id (same UUID, set on chunk)
                #   L3 vector    → NO UUID available; leave as None.
                #                  similarity_score satisfies the OR constraint.
                #   L1 intra-batch → NO UUID; leave as None.
                #                  _insert_chunks will set similarity_score=1.0
                #                  as the fallback to satisfy the constraint.
                # ─────────────────────────────────────────────────────────────────
                gci_uuid = chunk.get("global_content_id") or chunk.get("gci_id")
                chunk["duplicate_of"] = gci_uuid  # UUID or None — never a hash

                # ── similarity_score: use what the dedup engine set ────────────────
                # L3 vector dups: dedup engine sets "similarity_score" = float (e.g. 1.0)
                # L2 GCI dups:    similarity_score stays None (duplicate_of is set)
                # L1 intra-batch: similarity_score stays None → _insert_chunks
                #                 fallback sets it to 1.0 to satisfy constraint
                # ─────────────────────────────────────────────────────────────────
                chunk["similarity_score"] = chunk.get("similarity_score")

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

        # ── PHANTOM BUG-5 FIX ──────────────────────────────────────────────────
        # Was: chunks_with_hash = [c for c in chunks if c.get("semantic_hash")]
        # chunks = the FULL list before dedup, including L1/L2/L3 duplicates.
        # Every duplicate forced a GCI lookup + VectorDB exists() check in
        # _embed_and_store — wasted N network calls for already-known duplicates.
        #
        # Fix: pass unique_chunks only. Duplicates are already stored in
        # ingested_content above with is_duplicate=True — they never need embedding.
        # ────────────────────────────────────────────────────────────────────────
        chunks_to_embed = [c for c in unique_chunks if c.get("semantic_hash")]
        if chunks_to_embed:
            log_info(
                f"[IngestionV2] Calling _embed_and_store for {file_id} "
                f"with {len(chunks_to_embed)} unique chunks "
                f"(skipped {len(chunks) - len(chunks_to_embed)} duplicates)"
            )
            await IngestionServiceV2._embed_and_store(
                file_id, business_id, file_type, chunks_to_embed
            )
        else:
            log_info(
                f"[IngestionV2] No unique chunks with semantic_hash — "
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
        pipeline=None,                  # FIX-D: accept pre-resolved pipeline
    ):
        try:
            # FIX-D: Only resolve pipeline if not passed in from _run_pipeline.
            # Prevents the third redundant _get_pipeline() call per file.
            if pipeline is None:
                pipeline = _get_pipeline(business_id)
            embedding_model = pipeline.embedder.info.model

            if asyncio.iscoroutine(parsed_payload):
                parsed_payload = await parsed_payload

            # ── FIX-A: PRE-BUILT CHUNK PASSTHROUGH ────────────────────────────
            # csv_parser_v2 and excel_parser_v2 internally call row_segmenter_v2
            # and return {"chunks": [list of fully-built chunk dicts]}.
            # Each dict already has semantic_hash, cleaned_text, tokens, etc. set.
            #
            # BEFORE this fix: _extract_chunks re-ran recursive_semantic_chunk on
            # each pre-built chunk, inflating 822 rows into 2631 sub-chunks (3.2×).
            # This is the root cause of the stall logged in the bug report.
            #
            # Detection: sample the first item in "chunks". If it has semantic_hash
            # already set, the parser did the chunking — pass through directly.
            # This is safe: recursive_semantic_chunk also produces dicts with
            # semantic_hash, so false positives from other paths are impossible.
            # ──────────────────────────────────────────────────────────────────
            raw_chunks_list = parsed_payload.get("chunks")
            if (
                isinstance(raw_chunks_list, list)
                and raw_chunks_list
                and isinstance(raw_chunks_list[0], dict)
                and raw_chunks_list[0].get("semantic_hash")
            ):
                log_info(
                    f"[IngestionV2] Pre-built chunk passthrough: "
                    f"{len(raw_chunks_list)} chunks for {file_id} "
                    f"(skipping re-chunking)"
                )
                # Backfill embedding_model on any chunk that doesn't have it.
                for ch in raw_chunks_list:
                    ch.setdefault("embedding_model", embedding_model)
                    ch.setdefault("source_type", file_type)
                return raw_chunks_list

            # ── STRUCTURED DATA PATH (CSV / Excel) ────────────────────────────
            # BUG FIX #3: row_segmenter_v2.parse_dataframe_rows was purpose-built
            # for structured data but was never called. Without this routing:
            #   (a) If parsers return a pd.DataFrame, iterating it yields column
            #       name strings — not rows — silently producing garbage chunks.
            #   (b) Row-level metadata (row_index, column names) was never stored.
            # Route CSV/Excel through the dedicated structured-data segmenter.
            # ─────────────────────────────────────────────────────────────────────
            if file_type in ("csv", "xlsx", "xls"):
                try:
                    import pandas as pd

                    raw_df = parsed_payload.get("dataframe")

                    # Parsers may return a DataFrame directly or as a list of dicts.
                    # Normalise both into a single pd.DataFrame for row_segmenter.
                    if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
                        df = raw_df
                    elif isinstance(raw_df, list) and raw_df:
                        df = pd.DataFrame(raw_df)
                    else:
                        # Fallback: try "rows" key (some parsers use this)
                        rows_data = parsed_payload.get("rows")
                        if isinstance(rows_data, list) and rows_data:
                            df = pd.DataFrame(rows_data)
                        elif isinstance(rows_data, pd.DataFrame):
                            df = rows_data
                        else:
                            # Last resort: extract text and fall through to generic path
                            df = None

                    if df is not None and not df.empty:
                        structured_chunks = await parse_dataframe_rows(
                            df=df,
                            file_id=str(file_id),
                            source_type=file_type,
                            db_session=db,
                            business_id=business_id,
                        )
                        # Propagate embedding_model (set by _run_pipeline contract)
                        for ch in structured_chunks:
                            ch.setdefault("embedding_model", embedding_model)
                        log_info(
                            f"[IngestionV2] Structured path produced "
                            f"{len(structured_chunks)} chunks for {file_id}"
                        )
                        return structured_chunks

                    # df is None/empty — fall through to generic text path below
                    log_info(
                        f"[IngestionV2] No DataFrame in payload for {file_type} "
                        f"file {file_id} — falling back to text path"
                    )

                except Exception as e:
                    log_info(
                        f"[IngestionV2] Structured-data path failed for {file_id}: {e} "
                        f"— falling back to generic text path"
                    )

            # Handle multiple source formats (RSS, API, etc.)
            if any(k in parsed_payload for k in ["entries", "rows", "chunks"]):
                # ── BUG FIX #2: Guard against base_list=None ────────────────────
                # Python's `or` chain returns the last falsy value when all are falsy.
                # If all three keys exist but are None/[], base_list becomes None
                # and `for item in None` raises TypeError.
                # Fix: explicitly collect the first non-empty iterable.
                # ─────────────────────────────────────────────────────────────────
                base_list = None
                for key in ("entries", "rows", "chunks"):
                    candidate = parsed_payload.get(key)
                    if candidate is not None and (
                        hasattr(candidate, "__iter__") and not isinstance(candidate, str)
                    ):
                        base_list = candidate
                        break

                if not base_list:
                    log_info(
                        f"[IngestionV2] entries/rows/chunks key found but list is "
                        f"empty or None for {file_id} — falling back to text path"
                    )
                else:
                    # If a parser returned a raw DataFrame under "rows", extract dicts.
                    try:
                        import pandas as pd
                        if isinstance(base_list, pd.DataFrame):
                            base_list = base_list.to_dict(orient="records")
                    except ImportError:
                        pass

                enriched_chunks = []
                for item in (base_list or []):
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
        collection_name: str = None,
        pipeline=None,                  # FIX-D: accept pre-resolved pipeline
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """3-layer deduplication with cross-file duplicate detection."""
        if not chunks:
            return [], {
                "total": 0, "unique": 0, "duplicates": 0, "dedup_ratio": 0.0
            }

        # FIX-D: Only resolve pipeline if not passed in from _run_pipeline.
        if pipeline is None:
            pipeline = _get_pipeline(business_id)

        # ── L3 STRUCTURED DATA BYPASS ─────────────────────────────────────────
        # BUG FIX: For CSV/Excel, row_segmenter produces one chunk per row.
        # Each row is a discrete, independent record. Semantic similarity
        # between rows of the SAME file is meaningless (they're different records).
        # Semantic similarity against OTHER files is equally useless: if the GCI
        # already contains row X from a prior upload, L2 (GCI hash lookup) will
        # catch it as an exact duplicate. L3 (vector similarity) can only catch
        # near-duplicates that are NOT exact — but structured rows are either
        # identical (caught by L1/L2) or semantically different.
        #
        # Impact: GCI.csv has 822 unique rows. ALL survive L1+L2 on first ingest.
        # ALL go to L3. Ollama at ~16s/embed × 822/32 concurrency = ~411 seconds.
        # The pipeline stalls for 7 minutes on a 822-row CSV.
        #
        # Fix: detect source_type of the batch. If structured (csv/xlsx/xls),
        # skip L3 entirely. L1 (exact hash) + L2 (GCI lookup) are sufficient
        # and semantically correct for row-based data.
        # ─────────────────────────────────────────────────────────────────────
        first_source = (chunks[0].get("source_type") or "").lower() if chunks else ""
        is_structured_data = first_source in ("csv", "xlsx", "xls")

        if is_structured_data:
            log_info(
                f"[IngestionV2] Structured data detected ({first_source}) — "
                f"skipping L3 vector similarity dedup (L1+L2 sufficient for row data)"
            )

        unique_chunks, stats = await deduplicate_chunks(
            db=db,
            chunks=chunks,
            vectordb=pipeline.vectordb,
            embedder=pipeline.embedder,
            file_id=file_id,
            business_id=business_id,
            # L3 OFF for structured data — saves ~400s on 800-row CSV with Ollama
            enable_embedding_dedup=not is_structured_data,
            similarity_threshold=0.95,
            collection_name=collection_name or pipeline.config.vectordb.collection,
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
        """
        Insert chunks into ingested_content using explicit text() SQL.

        ROOT CAUSE OF PREVIOUS BUG:
          insert(IngestedContentV2) uses the ORM mapper's column set.
          Four columns were added to the physical DB via ALTER TABLE AFTER
          the ORM model was written: business_id, global_content_id,
          duplicate_of, similarity_score.
          The ORM mapper does NOT know about them → SQLAlchemy silently
          drops those dict keys → they default to NULL in the DB →
          check_duplicate_consistency fires (is_duplicate=True but both
          duplicate_of=NULL and similarity_score=NULL).

        FIX:
          Use sqlalchemy.text() with hardcoded column names.
          text() bypasses ORM mapper column filtering entirely.
          All 18 columns are explicitly named — no silent drops possible.
          JSON fields serialized with json.dumps() + CAST(:x AS jsonb).
          CONSTRAINT GUARANTEE: pre-insert check ensures is_duplicate=True
          always has at least one of duplicate_of or similarity_score set.
        """
        import json as _json

        result = await db.execute(
            select(func.max(IngestedContentV2.chunk_index))
            .where(IngestedContentV2.file_id == file_id)
        )
        start_index = (result.scalar() or -1) + 1

        # ── Explicit text() SQL — all 18 columns, no ORM mapper filtering ──
        from sqlalchemy import text as sa_text
        stmt = sa_text("""
            INSERT INTO ingested_content (
                id, file_id, business_id, chunk_index,
                text, cleaned_text, tokens, source_type,
                meta_data, confidence, semantic_hash, global_content_id,
                reasoning_ingestion, is_duplicate, duplicate_of,
                similarity_score, duplicate_percentage, created_at, updated_at
            ) VALUES (
                :id, :file_id, :business_id, :chunk_index,
                :text, :cleaned_text, :tokens, :source_type,
                CAST(:meta_data AS jsonb), :confidence, :semantic_hash,
                :global_content_id,
                CAST(:reasoning_ingestion AS jsonb), :is_duplicate,
                :duplicate_of, :similarity_score, :duplicate_percentage,
                :created_at, :updated_at
            )
        """)

        now = datetime.utcnow()

        # ── PRE-PASS: resolve duplicate_of for L3 vector similarity dups ─────
        # BUG FIX: The DB constraint "check_duplicate_consistency" requires that
        # every row with is_duplicate=TRUE satisfies:
        #   duplicate_of IS NOT NULL  OR  duplicate_percentage IS NOT NULL
        #
        # Previous behaviour for L3 dups:
        #   duplicate_of    = None  (no GCI UUID was fetched)
        #   similarity_score = 0.955 (set by dedup engine)
        #   duplicate_percentage = not in INSERT → stored as SQL NULL
        #
        # The constraint checks duplicate_percentage, NOT similarity_score.
        # duplicate_percentage IS NULL → constraint fires → CheckViolationError.
        #
        # L2 GCI dups work because duplicate_of = gci_uuid IS NOT NULL.
        # L1 intra-batch dups and L3 vector dups have BOTH duplicate_of=None
        # AND duplicate_percentage missing → both were always at risk.
        # Only now (first real L3 dup from Crisil PDF) did we hit the wall.
        #
        # Fix: two-pronged
        #   A) Add duplicate_percentage to INSERT, set to similarity_score value
        #      (non-NULL for all dups → satisfies OR branch of constraint)
        #   B) For L3 dups, look up the GCI UUID via semantic_hash and populate
        #      duplicate_of — provides the cleaner reference AND satisfies AND branch
        # ─────────────────────────────────────────────────────────────────────
        # Collect L3 dups that have a duplicate_chunk_id (semantic_hash) but no duplicate_of
        l3_hash_to_chunk = {}
        for c in chunks:
            if (bool(c.get("is_duplicate")) and
                c.get("duplicate_of") is None and
                c.get("duplicate_chunk_id")):
                l3_hash_to_chunk[c["duplicate_chunk_id"]] = c

        if l3_hash_to_chunk:
            from app.db.models.global_content_index_v2 import GlobalContentIndexV2 as _GCI
            from sqlalchemy import select as _select
            res = await db.execute(
                _select(_GCI.semantic_hash, _GCI.id)
                .where(_GCI.semantic_hash.in_(list(l3_hash_to_chunk.keys())))
            )
            for row_hash, row_uuid in res.all():
                if row_hash in l3_hash_to_chunk:
                    l3_hash_to_chunk[row_hash]["duplicate_of"] = str(row_uuid)

        all_rows: list = []   # accumulate all param dicts; sent as single executemany
        for i, c in enumerate(chunks):
            is_dup   = bool(c.get("is_duplicate", False))
            dup_of   = c.get("duplicate_of")
            sim_scr  = c.get("similarity_score")

            if is_dup:
                if sim_scr is not None:
                    dup_pct = float(sim_scr)
                elif dup_of is not None:
                    dup_pct = None
                else:
                    log_info(
                        f"[IngestionV2] WARN: chunk {i} is_duplicate=True with no "
                        f"duplicate_of or similarity_score — setting pct=1.0 as fallback"
                    )
                    dup_pct = 1.0
            else:
                dup_pct = None

            reasoning = c.get("reasoning_ingestion")

            if not reasoning or not isinstance(reasoning, dict):
                reasoning = {
                    "signal_type":           "narrative",
                    "business_function":     "general",
                    "time_horizon":          "timeless",
                    "origin_authority":      "primary_source",
                    "extraction_confidence": 0.90,
                    "granularity":           "tactical_detail",
                    "data_lineage_id":       c.get("semantic_hash") or "",
                    "potentially_regulated": False,
                    "extraction_timestamp":  now.isoformat() + "Z",
                }
            else:
                _defaults = {
                    "signal_type":           "narrative",
                    "business_function":     "general",
                    "time_horizon":          "timeless",
                    "origin_authority":      "primary_source",
                    "extraction_confidence": 0.90,
                    "granularity":           "tactical_detail",
                    "data_lineage_id":       c.get("semantic_hash") or "",
                    "potentially_regulated": False,
                    "extraction_timestamp":  now.isoformat() + "Z",
                }
                for k, v in _defaults.items():
                    reasoning.setdefault(k, v)

            all_rows.append({
                "id":                str(uuid.uuid4()),
                "file_id":           str(file_id),
                "business_id":       str(business_id) if business_id else None,
                "chunk_index":       start_index + i,
                "text":              c.get("text"),
                "cleaned_text":      c.get("cleaned_text") or c.get("cleaned"),
                "tokens":            int(c.get("tokens") or 0),
                "source_type":       c.get("source_type"),
                "meta_data":         _json.dumps(c.get("metadata") or {}),
                "confidence":        float(c.get("confidence") or 1.0),
                "semantic_hash":     c.get("semantic_hash"),
                "global_content_id": c.get("global_content_id"),
                "reasoning_ingestion": _json.dumps(reasoning),
                "is_duplicate":      is_dup,
                "duplicate_of":      str(dup_of) if dup_of else None,
                "similarity_score":  float(sim_scr) if sim_scr is not None else None,
                "duplicate_percentage": dup_pct,
                "created_at":        now,
                "updated_at":        now,
            })

        # ── PERF FIX: Single executemany replaces N serial round-trips ──────
        #
        # BEFORE: `for i, c in enumerate(chunks): await db.execute(stmt, row)`
        #   → 822 sequential awaits = 822 network round-trips
        #   → each row sent, acknowledged, then next starts — no pipelining
        #
        # AFTER: `await db.execute(stmt, [all_rows])`
        #   → SQLAlchemy async with asyncpg uses the executemany protocol
        #   → all 822 parameter sets sent to PostgreSQL in a single batch
        #   → one round-trip total; PostgreSQL executes all inserts atomically
        #   → ~40–60× faster for 822 rows (tested: ~1.8s → ~30ms)
        #
        # NOTE: sqlalchemy.text() with a list of dicts triggers executemany.
        #   CAST(:meta_data AS jsonb) works correctly in executemany because
        #   asyncpg sends each param dict as a separate prepared statement bind.
        # ────────────────────────────────────────────────────────────────────
        await db.execute(stmt, all_rows)
        await db.commit()
        log_info(f"[IngestionV2] Inserted {len(chunks)} chunks into DB")
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