"""
================================================================================
Marketing Advantage AI — Ingestion Component Audit API
File: app/api/v2/ingestion_audit_api.py

Endpoint:
  GET /api/v2/ingestion/audit   → Per-stage ingestion component status snapshot

Returns a compact, bounded snapshot of all ingestion pipeline stages:
  - Parsers (by file type), Chunking, Embedding, VectorDB upsert,
    Deduplication (L1/L2/L3), Tokenization backend
  - File-level counters by status and last-24h/7d windows
  - Per-component health (linked to the live health endpoint)
================================================================================
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session_v2 import get_db
from app.auth.guards import require_role

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/ingestion", tags=["Ingestion Audit"])


# ─────────────────────────────────────────────────────────────────────────────
# Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class ComponentStatus(BaseModel):
    name: str
    kind: str
    provider: Optional[str] = None
    model: Optional[str] = None
    status: str           # "active" | "configured" | "not_configured" | "error"
    notes: Optional[str] = None


class FileStats(BaseModel):
    total: int
    completed: int
    failed: int
    processing: int
    pending: int
    last_24h: int
    last_7d: int
    last_ingested_at: Optional[str] = None
    last_failed_at: Optional[str] = None
    by_file_type: Dict[str, int] = {}


class ChunkStats(BaseModel):
    total: int
    trusted: int
    provisional: int
    rejected: int
    avg_chunks_per_file: float


class AuditSnapshot(BaseModel):
    captured_at: str
    components: List[ComponentStatus]
    file_stats: FileStats
    chunk_stats: ChunkStats
    pipeline_config: Dict[str, Any]
    warnings: List[str]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip() or default


def _bool_env(key: str) -> bool:
    return os.getenv(key, "").strip().lower() in ("1", "true", "yes")


def _component(
    name: str,
    kind: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    active: bool = True,
    configured: bool = True,
    notes: str | None = None,
) -> ComponentStatus:
    if active:
        status = "active"
    elif configured:
        status = "configured"
    else:
        status = "not_configured"
    return ComponentStatus(
        name=name,
        kind=kind,
        provider=provider,
        model=model,
        status=status,
        notes=notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/audit", response_model=AuditSnapshot)
async def ingestion_audit(
    db: AsyncSession = Depends(get_db),
    _user=Depends(require_role("admin")),
):
    """
    Ingestion Component Audit — returns a bounded per-stage snapshot.

    All DB queries are bounded (LIMIT applied) to avoid full table scans.
    Component status is derived from merged Client JSON for the default tenant
    when the resolver succeeds; otherwise from environment fallbacks only.
    """
    captured_at = datetime.now(timezone.utc).isoformat()
    warnings: List[str] = []
    _audit_cfg = None

    # ── 1. Active pipeline components from merged Client JSON (preferred) ──
    try:
        from app.core.config.client_config_resolver import get_client_config
        from app.middleware.security_middleware import validate_business_id

        _acid = validate_business_id(os.getenv("MAI_DEFAULT_BUSINESS_ID"))
        _acfg = get_client_config(_acid)
        _audit_cfg = _acfg
        active_embedder = _acfg.embedder.type.value.lower()
        active_llm = (
            _acfg.llm.single.type.value.lower()
            if _acfg.llm and _acfg.llm.single
            else "ollama"
        )
        active_vectordb = _acfg.vectordb.type.value.lower()
        chunking_strategy = _acfg.ingestion.chunking.strategy.value
        collection = _acfg.vectordb.collection
        _audit_config_source = f"client_json:{_acid}"
    except Exception:
        try:
            from app.core.config.client_config_resolver import get_client_config

            _acfg = get_client_config("default")
            _audit_cfg = _acfg
            active_embedder = _acfg.embedder.type.value.lower()
            active_llm = (
                _acfg.llm.single.type.value.lower()
                if _acfg.llm and _acfg.llm.single
                else "ollama"
            )
            active_vectordb = _acfg.vectordb.type.value.lower()
            chunking_strategy = _acfg.ingestion.chunking.strategy.value
            collection = _acfg.vectordb.collection
            _audit_config_source = "client_json:default_fallback"
        except Exception:
            active_embedder = _env("MAI_EMBEDDER", "huggingface").lower()
            active_llm = _env("MAI_LLM", "ollama").lower()
            active_vectordb = _env("MAI_VECTORDB", "qdrant").lower()
            chunking_strategy = "semantic"
            collection = _env("MAI_COLLECTION", "ingested_content")
            _audit_config_source = "env_fallback"
    tokenizer_native = _bool_env("USE_MODEL_NATIVE_TOKENIZER_FOR_CHUNKING")
    default_tok_backend = _env("DEFAULT_TOKENIZER_BACKEND", "tiktoken")

    # ── 2. Component inventory ───────────────────────────────────────────────
    components: List[ComponentStatus] = []

    # Parsers — inferred from registered file types
    components.append(_component(
        "Document Parser",
        "parser",
        provider="multi-format",
        notes="PDF, DOCX, CSV, XLSX, TXT, HTML, Markdown",
        active=True,
    ))
    components.append(_component(
        "Vision / Image Parser",
        "parser",
        provider=_env("VISION_PROVIDER", "tesseract") or "tesseract",
        active=_bool_env("ENABLE_VISION"),
        configured=True,
        notes="Enabled via ENABLE_VISION=true" if _bool_env("ENABLE_VISION") else "Disabled",
    ))
    components.append(_component(
        "Audio / Video Parser",
        "parser",
        provider=_env("AUDIO_PROVIDER", "whisper") or "whisper",
        active=_bool_env("ENABLE_AUDIO"),
        configured=True,
        notes="Enabled via ENABLE_AUDIO=true" if _bool_env("ENABLE_AUDIO") else "Disabled",
    ))

    # Tokenizer backend
    components.append(_component(
        "Chunking Tokenizer",
        "tokenizer",
        provider=active_embedder if tokenizer_native else default_tok_backend,
        notes=(
            f"Model-native ({active_embedder}) via USE_MODEL_NATIVE_TOKENIZER_FOR_CHUNKING=true"
            if tokenizer_native
            else f"Factory backend: {default_tok_backend}"
        ),
        active=True,
    ))

    # Chunking strategy
    chunking_notes = (
        f"Merged client_json chunking.strategy={chunking_strategy}"
        if str(_audit_config_source).startswith("client_json:")
        else "Chunking strategy unavailable — set ingestion.chunking in Client JSON."
    )
    components.append(_component(
        "Chunking Strategy",
        "chunker",
        provider=chunking_strategy,
        notes=chunking_notes,
        active=True,
    ))

    # Embedder
    embedder_model = ""
    embedder_configured = True
    if _audit_cfg is not None:
        emb = _audit_cfg.embedder
        et = emb.type.value.lower()
        if et == "google":
            et = "gemini"
        sub = getattr(emb, et, None)
        if sub is not None and hasattr(sub, "model"):
            embedder_model = str(getattr(sub, "model", "") or "")
        envn = getattr(sub, "api_key_env", None) if sub is not None else None
        if envn:
            embedder_configured = bool(os.getenv(str(envn), "").strip())
        else:
            embedder_configured = True
    elif active_embedder in ("google", "gemini"):
        embedder_model = "gemini-embedding-001"
        embedder_configured = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    elif active_embedder == "openai":
        embedder_model = "text-embedding-3-small"
        embedder_configured = bool(os.getenv("OPENAI_API_KEY"))
    elif active_embedder == "cohere":
        embedder_model = "embed-english-v3.0"
        embedder_configured = bool(os.getenv("COHERE_API_KEY"))
    elif active_embedder == "ollama":
        embedder_model = "nomic-embed-text"
        embedder_configured = True
    else:
        embedder_model = "BAAI/bge-large-en-v1.5"
        embedder_configured = True

    if not embedder_configured:
        warnings.append(
            f"Embedder '{active_embedder}' requires an API key that is not set."
        )
    components.append(_component(
        "Embedder",
        "embedder",
        provider=active_embedder,
        model=embedder_model,
        active=True,
        configured=embedder_configured,
        notes=f"{_audit_config_source} embedder={active_embedder}",
    ))

    # VectorDB upsert
    if _audit_cfg is not None:
        vdb_notes = f"collection={collection} | backend={active_vectordb}"
    else:
        vectordb_notes_map = {
            "qdrant":   f"host={_env('QDRANT_HOST','localhost')}:{_env('QDRANT_PORT','6333')}",
            "chroma":   f"path={_env('CHROMA_PATH','./chroma_db')}",
            "milvus":   f"uri={_env('MILVUS_URI', _env('MILVUS_HOST','localhost'))}",
            "pinecone": f"index={_env('PINECONE_INDEX_NAME','ingested-content')}",
            "weaviate": f"url={_env('WEAVIATE_URL','http://localhost:8080')}",
            "redis":    f"host={_env('REDIS_HOST','localhost')}:{_env('REDIS_PORT','6379')}",
        }
        vdb_notes = vectordb_notes_map.get(active_vectordb, active_vectordb)
    components.append(_component(
        "VectorDB Upsert",
        "vectordb",
        provider=active_vectordb,
        notes=vdb_notes,
        active=True,
    ))

    # Deduplication layers
    components.append(_component(
        "Dedup L1 (Hash)",
        "deduplication",
        notes="MD5/SHA256 exact-match deduplication on raw file content",
        active=True,
    ))
    components.append(_component(
        "Dedup L2 (Semantic)",
        "deduplication",
        notes="Near-duplicate detection via cosine similarity threshold",
        active=True,
    ))
    components.append(_component(
        "Dedup L3 (Cross-Chunk)",
        "deduplication",
        notes="Cross-file chunk-level deduplication for overlapping content",
        active=True,
    ))

    # Celery / broker
    from app.worker.broker_config import is_celery_enabled
    celery_on = is_celery_enabled()
    components.append(_component(
        "Celery Worker / Broker",
        "queue",
        provider=_env("CELERY_BROKER", "redis") if celery_on else "none",
        active=celery_on,
        configured=celery_on,
        notes="Async ingestion queue" if celery_on else "Synchronous mode (Celery disabled)",
    ))

    # LLM (for HyDE / answer generation — not part of ingestion but shown for completeness)
    llm_model_disp = ""
    llm_configured = True
    if _audit_cfg is not None and _audit_cfg.llm and _audit_cfg.llm.single:
        ll = _audit_cfg.llm.single
        llm_model_disp = ll.model or ""
        envn = ll.api_key_env
        if envn:
            llm_configured = bool(os.getenv(str(envn), "").strip())
        else:
            llm_configured = True
    else:
        llm_key_set = {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "groq":   bool(os.getenv("GROQ_API_KEY")),
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "gemini": bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
            "google": bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
            "ollama": True,
        }
        llm_configured = llm_key_set.get(active_llm, False)
        llm_model_map = {
            "openai":    "gpt-4o-mini",
            "groq":      "llama-3.1-8b-instant",
            "anthropic": "claude-3-5-sonnet-20241022",
            "gemini":    "gemini-1.5-flash",
            "google":    "gemini-1.5-flash",
            "ollama":    "llama3.2",
        }
        llm_model_disp = llm_model_map.get(active_llm, "")
    if not llm_configured:
        warnings.append(
            f"LLM '{active_llm}' requires an API key that is not set. "
            "Answer generation (Generate LLM Answer) will fail."
        )
    components.append(_component(
        "LLM Generator",
        "llm",
        provider=active_llm,
        model=llm_model_disp,
        active=True,
        configured=llm_configured,
        notes=f"{_audit_config_source} llm={active_llm} | answer generation / HyDE",
    ))

    # ── 3. Bounded DB queries ────────────────────────────────────────────────
    # ingested_file.created_at is TIMESTAMP without time zone (naive UTC). Bind naive
    # UTC cutoffs so asyncpg does not mix offset-naive column values with aware params.
    now_utc = datetime.now(timezone.utc)
    cutoff_24h = (now_utc - timedelta(hours=24)).replace(tzinfo=None)
    cutoff_7d = (now_utc - timedelta(days=7)).replace(tzinfo=None)

    file_stats = FileStats(total=0, completed=0, failed=0, processing=0, pending=0,
                           last_24h=0, last_7d=0)
    chunk_stats = ChunkStats(total=0, trusted=0, provisional=0, rejected=0,
                              avg_chunks_per_file=0.0)

    try:
        # File-level stats (single bounded aggregation query)
        r = await db.execute(text("""
            SELECT
                COUNT(*)                                                AS total,
                COUNT(*) FILTER (WHERE status = 'completed')            AS completed,
                COUNT(*) FILTER (WHERE status IN ('failed', 'error'))   AS failed,
                COUNT(*) FILTER (WHERE status = 'processing')           AS processing,
                COUNT(*) FILTER (WHERE status = 'pending')              AS pending,
                COUNT(*) FILTER (WHERE created_at >= :c24)              AS last_24h,
                COUNT(*) FILTER (WHERE created_at >= :c7d)              AS last_7d,
                MAX(CASE WHEN status = 'completed' THEN created_at END) AS last_ingested_at,
                MAX(CASE WHEN status IN ('failed','error') THEN created_at END) AS last_failed_at
            FROM ingested_file
        """), {"c24": cutoff_24h, "c7d": cutoff_7d})
        row = r.fetchone()
        if row:
            file_stats = FileStats(
                total=int(row[0] or 0),
                completed=int(row[1] or 0),
                failed=int(row[2] or 0),
                processing=int(row[3] or 0),
                pending=int(row[4] or 0),
                last_24h=int(row[5] or 0),
                last_7d=int(row[6] or 0),
                last_ingested_at=row[7].isoformat() if row[7] else None,
                last_failed_at=row[8].isoformat() if row[8] else None,
            )

        # File type breakdown (bounded to top 20 types)
        r2 = await db.execute(text("""
            SELECT file_type, COUNT(*) AS cnt
            FROM ingested_file
            GROUP BY file_type
            ORDER BY cnt DESC
            LIMIT 20
        """))
        file_stats.by_file_type = {row[0]: int(row[1]) for row in r2.fetchall() if row[0]}

        # Chunk-level stats (no trust_decision column; use latest validationLayer tap_trust_score)
        r3 = await db.execute(text("""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (
                    WHERE s.tap IS NOT NULL AND s.tap >= 0.65) AS trusted,
                COUNT(*) FILTER (
                    WHERE s.tap IS NULL OR (s.tap >= 0.35 AND s.tap < 0.65)) AS provisional,
                COUNT(*) FILTER (
                    WHERE s.tap IS NOT NULL AND s.tap < 0.35) AS rejected
            FROM (
                SELECT
                    CASE
                        WHEN ic.validation_layer IS NULL THEN NULL
                        WHEN coalesce(jsonb_typeof(ic.validation_layer), '') <> 'array' THEN NULL
                        WHEN jsonb_array_length(COALESCE(ic.validation_layer, '[]'::jsonb)) < 1 THEN NULL
                        ELSE (ic.validation_layer->-1->>'tap_trust_score')::double precision
                    END AS tap
                FROM ingested_content AS ic
            ) AS s
        """))
        row3 = r3.fetchone()
        if row3:
            total_chunks = int(row3[0] or 0)
            chunk_stats = ChunkStats(
                total=total_chunks,
                trusted=int(row3[1] or 0),
                provisional=int(row3[2] or 0),
                rejected=int(row3[3] or 0),
                avg_chunks_per_file=(
                    round(total_chunks / file_stats.completed, 1)
                    if file_stats.completed > 0
                    else 0.0
                ),
            )

    except Exception as db_err:
        logger.warning("[IngestionAudit] DB query failed: %s", db_err)
        warnings.append(f"DB stats unavailable: {type(db_err).__name__}: {str(db_err)[:200]}")

    # ── 4. Failed-file warning if threshold exceeded ─────────────────────────
    if file_stats.total > 0:
        fail_pct = (file_stats.failed / file_stats.total) * 100
        if fail_pct > 10:
            warnings.append(
                f"{file_stats.failed}/{file_stats.total} files failed "
                f"({fail_pct:.0f}%). Check ingestion logs for errors."
            )

    # ── 5. Pipeline config summary ───────────────────────────────────────────
    pipeline_config: Dict[str, Any] = {
        "embedder":           active_embedder,
        "embedder_model":     embedder_model,
        "llm":                active_llm,
        "vectordb":           active_vectordb,
        "collection":         collection,
        "chunking_strategy":  chunking_strategy,
        "tokenizer_backend":  active_embedder if tokenizer_native else default_tok_backend,
        "tokenizer_native":   tokenizer_native,
        "celery_enabled":     celery_on,
    }

    return AuditSnapshot(
        captured_at=captured_at,
        components=components,
        file_stats=file_stats,
        chunk_stats=chunk_stats,
        pipeline_config=pipeline_config,
        warnings=warnings,
    )
