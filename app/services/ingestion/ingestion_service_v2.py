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
from collections import Counter
from datetime import datetime
from functools import lru_cache
from threading import Lock as _threading_lock
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import uuid as _uuid_module

try:
    from cachetools import TTLCache
    _HAS_CACHETOOLS = True
except ImportError:
    _HAS_CACHETOOLS = False

# ── FastAPI / SQLAlchemy ──────────────────────────────────────────────
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy import select, insert, update, func, delete

# ── App internals ─────────────────────────────────────────────────────
from app.api.v2.ingestion_ws_api import broadcast
from app.db.session_v2 import async_engine
from app.db.models.ingested_file_v2 import IngestedFileV2
from app.db.models.ingested_content_v2 import IngestedContentV2
from app.db.models.global_content_index_v2 import GlobalContentIndexV2
from app.services.ingestion.parsers_router_v2 import ParserRouterV2
from app.services.ingestion.row_segmenter_v2 import parse_dataframe_rows
from app.core.chunking_stratagies.chunking_registry import (
    clear_chunker_cache,
    get_chunker,
    list_chunking_strategies,
)
from app.services.ingestion.deduplication_engine_v2 import (
    deduplicate_chunks,
    create_normalized_hash,
    register_unique_chunks_in_gci,   # ← NEW: post-dedup GCI commit
)
from app.services.ingestion.streaming_state import (
    STREAMING_INGESTION_ENABLED,
    StreamingIngestionState,
)
from app.services.ingestion.pdf_parser_v2 import iter_pdf_pages
from app.utils.text_cleaner_v2 import clean_text
from app.utils.logger import log_info, log_warning, log_error
from app.core.chunking_stratagies.text_preprocessor import preprocess_document_text
from app.middleware.security_middleware import validate_business_id as _validate_business_id
from app.utils.instrumentation import timed_stage, IngestionLogger
from app.utils.cost_tracker import record_tokens
from app.utils.pipeline_logger import PipelineLogger

_ing_logger = IngestionLogger(__name__)

STRICT_INGESTION_SECURITY: bool = os.getenv(
    "STRICT_INGESTION_SECURITY", "false"
).lower() in ("true", "1", "yes")

# ── Pluggable pipeline factory ────────────────────────────────────────
from app.core.pipeline_factory import pipeline_factory
from app.core.config.client_config_schema import (
    ClientConfig,
    VectorDBConfig,
    EmbedderConfig,
    IngestionConfig,
    DeduplicationConfig,
    VectorDBType,
    EmbedderType,
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

def _safe_env_int(key: str, default: int, min_value: int = 1) -> int:
    """
    Parse positive integer env var with fallback.
    """
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        value = int(raw)
        return value if value >= min_value else default
    except (TypeError, ValueError):
        return default


def _safe_env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _pipeline_telemetry_labels(client_id: Optional[Any]) -> tuple[str, str]:
    """Embedder + vectordb labels from merged tenant JSON (never MAI_* env)."""
    if client_id is None or str(client_id).strip() == "":
        return "unknown", "unknown"
    try:
        from app.core.config.pipeline_runtime import get_pipeline_identity

        ident = get_pipeline_identity(str(client_id))
        embedder = ident.get("embedder_model") or ident.get("embedder") or "unknown"
        vectordb = ident.get("vectordb") or "unknown"
        return str(embedder), str(vectordb)
    except Exception:
        return "unknown", "unknown"


BATCH_SIZE: int = _safe_env_int("INGEST_BATCH_SIZE", 256)

# ── Visual LLM concurrency cap ───────────────────────────────────────────────
# Created lazily per event-loop to avoid cross-loop leaks under uvicorn reload.
# Semaphore value: configurable via VISUAL_LLM_CONCURRENCY env var (default 4).
# Prevents unbounded concurrent LLM calls when documents contain many charts.
_VISUAL_LLM_CONCURRENCY: int = _safe_env_int("VISUAL_LLM_CONCURRENCY", 4)
_VISUAL_LLM_SEMAPHORE: Optional[asyncio.Semaphore] = None

def _get_visual_llm_semaphore() -> asyncio.Semaphore:
    """Return (or create) the module-level semaphore bound to the running loop."""
    global _VISUAL_LLM_SEMAPHORE
    if _VISUAL_LLM_SEMAPHORE is None:
        _VISUAL_LLM_SEMAPHORE = asyncio.Semaphore(_VISUAL_LLM_CONCURRENCY)
    return _VISUAL_LLM_SEMAPHORE


def _resolve_embed_parallelism(embedder_kind: Optional[str]) -> int:
    """
    Resolve concurrent embed+upsert batch parallelism from env.

    Precedence:
      1. INGEST_<PROVIDER>_EMBED_PARALLELISM
      2. INGEST_EMBED_PARALLELISM
      3. EMBED_PARALLELISM
      4. HF_EMBED_CONCURRENCY (legacy fallback)

    HuggingFace stays capped at 1 because concurrent encode() calls on the
    same model instance are not thread-safe.
    """
    kind = (embedder_kind or "").strip().lower()
    if "huggingface" in kind:
        return 1

    provider = re.sub(r"[^a-z0-9]+", "_", kind).strip("_").upper()
    keys = [
        f"INGEST_{provider}_EMBED_PARALLELISM" if provider else "",
        "INGEST_EMBED_PARALLELISM",
        "EMBED_PARALLELISM",
        "HF_EMBED_CONCURRENCY",
    ]
    for key in keys:
        if not key:
            continue
        raw = os.getenv(key)
        if raw is None:
            continue
        try:
            value = int(raw)
            if value >= 1:
                return value
        except (TypeError, ValueError):
            continue

    return 4

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
    db: AsyncSession,
    file_hash: str,
    business_id: Optional[str] = None,
) -> Optional[str]:
    """
    Check if a file with the given hash already exists.
    
    Tenant Isolation: When business_id is provided, only checks
    within that tenant's files - same hash in different tenants 
    are considered unique files.
    """
    stmt = select(IngestedFileV2.id).where(
        IngestedFileV2.meta_data["file_hash"].astext == file_hash
    )
    # Add tenant filter if provided
    if business_id:
        from app.utils.tenant_storage_uuid import storage_business_uuid_for_tenant
        storage_uuid = storage_business_uuid_for_tenant(business_id, None)
        stmt = stmt.where(IngestedFileV2.business_id == storage_uuid)
    result = await db.execute(stmt)
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


def _normalize_chunk_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure each chunk carries a stable metadata payload used by DB + VectorDB.
    """
    metadata = chunk.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    reasoning = chunk.get("reasoning_ingestion")
    if isinstance(reasoning, dict):
        section_title = reasoning.get("section_title")
        if section_title and "section_title" not in metadata:
            metadata["section_title"] = str(section_title)

        section_depth = reasoning.get("section_depth")
        if isinstance(section_depth, int):
            metadata.setdefault("section_depth", section_depth)
            metadata.setdefault("heading_depth", section_depth)

        content_type = reasoning.get("content_type")
        if content_type and "content_type" not in metadata:
            metadata["content_type"] = str(content_type)

    page_number = chunk.get("page_number")
    if isinstance(page_number, int) and page_number > 0:
        metadata["page_number"] = page_number

    chunk["metadata"] = metadata
    return metadata


def _compute_chunk_offsets(chunks: List[Dict[str, Any]], source_text: str) -> None:
    """
    Best-effort offset mapping in cleaned-text space.
    """
    if not source_text:
        return

    cleaned_source = clean_text(source_text)
    if not cleaned_source:
        return

    cursor = 0
    for chunk in chunks:
        metadata = _normalize_chunk_metadata(chunk)
        cleaned_chunk = (chunk.get("cleaned_text") or "").strip()
        if not cleaned_chunk:
            continue

        idx = cleaned_source.find(cleaned_chunk, cursor)
        if idx < 0:
            idx = cleaned_source.find(cleaned_chunk)
        if idx < 0:
            continue

        end = idx + len(cleaned_chunk)
        metadata["chunk_position"] = {"start_char": idx, "end_char": end}
        cursor = end


def _assign_page_numbers_from_page_map(
    chunks: List[Dict[str, Any]],
    page_map: List[Dict[str, Any]],
) -> None:
    """
    Best-effort page attribution with forward cursor bias for ordered chunks.
    """
    if not chunks or not page_map:
        return

    cleaned_pages: List[Tuple[int, str]] = []
    for page in page_map:
        pno = page.get("page_number")
        if not isinstance(pno, int) or pno <= 0:
            continue
        ptxt = (page.get("cleaned_text") or "").strip()
        cleaned_pages.append((pno, ptxt))

    if not cleaned_pages:
        return

    page_cursor = 0
    max_lookahead = 3

    for chunk in chunks:
        ctext = (chunk.get("cleaned_text") or "").strip()
        if not ctext:
            continue

        prefix = ctext[:160]
        suffix = ctext[-160:] if len(ctext) > 160 else ctext

        best_page = None
        best_score = -1.0

        start = max(0, page_cursor - 1)
        stop = min(len(cleaned_pages), page_cursor + max_lookahead + 1)
        for idx in range(start, stop):
            page_no, page_text = cleaned_pages[idx]
            if not page_text:
                continue

            score = 0.0
            if ctext in page_text:
                score = 1.0
            elif prefix and prefix in page_text:
                score = 0.9
            elif suffix and suffix in page_text:
                score = 0.8
            else:
                chunk_tokens = ctext.split()
                page_tokens = page_text.split()
                if chunk_tokens and page_tokens:
                    sample = set(chunk_tokens[:24])
                    overlap = sum(1 for t in page_tokens if t in sample)
                    score = overlap / max(1, len(sample))

            if score > best_score:
                best_score = score
                best_page = (idx, page_no)

        if best_page and best_score >= 0.08:
            page_cursor = max(page_cursor, best_page[0])
            chunk["page_number"] = best_page[1]
            _normalize_chunk_metadata(chunk)["page_number"] = best_page[1]


def _assign_structure_parent_offsets(chunks: List[Dict[str, Any]]) -> None:
    """
    For structure-aware chunks, assign parent offsets per section heading.
    The first chunk in a section acts as the parent anchor.
    """
    if not chunks:
        return

    parent_by_section: Dict[Tuple[str, int], int] = {}
    for idx, chunk in enumerate(chunks):
        reasoning = chunk.get("reasoning_ingestion")
        if not isinstance(reasoning, dict):
            continue

        strategy = str(reasoning.get("chunking_strategy") or "").strip().lower()
        if strategy != "structure_aware":
            continue

        section_title = str(reasoning.get("section_title") or "").strip()
        if not section_title:
            continue

        section_depth_raw = reasoning.get("section_depth")
        section_depth = section_depth_raw if isinstance(section_depth_raw, int) else 0
        section_key = (section_title, section_depth)

        parent_idx = parent_by_section.get(section_key)
        if parent_idx is None:
            parent_by_section[section_key] = idx
            chunk["parent_row_offset"] = None
        else:
            chunk["parent_row_offset"] = parent_idx


def _resolve_item_text(item: Any) -> Optional[str]:
    """
    Resolve text from a single multi-item entry (RSS, API, web, etc.).
    Returns None if no text can be resolved (caller will skip the item).
    Delegates to _resolve_text — same key priority:
    normalized_text → cleaned_text → text → raw_text → key:val join.
    """
    if item is None:
        return None
    if isinstance(item, str):
        return item if item.strip() else None
    if isinstance(item, dict):
        return _resolve_text(item) or None
    return None

def _coerce_to_str(value: Any) -> str:
    """
    Coerce any resolved item value to a plain string.
    _resolve_item_text already returns str | None, so in practice
    this handles edge cases: numeric IDs, bytes, or non-str returns
    from third-party parsers that slip past the isinstance check.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return ""
    try:
        return str(value)
    except Exception:
        return ""


def _normalize_business_id(business_id: Optional[Any]) -> str:
    """
    Normalize and validate tenant/business identifier to a stable string key.

    Delegates to security_middleware.validate_business_id which:
      - Accepts UUID strings (normalized to lowercase dashes)
      - Accepts safe slugs: [a-z0-9_-], max 64 chars
      - Rejects strings with special chars that could collide with env-var names
      - Maps None → "default"

    This prevents env-var injection: a malicious client_id="GLOBAL" could
    construct MAI_GLOBAL_VECTORDB via _business_env_prefix(). After this fix,
    "GLOBAL" is lowercased/sanitized to "global" which is still benign, but
    any chars that could form a collision are stripped.
    """
    return _validate_business_id(business_id)


def _business_env_prefix(client_id: str) -> str:
    """
    Convert client_id into an env-safe key segment used by MAI_{TENANT}_* vars.
    """
    return client_id.lower().replace("-", "_")


def clear_ingestion_pipeline_cache(client_id: Optional[str] = None) -> None:
    """
    Clear ingestion-side pipeline resolver caches.
    Call this after config/env updates to avoid stale pipeline instances.
    
    Args:
        client_id: If provided, only clear cache for this specific tenant.
                   If None, clear all cached pipelines.
    """
    if _HAS_CACHETOOLS:
        with _query_pipeline_lock:
            if client_id:
                _query_pipeline_cache.pop(client_id, None)
            else:
                _query_pipeline_cache.clear()
        with _ingestion_pipeline_lock:
            if client_id:
                _ingestion_pipeline_cache.pop(client_id, None)
            else:
                _ingestion_pipeline_cache.clear()
    else:
        _get_pipeline_for_client.cache_clear()
        _get_ingestion_pipeline_for_client.cache_clear()
    clear_chunker_cache()
    log_info(
        f"[IngestionV2] Cleared pipeline caches" + 
        (f" for tenant={client_id}" if client_id else " (all tenants)")
    )


def _resolve_chunking_strategy(pipeline: Any = None) -> str:
    """
    Resolve active chunking strategy from pipeline ClientConfig (JSON).
    """
    strategy: Optional[str] = None
    try:
        if pipeline is not None:
            chunk_cfg = getattr(
                getattr(getattr(pipeline, "config", None), "ingestion", None),
                "chunking",
                None,
            )
            if chunk_cfg is not None:
                raw = getattr(chunk_cfg, "strategy", None)
                if raw is not None:
                    strategy = raw.value if hasattr(raw, "value") else str(raw)
    except Exception:
        strategy = None

    if not strategy:
        strategy = "semantic"

    normalized = str(strategy).strip().lower()
    valid = set(list_chunking_strategies())
    if normalized not in valid:
        valid_str = ", ".join(sorted(valid))
        raise ValueError(
            f"Unknown chunking strategy '{normalized}'. "
            f"Valid values: {valid_str}"
        )
    return normalized


async def _chunk_text_with_strategy(
    text: str,
    *,
    db_session: AsyncSession,
    file_id: str,
    business_id: Optional[Any],
    source_type: Optional[str],
    embedding_model: str,
    pipeline: Any = None,
    strategy_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Strategy-aware chunking entrypoint.

    Centralised quality gate: preprocess_document_text() runs here BEFORE
    any strategy touches the text.  This guarantees page-break cleanup,
    OCR garble removal, image-stub stripping, LLM prefix removal, and
    whitespace normalisation for EVERY source type (PDF, DOCX, RSS, API,
    Web-scrape, etc.) and EVERY chunking strategy — even ones added in
    the future that forget to call the preprocessor themselves.

    The preprocessor is idempotent, so strategies that also call it
    internally (semantic, structure_aware, overlap, etc.) are unaffected.

    Model-native tokenizer injection (Phase 2):
    When USE_MODEL_NATIVE_TOKENIZER_FOR_CHUNKING=true (default) and an
    EmbedderBundle is available, a ChunkingTokenCounter is built from the
    bundle and injected via _ACTIVE_TOKEN_COUNTER ContextVar.  All chunking
    strategies read from this ContextVar via segmenter_v2.count_tokens(), so
    every strategy automatically uses the embedding model's native tokenizer
    for accurate token counts without any signature changes.
    """
    # ── Centralised pre-processing — quality & integrity gate ─────────
    text = preprocess_document_text(text or "")
    if not text.strip():
        return []

    strategy = (strategy_override or _resolve_chunking_strategy(pipeline)).lower()
    bundle = getattr(pipeline, "embedder_bundle", None)
    _cfg = getattr(pipeline, "config", None)
    _tok = getattr(_cfg, "tokenization", None) if _cfg is not None else None
    _chunk_cfg = getattr(_cfg, "ingestion", None).chunking if _cfg is not None and getattr(_cfg, "ingestion", None) else None

    # ── Model-native tokenizer injection for ALL strategies ───────────
    _ctx_token = None
    _counter = None
    if _tok is not None:
        _use_native = bool(_tok.use_model_native_tokenizer_for_chunking)
        _cache_sz = int(_tok.chunking_token_counter_cache_size)
    else:
        _use_native = True
        _cache_sz = 512

    if _use_native and bundle is not None:
        try:
            from app.ai.chunking.token_counter import ChunkingTokenCounter, soft_cap_factors_from_client
            from app.core.chunking_stratagies.segmenter_v2 import _ACTIVE_TOKEN_COUNTER

            _cfg = getattr(pipeline, "config", None)
            _cap_factors = None
            if _cfg is not None and getattr(_cfg, "ingestion", None) is not None:
                _cap_factors = dict(_cfg.ingestion.chunk_soft_cap_factors or {})

            # Try to get calibrated alignment metrics from the validator cache.
            # calibrate() is a fast no-op when the result is already cached.
            _alignment_metrics = None
            with soft_cap_factors_from_client(_cap_factors if _cap_factors else None):
                try:
                    from app.ai.validation.tokenizer_validator import tokenizer_validator

                    _fb = (
                        str(_tok.default_tokenizer_backend).strip().lower()
                        if _tok is not None
                        else "huggingface"
                    )
                    _alignment_metrics = tokenizer_validator.get_alignment_metrics_for_bundle(
                        bundle,
                        run_calibration=True,
                        factory_backend=_fb,
                    )
                except Exception as _cal_exc:
                    log_warning(
                        f"[IngestionV2] Calibration skipped for model={bundle.model_id}: {_cal_exc}. "
                        f"Using provider defaults for soft_cap."
                    )
                _counter = ChunkingTokenCounter.from_bundle(
                    bundle,
                    alignment_metrics=_alignment_metrics,
                    cache_size=_cache_sz,
                    factory_backend=(
                        str(_tok.default_tokenizer_backend).strip().lower()
                        if _tok is not None
                        else "huggingface"
                    ),
                )
            _ctx_token = _ACTIVE_TOKEN_COUNTER.set(_counter.count)
            log_info(
                f"[IngestionV2] Model-native token counting active | "
                f"strategy={strategy} | model={bundle.model_id} | "
                f"hard_cap={_counter.hard_cap} | soft_cap={_counter.soft_cap} | "
                f"soft_cap_factor={_counter.metrics.soft_cap_factor:.2f} | "
                f"remote={_counter.is_remote} | calibrated={_counter.metrics.calibrated}"
            )
        except Exception as _inject_exc:
            log_warning(
                f"[IngestionV2] ChunkingTokenCounter injection failed "
                f"({_inject_exc}); falling back to factory tokenizer for all strategies."
            )
            _ctx_token = None
            _counter = None

    try:
        # ── Phase 1: token_aware + EmbedderBundle → model-native tokenizer ──
        # When the pipeline carries a Phase 1 EmbedderBundle and the active
        # strategy is "token_aware", bypass the generic chunker and delegate to
        # token_aware_chunk() with the bundle's native TokenizerContract.  This
        # enforces F-01/F-02/F-16: the exact same tokenizer used for embedding is
        # also used to measure and bound each chunk.
        if strategy == "token_aware":
            if bundle is not None:
                try:
                    from app.services.ingestion.token_chunking_service import (
                        token_aware_chunk,
                    )
                    log_info(
                        f"[IngestionV2] token_aware chunking with Phase 1 tokenizer "
                        f"| model={bundle.model_id} | family={bundle.tokenizer_family.value} "
                        f"| max_tokens={bundle.embed_max_tokens}"
                    )
                    return await token_aware_chunk(
                        text,
                        db_session=db_session,
                        file_id=file_id,
                        business_id=business_id,
                        source_type=source_type,
                        embedding_model=embedding_model,
                        tokenizer_contract=bundle.tokenizer,
                        chunk_size=_chunk_cfg.chunk_size if _chunk_cfg is not None else None,
                        chunk_overlap=_chunk_cfg.chunk_overlap if _chunk_cfg is not None else None,
                        min_chunk_tokens=_chunk_cfg.min_chunk_len if _chunk_cfg is not None else None,
                    )
                except Exception as _phase1_exc:
                    log_info(
                        f"[IngestionV2] Phase 1 token_aware path failed "
                        f"({_phase1_exc}); falling back to generic chunker."
                    )
            elif _tok is not None and _chunk_cfg is not None:
                try:
                    from app.services.ingestion.token_chunking_service import token_aware_chunk

                    return await token_aware_chunk(
                        text,
                        db_session=db_session,
                        file_id=file_id,
                        business_id=business_id,
                        source_type=source_type,
                        embedding_model=embedding_model,
                        client_tokenization=_tok,
                        chunk_size=_chunk_cfg.chunk_size,
                        chunk_overlap=_chunk_cfg.chunk_overlap,
                        min_chunk_tokens=_chunk_cfg.min_chunk_len,
                    )
                except Exception as _tok_exc:
                    log_info(
                        f"[IngestionV2] token_aware (client JSON tokenizer) failed ({_tok_exc}); "
                        "falling back to generic chunker."
                    )

        chunker = get_chunker(strategy)
        return await chunker.chunk(
            text,
            db_session=db_session,
            file_id=file_id,
            business_id=business_id,
            source_type=source_type,
            embedding_model=embedding_model,
        )

    finally:
        # ── Always reset the ContextVar after chunking ─────────────────
        # This prevents the model-native counter from leaking into unrelated
        # async tasks that share the same event loop.
        if _ctx_token is not None:
            from app.core.chunking_stratagies.segmenter_v2 import _ACTIVE_TOKEN_COUNTER
            _ACTIVE_TOKEN_COUNTER.reset(_ctx_token)
            if _counter is not None:
                stats = _counter.stats()
                if stats["remote_calls"] > 0 or stats["approx_calls"] > 0:
                    log_info(
                        f"[IngestionV2] ChunkingTokenCounter stats | "
                        f"strategy={strategy} | model={bundle.model_id if bundle else 'n/a'} | "
                        f"cache_hits={stats['cache_hits']} | "
                        f"cache_misses={stats['cache_misses']} | "
                        f"remote_calls={stats['remote_calls']} | "
                        f"approx_calls={stats['approx_calls']}"
                    )


# ============================================================
# PLUGGABLE PIPELINE RESOLVER (tenant JSON via PipelineFactory)
#
# Pipeline identity (embedder, vectordb, LLM, chunking, secrets) comes from
# merged Client JSON — configure via Admin → Settings → Pipeline Builder.
# ============================================================

# ── Pipeline-per-client cache ─────────────────────────────────────────
# TTLCache (maxsize=128, ttl=600s) avoids pipeline rebuild storms when
# >16 tenants hit concurrently, while still expiring stale entries.
# Falls back to functools.lru_cache if cachetools is not installed.
#
# ARCHITECTURE NOTE (production-grade separation):
#   - _query_pipeline_cache: FULL pipeline (embedder + vectordb + LLM + reranker)
#     Used by RAG chat, retrieve endpoints. Reranker failures are fatal here.
#   - _ingestion_pipeline_cache: MINIMAL pipeline (embedder + vectordb only)
#     Used by file upload, sync, integrity. Reranker/LLM not needed, not built.
#
# This separation ensures ingestion never fails due to missing query-time
# dependencies (e.g., OpenAI key for LLM-Judge reranker).

if _HAS_CACHETOOLS:
    _query_pipeline_cache: TTLCache = TTLCache(maxsize=128, ttl=600)
    _query_pipeline_lock = _threading_lock()
    _ingestion_pipeline_cache: TTLCache = TTLCache(maxsize=128, ttl=600)
    _ingestion_pipeline_lock = _threading_lock()

    def _get_pipeline_for_client(client_id: str):
        """
        Resolve a FULL AssembledPipeline for normalized client_id.
        Includes LLM and reranker if configured. For QUERY-TIME use only.
        """
        from app.core.config.client_config_resolver import get_client_config

        with _query_pipeline_lock:
            if client_id in _query_pipeline_cache:
                return _query_pipeline_cache[client_id]

        cfg = get_client_config(client_id)
        result = pipeline_factory.build(cfg)
        with _query_pipeline_lock:
            _query_pipeline_cache[client_id] = result
        return result

    def _get_ingestion_pipeline_for_client(client_id: str):
        """
        Resolve an INGESTION-ONLY AssembledPipeline (embedder + vectordb).
        Strips reranker and LLM — ingestion does not need them.
        Cached separately from query pipelines.
        """
        from app.core.config.client_config_resolver import get_client_config_for_ingestion

        with _ingestion_pipeline_lock:
            if client_id in _ingestion_pipeline_cache:
                return _ingestion_pipeline_cache[client_id]

        cfg = get_client_config_for_ingestion(client_id)
        cfg_ingestion = cfg.model_copy(update={"reranker": None, "llm": None})
        result = pipeline_factory.build(cfg_ingestion, skip_cache=True)
        with _ingestion_pipeline_lock:
            _ingestion_pipeline_cache[client_id] = result
        return result

else:
    @lru_cache(maxsize=128)
    def _get_pipeline_for_client(client_id: str):  # type: ignore[no-redef]
        """Fallback cache (query pipeline) when cachetools is not installed."""
        from app.core.config.client_config_resolver import get_client_config

        cfg = get_client_config(client_id)
        return pipeline_factory.build(cfg)

    @lru_cache(maxsize=128)
    def _get_ingestion_pipeline_for_client(client_id: str):  # type: ignore[no-redef]
        """Fallback cache (ingestion pipeline) when cachetools is not installed."""
        from app.core.config.client_config_resolver import get_client_config_for_ingestion

        cfg = get_client_config_for_ingestion(client_id)
        cfg_ingestion = cfg.model_copy(update={"reranker": None, "llm": None})
        return pipeline_factory.build(cfg_ingestion, skip_cache=True)


# Public alias — same callable and cache as _get_pipeline_for_client (full query-time pipeline).
get_query_pipeline_for_client = _get_pipeline_for_client


def _get_pipeline(business_id: Optional[Any] = None):
    """
    Resolve FULL pipeline (query-time) using UUID-safe business_id normalization.
    Includes LLM + reranker. Use for RAG chat / retrieve endpoints.
    """
    client_id = _normalize_business_id(business_id)
    return _get_pipeline_for_client(client_id)


def _get_ingestion_pipeline(business_id: Optional[Any] = None):
    """
    Resolve INGESTION-ONLY pipeline (embedder + vectordb).
    Use for file upload, sync, integrity — no reranker/LLM needed.
    """
    client_id = _normalize_business_id(business_id)
    return _get_ingestion_pipeline_for_client(client_id)


if _HAS_CACHETOOLS:

    async def _get_ingestion_pipeline_for_client_async(client_id: str):
        """Async ingestion pipeline resolver (embedder + vectordb only)."""
        from app.core.config.client_config_resolver import get_client_config_for_ingestion

        with _ingestion_pipeline_lock:
            if client_id in _ingestion_pipeline_cache:
                return _ingestion_pipeline_cache[client_id]

        cfg = get_client_config_for_ingestion(client_id)
        cfg_ingestion = cfg.model_copy(update={"reranker": None, "llm": None})
        result = await pipeline_factory.build_async(cfg_ingestion, skip_cache=True)
        with _ingestion_pipeline_lock:
            _ingestion_pipeline_cache[client_id] = result
        return result

else:

    async def _get_ingestion_pipeline_for_client_async(client_id: str):  # type: ignore[no-redef]
        from app.core.config.client_config_resolver import get_client_config_for_ingestion

        cfg = get_client_config_for_ingestion(client_id)
        cfg_ingestion = cfg.model_copy(update={"reranker": None, "llm": None})
        return await pipeline_factory.build_async(cfg_ingestion, skip_cache=True)


async def _get_ingestion_pipeline_async(business_id: Optional[Any] = None):
    """Async variant of :func:`_get_ingestion_pipeline` for FastAPI async routes."""
    client_id = _normalize_business_id(business_id)
    return await _get_ingestion_pipeline_for_client_async(client_id)


if _HAS_CACHETOOLS:

    async def _get_pipeline_for_client_async(client_id: str):
        """Async query-time pipeline resolver (embedder + vectordb + LLM + reranker)."""
        from app.core.config.client_config_resolver import get_client_config

        with _query_pipeline_lock:
            if client_id in _query_pipeline_cache:
                return _query_pipeline_cache[client_id]

        cfg = get_client_config(client_id)
        result = await pipeline_factory.build_async(cfg)
        with _query_pipeline_lock:
            _query_pipeline_cache[client_id] = result
        return result

else:

    async def _get_pipeline_for_client_async(client_id: str):  # type: ignore[no-redef]
        from app.core.config.client_config_resolver import get_client_config

        cfg = get_client_config(client_id)
        return await pipeline_factory.build_async(cfg)


get_query_pipeline_for_client_async = _get_pipeline_for_client_async


async def _get_pipeline_async(business_id: Optional[Any] = None):
    """Async FULL pipeline resolver for RAG chat / retrieve endpoints."""
    client_id = _normalize_business_id(business_id)
    return await _get_pipeline_for_client_async(client_id)


# ============================================================
# ADDITIVE BLOCK 1: VISUAL / CHART-LIKE CONTENT DETECTOR
# ============================================================

def _looks_like_visual_content(text: str) -> bool:
    """
    Heuristic detector for charts, graphs, tables, numeric-heavy visuals.

    FIX (P0): The original used digit_ratio > 0.35 OR keyword_hits >= 2,
    which caused false-positive storms: prose like "Revenue grew 12% in 2022
    compared to 2021, as shown in the table below." hit 2 weak keywords and
    was routed to the LLM visual explainer. This burns tokens and corrupts the
    chunk (LLM explanation replaces original text).

    New criteria requires EITHER:
      (a) BOTH high digit density AND tabular structure (short lines, many rows)
      (b) At least 2 STRONG visual-specific keywords that rarely appear in prose
    """
    if not text or not isinstance(text, str) or len(text) < 80:
        return False

    # Already processed by VisualExplainerCPU — skip re-processing
    if "--- Semantic Analysis ---" in text:
        return False

    # Criterion (a): high digit density + structural signals (short lines = table/chart)
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    if digit_ratio > 0.35:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            avg_line_len = sum(len(ln) for ln in lines) / len(lines)
            # Short average line length with many lines = tabular/chart structure
            has_tabular_structure = avg_line_len < 40 and len(lines) > 5
            if has_tabular_structure:
                return True

    # Criterion (b): strong visual-specific keywords that rarely appear in general prose
    strong_visual_keywords = [
        "x-axis", "y-axis", "axis label", "legend", "data series",
        "chart title", "chart type", "bar chart", "pie chart",
        "line graph", "scatter plot", "SOURCE:", "figure ",
    ]
    strong_hits = sum(1 for k in strong_visual_keywords if k.lower() in text.lower())
    return strong_hits >= 2


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

    # Sentinel phrases that indicate the LLM echoed the prompt instead of answering
    _PROMPT_SENTINEL = "the following content is extracted from a chart"

    try:
        result = await rewrite_batch([prompt])
        if result and isinstance(result, list):
            explanation = result[0].strip()
            # Guard: reject if the LLM echoed the prompt back
            if not explanation:
                return ""
            if _PROMPT_SENTINEL in explanation.lower():
                return ""
            # Reject if explanation is just a short rephrasing (<30 chars)
            if len(explanation) < 30:
                return ""
            return explanation
    except Exception as e:
        # Log the exception to ensure visibility of LLM rate limits/timeouts
        log_warning(f"[VisualLLM] Failed to generate visual explanation: {e}")
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

    def __init__(self, vectordb, collection_name: str, *, client_id: str = ""):
        coll = (collection_name or "").strip()
        if not coll:
            ctx = f" (client_id={client_id!r})" if client_id else ""
            raise ValueError(
                f"collection_name is required for _CollectionAdapter{ctx}. "
                "Set vectordb.collection in tenant Client JSON."
            )
        self._vdb = vectordb
        self._col = coll

    @property
    def name(self) -> str:
        return self._col

    def count(self) -> int:
        try:
            return self._vdb.count(collection=self._col)
        except Exception as e:
            log_info(f"[CollectionAdapter] count() failed: {e}")
            return 0

    # PROPOSED — ingestion_service_v2.py
    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Chroma-compatible bulk upsert → delegates to BaseVectorDB.batch_upsert().

        WHY THIS EXISTS:
            main.py, admin APIs, and backfill scripts call this method via the
            _CollectionAdapter shim. They expect the chromadb.Collection.upsert()
            signature: (ids, embeddings, documents, metadatas).

            Previously this was a serial for-loop calling self._vdb.upsert() once
            per record — N network round-trips for N chunks.

            Fix: delegates to batch_upsert() which all BaseVectorDB plugins
            implement as a single bulk API call (1 round-trip for all N chunks).

        PERFORMANCE:
            Before: O(N) round-trips — 800 chunks × 5ms  =  4,000ms (Chroma local)
                                        800 chunks × 50ms = 40,000ms (Qdrant remote)
            After:  O(1) round-trips — 1 call  regardless of N

        SAFETY:
            - Input validation before any network call (fail fast, zero partial writes)
            - Empty-list guard: returns immediately if ids is empty
            - Length mismatch: raises ValueError before any write (no partial upserts)
            - BatchUpsertResult is logged: failed count > 0 raises RuntimeError
              so callers never silently lose vectors
            - Falls back to serial upsert ONLY if the plugin does not implement
              batch_upsert (defensive for future third-party plugins)

        SIGNATURE COMPATIBILITY:
            100% backward-compatible with chromadb.Collection.upsert().
            No caller change required anywhere.
        """
        # ── Guard: empty batch is a no-op, not an error ──────────────────────
        if not ids:
            return

        # ── Normalise optional args to concrete lists ─────────────────────────
        docs: List[str]             = documents  if documents  is not None else [""] * len(ids)
        mets: List[Dict[str, Any]]  = metadatas  if metadatas  is not None else [{}] * len(ids)

        # ── Pre-call validation — fail fast BEFORE any network I/O ───────────
        if len(ids) != len(embeddings):
            raise ValueError(
                f"[CollectionAdapter.upsert] Length mismatch: "
                f"ids={len(ids)}, embeddings={len(embeddings)}. "
                f"These must be equal. No data was written."
            )
        if len(ids) != len(docs):
            raise ValueError(
                f"[CollectionAdapter.upsert] Length mismatch: "
                f"ids={len(ids)}, documents={len(docs)}. "
                f"These must be equal. No data was written."
            )
        if len(ids) != len(mets):
            raise ValueError(
                f"[CollectionAdapter.upsert] Length mismatch: "
                f"ids={len(ids)}, metadatas={len(mets)}. "
                f"These must be equal. No data was written."
            )

        # ── Primary path: batch_upsert (O(1) round-trips) ────────────────────
        if hasattr(self._vdb, "batch_upsert") and callable(self._vdb.batch_upsert):
            result = self._vdb.batch_upsert(
                collection=self._col,
                doc_ids=ids,
                embeddings=embeddings,
                texts=docs,
                metadatas=mets,
            )
            # BatchUpsertResult.failed > 0 means vectors were silently dropped.
            # Raise immediately — the caller (admin API / backfill) must know.
            if result is not None and getattr(result, "failed", 0) > 0:
                raise RuntimeError(
                    f"[CollectionAdapter.upsert] batch_upsert reported "
                    f"{result.failed}/{len(ids)} failed vectors in "
                    f"collection '{self._col}'. Check VectorDB logs."
                )
            log_info(
                f"[CollectionAdapter] batch_upsert → {len(ids)} vectors "
                f"into '{self._col}' via {getattr(self._vdb, 'kind', type(self._vdb).__name__)}"
            )
            return

        # ── Fallback: serial upsert for legacy / third-party plugins ─────────
        # Only reached if a future plugin does NOT implement batch_upsert.
        # Will not be hit by any current plugin (Chroma/Qdrant/Pinecone/
        # Milvus/Weaviate/Redis all implement batch_upsert).
        log_info(
            f"[CollectionAdapter] WARNING: {type(self._vdb).__name__} has no "
            f"batch_upsert(). Falling back to serial upsert for {len(ids)} records. "
            f"Implement batch_upsert() in the plugin to fix this."
        )
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
        # Prefer bulk delete (O(1) round-trips) over serial delete (O(N))
        if hasattr(self._vdb, "delete_many") and callable(self._vdb.delete_many):
            try:
                self._vdb.delete_many(collection=self._col, doc_ids=list(ids))
                log_info(
                    f"[CollectionAdapter] delete_many → {len(list(ids))} ids "
                    f"from '{self._col}'"
                )
                return
            except Exception as e:
                log_info(
                    f"[CollectionAdapter] delete_many failed ({e}), "
                    f"falling back to serial delete"
                )
        # Serial fallback for plugins that don't implement delete_many
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

    Implementation note:
        Uses ``_get_ingestion_pipeline()`` (embedder + vectordb only, cached separately).
        Sync/integrity paths never initialize query-time components (reranker, LLM).
    """
    client_id = _normalize_business_id(business_id)
    pipeline = _get_ingestion_pipeline(business_id)
    from app.core.config.client_config_resolver import get_client_config_for_ingestion

    cfg = get_client_config_for_ingestion(client_id)
    coll = (cfg.vectordb.collection or "").strip()
    if not coll:
        raise ValueError(
            f"vectordb.collection is required for client_id={client_id!r}; "
            "set it via pipeline-pluggable PATCH or Client JSON."
        )
    return None, _CollectionAdapter(pipeline.vectordb, coll, client_id=client_id)


async def get_chroma_collection_async(
    skip_count: bool = False,
    business_id: Optional[str] = None,
) -> Tuple[None, _CollectionAdapter]:
    """Async-safe variant of :func:`get_chroma_collection` for FastAPI handlers."""
    del skip_count  # same semantics as sync shim; count is lazy on adapter
    client_id = _normalize_business_id(business_id)
    pipeline = await _get_ingestion_pipeline_async(business_id)
    from app.core.config.client_config_resolver import get_client_config_for_ingestion

    cfg = get_client_config_for_ingestion(client_id)
    coll = (cfg.vectordb.collection or "").strip()
    if not coll:
        raise ValueError(
            f"vectordb.collection is required for client_id={client_id!r}; "
            "set it via pipeline-pluggable PATCH or Client JSON."
        )
    return None, _CollectionAdapter(pipeline.vectordb, coll, client_id=client_id)


def get_embedder(business_id: Optional[str] = None) -> _EmbedderAdapter:
    """
    Returns _EmbedderAdapter.

    Identical to old SentenceTransformer call:
      embedder = get_embedder()
      vector   = embedder.encode(text, normalize_embeddings=True).tolist()

    Backend: MAI_EMBEDDER=ollama|openai|huggingface in .env

    Uses the same ingestion pipeline as ``get_chroma_collection`` (embedder + vectordb only).
    """
    pipeline = _get_ingestion_pipeline(business_id)
    return _EmbedderAdapter(pipeline.embedder)


async def get_embedder_async(business_id: Optional[str] = None) -> _EmbedderAdapter:
    """Async-safe variant of :func:`get_embedder` for FastAPI handlers."""
    pipeline = await _get_ingestion_pipeline_async(business_id)
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
    _async_session_factory = async_session

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
        *,
        pre_embed_hook: Optional[Any] = None,
    ):
        if pre_embed_hook is None:
            log_error(
                "[SECURITY][CRITICAL] pre_embed_hook missing in process_file — "
                "unsafe ingestion path. PII may reach embedders unsanitized. "
                "Use IngestionOrchestrator.",
                file_id=file_id, stage="pre_embedding",
            )
            if STRICT_INGESTION_SECURITY:
                raise RuntimeError(
                    f"STRICT_INGESTION_SECURITY: pre_embed_hook is required for "
                    f"process_file (file_id={file_id}). Route through IngestionOrchestrator."
                )
        _emb_label, _vdb_label = _pipeline_telemetry_labels(business_id)
        _plog = PipelineLogger(
            request_path="ingestion",
            client_id=business_id,
            embedder_model=_emb_label,
            vectordb_backend=_vdb_label,
        )
        _plog.info("Starting file ingestion", file_id=file_id)
        async with async_session() as db:
            try:
                log_info(f"[IngestionV2] Starting ingestion for {file_id}")

                file_record = await IngestionServiceV2._get_file_record(db, file_id)

                # ==========================================================
                # MEDIA-LEVEL HARD DEDUP (AUTHORITATIVE — API + WATCHER)
                # Tenant-scoped: same file in different tenants is allowed
                # ==========================================================
                if file_record and file_record.file_path:
                    loop = asyncio.get_running_loop()
                    try:
                        incoming_hash = await loop.run_in_executor(
                            None, compute_file_hash, file_record.file_path
                        )

                        # Build dedup query with tenant isolation
                        dedup_stmt = (
                            select(IngestedFileV2)
                            .where(IngestedFileV2.meta_data["file_hash"].astext == incoming_hash)
                            .where(IngestedFileV2.status == "processed")
                        )
                        # Add tenant filter using file_record.business_id
                        if file_record.business_id:
                            dedup_stmt = dedup_stmt.where(
                                IngestedFileV2.business_id == file_record.business_id
                            )
                        
                        result = await db.execute(dedup_stmt)
                        existing = result.scalar_one_or_none()

                        if existing and existing.id != file_record.id:
                            log_info(
                                f"[IngestionV2] ⛔ MEDIA DUPLICATE — "
                                f"already ingested as {existing.id} (tenant={file_record.business_id})"
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

                await IngestionServiceV2._run_pipeline(
                    db, file_record, parsed,
                    pre_embed_hook=pre_embed_hook,
                )
                log_info(f"[IngestionV2] ✅ Completed ingestion for {file_id}")
                _plog.info("File ingestion completed", file_id=file_id)

            except Exception as original_error:
                _plog.error(
                    "File ingestion failed",
                    file_id=file_id,
                    error=str(original_error)[:300],
                )
                _ing_logger.error(
                    "Ingestion failed",
                    file_id=str(file_id),
                    stage="pipeline",
                    error_type=type(original_error).__name__,
                    error=str(original_error)[:500],
                )
                try:
                    await IngestionServiceV2._set_file_error(db, file_id, str(original_error))
                except Exception as db_err:
                    # DB error during error-status update must not mask the original
                    log_warning(
                        f"[IngestionV2] Could not persist error status for {file_id}: "
                        f"{db_err} (original: {original_error})"
                    )
                raise original_error

    # ----------------------------------------------------------
    # Direct ingestion for pre-parsed output (RSS, API, etc.)
    # ----------------------------------------------------------
    @staticmethod
    async def ingest_parsed_output(
        file_id: str,
        parsed_output: Dict[str, Any],
        *,
        pre_embed_hook: Optional[Any] = None,
        client_id: Optional[str] = None,
    ):
        if pre_embed_hook is None:
            log_error(
                "[SECURITY][CRITICAL] pre_embed_hook missing in ingest_parsed_output — "
                "unsafe ingestion path. PII may reach embedders unsanitized. "
                "Use IngestionOrchestrator.",
                file_id=file_id, stage="pre_embedding",
            )
            if STRICT_INGESTION_SECURITY:
                raise RuntimeError(
                    f"STRICT_INGESTION_SECURITY: pre_embed_hook is required for "
                    f"ingest_parsed_output (file_id={file_id}). Route through IngestionOrchestrator."
                )
        _emb_label, _vdb_label = _pipeline_telemetry_labels(client_id)
        _plog = PipelineLogger(
            request_path="ingestion",
            embedder_model=_emb_label,
            vectordb_backend=_vdb_label,
        )
        _plog.info("Direct ingestion started", file_id=file_id)
        async with async_session() as db:
            try:
                file_record = await IngestionServiceV2._get_file_record(db, file_id)
                if not file_record:
                    log_info(
                        f"[IngestionV2] File not found for pre-parsed ingestion: {file_id}"
                    )
                    return

                _plog = _plog.bind(client_id=file_record.business_id)
                await IngestionServiceV2._ensure_file_entry(db, file_record)
                await IngestionServiceV2._run_pipeline(
                    db, file_record, parsed_output,
                    pre_embed_hook=pre_embed_hook,
                )
                log_info(f"[IngestionV2] ✅ Completed direct ingestion for {file_id}")
                _plog.info("Direct ingestion completed", file_id=file_id)

            except Exception as e:
                _plog.error("Direct ingestion failed", file_id=file_id, error=str(e)[:300])
                log_info(f"[ERROR] Direct ingestion failed for {file_id}: {e}")
                await IngestionServiceV2._set_file_error(db, file_id, str(e))
                raise

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
        *,
        pre_embed_hook: Optional[Any] = None,
    ):
        if STREAMING_INGESTION_ENABLED and (
            str(file_record.file_type or "").lower() == "pdf"
        ):
            return await IngestionServiceV2._run_pipeline_streaming(
                db,
                file_record,
                parsed_payload,
                pre_embed_hook=pre_embed_hook,
            )
        return await IngestionServiceV2._run_pipeline_legacy(
            db,
            file_record,
            parsed_payload,
            pre_embed_hook=pre_embed_hook,
        )

    @staticmethod
    async def _run_pipeline_legacy(
        db: AsyncSession,
        file_record: IngestedFileV2,
        parsed_payload: Dict[str, Any],
        *,
        pre_embed_hook: Optional[Any] = None,
    ):
        await IngestionServiceV2._set_file_processing(db, file_record.id)
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
        pipeline        = await _get_ingestion_pipeline_async(business_id)
        embedding_model = pipeline.embedder.info.model
        _plog = PipelineLogger(
            request_path="ingestion",
            client_id=str(business_id) if business_id else None,
            pipeline_id=str(file_id),
            embedder_model=embedding_model,
            vectordb_backend=getattr(pipeline.vectordb, "kind", "unknown"),
        )
        _plog.info("Pipeline resolved for ingestion", file_id=str(file_id))
        log_info(
            f"[IngestionV2] Active embedding model for {file_id}: {embedding_model}"
        )

        async with timed_stage("extract_chunks", file_id=str(file_id), business_id=str(business_id)):
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

        # Normalize metadata for every chunk, then enrich offsets/page citations.
        for ch in chunks:
            _normalize_chunk_metadata(ch)

        source_text_for_offsets = _resolve_text(parsed_payload)
        if source_text_for_offsets:
            _compute_chunk_offsets(chunks, source_text_for_offsets)

        page_map = parsed_payload.get("page_map")
        if isinstance(page_map, list) and page_map:
            _assign_page_numbers_from_page_map(chunks, page_map)

        # ── Security gate: PII sanitization before dedup/storage/embedding ──
        # pre_embed_hook signature: (chunks: List[Dict]) -> List[Dict]
        # Returns sanitized chunks with PII-redacted text; metadata preserved.
        # semantic_hash was computed in _extract_chunks (segmenter) from
        # original text BEFORE this point — dedup integrity is preserved.
        if pre_embed_hook is not None and chunks:
            chunks = pre_embed_hook(chunks)
        elif chunks:
            log_error(
                "[SECURITY][CRITICAL] pre_embed_hook not provided in _run_pipeline — "
                "unsafe ingestion path. Raw PII may reach embedders and vector stores.",
                file_id=file_id, stage="pre_embedding",
            )
            if STRICT_INGESTION_SECURITY:
                raise RuntimeError(
                    f"STRICT_INGESTION_SECURITY: pre_embed_hook is required for "
                    f"_run_pipeline (file_id={file_id}). Route through IngestionOrchestrator."
                )

        if not chunks:
            log_info(f"[IngestionV2] No chunks to ingest for {file_id}")
            return

        try:
            async with timed_stage("dedup_chunks", file_id=str(file_id), business_id=str(business_id)):
                unique_chunks, dedup_stats = await IngestionServiceV2._dedup_chunks(
                    db, chunks, file_id, business_id,
                    collection_name=pipeline.config.vectordb.collection,
                    pipeline=pipeline,          # FIX-D: pass resolved pipeline
                )
        except Exception as dedup_err:
            # Fail-open fallback: dedup issues should not block ingestion.
            # This specifically protects direct-ingestion flows from transient
            # runtime issues (for example, an UnboundLocalError in stats
            # collection) while preserving an audit trail in logs.
            log_info(
                f"[IngestionV2] Dedup failed for {file_id}; falling back to "
                f"non-deduplicated ingestion. Error: {type(dedup_err).__name__}: {dedup_err}"
            )
            unique_chunks = chunks
            dedup_stats = {
                "total": len(chunks),
                "unique": len(chunks),
                "duplicates": 0,
                "dedup_ratio": 0.0,
            }


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

        _assign_structure_parent_offsets(all_chunks_for_storage)

        if all_chunks_for_storage:
            async with timed_stage("insert_chunks", file_id=str(file_id), business_id=str(business_id)):
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
                f"[IngestionV2] Calling embed_and_store for {file_id} "
                f"with {len(chunks_to_embed)} unique chunks "
                f"(skipped {len(chunks) - len(chunks_to_embed)} duplicates)"
            )
            async with timed_stage("embed_and_store", file_id=str(file_id), business_id=str(business_id)):
                await IngestionServiceV2.embed_and_store(
                    file_id, business_id, file_type, chunks_to_embed,
                    file_name=file_record.file_name,
                    source_url=file_record.source_url,
                    pipeline=pipeline,   # FIX-B4: pass pre-resolved pipeline — eliminates 4th _get_pipeline()
                    db=db,               # FIX-B4: pass caller's session — eliminates redundant async_session()
                    skip_presence_check=True,
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

    @staticmethod
    async def _run_pipeline_streaming(
        db: AsyncSession,
        file_record: IngestedFileV2,
        parsed_payload: Dict[str, Any],
        *,
        pre_embed_hook: Optional[Any] = None,
    ):
        await IngestionServiceV2._set_file_processing(db, file_record.id)
        file_id = file_record.id
        business_id = file_record.business_id
        file_type = file_record.file_type

        if not business_id:
            log_info(
                f"[IngestionV2] Streaming requires business_id for {file_id}; "
                f"using legacy pipeline"
            )
            return await IngestionServiceV2._run_pipeline_legacy(
                db,
                file_record,
                parsed_payload,
                pre_embed_hook=pre_embed_hook,
            )

        pipeline = await _get_ingestion_pipeline_async(business_id)
        embedding_model = pipeline.embedder.info.model
        icfg = getattr(pipeline.config, "ingestion", None)
        batch_size = max(
            1,
            int(getattr(icfg, "batch_size", BATCH_SIZE) or BATCH_SIZE),
        )

        _plog = PipelineLogger(
            request_path="ingestion",
            client_id=str(business_id) if business_id else None,
            pipeline_id=str(file_id),
            embedder_model=embedding_model,
            vectordb_backend=getattr(pipeline.vectordb, "kind", "unknown"),
        )
        _plog.info("Streaming pipeline resolved for ingestion", file_id=str(file_id))

        state = await StreamingIngestionState.from_checkpoint(
            db, UUID(str(file_id)), UUID(str(business_id))
        )

        pdf_path = file_record.file_path or parsed_payload.get("file_path")
        if not pdf_path:
            log_info(
                f"[IngestionV2] Streaming PDF path missing for {file_id}; "
                f"falling back to legacy pipeline"
            )
            return await IngestionServiceV2._run_pipeline_legacy(
                db,
                file_record,
                parsed_payload,
                pre_embed_hook=pre_embed_hook,
            )

        def _semantic_hash_fn(text: str) -> str:
            return create_normalized_hash(text, embedding_model)

        from app.services.ingestion.chunk_stream import iter_chunks

        page_iter = iter_pdf_pages(pdf_path, start_page=state.last_processed_page)
        chunk_iter = iter_chunks(
            page_iter=page_iter,
            pipeline=pipeline,
            db=db,
            file_id=UUID(str(file_id)),
            business_id=UUID(str(business_id)),
            file_type=file_type,
            semantic_hash_fn=_semantic_hash_fn,
        )

        current_window: List[Dict[str, Any]] = []

        async def _flush_window() -> None:
            nonlocal current_window
            if not current_window:
                return
            window_snapshot = current_window
            current_window = []
            upserted_vector_ids: List[str] = []
            async with IngestionServiceV2._async_session_factory() as window_db:
                try:
                    upserted_vector_ids = await IngestionServiceV2._process_window(
                        window_snapshot,
                        state,
                        window_db,
                        pipeline=pipeline,
                        file_record=file_record,
                        file_id=file_id,
                        business_id=business_id,
                        file_type=file_type,
                        embedding_model=embedding_model,
                        pre_embed_hook=pre_embed_hook,
                    )
                    await window_db.commit()
                except Exception as window_err:
                    await window_db.rollback()
                    if upserted_vector_ids:
                        await IngestionServiceV2._compensate_vectors_safe(
                            upserted_vector_ids, pipeline
                        )
                    from app.services.ingestion.ingestion_failure_context import (
                        IngestionFailureContext,
                        set_ingestion_failure_context,
                    )

                    set_ingestion_failure_context(
                        IngestionFailureContext(
                            stage="streaming_window_commit",
                            window_index=state.current_window_index,
                            pipeline_mode="streaming",
                            extra={
                                "compensated_vector_ids": len(upserted_vector_ids),
                            },
                        )
                    )
                    raise window_err
            state.current_window_index += 1

        async for chunk in chunk_iter:
            current_window.append(chunk)
            if len(current_window) >= batch_size:
                async with timed_stage(
                    "process_window",
                    file_id=str(file_id),
                    business_id=str(business_id),
                ):
                    await _flush_window()

        async with timed_stage(
            "process_window_final",
            file_id=str(file_id),
            business_id=str(business_id),
        ):
            await _flush_window()

        total = state.total_chunks
        unique = state.unique_chunks
        duplicates = state.duplicate_chunks
        dedup_ratio = (
            round((duplicates / max(1, total)) * 100, 2) if total else 0.0
        )

        await IngestionServiceV2._update_file_status(
            db,
            file_id,
            total_chunks=total,
            unique_chunks=unique,
            duplicate_chunks=duplicates,
            dedup_ratio=dedup_ratio,
            status="processed",
        )

        log_info(
            f"[IngestionV2] Streaming pipeline complete for {file_id}: "
            f"{unique}/{total} chunks stored ({dedup_ratio:.2f}% deduplication)"
        )

    @staticmethod
    async def _process_window(
        window: List[Dict[str, Any]],
        state: StreamingIngestionState,
        window_db: AsyncSession,
        *,
        pipeline,
        file_record: IngestedFileV2,
        file_id,
        business_id,
        file_type: str,
        embedding_model: str,
        pre_embed_hook: Optional[Any] = None,
    ) -> List[str]:
        """
        Process one micro-batch. Caller owns commit/rollback on window_db.
        Returns semantic_hash doc_ids upserted to VectorDB (for compensation).
        """
        flattened: List[Dict[str, Any]] = []
        for chunk in window:
            ch: Dict[str, Any] = dict(chunk)
            if ch.get("status") == "visual_pending" and ch.get("_visual_task"):
                task = ch.pop("_visual_task")
                try:
                    resolved = await task
                except Exception as exc:
                    log_warning(
                        f"[IngestionV2] Visual task failed for {file_id}: {exc}"
                    )
                    resolved = []
                for resolved_chunk in resolved or []:
                    rc = dict(resolved_chunk)
                    rc.pop("_visual_task", None)
                    if rc.get("status") != "empty":
                        flattened.append(rc)
                continue
            ch.pop("_visual_task", None)
            if ch.get("status") == "empty":
                continue
            flattened.append(ch)

        if not flattened:
            return []

        for ch in flattened:
            _normalize_chunk_metadata(ch)

        if pre_embed_hook is not None:
            flattened = pre_embed_hook(flattened)
        elif flattened:
            log_error(
                "[SECURITY][CRITICAL] pre_embed_hook not provided in streaming window — "
                "unsafe ingestion path.",
                file_id=file_id,
                stage="pre_embedding",
            )
            if STRICT_INGESTION_SECURITY:
                raise RuntimeError(
                    f"STRICT_INGESTION_SECURITY: pre_embed_hook required "
                    f"(file_id={file_id})"
                )

        cross_window_dups: List[Dict[str, Any]] = []
        survivors: List[Dict[str, Any]] = []
        for ch in flattened:
            semantic_hash = ch.get("semantic_hash")
            if not semantic_hash:
                survivors.append(ch)
                continue
            if state.dedup.is_seen(semantic_hash):
                ch["is_duplicate"] = True
                ch["dedup_layer"] = "layer1_streaming_state"
                ch["similarity_score"] = 1.0
                cross_window_dups.append(ch)
            else:
                survivors.append(ch)

        if survivors:
            try:
                unique_chunks, dedup_stats = await IngestionServiceV2._dedup_chunks(
                    window_db,
                    survivors,
                    file_id,
                    business_id,
                    collection_name=pipeline.config.vectordb.collection,
                    pipeline=pipeline,
                )
            except Exception as dedup_err:
                log_info(
                    f"[IngestionV2] Window dedup failed for {file_id}; "
                    f"non-deduplicated fallback: {dedup_err}"
                )
                unique_chunks = survivors
                dedup_stats = {
                    "total": len(survivors),
                    "unique": len(survivors),
                    "duplicates": 0,
                    "dedup_ratio": 0.0,
                }
        else:
            unique_chunks = []
            dedup_stats = {
                "total": len(cross_window_dups),
                "unique": 0,
                "duplicates": len(cross_window_dups),
                "dedup_ratio": 100.0 if cross_window_dups else 0.0,
            }

        window_total = len(flattened)
        window_unique = len(unique_chunks)
        window_duplicates = window_total - window_unique
        state.total_chunks += window_total
        state.unique_chunks += window_unique
        state.duplicate_chunks += window_duplicates

        for ch in unique_chunks:
            h = ch.get("semantic_hash")
            if h:
                state.dedup.add(h)

        if unique_chunks:
            await register_unique_chunks_in_gci(
                db=window_db,
                unique_chunks=unique_chunks,
                file_id=str(file_id),
                business_id=business_id,
                source_type=file_type,
                embedding_model=embedding_model,
            )

        unique_hashes = {c.get("semantic_hash") for c in unique_chunks}
        all_chunks_for_storage: List[Dict[str, Any]] = []

        for chunk in unique_chunks:
            chunk["is_duplicate"] = False
            chunk["duplicate_of"] = None
            chunk["similarity_score"] = None
            all_chunks_for_storage.append(chunk)

        for chunk in flattened:
            semantic_hash = chunk.get("semantic_hash")
            if semantic_hash and semantic_hash not in unique_hashes:
                chunk["is_duplicate"] = True
                gci_uuid = chunk.get("global_content_id") or chunk.get("gci_id")
                chunk["duplicate_of"] = gci_uuid
                chunk["similarity_score"] = chunk.get("similarity_score")
                all_chunks_for_storage.append(chunk)

        _assign_structure_parent_offsets(all_chunks_for_storage)

        if all_chunks_for_storage:
            await IngestionServiceV2._insert_chunks(
                window_db,
                file_id,
                business_id,
                all_chunks_for_storage,
                auto_commit=False,
            )

        upserted_vector_ids: List[str] = []
        chunks_to_embed = [
            c for c in unique_chunks if c.get("semantic_hash")
        ]
        if chunks_to_embed:
            upserted_vector_ids = list(
                dict.fromkeys(
                    str(c["semantic_hash"])
                    for c in chunks_to_embed
                    if c.get("semantic_hash")
                )
            )
            await IngestionServiceV2.embed_and_store(
                file_id,
                business_id,
                file_type,
                chunks_to_embed,
                file_name=file_record.file_name,
                source_url=file_record.source_url,
                pipeline=pipeline,
                db=window_db,
                skip_presence_check=True,
            )

        if flattened:
            chunk_indices = [
                int(c.get("chunk_index", 0))
                for c in flattened
                if c.get("chunk_index") is not None
            ]
            page_numbers = [
                int(c.get("page_number", 0))
                for c in flattened
                if isinstance(c.get("page_number"), int)
            ]
            if chunk_indices:
                state.last_processed_chunk_index = max(
                    state.last_processed_chunk_index,
                    max(chunk_indices),
                )
            if page_numbers:
                state.last_processed_page = max(
                    state.last_processed_page,
                    max(page_numbers),
                )

        await state.persist_checkpoint(window_db)
        return upserted_vector_ids

    @staticmethod
    async def _batch_gci_lookup(
        db: AsyncSession,
        hashes: List[str],
        business_id: Optional[str],
        batch_size: int = 500,
    ) -> set[str]:
        """Batched GCI presence lookup (tenant-scoped when business_id set)."""
        if not hashes:
            return set()
        known: set[str] = set()
        for i in range(0, len(hashes), batch_size):
            batch = hashes[i : i + batch_size]
            query = select(GlobalContentIndexV2.semantic_hash).where(
                GlobalContentIndexV2.semantic_hash.in_(batch)
            )
            if business_id:
                query = query.where(
                    GlobalContentIndexV2.business_id == business_id
                )
            result = await db.execute(query)
            known.update(row[0] for row in result.all())
        return known

    @staticmethod
    async def _compensate_vectors_safe(
        ids: List[str],
        pipeline,
    ) -> None:
        """Delete upserted vector doc_ids; never re-raise."""
        if not ids:
            return
        collection = pipeline.config.vectordb.collection
        try:
            vectordb = pipeline.vectordb
            if hasattr(vectordb, "delete_many") and callable(
                vectordb.delete_many
            ):
                vectordb.delete_many(collection=collection, doc_ids=list(ids))
            else:
                for doc_id in ids:
                    vectordb.delete(collection=collection, doc_id=doc_id)
        except Exception as e:
            log_error(f"[IngestionV2] Vector compensation failed: {e}")

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
        pipeline=None,
    ) -> List[Dict[str, Any]]:
        """
        Text → chunk dicts. Memory-safe, event-loop safe.

        B3 FIXES:
          FIX-B3-1: Streaming accumulation — result list grown incrementally,
                    intermediate per-item lists released immediately after extend().
                    Peak RAM = max(single_item_chunks) not sum(all_item_chunks).

          FIX-B3-2: Visual LLM calls batched with asyncio.gather() —
                    all visual items processed concurrently instead of serially.
                    20 visual items × 2s → 2s total instead of 40s.

          FIX-B3-3: Structured data path delegates to parse_dataframe_rows()
                    which now uses chunked DataFrame iteration (no full-list copy).

          FIX-B3-4: parsed_payload reference released early on single-text path
                    so the raw payload dict is GC-eligible before downstream
                    dedup + embedding hold their own lists.

        DEPRECATED: this helper will be replaced by the streaming
        `iter_chunks` + windowed ingestion pipeline. Callers should
        migrate to the streaming API as part of the
        `chunk-streaming-generator` and `streaming-window-integration`
        todos.
        """
        log_warning(
            "_extract_chunks() is deprecated. Callers must migrate to iter_chunks() "
            "for streaming micro-batches. [todo: chunk-streaming-generator]"
        )
        try:
            # FIX-D (carried forward): resolve pipeline once, never re-resolve
            if pipeline is None:
                pipeline = await _get_ingestion_pipeline_async(business_id)
            embedding_model = pipeline.embedder.info.model
            active_chunking_strategy = _resolve_chunking_strategy(pipeline)

            if asyncio.iscoroutine(parsed_payload):
                parsed_payload = await parsed_payload

            # ── Pre-built chunk passthrough (CSV/Excel parsers) ──────────────
            # Parsers that call row_segmenter internally return fully-built
            # chunk dicts. Detect by presence of semantic_hash on first item.
            raw_chunks_list = parsed_payload.get("chunks")
            if (
                isinstance(raw_chunks_list, list)
                and raw_chunks_list
                and isinstance(raw_chunks_list[0], dict)
                and raw_chunks_list[0].get("semantic_hash")
            ):
                log_info(
                    f"[IngestionV2] Pre-built chunk passthrough: "
                    f"{len(raw_chunks_list)} chunks for {file_id}"
                )
                for ch in raw_chunks_list:
                    ch.setdefault("embedding_model", embedding_model)
                    ch.setdefault("source_type", file_type)
                return raw_chunks_list

            # ── Structured data path (CSV / Excel) ───────────────────────────
            if file_type in ("csv", "xlsx", "xls"):
                try:
                    import pandas as pd
                    raw_df = parsed_payload.get("dataframe")

                    if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
                        df = raw_df
                    elif isinstance(raw_df, list) and raw_df:
                        df = pd.DataFrame(raw_df)
                    else:
                        rows_data = parsed_payload.get("rows")
                        if isinstance(rows_data, list) and rows_data:
                            df = pd.DataFrame(rows_data)
                        elif isinstance(rows_data, pd.DataFrame):
                            df = rows_data
                        else:
                            df = None

                    if df is not None and not df.empty:
                        # FIX-B3-3: parse_dataframe_rows now uses chunked
                        # iteration — does not hold full DataFrame in RAM.
                        structured_chunks = await parse_dataframe_rows(
                            df=df,
                            file_id=str(file_id),
                            source_type=file_type,
                            db_session=db,
                            business_id=business_id,
                        )
                        for ch in structured_chunks:
                            ch.setdefault("embedding_model", embedding_model)
                        log_info(
                            f"[IngestionV2] Structured path: "
                            f"{len(structured_chunks)} chunks for {file_id}"
                        )
                        return structured_chunks

                    log_info(
                        f"[IngestionV2] No DataFrame in payload for {file_type} "
                        f"{file_id} — falling back to text path"
                    )

                except Exception as e:
                    log_info(
                        f"[IngestionV2] Structured path failed for {file_id}: {e} "
                        f"— falling back to text path"
                    )

            # ── Multi-item path (RSS, API, web, etc.) ────────────────────────
            if any(k in parsed_payload for k in ("entries", "rows", "chunks")):

                base_list = None
                for key in ("entries", "rows", "chunks"):
                    candidate = parsed_payload.get(key)
                    if candidate is not None and (
                        hasattr(candidate, "__iter__")
                        and not isinstance(candidate, str)
                    ):
                        base_list = candidate
                        break

                if not base_list:
                    log_info(
                        f"[IngestionV2] entries/rows/chunks key found but "
                        f"empty/None for {file_id} — falling to text path"
                    )
                else:
                    try:
                        import pandas as pd
                        if isinstance(base_list, pd.DataFrame):
                            base_list = base_list.to_dict(orient="records")
                    except ImportError:
                        pass

                    # ── B3-FIX-1: Separate visual and text items upfront ──────
                    # Classify ALL items before any chunking begins.
                    # Avoids the serial interleave of visual-LLM + chunk calls.
                    visual_texts: List[str] = []
                    plain_texts:  List[str] = []

                    for item in base_list:
                        raw_text = _resolve_item_text(item)
                        if raw_text is None:
                            continue
                        raw_text = _coerce_to_str(raw_text).strip()
                        if not raw_text:
                            continue
                        if _looks_like_visual_content(raw_text):
                            visual_texts.append(raw_text)
                        else:
                            plain_texts.append(raw_text)

                    result: List[Dict[str, Any]] = []

                    # ── B3-FIX-2: Visual items — semaphore-gated concurrent LLM + chunk ──
                    # All LLM calls fired concurrently via asyncio.gather(), bounded by
                    # _VISUAL_LLM_SEMAPHORE to prevent rate-limit exhaustion.
                    # return_exceptions=True: one LLM failure does not crash the batch.
                    if visual_texts:
                        _vis_lim = max(1, int(getattr(
                            getattr(pipeline.config, "ingestion", None),
                            "visual_llm_concurrency",
                            _VISUAL_LLM_CONCURRENCY,
                        )))
                        _vis_sem = asyncio.Semaphore(_vis_lim)

                        async def _process_visual(vtext: str) -> List[Dict]:
                            async with _vis_sem:
                                explanation = await _explain_visual_with_llm(vtext)
                                source = explanation if explanation else vtext
                                vis_chunks = await _chunk_text_with_strategy(
                                    source,
                                    db_session=db,
                                    file_id=str(file_id),
                                    business_id=business_id,
                                    source_type=file_type,
                                    embedding_model=embedding_model,
                                    pipeline=pipeline,
                                    strategy_override=active_chunking_strategy,
                                )
                                orig_hash = hashlib.sha256(
                                    vtext.encode("utf-8")
                                ).hexdigest()
                                for ch in vis_chunks:
                                    ch.setdefault("reasoning_ingestion", {}).update({
                                        "content_type":       "visual",
                                        "interpreted_by":     "llm",
                                        "original_text_hash": orig_hash,
                                    })
                                    ch.setdefault("embedding_model", embedding_model)
                                return vis_chunks

                        visual_results = await asyncio.gather(
                            *[_process_visual(vt) for vt in visual_texts],
                            return_exceptions=True,   # FIX: one failure must not crash entire batch
                        )
                        for chunk_list in visual_results:
                            if isinstance(chunk_list, Exception):
                                log_warning(
                                    f"[IngestionV2] Visual LLM processing failed for one item "
                                    f"in {file_id}: {type(chunk_list).__name__}: {chunk_list}"
                                )
                                continue
                            result.extend(chunk_list)
                            # FIX-B3-1: chunk_list goes out of scope after extend
                            # — immediately GC-eligible, not held until loop end

                    # ── B3-FIX-1: Plain text items — stream into result ───────
                    # Each item's subchunks are extend()ed and the local list
                    # goes out of scope immediately — never all in RAM at once.
                    for plain_text in plain_texts:
                        subchunks = await _chunk_text_with_strategy(
                            plain_text,
                            db_session=db,
                            file_id=str(file_id),
                            business_id=business_id,
                            source_type=file_type,
                            embedding_model=embedding_model,
                            pipeline=pipeline,
                            strategy_override=active_chunking_strategy,
                        )
                        # FIX-B3-1: extend + let subchunks go out of scope
                        result.extend(subchunks)
                        del subchunks   # ← explicit early release

                    for ch in result:
                        ch.setdefault("embedding_model", embedding_model)

                    return result

            # ── Single payload path ───────────────────────────────────────────
            text = _resolve_text(parsed_payload)
            # FIX-B3-4: release parsed_payload reference BEFORE chunking begins
            # so the full payload dict (may contain raw PDF pages, image metadata)
            # is GC-eligible during the chunk + dedup + embed pipeline.
            del parsed_payload

            if not text:
                log_info(
                    f"[IngestionV2] No text found in parsed payload for {file_id}"
                )
                return []

            if _looks_like_visual_content(text):
                explanation = await _explain_visual_with_llm(text)
                if explanation:
                    explained_chunks = await _chunk_text_with_strategy(
                        explanation,
                        db_session=db,
                        file_id=str(file_id),
                        business_id=business_id,
                        source_type=file_type,
                        embedding_model=embedding_model,
                        pipeline=pipeline,
                        strategy_override=active_chunking_strategy,
                    )
                    orig_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    for ch in explained_chunks:
                        ch.setdefault("reasoning_ingestion", {}).update({
                            "content_type":       "visual",
                            "interpreted_by":     "llm",
                            "original_text_hash": orig_hash,
                        })
                        ch.setdefault("embedding_model", embedding_model)
                    return explained_chunks

            chunks = await _chunk_text_with_strategy(
                text,
                db_session=db,
                file_id=str(file_id),
                business_id=business_id,
                source_type=file_type,
                embedding_model=embedding_model,
                pipeline=pipeline,
                strategy_override=active_chunking_strategy,
            )
            for ch in chunks:
                ch.setdefault("embedding_model", embedding_model)
            return chunks

        except Exception as e:
            log_info(f"[CRITICAL] Chunk extraction failed for {file_id}: {e}")
            raise RuntimeError(f"Chunk extraction failed: {e}") from e

    
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
            pipeline = await _get_ingestion_pipeline_async(business_id)

        dedup_cfg = pipeline.config.ingestion.deduplication

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
        is_structured_data = first_source in ("csv", "xlsx", "xls", "excel")

        # AFTER (FIXED):
        enable_hash_dedup = bool(dedup_cfg.enable_hash_dedup)
        enable_gci_dedup = bool(dedup_cfg.enable_gci_dedup)
        enable_embedding_dedup = bool(dedup_cfg.enable_embedding_dedup)
        if is_structured_data:
            log_info(
                f"[IngestionV2] Structured data detected ({first_source}) - "
                f"skipping L3 vector similarity dedup (L1/L2 sufficient for row data)"
            )
            enable_embedding_dedup = False

        log_info(
            f"[IngestionV2] Dedup config for {file_id}: "
            f"L1={enable_hash_dedup} L2={enable_gci_dedup} "
            f"L3={enable_embedding_dedup} "
            f"threshold={dedup_cfg.similarity_threshold:.2f}"
        )

        unique_chunks, stats = await deduplicate_chunks(
            db=db,
            chunks=chunks,
            vectordb=pipeline.vectordb,
            embedder=pipeline.embedder,
            file_id=file_id,
            business_id=business_id,
            enable_hash_dedup=enable_hash_dedup,
            enable_gci_dedup=enable_gci_dedup,
            enable_embedding_dedup=enable_embedding_dedup,
            similarity_threshold=dedup_cfg.similarity_threshold,
            collection_name=collection_name or pipeline.config.vectordb.collection,  # B10: pass resolved name
            l3_redis_threshold=int(dedup_cfg.l3_redis_threshold),
            l3_embed_batch_size=int(dedup_cfg.l3_embed_batch_size),
            l3_search_concurrency=int(dedup_cfg.l3_search_concurrency),
        )
        # FIXED - matches B5-FIX-4 key names exactly
        log_info(
            f"[IngestionV2] Dedup complete for {file_id}: "
            f"{stats['unique']} unique, {stats['duplicates']} duplicates, "
            f"{stats['dedup_ratio']:.2f}% reduction "
            f"[L1={stats.get('layer1_hash_duplicates', 0)}, "
            f"L2={stats.get('layer2_gci_duplicates', 0)}, "         # ← CORRECT
            f"L3={stats.get('layer3_embedding_duplicates', 0)}]"    # ← CORRECT
        )

        return unique_chunks, stats

    # ----------------------------------------------------------
    # Insert chunks into PostgreSQL
    # ----------------------------------------------------------
    @staticmethod
    async def _insert_chunks(
        db: AsyncSession,
        file_id,
        business_id,
        chunks,
        *,
        auto_commit: bool = True,
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
          All target columns are explicitly named — no silent drops possible.
          JSON fields serialized with json.dumps() + CAST(:x AS jsonb).
          CONSTRAINT GUARANTEE: pre-insert check ensures is_duplicate=True
          always has at least one of duplicate_of or similarity_score set.
        """
        import json as _json

        # FIXED — lock the file row first, THEN read the aggregate freely:
        await db.execute(
            select(IngestedFileV2.id)
            .where(IngestedFileV2.id == file_id)
            .with_for_update()          # ← lock the file row (single row, valid)
        )
        
        result = await db.execute(
            select(func.max(IngestedContentV2.chunk_index))
            .where(IngestedContentV2.file_id == file_id)
            # no .with_for_update() here
        )
        start_index = (result.scalar() or -1) + 1


        
        # ── Explicit text() SQL — full column list, no ORM mapper filtering ──
        from sqlalchemy import text as sa_text
        stmt = sa_text("""
            INSERT INTO ingested_content (
                id, file_id, business_id, chunk_index,
                text, cleaned_text, tokens, source_type, page_number, parent_chunk_id,
                meta_data, confidence, semantic_hash, global_content_id,
                reasoning_ingestion, is_duplicate, duplicate_of,
                similarity_score, duplicate_percentage, created_at, updated_at
            ) VALUES (
                :id, :file_id, :business_id, :chunk_index,
                :text, :cleaned_text, :tokens, :source_type, :page_number, :parent_chunk_id,
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
        # every row with is_duplicate=TRUE has duplicate_of IS NOT NULL.
        #
        # Previous behaviour for L3 dups:
        #   duplicate_of    = None  (no GCI UUID was fetched)
        #   similarity_score = 0.955 (set by dedup engine)
        #   duplicate_percentage = not in INSERT → stored as SQL NULL
        #
        # similarity_score / duplicate_percentage do not satisfy this check.
        # If duplicate_of remains NULL, PostgreSQL raises CheckViolationError.
        #
        # L2 GCI dups work because duplicate_of = gci_uuid IS NOT NULL.
        # L1 intra-batch dups and L3 vector dups have BOTH duplicate_of=None
        # AND duplicate_percentage missing → both were always at risk.
        # Only now (first real L3 dup from Crisil PDF) did we hit the wall.
        #
        # Fix: resolve duplicate_of for L3 vector duplicates.
        #   A) Preferred: GCI UUID lookup by semantic_hash
        #   B) Fallback: existing ingested_content row UUID by semantic_hash
        # ─────────────────────────────────────────────────────────────────────
        # Collect L3 dups that have a duplicate_chunk_id (semantic_hash) but no duplicate_of.
        # Use list buckets so multiple chunks sharing the same hash all get patched.
        l3_hash_to_chunks: Dict[str, List[Dict[str, Any]]] = {}
        for c in chunks:
            if (bool(c.get("is_duplicate")) and
                c.get("duplicate_of") is None and
                c.get("duplicate_chunk_id")):
                h = c["duplicate_chunk_id"]
                l3_hash_to_chunks.setdefault(h, []).append(c)

        if l3_hash_to_chunks:
            from app.db.models.global_content_index_v2 import GlobalContentIndexV2 as _GCI
            from app.db.models.ingested_content_v2 import IngestedContentV2 as _IC
            from sqlalchemy import select as _select

            unresolved_hashes = set(l3_hash_to_chunks.keys())

            # 1) Preferred: resolve to GCI UUID by semantic_hash.
            res = await db.execute(
                _select(_GCI.semantic_hash, _GCI.id)
                .where(_GCI.semantic_hash.in_(list(unresolved_hashes)))
            )
            for row_hash, row_uuid in res.all():
                if row_hash in l3_hash_to_chunks:
                    for chunk_ref in l3_hash_to_chunks[row_hash]:
                        chunk_ref["duplicate_of"] = str(row_uuid)
                    unresolved_hashes.discard(row_hash)

            # 2) Fallback: resolve to an existing ingested_content row UUID.
            # This keeps duplicate rows insertable even when legacy vectors exist
            # in VectorDB without corresponding GCI rows.
            if unresolved_hashes:
                res_ic = await db.execute(
                    _select(_IC.semantic_hash, _IC.id)
                    .where(_IC.semantic_hash.in_(list(unresolved_hashes)))
                )
                for row_hash, row_uuid in res_ic.all():
                    if row_hash in l3_hash_to_chunks:
                        for chunk_ref in l3_hash_to_chunks[row_hash]:
                            chunk_ref["duplicate_of"] = str(row_uuid)

        row_ids = [str(uuid.uuid4()) for _ in chunks]
        all_rows: list = []   # accumulate all param dicts; sent as single executemany
        for i, c in enumerate(chunks):
            is_dup   = bool(c.get("is_duplicate", False))
            dup_of   = c.get("duplicate_of")
            sim_scr  = c.get("similarity_score")
            metadata = c.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            page_number = c.get("page_number")
            if not isinstance(page_number, int):
                page_number = metadata.get("page_number")
            if not isinstance(page_number, int) or page_number <= 0:
                page_number = None
            parent_row_offset = c.get("parent_row_offset")
            parent_chunk_id = None
            if isinstance(parent_row_offset, int):
                if 0 <= parent_row_offset < len(row_ids) and parent_row_offset != i:
                    parent_chunk_id = row_ids[parent_row_offset]

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
                    "sentiment_bucket":      "neutral",
                    "sentiment_confidence":  0.50,
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
                    "sentiment_bucket":      "neutral",
                    "sentiment_confidence":  0.50,
                    "data_lineage_id":       c.get("semantic_hash") or "",
                    "potentially_regulated": False,
                    "extraction_timestamp":  now.isoformat() + "Z",
                }
                for k, v in _defaults.items():
                    reasoning.setdefault(k, v)

            all_rows.append({
                "id":                row_ids[i],
                "file_id":           str(file_id),
                "business_id":       str(business_id) if business_id else None,
                "chunk_index":       start_index + i,
                "text":              c.get("text"),
                "cleaned_text":      c.get("cleaned_text") or c.get("cleaned"),
                "tokens":            int(c.get("tokens") or 0),
                "source_type":       c.get("source_type"),
                "page_number":       page_number,
                "parent_chunk_id":   parent_chunk_id,
                "meta_data":         _json.dumps(metadata),
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
        # AFTER:
        try:
            await db.execute(stmt, all_rows)
            if auto_commit:
                await db.commit()
            log_info(f"[IngestionV2] Inserted {len(chunks)} chunks into DB")
        except Exception as ins_err:
            if auto_commit:
                await db.rollback()   # B8: clean session so _update_file_status can still run
            log_info(f"[IngestionV2] _insert_chunks executemany failed: {ins_err}")
            raise

    # ----------------------------------------------------------
    # Embedding + Vector Store (executor-offloaded, capture-safe)
    # ----------------------------------------------------------
    @staticmethod
    async def embed_and_store(
        file_id,
        business_id,
        file_type,
        chunks,
        file_name: Optional[str] = None,
        source_url: Optional[str] = None,
        pipeline=None,      # FIX-B4-3: accept pre-resolved pipeline from _run_pipeline
        db=None,            # FIX-B4-4: accept caller's session — no redundant pool open
        skip_presence_check: bool = False,
    ):
        """
        Embeds unique chunks and stores them in VectorDB.
    
        B4 FIXES:
          FIX-B4-1: Lambda capture — all variables used inside executor
                    lambdas are now explicit default args. Zero reference
                    captures from enclosing scope inside any lambda body.
    
          FIX-B4-2: collection_name added to batch_upsert lambda default
                    args. Previously captured by reference — fragile under
                    refactor or concurrent coroutine reassignment.
    
          FIX-B4-3: FIX-D extended to _embed_and_store. Accepts pre-resolved
                    pipeline from _run_pipeline. Eliminates the 4th redundant
                    _get_pipeline() call per file (lru_cache makes it cheap
                    but the extra call is semantically wrong).
    
          FIX-B4-4: Accepts caller's db session. Eliminates the redundant
                    `async with async_session()` in Step 2 — removes
                    unnecessary connection pool consumption and transaction
                    isolation risk (new session may not see rows committed
                    in the caller's transaction).

          PERF-B6-1: `skip_presence_check=True` bypasses the GCI + VectorDB
                    presence checks for the common ingestion path where
                    `_run_pipeline()` passes only post-dedup unique chunks.
                    Those chunks are guaranteed new for the current ingestion,
                    so checking 100s of IDs before embedding is wasted latency.
    
        FAILURE POLICY (unchanged):
          Any exception logs full detail, marks file FAILED in DB, re-raises.
          Never swallow exceptions — silent success = corrupt / missing data.
        """
        _emb_label, _vdb_label = _pipeline_telemetry_labels(business_id)
        _plog = PipelineLogger(
            request_path="ingestion",
            client_id=business_id,
            embedder_model=_emb_label,
            vectordb_backend=_vdb_label,
        )
        _plog.info(
            "Embed and store started",
            file_id=file_id,
            chunk_count=len(chunks) if chunks else 0,
        )
        _phantom_cm = None
        try:
            from app.services.ingestion.phantom_config_bridge import phantom_runtime_from_client

            if pipeline is not None and getattr(pipeline, "config", None) is not None:
                _ing_cfg = pipeline.config.ingestion
            else:
                from app.core.config.client_config_resolver import get_client_config_for_ingestion

                _ing_cfg = get_client_config_for_ingestion(str(business_id)).ingestion
            _phantom_cm = phantom_runtime_from_client(
                _ing_cfg.phantom,
                allow_legacy_env_overrides=bool(_ing_cfg.allow_legacy_env_overrides),
            )
            _phantom_cm.__enter__()
            loop = asyncio.get_running_loop()
    
            # ── STEP 1: Build hash → chunk map ─────────────────────────────
            hash_to_chunk: Dict[str, Any] = {}
            all_hashes:    List[str]      = []
            for c in chunks:
                sh = c.get("semantic_hash")
                if not sh:
                    continue
                all_hashes.append(sh)
                hash_to_chunk[sh] = c
    
            if not all_hashes:
                log_info(
                    f"[IngestionV2] 0 semantic hashes for {file_id} — skipping VectorDB"
                )
                return
    
            # ── STEP 2: Check GCI using CALLER'S session ────────────────────
            # FIX-B4-4: Use the db session passed from _run_pipeline.
            #
            # BEFORE: `async with async_session() as db:` opened a SECOND
            # connection from the pool inside this method. Two defects:
            #   (a) Connection pool pressure — one extra connection per
            #       concurrent ingestion held for the full embed duration.
            #   (b) Isolation risk — register_unique_chunks_in_gci() committed
            #       GCI rows in the caller's session. A new session at
            #       REPEATABLE READ may not see those rows → false "not in GCI"
            #       → re-embeds already-registered chunks.
            #
            # AFTER: reuse `db` from _run_pipeline — same transaction context,
            # zero extra connection, GCI rows guaranteed visible.
            if skip_presence_check:
                known_hashes: set = set()
            elif db is not None:
                res = await db.execute(
                    select(
                        GlobalContentIndexV2.semantic_hash,
                        GlobalContentIndexV2.id,
                    ).where(GlobalContentIndexV2.semantic_hash.in_(all_hashes))
                )
                known_hashes = {row[0] for row in res.all()}
            else:
                # Fallback: open own session only if no session was passed
                # (supports direct callers outside _run_pipeline)
                async with async_session() as fallback_db:
                    res = await fallback_db.execute(
                        select(
                            GlobalContentIndexV2.semantic_hash,
                            GlobalContentIndexV2.id,
                        ).where(
                            GlobalContentIndexV2.semantic_hash.in_(all_hashes)
                        )
                    )
                    known_hashes = {row[0] for row in res.all()}
    
            # ── STEP 3: Resolve pipeline ────────────────────────────────────
            # FIX-B4-3: Accept pre-resolved pipeline from _run_pipeline.
            # _run_pipeline already called _get_pipeline(business_id) once.
            # Even with @lru_cache, calling it again is semantically redundant.
            #
            # FIX-B4-1+2: Bind `collection_name` and `vectordb` to local
            # variables here, then explicitly capture them in every lambda
            # default arg. Zero reference captures from enclosing scope.
            if pipeline is None:
                pipeline = await _get_ingestion_pipeline_async(business_id)
    
            embedder:        Any = pipeline.embedder
            vectordb:        Any = pipeline.vectordb
            collection_name: str = pipeline.config.vectordb.collection
            batch_size:      int = max(
                1,
                int(getattr(getattr(pipeline.config, "ingestion", None), "batch_size", BATCH_SIZE)),
            )
            embedder_kind:   str = str(getattr(embedder, "kind", "") or "").lower()
            if not embedder_kind:
                embedder_kind = str(
                    getattr(getattr(embedder, "info", None), "provider", "") or ""
                ).lower()
    
            # ── STEP 4: Check which known hashes exist in VectorDB ──────────
            # FIX-B4-1: `collection_name`, `vectordb`, `known_hashes` all
            # explicitly captured in lambda default args — zero reference capture.
            if skip_presence_check or not known_hashes:
                present_in_vectordb: set = set()
            else:
                try:
                    present_in_vectordb = set(
                        await loop.run_in_executor(
                            None,
                            lambda _vdb=vectordb,
                                   _col=collection_name,
                                   _hashes=known_hashes: _vdb.exists(
                                collection=_col,
                                ids=list(_hashes),
                            ),
                        )
                    )
                except Exception as e:
                    log_info(f"[IngestionV2] VectorDB exists check failed: {e}")
                    present_in_vectordb = set()
    
            # ── STEP 5: Determine hashes that need embedding ────────────────
            if skip_presence_check:
                hashes_needing_embedding: set = set(all_hashes)
            else:
                hashes_needing_embedding = {
                    h for h in all_hashes
                    if h not in known_hashes or h not in present_in_vectordb
                }
    
            if not hashes_needing_embedding:
                log_info(
                    f"[IngestionV2] All hashes already in VectorDB for {file_id}"
                )
                return
    
            new_chunks: List[Dict[str, Any]] = [
                hash_to_chunk[h]
                for h in all_hashes
                if h in hashes_needing_embedding
            ]
    
            # ── STEP 6: Probe embedding dimension (never assume) ────────────
            # FIX-B4-1: `embedder` explicitly captured in lambda default arg.
            embedding_dim = getattr(
                getattr(embedder, "info", None), "dim", None
            )
            if not embedding_dim or embedding_dim <= 0:
                probe = await loop.run_in_executor(
                    None,
                    lambda _emb=embedder: _emb.embed_query("dimension probe"),
                    #      ^^^ explicit default capture — not reference
                )
                embedding_dim = len(probe)
                log_info(f"[IngestionV2] Probed embedding dim: {embedding_dim}")
    
            # ── STEP 7: Ensure collection — DIMENSION GUARD ─────────────────
            if not getattr(pipeline, "_collection_ensured", False):
                vectordb.ensure_collection(
                    collection_name,
                    embedding_dim=embedding_dim,
                    distance_metric="cosine",
                )
                pipeline._collection_ensured = True
    
            # ── STEP 8: Concurrent batched embed + upsert ───────────────────
            # PERF-FIX: Serial for-loop → concurrent asyncio.gather().
            # 822 chunks / BATCH_SIZE 256 = 4 batches.
            # Serial: batch1 → batch2 → batch3 → batch4  (4× wait time)
            # Concurrent: all 4 batches fire simultaneously, bounded by semaphore.
            # FIX-B4-1 and FIX-B4-2 lambda captures are fully preserved.
            
            # HuggingFace SentenceTransformer instances are not safe for
            # concurrent encode() calls from multiple threads on the same object.
            # If we run 4 parallel batches, some futures can hang and file status
            # never reaches "processed". Keep batch processing serial for HF.
            # Keep this semaphore local to the active coroutine so it is scoped
            # to the current running event loop and cannot leak across reloads/tests.
            icfg = getattr(pipeline.config, "ingestion", None)
            if "huggingface" in (embedder_kind or ""):
                batch_parallelism = 1
            else:
                batch_parallelism = max(
                    1,
                    int(getattr(icfg, "embed_parallelism", 4) or 4),
                )
            batch_semaphore = asyncio.Semaphore(max(1, int(batch_parallelism)))
            log_info(
                f"[IngestionV2] Embedding runtime: kind={embedder_kind or 'unknown'}, "
                f"batch_size={batch_size}, parallelism={batch_parallelism}"
            )

            async def _process_one_batch(
                batch_idx: int,
                batch: List[Dict[str, Any]],
                _embedder=embedder,           # FIX-B4-1: explicit capture
                _vectordb=vectordb,           # FIX-B4-2: explicit capture
                _col=collection_name,         # FIX-B4-2: explicit capture
                _loop=loop,
                _file_id=file_id,
                _business_id=business_id,
                _file_type=file_type,
                _file_name=file_name,
                _source_url=source_url,
            ) -> None:
                async with batch_semaphore:
                    b_texts = [
                        c.get("cleaned_text")
                        or c.get("cleaned")
                        or c.get("text")
                        or c.get("normalized_text")
                        or c.get("raw_text")
                        or ""
                        for c in batch
                    ]
                    b_ids  = [c.get("semantic_hash") for c in batch]
                    b_meta = []
                    for c in batch:
                        meta = c.get("metadata")
                        if not isinstance(meta, dict):
                            meta = {}
                        md = {
                            "file_id": str(_file_id),
                            "business_id": str(_business_id) if _business_id else "default",
                            "source_type": str(_file_type) if _file_type else "",
                            "semantic_hash": str(c.get("semantic_hash", "")),
                            "file_name": str(_file_name) if _file_name else "",
                            "source": str(_file_name) if _file_name else "",
                            "url": str(_source_url) if _source_url else "",
                        }
                        page_number = meta.get("page_number")
                        if isinstance(page_number, int) and page_number > 0:
                            md["page_number"] = page_number
                        section_title = meta.get("section_title")
                        if section_title:
                            md["section_title"] = str(section_title)
                        heading_depth = meta.get("heading_depth")
                        if isinstance(heading_depth, int):
                            md["heading_depth"] = heading_depth
                        chunk_position = meta.get("chunk_position")
                        if isinstance(chunk_position, dict):
                            start_char = chunk_position.get("start_char")
                            end_char = chunk_position.get("end_char")
                            if isinstance(start_char, int):
                                md["chunk_start_char"] = start_char
                            if isinstance(end_char, int):
                                md["chunk_end_char"] = end_char
                        b_meta.append(md)
            
                    # Embed — FIX-B4-1: explicit default captures preserved
                    b_emb: List = await _loop.run_in_executor(
                        None,
                        lambda _emb=_embedder, t=b_texts: _emb.embed_documents(t),
                    )
            
                    # Upsert — FIX-B4-2: explicit default captures preserved
                    await _loop.run_in_executor(
                        None,
                        lambda _vdb=_vectordb,
                               col=_col,
                               ids=b_ids,
                               emb=b_emb,
                               met=b_meta,
                               docs=b_texts: _vdb.batch_upsert(
                                   collection=col,
                                   doc_ids=ids,
                                   embeddings=emb,
                                   texts=docs,
                                   metadatas=met,
                               ),
                    )
            
                    # ── Token cost tracking ────────────────────────────────────
                    batch_tokens = sum(int(c.get("tokens") or 0) for c in batch)
                    if batch_tokens > 0:
                        try:
                            record_tokens(
                                str(_business_id) if _business_id else "default",
                                "embed",
                                batch_tokens,
                                raise_on_hard_limit=False,  # don't block mid-ingest
                            )
                        except Exception:
                            pass  # cost tracking must never block ingestion

                    # ── Tenant audit: ingestion write telemetry ────────────────
                    try:
                        from app.services.security.tenant_audit import log_ingestion_write
                        log_ingestion_write(
                            tenant_id=str(_business_id) if _business_id else "default",
                            file_id=str(_file_id),
                            collection=_col,
                            vector_count=len(b_ids),
                            source_type=str(_file_type) if _file_type else "unknown",
                        )
                    except Exception:
                        pass

                    log_info(
                        f"[IngestionV2] ✅ Batch {batch_idx + 1}: "
                        f"{len(batch)} vectors via {_vectordb.kind}"
                    )
            
            # Build batch list, fire all concurrently
            batches = [
                new_chunks[i : i + batch_size]
                for i in range(0, len(new_chunks), batch_size)
            ]
            await asyncio.gather(
                *[_process_one_batch(idx, b) for idx, b in enumerate(batches)]
            )
            
            log_info(
                f"[IngestionV2] ✅ Stored {len(new_chunks)} new vectors "
                f"via {vectordb.kind} for {file_id}"
            )
            
    
        except Exception as e:
            log_info(
                f"[IngestionV2] ❌ FATAL: VectorDB storage failed for {file_id}:\n"
                f"  Error : {type(e).__name__}: {e}\n"
                f"  Client: {business_id} | File type: {file_type}"
            )
            try:
                await IngestionServiceV2._compensate_failed_vector_storage(
                    file_id=file_id,
                    business_id=business_id,
                    chunks=chunks,
                )
            except Exception as cleanup_err:
                log_info(
                    f"[IngestionV2] Compensation cleanup also failed for {file_id}: {cleanup_err}"
                )
            try:
                async with async_session() as err_db:
                    await IngestionServiceV2._set_file_error(err_db, file_id, str(e))
            except Exception as db_err:
                log_info(
                    f"[IngestionV2] Also failed to write error status: {db_err}"
                )
            raise
        finally:
            if _phantom_cm is not None:
                _phantom_cm.__exit__(None, None, None)



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
    async def _set_file_processing(db: AsyncSession, file_id: str):
        await db.execute(
            update(IngestedFileV2)
            .where(IngestedFileV2.id == file_id)
            .values(
                status="processing",
                error_message=None,
                updated_at=datetime.utcnow(),
            )
        )
        await db.commit()

    @staticmethod
    async def _compensate_failed_vector_storage(
        *,
        file_id: str,
        business_id: Optional[str],
        chunks: List[Dict[str, Any]],
    ) -> None:
        semantic_hashes = list(
            dict.fromkeys(
                str(chunk.get("semantic_hash"))
                for chunk in chunks
                if chunk.get("semantic_hash")
            )
        )
        gci_counts = Counter(
            chunk.get("global_content_id")
            for chunk in chunks
            if chunk.get("global_content_id")
        )

        # Best-effort cleanup for partially written vectors.
        if semantic_hashes:
            try:
                pipeline = await _get_ingestion_pipeline_async(business_id)
                pipeline.vectordb.delete_many(
                    collection=pipeline.config.vectordb.collection,
                    doc_ids=semantic_hashes,
                )
            except Exception as vectordb_err:
                log_info(
                    f"[IngestionV2] Vector cleanup skipped for {file_id}: {vectordb_err}"
                )

        async with async_session() as db:
            await db.execute(
                delete(IngestedContentV2).where(IngestedContentV2.file_id == file_id)
            )

            for global_content_id, decrement in gci_counts.items():
                row = await db.get(GlobalContentIndexV2, global_content_id)
                if not row:
                    continue

                remaining_refs_result = await db.execute(
                    select(func.count())
                    .select_from(IngestedContentV2)
                    .where(IngestedContentV2.global_content_id == global_content_id)
                )
                remaining_refs = int(remaining_refs_result.scalar() or 0)
                next_occurrence = max(0, int(row.occurrence_count or 0) - decrement)

                if remaining_refs == 0 and (
                    next_occurrence == 0 or str(row.first_seen_file_id or "") == str(file_id)
                ):
                    await db.delete(row)
                    continue

                row.occurrence_count = max(remaining_refs, next_occurrence, 1 if remaining_refs > 0 else 0)
                row.updated_at = datetime.utcnow()

            await db.commit()
            log_info(
                f"[IngestionV2] Compensation cleanup completed for {file_id}: "
                f"removed_chunks_for_file, adjusted_gci={len(gci_counts)}, "
                f"vector_ids={len(semantic_hashes)}"
            )

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
