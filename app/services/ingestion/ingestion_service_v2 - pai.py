# =============================================
# ingestion_service_v2.py — Unified Ingestion Service (UI + Bulk)
# Fully aligned with PostgreSQL schema, Chroma, and FK-safe
# Now includes: Direct ingestion support for pre-parsed inputs (RSS, API, etc.)
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
    ClientConfig, VectorDBConfig, EmbedderConfig,
    VectorDBType, EmbedderType,
    ChromaConfig, OllamaEmbedderConfig,
)

# ── Constants ─────────────────────────────────────────────────────────
BATCH_SIZE = 256

async_session = async_sessionmaker(async_engine, expire_on_commit=False, autoflush=False)


# =============================================
# HELPERS
# =============================================

async def _check_existing_file_by_hash(db: AsyncSession, file_hash: str) -> Optional[str]:
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

    # defensive flatten
    try:
        return " | ".join(
            f"{k}: {v}" for k, v in payload.items() if isinstance(v, (str, int, float))
        )
    except Exception:
        return ""
        

# ============================================================
# WHAT THIS FUNCTION IS:
#   The single place that resolves which pipeline to use.
#   Priority order:
#     1. Per-business config from environment variables
#     2. Global default from environment variables
#     3. Raise clearly if nothing is configured
#
# HOW TO CONFIGURE PER-BUSINESS:
#   Set env vars: MAI_ACME_VECTORDB=qdrant, MAI_ACME_EMBEDDER=openai
#   These override the global defaults for business_id="acme"
# ============================================================

def _get_pipeline(business_id: Optional[str] = None):
    """
    Resolve a live AssembledPipeline for the given business.
    
    Config is read from environment variables — no JSON files on disk.
    This makes the system work identically in dev, staging, and production
    without copying config files around.
    """
    b = (business_id or "default").lower().replace("-", "_")

    # ── Read vectordb config from env ────────────────────────────────
    vectordb_type = os.getenv(
        f"MAI_{b.upper()}_VECTORDB",        # per-business override
        os.getenv("MAI_VECTORDB", "chroma")  # global default
    ).lower()

    # ── Read embedder config from env ─────────────────────────────────
    embedder_type = os.getenv(
        f"MAI_{b.upper()}_EMBEDDER",
        os.getenv("MAI_EMBEDDER", "ollama")
    ).lower()

    # ── Read LLM config from env ───────────────────────────────────────
    llm_type = os.getenv(
        f"MAI_{b.upper()}_LLM",
        os.getenv("MAI_LLM", "ollama")
    ).lower()

    # ── Build a ClientConfig from env vars ────────────────────────────
    # This is passed to pipeline_factory.build() which handles caching
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

    # VectorDB config
    if vectordb_type == "chroma":
        from app.core.config.client_config_schema import ChromaConfig
        vdb_cfg = VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection=os.getenv("MAI_COLLECTION", "ingested_content"),
            chroma=ChromaConfig(
                persist_directory=os.getenv("CHROMA_PATH", "./chroma_db"),
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
    else:
        raise ValueError(f"Unknown vectordb type from env: '{vectordb_type}'")

    # Embedder config
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
    else:
        raise ValueError(f"Unknown embedder type from env: '{embedder_type}'")

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
    Converts visual / chart / table text into semantic explanation.
    Fail-safe: returns empty string on failure.
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


async def log_event(file_name: str, stage: str, status: str, message: str = ""):
    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "file": file_name,
        "stage": stage,
        "status": status,
        "message": message
    }
    await broadcast(event)

# =============================================
# INGESTION SERVICE V2 CLASS
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
                # compute_file_hash can be blocking — run in executor
                loop = asyncio.get_running_loop()
                try:
                    file_hash = await loop.run_in_executor(None, compute_file_hash, file_record.file_path)
                    meta_data["file_hash"] = file_hash
                except Exception as e:
                    log_info(f"[IngestionV2] Failed to compute file hash for {file_record.file_path}: {e}")

            file_entry = {
                "id": file_record.id,
                "business_id": file_record.business_id,
                "file_name": getattr(file_record, "file_name", str(file_record.id)),
                "file_type": file_record.file_type,
                "file_path": getattr(file_record, "file_path", None),
                "source_url": getattr(file_record, "source_url", None),
                "source_type": getattr(file_record, "source_type", file_record.file_type),
                "meta_data": meta_data,
                "parser_used": getattr(file_record, "parser_used", None),
                "status": "uploaded",
                "total_chunks": 0,
                "unique_chunks": 0,
                "duplicate_chunks": 0,
                "dedup_ratio": 0.0,
                "error_message": None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
            await db.execute(insert(IngestedFileV2), [file_entry])
            await db.commit()

    # ----------------------------------------------------------
    # Primary file ingestion entrypoint (UI + Bulk safe)
    # ----------------------------------------------------------
    @staticmethod
    async def process_file(file_id: str, file_path: Optional[str] = None, business_id: Optional[str] = None):
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
                
                        # 🔒 Global dedup across ALL sources
                        result = await db.execute(
                            select(IngestedFileV2)
                            .where(IngestedFileV2.meta_data["file_hash"].astext == incoming_hash)
                            .where(IngestedFileV2.status == "processed")
                        )
                        existing = result.scalar_one_or_none()
                
                        if existing and existing.id != file_record.id:
                            log_info(
                                f"[IngestionV2] ⛔ MEDIA DUPLICATE — already ingested as {existing.id}"
                            )
                
                            await IngestionServiceV2._update_file_status(
                                db,
                                file_record.id,
                                total_chunks=0,
                                status="duplicate",
                            )
                
                            # 🚫 HARD STOP — NOTHING BELOW RUNS
                            return
                
                        # Persist hash early so API + watcher converge
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
                        #media_hash=media_hash,
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
    async def ingest_parsed_output(file_id: str, parsed_output: Dict[str, Any]):
        async with async_session() as db:
            try:
                file_record = await IngestionServiceV2._get_file_record(db, file_id)
                if not file_record:
                    log_info(f"[IngestionV2] File not found for pre-parsed ingestion: {file_id}")
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
        result = await db.execute(select(IngestedFileV2).where(IngestedFileV2.id == file_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def _parse_file(file_path: str, file_type: str, db: AsyncSession, file_id: str):
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
    async def _run_pipeline(db: AsyncSession, file_record: IngestedFileV2, parsed_payload: Dict[str, Any]):
        await IngestionServiceV2._ensure_file_entry(db, file_record)

        file_id = file_record.id
        business_id = file_record.business_id
        file_type = file_record.file_type

        chunks = await IngestionServiceV2._extract_chunks(parsed_payload, file_id, file_type, business_id, db)

        if not chunks:
            log_info(f"[IngestionV2] No chunks to ingest for {file_id}")
            return

        unique_chunks, dedup_stats = await IngestionServiceV2._dedup_chunks(
            db, chunks, file_id, business_id
        )

        # ═══════════════════════════════════════════════════════════
        # ✅ FIX: Store ALL chunks in ingested_content (unique + duplicates)
        # ═══════════════════════════════════════════════════════════

        # Identify which chunks are unique
        unique_hashes = {c.get('semantic_hash') for c in unique_chunks}

        # Prepare all chunks for storage
        all_chunks_for_storage = []

        # Add unique chunks (marked as not duplicate)
        for chunk in unique_chunks:
            chunk['is_duplicate'] = False
            chunk['duplicate_of'] = None
            chunk['similarity_score'] = None
            all_chunks_for_storage.append(chunk)

        # Add duplicate chunks (marked appropriately)
        for chunk in chunks:
            semantic_hash = chunk.get('semantic_hash')
            if semantic_hash not in unique_hashes:
                # This is a duplicate chunk
                chunk['is_duplicate'] = True
                chunk['duplicate_of'] = chunk.get('global_content_id')  # Link to GCI
                chunk['similarity_score'] = chunk.get('similarity', None)
                all_chunks_for_storage.append(chunk)

        # ✅ ALWAYS insert chunks (unique + duplicates)
        if all_chunks_for_storage:
            await IngestionServiceV2._insert_chunks(
                db, file_id, business_id, all_chunks_for_storage
            )
            log_info(
                f"[IngestionV2] Inserted {len(all_chunks_for_storage)} chunks into ingested_content: "
                f"{len(unique_chunks)} unique, {len(all_chunks_for_storage) - len(unique_chunks)} duplicates"
            )

        # ✅ Only embed and store unique chunks in VectorDB (avoid duplicate vectors)
        
        # ✅ Embed any GCI-known hashes that are missing from VectorDB
        # Pass ALL chunks — _embed_and_store internally skips hashes already in Chroma
        chunks_with_hash = [c for c in chunks if c.get("semantic_hash")]
        if chunks_with_hash:
            log_info(f"[IngestionV2] Calling _embed_and_store for {file_id} with {len(chunks_with_hash)} chunks")
            await IngestionServiceV2._embed_and_store(file_id, business_id, file_type, chunks_with_hash)
        else:
            log_info(f"[IngestionV2] No chunks with semantic_hash — skipping VectorDB embed for {file_id}")
        

        # ✅ Update file status with detailed stats
        await IngestionServiceV2._update_file_status(
            db,
            file_id,
            total_chunks=dedup_stats['total'],
            unique_chunks=dedup_stats['unique'],
            duplicate_chunks=dedup_stats['duplicates'],
            dedup_ratio=dedup_stats['dedup_ratio'],
            status="processed"
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
    async def _extract_chunks(parsed_payload, file_id, file_type, business_id, db):
        try:
            if asyncio.iscoroutine(parsed_payload):
                parsed_payload = await parsed_payload

            # ✅ New addition: handle multiple source formats (RSS, API, etc.)
            if any(k in parsed_payload for k in ["entries", "rows", "chunks"]):
                base_list = parsed_payload.get("entries") or parsed_payload.get("rows") or parsed_payload.get("chunks")
                enriched_chunks = []
                for item in base_list:
                    # Defensive coercion: handle dict/list/str cases safely
                    raw_text = None
                    if isinstance(item, dict):
                        raw_text = (
                            item.get("cleaned_text")
                            or item.get("text")
                            or item.get("summary")
                            or item.get("description")
                        )
                        if raw_text is None:
                            # flatten small dict into readable text
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
                        # item is not a dict (could be string)
                        if hasattr(item, "get") and callable(item.get):
                            raw_text = item.get("text") or item.get("cleaned_text") or item.get("summary")
                        else:
                            raw_text = item

                    # coerce to string
                    if raw_text is None:
                        continue
                    if not isinstance(raw_text, str):
                        if isinstance(raw_text, list):
                            raw_text = " ".join(str(x) for x in raw_text if x is not None)
                        elif isinstance(raw_text, dict):
                            raw_text = " | ".join(f"{k}: {v}" for k, v in raw_text.items())
                        else:
                            raw_text = str(raw_text)

                    #text = raw_text.strip()
                    raw_text_original = raw_text.strip()
                    text = raw_text_original  # keep existing behavior unchanged
                    if not text:
                        continue
                    # ====================================================
                    # ADDITIVE BLOCK 3: VISUAL INTERCEPTION (MULTI)
                    # ====================================================
                    if _looks_like_visual_content(text):
                        explanation = await _explain_visual_with_llm(text)
                        if explanation:
                            explained_chunks = await recursive_semantic_chunk(
                                explanation,
                                db_session=db,
                                file_id=str(file_id),
                                business_id=business_id,
                                source_type=file_type,
                            )
                            for ch in explained_chunks:
                                ch.setdefault("reasoning_ingestion", {})
                                ch["reasoning_ingestion"].update({
                                    "content_type": "visual",
                                    "interpreted_by": "llm",
                                    "original_text_hash": hashlib.sha256(
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
                    )
                    enriched_chunks.extend(subchunks)
                return enriched_chunks

            text = _resolve_text(parsed_payload)
            if not text:
                log_info(f"[IngestionV2] No text found in parsed payload for {file_id}")
                return []
            
            # ====================================================
            # ADDITIVE BLOCK 4: VISUAL INTERCEPTION (SINGLE)
            # ====================================================
            if _looks_like_visual_content(text):
                explanation = await _explain_visual_with_llm(text)
                if explanation:
                    explained_chunks = await recursive_semantic_chunk(
                        explanation,
                        db_session=db,
                        file_id=str(file_id),
                        business_id=business_id,
                        source_type=file_type,
                    )
                    for ch in explained_chunks:
                        ch.setdefault("reasoning_ingestion", {})
                        ch["reasoning_ingestion"].update({
                            "content_type": "visual",
                            "interpreted_by": "llm",
                            "original_text_hash": hashlib.sha256(
                                text.encode("utf-8")
                            ).hexdigest(),
                        })
                    return explained_chunks
            # ====================================================            

            return await recursive_semantic_chunk(
                text,
                db_session=db,
                file_id=str(file_id),
                business_id=business_id,
                source_type=file_type,
            )

        except Exception as e:
            log_info(f"[CRITICAL] Chunk extraction failed: {e}")
            raise RuntimeError(f"Chunk extraction failed: {e}")

            
        

    # ----------------------------------------------------------
    # Deduplication (batched for performance)
    # ----------------------------------------------------------
    @staticmethod
    async def _dedup_chunks(
        db: AsyncSession,
        chunks: List[Dict[str, Any]],
        file_id: str,
        business_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """UPDATED: 3-layer deduplication with cross-file duplicate detection."""
        if not chunks:
            return [], {"total": 0, "unique": 0, "duplicates": 0, "dedup_ratio": 0.0}
    
        # ── Gap-2 Change-3: Replace get_chroma_collection() + get_embedder() ──
        pipeline = _get_pipeline(business_id)          # reuse Gap-1 helper
    
        unique_chunks, stats = await deduplicate_chunks(
            db=db,
            chunks=chunks,
            vectordb=pipeline.vectordb,      # BaseVectorDB ← pluggable
            embedder=pipeline.embedder,      # BaseEmbedder ← pluggable
            file_id=file_id,
            business_id=business_id,
            enable_embedding_dedup=True,     # Layer 2 semantic similarity
            similarity_threshold=0.95        # 95% similarity threshold
        )
        # ──────────────────────────────────────────────────────────────────────
    
        log_info(
            f"[IngestionV2] Dedup complete for file {file_id}: "
            f"{stats['unique']} unique, {stats['duplicates']} duplicates "
            f"{stats['dedup_ratio']:.2f}% reduction "
            f"[L1={stats.get('layer1_hash_duplicates', 0)}, "
            f"L2={stats.get('layer2_embedding_duplicates', 0)}, "
            f"L3={stats.get('layer3_gci_duplicates', 0)}]"
        )
        return unique_chunks, stats


    # ----------------------------------------------------------
    # Insert + Embeddings (Unified Chroma Dedup Safe)
    # ----------------------------------------------------------
    @staticmethod
    async def _insert_chunks(db: AsyncSession, file_id, business_id, chunks):
        # --------------------------------------------------
        # FIX: determine starting chunk_index offset
        # --------------------------------------------------
        result = await db.execute(
            select(func.max(IngestedContentV2.chunk_index))
            .where(IngestedContentV2.file_id == file_id)
        )
        start_index = (result.scalar() or -1) + 1
    
        db_rows = [
            {
                "id": uuid.uuid4(),
                "file_id": file_id,
                "business_id": business_id,
                "chunk_index": start_index + i,   # ✅ OFFSET APPLIED
                "text": c.get("text"),
                "cleaned_text": c.get("cleaned_text", c.get("cleaned")),
                "tokens": c.get("tokens"),
                "source_type": c.get("source_type"),
                "meta_data": c.get("metadata", {}),
                "confidence": c.get("confidence", 1.0),
                "semantic_hash": c.get("semantic_hash"),
                "global_content_id": c.get("global_content_id"),
                "reasoning_ingestion": c.get("reasoning_ingestion"),
                "is_duplicate": c.get("is_duplicate", False),  # ✅ USE THE FLAG FROM CHUNK!
                "duplicate_of": c.get("duplicate_of"),  # ✅ ADD THIS LINE TOO
                "similarity_score": c.get("similarity_score"),  # ✅ ADD THIS LINE TOO
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
            for i, c in enumerate(chunks)
        ]
    
        await db.execute(insert(IngestedContentV2), db_rows)
        await db.commit()
        log_info(f"[IngestionV2] Inserted {len(db_rows)} chunks into DB")

    # ----------------------------------------------------------
    # Embedding + Vector Store (executor-offloaded + DB-and-VectorDB-checked)
    # ----------------------------------------------------------
    @staticmethod
    async def _embed_and_store(file_id, business_id, file_type, chunks):
        try:
            # normalize cleaned texts
            clean_texts = [
                re.sub(r'\s*---(BLOCK|ENTRY) BREAK---\s*', '\n\n',
                       c.get("cleaned_text", c.get("cleaned", "")))
                for c in chunks
            ]
    
            # ✅ FIX 1: Define loop FIRST — it is used throughout this entire method
            loop = asyncio.get_running_loop()
    
            # ── Resolve pipeline from registry ───────────────────────────
            pipeline = _get_pipeline(business_id)
            embedder = pipeline.embedder   # BaseEmbedder (pluggable)
            vectordb = pipeline.vectordb   # BaseVectorDB (pluggable)
    
            # ✅ FIX 2: Probe actual dimension — never assume 1024
            embedding_dim = embedder.info.dim
            if not embedding_dim or embedding_dim <= 0:
                # Ask the embedder directly — works for any model
                probe = await loop.run_in_executor(
                    None, lambda: embedder.embed_query("dimension probe")
                )
                embedding_dim = len(probe)
                log_info(f"[IngestionV2] Probed embedding dim: {embedding_dim}")
    
            vectordb.ensure_collection(
                "ingested_content",
                embedding_dim=embedding_dim,
                distance_metric="cosine",
            )
    
            # gather semantic_hashes for all chunks (keep mapping)
            hash_to_chunk = {}
            all_hashes = []
            for c in chunks:
                sh = c.get("semantic_hash")
                if not sh:
                    continue
                all_hashes.append(sh)
                hash_to_chunk[sh] = c
    
            if not all_hashes:
                log_info(f"[IngestionV2] 0 semantic hashes for embedding for file {file_id}")
                return
    
            # Query GlobalContentIndexV2 for known hashes
            async with async_session() as db:
                q = select(GlobalContentIndexV2.semantic_hash, GlobalContentIndexV2.id).where(
                    GlobalContentIndexV2.semantic_hash.in_(all_hashes)
                )
                res = await db.execute(q)
                rows = res.all()
                known_hashes = {row[0] for row in rows}
    
            # ── Replace _chroma_get block with vectordb.exists() ──────────
            try:
                present_in_vectordb = set(
                    await loop.run_in_executor(
                        None,
                        lambda: vectordb.exists(
                            collection="ingested_content",
                            ids=list(known_hashes)
                        )
                    )
                )
            except Exception as e:
                log_info(f"[IngestionV2] Warning: VectorDB exists check failed: {e}")
                present_in_vectordb = set()
            # ──────────────────────────────────────────────────────────────
    
            # Determine which hashes truly need embedding
            hashes_needing_embedding = set()
            for h in all_hashes:
                if h not in known_hashes:
                    hashes_needing_embedding.add(h)
                elif h not in present_in_vectordb:
                    hashes_needing_embedding.add(h)
    
            if not hashes_needing_embedding:
                log_info(f"[IngestionV2] No new vectors to add (VectorDB already has all hashes)")
                return
    
            # Build list of chunks to embed (preserve order)
            new_chunks = [hash_to_chunk[h] for h in all_hashes if h in hashes_needing_embedding]
    
            # Batched embedding + upsert
            for i in range(0, len(new_chunks), BATCH_SIZE):
                batch     = new_chunks[i:i + BATCH_SIZE]
                texts     = [c.get("cleaned_text", c.get("cleaned", "")) for c in batch]
                batch_ids = [c.get("semantic_hash") for c in batch]
    
                # ── BaseEmbedder.embed_documents() — pluggable, no .encode() ──
                emb_list = await loop.run_in_executor(
                    None, lambda: embedder.embed_documents(texts)
                )
    
                metadatas = [
                    {
                        "file_id":       str(file_id),
                        "business_id":   str(business_id) if business_id else "",
                        "source_type":   str(file_type) if file_type else "",
                        "semantic_hash": str(c.get("semantic_hash", "")),
                    }
                    for c in batch
                ]
    
                # ── Single batch call — 1 API call instead of N calls ─────
                result = await loop.run_in_executor(
                    None,
                    lambda ids=batch_ids, emb=emb_list, met=metadatas, docs=texts:
                        vectordb.batch_upsert(
                            collection="ingested_content",
                            doc_ids=ids,
                            embeddings=emb,
                            texts=docs,
                            metadatas=met,
                        )
                )
                log_info(
                    f"[IngestionV2] ✅ Batch {i//BATCH_SIZE + 1}: "
                    f"+{result.inserted} new, ~{result.updated} updated via {vectordb.kind}"
                )
    
            log_info(f"[IngestionV2] Stored {len(new_chunks)} new unique vectors via {vectordb.kind} for file {file_id}")
    
        except Exception as e:
            log_info(f"[ERROR] Embedding or VectorDB storage failed: {e}")
            return

    
    # ----------------------------------------------------------
    # Error Handling + File Status
    # ----------------------------------------------------------
    @staticmethod
    async def _set_file_error(db: AsyncSession, file_id: str, error_message: str):
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
        total_chunks: int = 0,
        unique_chunks: int = 0,
        duplicate_chunks: int = 0,
        dedup_ratio: float = 0.0,
        status: str = "processed",
    ):
        """
        ✅ UPDATED: Track comprehensive deduplication metrics.
        """
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
                f"[IngestionV2] File {file_id} status updated: "
                f"status={status}, chunks={unique_chunks}/{total_chunks}, "
                f"dedup={dedup_ratio:.2f}%"
            )
    
        except Exception as e:
            log_info(f"[IngestionV2] Failed to update file status: {e}")
            await db.rollback()
