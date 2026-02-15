# =============================================
# ingestion_service_v2.py — Unified Ingestion Service (UI + Bulk)
# Fully aligned with PostgreSQL schema, Chroma, and FK-safe
# Now includes: Direct ingestion support for pre-parsed inputs (RSS, API, etc.)
# =============================================

import uuid
import hashlib
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional
import re
import asyncio
from app.api.v2.ingestion_ws_api import broadcast
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy import select, insert, update, func

from app.db.session_v2 import async_engine
from app.db.models.ingested_file_v2 import IngestedFileV2
from app.db.models.ingested_content_v2 import IngestedContentV2
from app.db.models.global_content_index_v2 import GlobalContentIndexV2

from app.services.ingestion.parsers_router_v2 import ParserRouterV2
from app.services.ingestion.row_segmenter_v2 import parse_dataframe_rows
from app.services.ingestion.segmenter_v2 import recursive_semantic_chunk
from app.utils.logger import log_info
from app.config.ingestion_settings import EMBEDDING_MODEL_NAME
from sentence_transformers import SentenceTransformer

import chromadb


# =============================================
# CONFIGURATION
# =============================================
CHROMA_PATH = "./chroma_db"
#EMBED_MODEL_NAME = "BAAI/bge-large-en"
BATCH_SIZE = 256

# Lazy-initialized resources (do not create heavy clients/models at import time)
CHROMA_CLIENT = None
COLLECTION = None
EMBEDDER = None

# ✅ NEW: Cached collection count (refreshes every 5 minutes)
_COLLECTION_COUNT_CACHE = {
    "count": 0,
    "last_updated": None
}

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


def get_chroma_collection(skip_count: bool = False):
    """
    Lazily create and return a chroma collection.
    
    Args:
        skip_count: If True, skip the expensive count() operation.
                   Recommended for startup to avoid delays with large collections.
    
    Returns:
        Tuple[chromadb.Client, chromadb.Collection]
    """
    global CHROMA_CLIENT, COLLECTION
    
    if CHROMA_CLIENT is None:
        try:
            CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_PATH)
            log_info(f"[IngestionV2] ChromaDB client initialized at {CHROMA_PATH}")
        except Exception as e:
            log_info(f"[ERROR] Failed to initialize ChromaDB client: {e}")
            raise RuntimeError(f"ChromaDB initialization failed: {e}")
    
    if COLLECTION is None:
        try:
            # List all collections first
            existing = CHROMA_CLIENT.list_collections()
            collection_names = [col.name for col in existing]
            log_info(f"[IngestionV2] Found ChromaDB collections: {collection_names}")
            
            # Check if our collection exists
            if "ingested_content" in collection_names:
                COLLECTION = CHROMA_CLIENT.get_collection(name="ingested_content")
                
                # ✅ Only count if explicitly requested (avoid startup delay)
                if not skip_count:
                    try:
                        count = COLLECTION.count()
                        log_info(f"[IngestionV2] Connected to existing collection 'ingested_content' ({count:,} vectors)")
                    except Exception as count_err:
                        log_info(f"[IngestionV2] Connected to existing collection 'ingested_content' (count failed: {count_err})")
                else:
                    log_info(f"[IngestionV2] ✅ Connected to existing collection 'ingested_content'")
            else:
                # Collection doesn't exist, create it
                log_info("[IngestionV2] Collection 'ingested_content' not found, creating...")
                COLLECTION = CHROMA_CLIENT.create_collection(
                    name="ingested_content",
                    metadata={"hnsw:space": "cosine"}
                )
                log_info("[IngestionV2] ✅ Created new collection 'ingested_content' (0 vectors)")
                
        except Exception as e:
            log_info(f"[ERROR] Failed to get/create ChromaDB collection: {e}")
            import traceback
            log_info(traceback.format_exc())
            raise RuntimeError(f"ChromaDB collection setup failed: {e}")
    
    return CHROMA_CLIENT, COLLECTION


def get_collection_count_cached(force_refresh: bool = False):
    """
    Get collection count with caching to avoid expensive operations on large collections.
    
    Cache Strategy:
    - First call: Fetches count and caches it
    - Subsequent calls: Returns cached value if <5 minutes old
    - force_refresh=True: Bypasses cache and fetches fresh count
    
    Args:
        force_refresh: If True, ignore cache and fetch fresh count
    
    Returns:
        int: Number of vectors in collection (0 if unavailable)
    """
    global _COLLECTION_COUNT_CACHE
    
    now = datetime.utcnow()
    last_updated = _COLLECTION_COUNT_CACHE.get("last_updated")
    
    # Cache duration: 5 minutes (300 seconds)
    CACHE_DURATION = 300
    
    # Return cached value if still fresh
    if not force_refresh and last_updated:
        age_seconds = (now - last_updated).total_seconds()
        if age_seconds < CACHE_DURATION:
            cached_count = _COLLECTION_COUNT_CACHE.get("count", 0)
            log_info(f"[IngestionV2] Returning cached vector count: {cached_count:,} (age: {int(age_seconds)}s)")
            return cached_count
    
    # Fetch fresh count
    try:
        _, collection = get_chroma_collection(skip_count=True)  # Don't double-count
        count = collection.count()
        
        # Update cache
        _COLLECTION_COUNT_CACHE["count"] = count
        _COLLECTION_COUNT_CACHE["last_updated"] = now
        
        log_info(f"[IngestionV2] Fetched fresh vector count: {count:,}")
        return count
        
    except Exception as e:
        log_info(f"[IngestionV2] Failed to get collection count: {e}")
        # Return stale cache if available, otherwise 0
        return _COLLECTION_COUNT_CACHE.get("count", 0)




def get_embedder():
    """
    Lazily create and return the embedder model.
    """
    global EMBEDDER  # ✅ Fixed - no underscore
    if EMBEDDER is None:  # ✅ Fixed
        EMBEDDER = SentenceTransformer(EMBEDDING_MODEL_NAME)  # ✅ Fixed
    return EMBEDDER  # ✅ Fixed



async def log_event(file_name: str, stage: str, status: str, message: str = ""):
    event = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
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
                # ==========================================================
                # HARD MEDIA-LEVEL DEDUP (AUTHORITATIVE — API + WATCHER)
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
                        media_hash=media_hash,
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

        unique_chunks = await IngestionServiceV2._dedup_chunks(db, chunks, file_id)
        if not unique_chunks:
            log_info(f"[IngestionV2] All chunks duplicates for {file_id}")
            await IngestionServiceV2._update_file_status(db, file_id, 0, "processed")
            return

        await IngestionServiceV2._insert_chunks(db, file_id, business_id, unique_chunks)
        await IngestionServiceV2._embed_and_store(file_id, business_id, file_type, unique_chunks)
        await IngestionServiceV2._update_file_status(db, file_id, len(unique_chunks), "processed")

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
    async def _dedup_chunks(db: AsyncSession, chunks: List[Dict[str, Any]], file_id: str) -> List[Dict[str, Any]]:
        # Batch-query existing semantic_hash values for this file to avoid per-chunk queries
        hashes = [c.get("semantic_hash") for c in chunks if c.get("semantic_hash")]
        if not hashes:
            log_info(f"[IngestionV2] 0 unique chunks retained (no semantic_hashes present)")
            return []

        q = select(IngestedContentV2.semantic_hash).where(
            IngestedContentV2.file_id == file_id,
            IngestedContentV2.semantic_hash.in_(hashes),
        )
        result = await db.execute(q)
        existing = set(result.scalars().all())

        unique_chunks = [c for c in chunks if c.get("semantic_hash") not in existing]
        log_info(f"[IngestionV2] {len(unique_chunks)} unique chunks retained")
        return unique_chunks

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
                "is_duplicate": False,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
            for i, c in enumerate(chunks)
        ]
    
        await db.execute(insert(IngestedContentV2), db_rows)
        await db.commit()
        log_info(f"[IngestionV2] Inserted {len(db_rows)} chunks into DB")


    # ----------------------------------------------------------
    # Embedding + Vector Store (executor-offloaded + DB-and-Chroma-checked)
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

            # ensure chroma collection exists (lazy)
            _, collection = get_chroma_collection()

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

            # Query GlobalContentIndexV2 for known hashes (mapping semantic_hash -> gci_id)
            async with async_session() as db:
                q = select(GlobalContentIndexV2.semantic_hash, GlobalContentIndexV2.id).where(
                    GlobalContentIndexV2.semantic_hash.in_(all_hashes)
                )
                res = await db.execute(q)
                rows = res.all()
                # build set of known hashes (present in DB)
                known_hashes = {row[0] for row in rows}

            # For the known hashes, check Chroma whether a vector exists (by id = semantic_hash)
            # Note: calling collection.get for specific ids is OK (not enumerating whole collection).
            loop = asyncio.get_running_loop()
            present_in_chroma = set()
            try:
                # run collection.get(ids=...) in executor to avoid blocking
                def _chroma_get(ids):
                    try:
                        # call collection.get with ids — returns entries for found ids
                        return collection.get(ids=ids)
                    except Exception:
                        # if Chroma client raises (e.g. unsupported param), return empty
                        return {}

                chroma_resp = await loop.run_in_executor(None, lambda: _chroma_get(list(known_hashes)))
                # chroma_resp can be dict-like with 'ids' or 'metadatas' depending on client;
                # defensively extract returned ids or metadatas
                found_ids = set()
                if isinstance(chroma_resp, dict):
                    # Try common keys
                    if "ids" in chroma_resp and isinstance(chroma_resp["ids"], list):
                        found_ids.update(chroma_resp["ids"])
                    elif "metadatas" in chroma_resp and isinstance(chroma_resp["metadatas"], list):
                        # If metadatas present, assume same ordering and use any id list if available
                        # best-effort: try to extract 'id' inside each meta if exists
                        for md in chroma_resp["metadatas"]:
                            if isinstance(md, dict) and "semantic_hash" in md:
                                found_ids.add(md["semantic_hash"])
                elif isinstance(chroma_resp, list):
                    # some clients may return list of item dicts
                    for item in chroma_resp:
                        if isinstance(item, dict) and item.get("id"):
                            found_ids.add(item.get("id"))
                # fallback: assume no ids found if we can't parse
                present_in_chroma = set(found_ids) & set(known_hashes)
            except Exception as e:
                # If Chroma check fails, log and continue — we will attempt to embed all hashes not found in GCI
                log_info(f"[IngestionV2] Warning: Chroma check failed: {e}")
                present_in_chroma = set()

            # Determine which hashes truly need embedding:
            # - If hash not found in GCI -> treat as new (should be embedded)
            # - If hash found in GCI but not present_in_chroma -> needs embedding
            hashes_needing_embedding = set()
            for h in all_hashes:
                if h not in known_hashes:
                    hashes_needing_embedding.add(h)
                elif h not in present_in_chroma:
                    hashes_needing_embedding.add(h)

            if not hashes_needing_embedding:
                log_info(f"[IngestionV2] No new vectors to add (Chroma already has vectors for all hashes)")
                return

            # Build list of chunks to embed (preserve order)
            new_chunks = [hash_to_chunk[h] for h in all_hashes if h in hashes_needing_embedding]

            # Prepare ids to be semantic_hash (tie Chroma id to semantic_hash)
            # This makes future existence checks straightforward and idempotent.
            ids = [c.get("semantic_hash") for c in new_chunks]

            # Batched embedding + upsert (embedder offloaded to executor)
            embedder = get_embedder()
            for i in range(0, len(new_chunks), BATCH_SIZE):
                batch = new_chunks[i:i + BATCH_SIZE]
                texts = [c.get("cleaned_text", c.get("cleaned", "")) for c in batch]
                batch_ids = [c.get("semantic_hash") for c in batch]

                # Offload embedding to executor to avoid blocking event loop
                embeddings = await loop.run_in_executor(None, lambda: embedder.encode(texts, normalize_embeddings=True))

                # Normalize embedding output to list-of-lists
                try:
                    emb_list = embeddings.tolist()
                except Exception:
                    emb_list = list(embeddings)

                metadatas = [
                    {
                        "file_id": str(file_id),
                        "business_id": str(business_id) if business_id else None,
                        "source_type": file_type,
                        "semantic_hash": c.get("semantic_hash"),
                    }
                    for c in batch
                ]

                # Upsert to Chroma using semantic_hash as id — offload to executor
                await loop.run_in_executor(
                    None,
                    lambda ids=batch_ids, emb=emb_list, met=metadatas, docs=texts: collection.upsert(
                        ids=ids, embeddings=emb, metadatas=met, documents=docs
                    ),
                )

            log_info(f"[IngestionV2] Stored {len(new_chunks)} new unique vectors in ChromaDB for file {file_id}")

        except Exception as e:
            log_info(f"[ERROR] Embedding or Chroma storage failed: {e}")
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
    async def _update_file_status(db: AsyncSession, file_id, total_chunks, status):
        await db.execute(
            update(IngestedFileV2)
            .where(IngestedFileV2.id == file_id)
            .values(
                total_chunks=total_chunks,
                status=status,
                updated_at=datetime.utcnow(),
            )
        )
        await db.commit()
        log_info(f"[IngestionV2] File {file_id} status updated to {status}")
