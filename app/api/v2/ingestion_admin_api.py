# =============================================
# app/api/v2/ingestion_admin_api.py
# =============================================
import asyncio

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from app.db.models.admin_audit_log import AdminAuditLog
from app.db.models.ingested_file_v2 import IngestedFileV2
from app.db.models.ingested_content_v2 import IngestedContentV2
from app.db.models.global_content_index_v2 import GlobalContentIndexV2

# ✅ PERMANENT FIX: removed get_embedder, get_chroma_collection
# Use _get_pipeline() directly — works with ANY backend (Chroma, Qdrant, Milvus...)
from app.services.ingestion.ingestion_service_v2 import IngestionServiceV2, _get_pipeline
from app.llm.llm_client import run_llm_normalization
from app.db.session_v2 import get_db
from app.auth.guards import require_role

router = APIRouter(
    prefix="/api/v2/ingestion-admin",
    tags=["Ingestion Admin"],
)

# ===========================================================
# 1️⃣  LIST ALL INGESTED FILES
# ===========================================================
@router.get("/files")
async def list_ingested_files(
    db: AsyncSession = Depends(get_db),
    user=Depends(require_role("admin", "editor", "viewer")),
):
    """Returns a list of all ingested files with metadata."""
    result = await db.execute(
        select(IngestedFileV2).order_by(IngestedFileV2.created_at.desc())
    )
    files = result.scalars().all()
    return [
        {
            "id":               str(f.id),
            "file_name":        f.file_name,
            "file_type":        f.file_type,
            "source_type":      f.source_type,
            "status":           f.status,
            "parser_used":      f.parser_used,
            "total_chunks":     f.total_chunks,
            "unique_chunks":    f.unique_chunks,
            "duplicate_chunks": f.duplicate_chunks,
            "dedup_ratio":      f.dedup_ratio,
            "error_message":    f.error_message,
            "created_at":       f.created_at,
            "updated_at":       f.updated_at,
        }
        for f in files
    ]

# ===========================================================
# 2️⃣  GET FILE DETAIL
# ===========================================================
@router.get("/files/{file_id}")
async def get_file_detail(
    file_id: str,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_role("admin", "editor", "viewer")),
):
    file = await db.get(IngestedFileV2, file_id)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    return {
        "id":               str(file.id),
        "file_name":        file.file_name,
        "file_type":        file.file_type,
        "file_path":        file.file_path,
        "source_url":       file.source_url,
        "source_type":      file.source_type,
        "status":           file.status,
        "parser_used":      file.parser_used,
        "meta_data":        file.meta_data,
        "total_chunks":     file.total_chunks,
        "unique_chunks":    file.unique_chunks,
        "duplicate_chunks": file.duplicate_chunks,
        "dedup_ratio":      file.dedup_ratio,
        "error_message":    file.error_message,
        "created_at":       file.created_at,
        "updated_at":       file.updated_at,
    }

# ===========================================================
# 3️⃣  LIST CHUNKS FOR A FILE
# ===========================================================
@router.get("/files/{file_id}/chunks")
async def list_file_chunks(
    file_id: str,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_role("admin", "editor", "viewer")),
):
    result = await db.execute(
        select(IngestedContentV2, GlobalContentIndexV2)
        .outerjoin(
            GlobalContentIndexV2,
            IngestedContentV2.global_content_id == GlobalContentIndexV2.id,
        )
        .where(IngestedContentV2.file_id == file_id)
        .order_by(IngestedContentV2.chunk_index)
    )
    rows = result.all()
    return [
        {
            "id":                  str(chunk.id),
            "chunk_index":         chunk.chunk_index,
            "text":                chunk.text,
            "cleaned_text":        chunk.cleaned_text,
            "semantic_hash":       chunk.semantic_hash,
            "confidence":          chunk.confidence,
            "is_duplicate":        chunk.is_duplicate,
            "global_content_id":   str(chunk.global_content_id) if chunk.global_content_id else None,
            "gci_occurrence_count": gci.occurrence_count if gci else None,
            "created_at":          chunk.created_at,
        }
        for chunk, gci in rows
    ]

# ===========================================================
# 4️⃣  RETRY INGESTION
# ===========================================================
@router.post("/files/{file_id}/retry")
async def retry_ingestion(
    file_id: str,
    user=Depends(require_role("admin")),
):
    """Re-runs full ingestion for the file. Role: admin only."""
    await IngestionServiceV2.process_file(file_id)
    return {"status": "retry_started", "file_id": file_id}

# ===========================================================
# 5️⃣  CHUNK EDIT / LLM NORMALIZATION
# ===========================================================
class ChunkUpdatePayload(BaseModel):
    cleaned_text: str
    llm_mode: str | None = None   # None | "factual" | "creative"


@router.put("/chunks/{chunk_id}")
async def update_chunk(
    chunk_id: str,
    payload: ChunkUpdatePayload,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_role("admin", "editor")),
):
    """
    Edits a chunk's cleaned_text, optionally using LLM normalization,
    then re-embeds and upserts the vector via the pluggable pipeline.
    Role: admin, editor
    """
    # ── 1. Fetch chunk ────────────────────────────────────────────
    chunk = await db.get(IngestedContentV2, chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")

    original_text = chunk.cleaned_text or chunk.text
    final_text    = payload.cleaned_text

    # ── 2. Optional LLM normalization ────────────────────────────
    if payload.llm_mode in ("factual", "creative"):
        final_text = await run_llm_normalization(
            text=payload.cleaned_text,
            mode=payload.llm_mode,
        )

    # ── 3. Update chunk in DB ─────────────────────────────────────
    chunk.cleaned_text = final_text
    chunk.meta_data    = chunk.meta_data or {}
    chunk.meta_data.update({
        "manually_edited": True,
        "llm_mode":        payload.llm_mode,
    })

    # ── 4. Update GCI ─────────────────────────────────────────────
    gci = await db.get(GlobalContentIndexV2, chunk.global_content_id)
    if not gci:
        raise HTTPException(status_code=404, detail="GlobalContentIndex not found")
    gci.cleaned_text = final_text

    # ── 5. Re-embed + upsert via pluggable pipeline ───────────────
    # ✅ PERMANENT FIX:
    #   OLD: embedder.encode(text, normalize_embeddings=True).tolist()
    #        → SentenceTransformer-only, not pluggable
    #   OLD: collection.upsert(ids, embeddings, documents, metadatas)
    #        → raw Chroma API, not pluggable
    #
    #   NEW: embedder.embed_query(text)  → works for Ollama/OpenAI/HuggingFace
    #        vectordb.upsert(...)        → works for Chroma/Qdrant/Milvus/Pinecone
    #
    # Use chunk.business_id so per-business pipelines are respected.
    pipeline = _get_pipeline(chunk.business_id)
    embedder = pipeline.embedder
    vectordb = pipeline.vectordb

    loop   = asyncio.get_running_loop()
    vector = await loop.run_in_executor(
        None, lambda: embedder.embed_query(final_text)
    )

    await loop.run_in_executor(
        None,
        lambda: vectordb.upsert(
            collection="ingested_content",
            doc_id=gci.semantic_hash,
            embedding=vector,
            text=final_text,
            metadata={
                "edited":   True,
                "llm_mode": payload.llm_mode,
            },
        ),
    )

    # ── 6. Audit log ──────────────────────────────────────────────
    audit = AdminAuditLog(
        action="chunk_edit",
        entity_type="ingested_content",
        entity_id=chunk.id,
        before_value=original_text,
        after_value=final_text,
        meta_data={
            "llm_mode":      payload.llm_mode,
            "semantic_hash": gci.semantic_hash,
            "edited_by":     user["sub"],
            "vectordb_kind": vectordb.kind,
            "embedder_kind": embedder.kind,
        },
    )
    db.add(audit)
    await db.commit()

    return {
        "status":        "updated",
        "chunk_id":      chunk_id,
        "semantic_hash": gci.semantic_hash,
        "llm_mode":      payload.llm_mode,
        "vectordb_used": vectordb.kind,
        "embedder_used": embedder.kind,
    }
