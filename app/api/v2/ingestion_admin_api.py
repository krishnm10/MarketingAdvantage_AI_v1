# =============================================
# app/api/v2/ingestion_admin_api.py
# =============================================
import asyncio
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import or_, select
from pydantic import BaseModel

from app.db.models.admin_audit_log import AdminAuditLog
from app.db.models.ingested_file_v2 import IngestedFileV2
from app.db.models.ingested_content_v2 import IngestedContentV2
from app.db.models.global_content_index_v2 import GlobalContentIndexV2

# ✅ PERMANENT FIX: removed get_embedder, get_chroma_collection
# Use _get_pipeline() directly — works with ANY backend (Chroma, Qdrant, Milvus...)
from app.services.ingestion.ingestion_service_v2 import IngestionServiceV2, _get_pipeline
from app.services.ingestion.ingestion_orchestrator import IngestionOrchestrator
from app.services.ingestion.tenant_guard import resolve_tenant_from_db_record
from app.llm.llm_client import run_llm_normalization
from app.db.session_v2 import get_db
from app.auth.guards import require_role
from app.utils.tenant_validator import validate_tenant_id, TenantValidationError
from app.utils.tenant_storage_uuid import storage_business_uuid_for_tenant

router = APIRouter(
    prefix="/api/v2/ingestion-admin",
    tags=["Ingestion Admin"],
)


def _assert_filescoped_to_tenant(file: IngestedFileV2, tenant_id: Optional[str]) -> None:
    """
    Enforce isolation when requesting file detail/chunks scoped to tenant_id.
    """
    if not tenant_id or not str(tenant_id).strip():
        return
    try:
        ctx = validate_tenant_id(
            str(tenant_id).strip(),
            source="query",
            endpoint="ingestion_admin_scope",
            allow_default=True,
        )
    except TenantValidationError:
        raise HTTPException(status_code=422, detail="Invalid tenant_id") from None

    scoped_uuid = storage_business_uuid_for_tenant(ctx.tenant_id, None)
    if file.business_id == scoped_uuid:
        return
    # Legacy uploads before deterministic UUID wiring (typically default tenant).
    if ctx.tenant_id == "default" and file.business_id is None:
        return

    raise HTTPException(status_code=404, detail="File not found")


# ===========================================================
# 1️⃣  LIST ALL INGESTED FILES
# ===========================================================
@router.get("/files")
async def list_ingested_files(
    tenant_id: Optional[str] = Query(
        None,
        description="When set, return only ingestion rows belonging to this tenant (isolated slice).",
    ),
    db: AsyncSession = Depends(get_db),
    user=Depends(require_role("admin", "editor", "viewer")),
):
    """Returns a list of ingested files; optional tenant isolation via tenant_id."""
    stmt = select(IngestedFileV2)
    if tenant_id is not None and str(tenant_id).strip():
        try:
            ctx = validate_tenant_id(
                str(tenant_id).strip(),
                source="query",
                endpoint="ingestion_admin_files",
                allow_default=True,
            )
        except TenantValidationError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

        scoped_uuid = storage_business_uuid_for_tenant(ctx.tenant_id, None)
        if ctx.tenant_id == "default":
            stmt = stmt.where(
                or_(
                    IngestedFileV2.business_id == scoped_uuid,
                    IngestedFileV2.business_id.is_(None),
                )
            )
        else:
            stmt = stmt.where(IngestedFileV2.business_id == scoped_uuid)

    stmt = stmt.order_by(IngestedFileV2.created_at.desc())
    result = await db.execute(stmt)
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
    tenant_id: Optional[str] = Query(None, description="Scope check — rejects files outside tenant"),
    db: AsyncSession = Depends(get_db),
    user=Depends(require_role("admin", "editor", "viewer")),
):
    file = await db.get(IngestedFileV2, file_id)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    _assert_filescoped_to_tenant(file, tenant_id)
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
    tenant_id: Optional[str] = Query(None, description="Scope check — rejects files outside tenant"),
    db: AsyncSession = Depends(get_db),
    user=Depends(require_role("admin", "editor", "viewer")),
):
    parent = await db.get(IngestedFileV2, file_id)
    if not parent:
        raise HTTPException(status_code=404, detail="File not found")
    _assert_filescoped_to_tenant(parent, tenant_id)

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
            "page_number":         chunk.page_number,
            "parent_chunk_id":     str(chunk.parent_chunk_id) if chunk.parent_chunk_id else None,
            "text":                chunk.text,
            "cleaned_text":        chunk.cleaned_text,
            "tokens":              chunk.tokens,
            "source_type":         chunk.source_type,
            "semantic_hash":       chunk.semantic_hash,
            "confidence":          chunk.confidence,
            "is_duplicate":        chunk.is_duplicate,
            "duplicate_of":        str(chunk.duplicate_of) if chunk.duplicate_of else None,
            "similarity_score":    chunk.similarity_score,
            "global_content_id":   str(chunk.global_content_id) if chunk.global_content_id else None,
            "gci_occurrence_count": gci.occurrence_count if gci else None,
            "meta_data":           chunk.meta_data or {},
            "reasoning_ingestion": chunk.reasoning_ingestion or {},
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
    db: AsyncSession = Depends(get_db),
    user=Depends(require_role("admin")),
):
    """Re-runs full ingestion for the file. Role: admin only.
    Recovers tenant_id from the existing DB record — never defaults blindly."""
    file_record = await db.scalar(
        select(IngestedFileV2).where(IngestedFileV2.id == file_id)
    )
    if not file_record:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
    tenant_ctx = resolve_tenant_from_db_record(file_record, source="admin_retry")
    await IngestionOrchestrator().ingest_file(
        file_id=file_id, client_id=tenant_ctx.tenant_id,
    )
    return {"status": "retry_started", "file_id": file_id, "tenant_id": tenant_ctx.tenant_id}

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
    gci = None
    if chunk.global_content_id:
        gci = await db.get(GlobalContentIndexV2, chunk.global_content_id)
        if gci:
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
            collection=os.getenv("MAI_COLLECTION", "ingested_content"),
            doc_id=(gci.semantic_hash if gci else chunk.semantic_hash),
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
            "semantic_hash": gci.semantic_hash if gci else chunk.semantic_hash,
            "edited_by":     user["sub"],
            "vectordb_kind": vectordb.kind,
            "embedder_kind": embedder.kind,
            "gci_linked":    bool(gci),
        },
    )
    db.add(audit)
    await db.commit()

    return {
        "status":        "updated",
        "chunk_id":      chunk_id,
        "semantic_hash": gci.semantic_hash if gci else chunk.semantic_hash,
        "gci_linked":    bool(gci),
        "llm_mode":      payload.llm_mode,
        "vectordb_used": vectordb.kind,
        "embedder_used": embedder.kind,
    }
