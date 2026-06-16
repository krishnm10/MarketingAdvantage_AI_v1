# =============================================
# app/api/v2/ingestion_admin_api.py
# =============================================
import asyncio
import logging
import os
from collections import Counter
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete, func, or_, select
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from app.db.models.admin_audit_log import AdminAuditLog
from app.db.models.ingested_file_v2 import IngestedFileV2
from app.db.models.ingested_content_v2 import IngestedContentV2
from app.db.models.global_content_index_v2 import GlobalContentIndexV2

# ✅ PERMANENT FIX: removed get_embedder, get_chroma_collection
# Use _get_pipeline() directly — works with ANY backend (Chroma, Qdrant, Milvus...)
from app.services.ingestion.ingestion_service_v2 import (
    IngestionServiceV2,
    _get_pipeline,
    _get_ingestion_pipeline_async,
    _normalize_business_id,
)
from app.services.ingestion.ingestion_orchestrator import IngestionOrchestrator
from app.services.ingestion.tenant_guard import resolve_tenant_from_db_record
from app.llm.llm_client import run_llm_normalization
from app.db.session_v2 import get_db
from app.auth.guards import require_role
from app.utils.tenant_validator import (
    validate_tenant_id_strict,
    get_storage_uuid,
    TenantValidationError,
    TenantContext,
)
from app.utils.tenant_storage_uuid import storage_business_uuid_for_tenant

router = APIRouter(
    prefix="/api/v2/ingestion-admin",
    tags=["Ingestion Admin"],
)


def _assert_filescoped_to_tenant(file: IngestedFileV2, tenant_ctx: TenantContext) -> None:
    """
    Enforce isolation when requesting file detail/chunks scoped to tenant_id.
    
    Args:
        file: The file record to check
        tenant_ctx: Validated tenant context (REQUIRED - caller must validate first)
    
    Raises:
        HTTPException 404 if file does not belong to the tenant
    """
    scoped_uuid = get_storage_uuid(tenant_ctx)
    if file.business_id == scoped_uuid:
        return
    # Legacy uploads before deterministic UUID wiring (typically default tenant).
    if tenant_ctx.tenant_id == "default" and file.business_id is None:
        return

    raise HTTPException(status_code=404, detail="File not found")


# ===========================================================
# 1️⃣  LIST ALL INGESTED FILES
# ===========================================================
@router.get("/files")
async def list_ingested_files(
    tenant_id: str = Query(
        ...,
        min_length=1,
        max_length=64,
        description="REQUIRED: Tenant identifier to scope file listing (tenant isolation).",
    ),
    db: AsyncSession = Depends(get_db),
    user=Depends(require_role("admin", "editor", "viewer")),
):
    """Returns a list of ingested files for the specified tenant (mandatory tenant isolation)."""
    try:
        tenant_ctx = validate_tenant_id_strict(
            tenant_id,
            source="query",
            endpoint="ingestion_admin_files",
        )
    except TenantValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    scoped_uuid = get_storage_uuid(tenant_ctx)
    stmt = select(IngestedFileV2)
    if tenant_ctx.tenant_id == "default":
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
    tenant_id: str = Query(
        ...,
        min_length=1,
        max_length=64,
        description="REQUIRED: Tenant identifier to scope file access (tenant isolation).",
    ),
    db: AsyncSession = Depends(get_db),
    user=Depends(require_role("admin", "editor", "viewer")),
):
    try:
        tenant_ctx = validate_tenant_id_strict(
            tenant_id,
            source="query",
            endpoint="ingestion_admin_file_detail",
        )
    except TenantValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    file = await db.get(IngestedFileV2, file_id)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    _assert_filescoped_to_tenant(file, tenant_ctx)
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
    tenant_id: str = Query(
        ...,
        min_length=1,
        max_length=64,
        description="REQUIRED: Tenant identifier to scope chunk access (tenant isolation).",
    ),
    db: AsyncSession = Depends(get_db),
    user=Depends(require_role("admin", "editor", "viewer")),
):
    try:
        tenant_ctx = validate_tenant_id_strict(
            tenant_id,
            source="query",
            endpoint="ingestion_admin_chunks",
        )
    except TenantValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    parent = await db.get(IngestedFileV2, file_id)
    if not parent:
        raise HTTPException(status_code=404, detail="File not found")
    _assert_filescoped_to_tenant(parent, tenant_ctx)

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
        file_id=file_id,
        client_id=tenant_ctx.tenant_id,
        kind="http_async",
    )
    return {"status": "retry_started", "file_id": file_id, "tenant_id": tenant_ctx.tenant_id}


# ===========================================================
# 4b️⃣  DELETE SINGLE INGESTED FILE
# ===========================================================
@router.delete("/files/{file_id}")
async def delete_ingested_file(
    file_id: str,
    tenant_id: str = Query(
        ...,
        min_length=1,
        max_length=64,
        description="REQUIRED: Tenant identifier to scope file deletion.",
    ),
    db: AsyncSession = Depends(get_db),
    _user=Depends(require_role("admin")),
):
    """Remove one ingested file, its chunks, vectors, and optional disk copy. Admin only."""
    try:
        tenant_ctx = validate_tenant_id_strict(
            tenant_id,
            source="query",
            endpoint="ingestion_admin_delete_file",
        )
    except TenantValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    file_record = await db.get(IngestedFileV2, file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")
    _assert_filescoped_to_tenant(file_record, tenant_ctx)

    chunk_rows = (
        await db.execute(
            select(IngestedContentV2).where(IngestedContentV2.file_id == file_id)
        )
    ).scalars().all()

    semantic_hashes = list(
        dict.fromkeys(
            str(c.semantic_hash) for c in chunk_rows if c.semantic_hash
        )
    )
    gci_counts = Counter(
        c.global_content_id for c in chunk_rows if c.global_content_id
    )

    deleted_vectors = 0
    if semantic_hashes:
        try:
            pipeline = await _get_ingestion_pipeline_async(tenant_ctx.tenant_id)
            deleted_vectors = await asyncio.to_thread(
                pipeline.vectordb.delete_many,
                collection=pipeline.config.vectordb.collection,
                doc_ids=semantic_hashes,
            )
        except Exception as exc:
            logger.warning(
                "Vector delete failed for file_id=%s: %s", file_id, exc
            )

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
            next_occurrence == 0
            or str(row.first_seen_file_id or "") == str(file_id)
        ):
            await db.delete(row)
            continue

        row.occurrence_count = max(
            remaining_refs, next_occurrence, 1 if remaining_refs > 0 else 0
        )
        row.updated_at = datetime.utcnow()

    if file_record.file_path and os.path.isfile(file_record.file_path):
        try:
            os.remove(file_record.file_path)
        except OSError as exc:
            logger.warning(
                "Failed to remove file %s: %s", file_record.file_path, exc
            )

    await db.delete(file_record)
    await db.commit()

    return {
        "status": "deleted",
        "file_id": file_id,
        "tenant_id": tenant_ctx.tenant_id,
        "deleted_chunks": len(chunk_rows),
        "deleted_vectors": deleted_vectors,
    }


# ===========================================================
# 5️⃣  CHUNK EDIT / LLM NORMALIZATION
# ===========================================================
class ChunkUpdatePayload(BaseModel):
    cleaned_text: str
    tenant_id: str   # REQUIRED: tenant scope for update validation
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
    
    Tenant isolation: Validates that the chunk belongs to the specified tenant.
    Role: admin, editor
    """
    # ── 0. Validate tenant ────────────────────────────────────────
    try:
        tenant_ctx = validate_tenant_id_strict(
            payload.tenant_id,
            source="body",
            endpoint="chunk_update",
        )
    except TenantValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    # ── 1. Fetch chunk ────────────────────────────────────────────
    chunk = await db.get(IngestedContentV2, chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")
    
    # ── 1b. Verify tenant ownership ───────────────────────────────
    scoped_uuid = get_storage_uuid(tenant_ctx)
    chunk_bid = chunk.business_id
    if chunk_bid and str(chunk_bid) != str(scoped_uuid):
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

    from app.core.config.client_config_resolver import get_client_config

    vdb_collection = get_client_config(_normalize_business_id(chunk.business_id)).vectordb.collection

    loop   = asyncio.get_running_loop()
    vector = await loop.run_in_executor(
        None, lambda: embedder.embed_query(final_text)
    )

    await loop.run_in_executor(
        None,
        lambda: vectordb.upsert(
            collection=vdb_collection,
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


BATCH_ERASE_SIZE = 1000


@router.delete("/tenant/{tenant_id}/corpus")
async def erase_tenant_corpus(
    tenant_id: str,
    db: AsyncSession = Depends(get_db),
    _user=Depends(require_role("admin")),
):
    """
    GDPR right-to-erasure for a tenant corpus.

    Deletes vectors via batched ``delete_many`` over PG-sourced chunk IDs —
    never ``delete_collection`` (shared-collection backends would wipe all tenants).
    """
    try:
        tenant_ctx = validate_tenant_id_strict(
            tenant_id, source="path", endpoint="erase_tenant_corpus",
        )
    except TenantValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    storage_uid = get_storage_uuid(tenant_ctx)
    storage_uid_str = str(storage_uid)

    pipeline = _get_pipeline(tenant_ctx.tenant_id)
    collection = pipeline.config.vectordb.collection
    vectordb = pipeline.vectordb

    deleted_chunks = 0
    failed_chunks = 0
    offset = 0

    while True:
        chunk_rows = (
            await db.execute(
                select(IngestedContentV2.id)
                .where(IngestedContentV2.business_id == storage_uid)
                .limit(BATCH_ERASE_SIZE)
                .offset(offset)
            )
        ).scalars().all()
        if not chunk_rows:
            break

        ids = [str(cid) for cid in chunk_rows]
        try:
            deleted_chunks += await asyncio.to_thread(
                vectordb.delete_many,
                collection=collection,
                doc_ids=ids,
            )
        except Exception as exc:
            failed_chunks += len(ids)
            logger.error(
                "Erasure batch failed tenant=%s ids=%s error=%s",
                tenant_ctx.tenant_id,
                ids,
                exc,
            )
        offset += BATCH_ERASE_SIZE

    file_rows = (
        await db.execute(
            select(IngestedFileV2).where(IngestedFileV2.business_id == storage_uid)
        )
    ).scalars().all()
    for row in file_rows:
        if row.file_path and os.path.isfile(row.file_path):
            try:
                os.remove(row.file_path)
            except OSError as exc:
                logger.warning("Failed to remove file %s: %s", row.file_path, exc)

    await db.execute(
        delete(IngestedContentV2).where(IngestedContentV2.business_id == storage_uid)
    )
    await db.execute(
        delete(GlobalContentIndexV2).where(
            GlobalContentIndexV2.business_id == storage_uid
        )
    )
    await db.execute(
        delete(IngestedFileV2).where(IngestedFileV2.business_id == storage_uid)
    )
    await db.commit()

    return {
        "tenant_id": tenant_ctx.tenant_id,
        "deleted_chunks": deleted_chunks,
        "failed_chunks": failed_chunks,
        "deleted_files": len(file_rows),
    }
