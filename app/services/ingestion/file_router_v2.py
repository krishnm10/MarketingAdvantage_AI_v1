# file_router_v2.py — Production-Safe Unified Version with DB-Based Deduplication
# Updated to integrate with enhanced ingestion_service_v2 (GlobalContentIndex-ready)
# (PATCH: filename sanitization + unique saved filename to prevent path traversal/overwrite)
# Gap-2: replaced hardwired if/elif connector routing with ingestor_registry lookup
import os
import uuid

from app.utils.tenant_storage_uuid import storage_business_uuid_for_tenant
import hashlib
import asyncio
import aiofiles
from datetime import datetime
from fastapi import UploadFile, HTTPException
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy import insert, select
from pathlib import Path

from app.services.ingestion.pdf_parser_v2 import parse_pdf
from app.services.ingestion.docx_parser_v2 import parse_docx
from app.services.ingestion.excel_parser_v2 import parse_excel
from app.services.ingestion.csv_parser_v2 import parse_csv
from app.services.ingestion.text_parser_v2 import parse_text
from app.services.ingestion.json_parser_v2 import parse_json
from app.services.ingestion.xml_parser_v2 import parse_xml


from app.services.ingestion.ingestion_service_v2 import IngestionServiceV2
from app.services.ingestion.ingestion_orchestrator import IngestionOrchestrator
from app.services.ingestion.media.media_ingestion_hook_v1 import MediaIngestionHookV1
from app.services.ingestion.tenant_guard import (
    resolve_ingestion_tenant,
    IngestionTenantViolation,
    log_ingestion_telemetry,
)
from app.db.models.ingested_file_v2 import IngestedFileV2
from app.db.session_v2 import async_engine
from app.utils.logger import log_info, log_warning

# ✅ Gap-2: registry imports
from app.core.plugin_registry import ingestor_registry, PluginNotFoundError

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
UPLOAD_DIR = os.path.join("static", "uploads", "api")
LOG_PATH = os.path.join("logs", "ingestion.log")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

async_session = async_sessionmaker(async_engine, expire_on_commit=False, autoflush=False)

# -----------------------------------------------------------
# ROUTER MAP (EXTENSION → PARSER)
# -----------------------------------------------------------
PARSER_MAP = {
    ".pdf":  parse_pdf,
    ".docx": parse_docx,
    ".xlsx": parse_excel,
    ".xls":  parse_excel,
    ".csv":  parse_csv,
    ".txt":  parse_text,
    ".md":   parse_text,
    ".markdown": parse_text,
    ".json": parse_json,
    ".xml":  parse_xml,
}

# -----------------------------------------------------------
# MEDIA EXTENSION → KIND MAP (images, audio, video)
# -----------------------------------------------------------
MEDIA_EXT_MAP = {
    # Images
    ".jpg": "image", ".jpeg": "image", ".png": "image", ".gif": "image",
    ".bmp": "image", ".webp": "image", ".tiff": "image", ".svg": "image",
    # Audio
    ".mp3": "audio", ".wav": "audio", ".flac": "audio", ".ogg": "audio",
    ".aac": "audio", ".m4a": "audio", ".wma": "audio", ".opus": "audio",
    # Video
    ".mp4": "video", ".avi": "video", ".mov": "video", ".mkv": "video",
    ".webm": "video", ".flv": "video", ".wmv": "video", ".m4v": "video",
}

MEDIA_UPLOAD_DIR = os.path.join("static", "uploads", "media")
os.makedirs(MEDIA_UPLOAD_DIR, exist_ok=True)

# -----------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------
def _write_log(message: str):
    """Append a log line. Offloaded to thread to avoid blocking the event loop."""
    line = f"{datetime.now().isoformat()} | {message}\n"
    try:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, _write_log_sync, line)
    except RuntimeError:
        _write_log_sync(line)


def _write_log_sync(line: str):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)


def _validate_file_extension(file_name: str):
    _, ext = os.path.splitext(file_name.lower())
    if ext not in PARSER_MAP and ext not in MEDIA_EXT_MAP:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    return ext


def _compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of file for deduplication."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


# -----------------------------------------------------------
# MAIN FILE INGESTION ROUTER (Unified + DB Dedup-Safe)
# -----------------------------------------------------------
async def route_file_ingestion(file: UploadFile, business_id: str = None):
    try:
        original_file_name = file.filename
        file_ext = _validate_file_extension(original_file_name)

        # ── Tenant enforcement: validate and lock before any processing ──
        tenant_ctx = resolve_ingestion_tenant(
            business_id=business_id,
            file_id="",
            source="file_upload",
            allow_default=True,
        )

        # ── Media files (image/audio/video) → route through MediaIngestionHookV1 ──
        if file_ext in MEDIA_EXT_MAP:
            storage_uid_media = storage_business_uuid_for_tenant(
                tenant_ctx.tenant_id, business_id,
            )
            return await _route_media_ingestion(file, file_ext, storage_uid_media)

        parser_func = PARSER_MAP[file_ext]

        # -----------------------
        # SANITIZE + UNIQUE SAVE
        # -----------------------
        safe_name = Path(original_file_name).name
        safe_stem = Path(safe_name).stem
        safe_suffix = Path(safe_name).suffix or file_ext
        unique_suffix = uuid.uuid4().hex
        saved_file_name = f"{safe_stem}_{unique_suffix}{safe_suffix}"
        saved_path = os.path.join(UPLOAD_DIR, saved_file_name)

        content = await file.read()
        temp_path = f"{saved_path}.tmp"
        async with aiofiles.open(temp_path, "wb") as tmpf:
            await tmpf.write(content)

        file_hash = _compute_file_hash(temp_path)
        _write_log(f"[HASH] {original_file_name} → {file_hash}")

        # ✅ DB deduplication check (hash-only, tenant-scoped by stable UUID)
        storage_uid = storage_business_uuid_for_tenant(tenant_ctx.tenant_id, business_id)
        async with async_session() as db:
            dedup_query = select(IngestedFileV2).where(
                IngestedFileV2.meta_data["file_hash"].as_string() == file_hash
            )
            dedup_query = dedup_query.where(IngestedFileV2.business_id == storage_uid)

            existing = await db.scalar(dedup_query)
            if existing:
                log_warning(f"[file_router_v2] DB duplicate detected: {original_file_name}")
                _write_log(f"[SKIPPED_DB_DUPLICATE] {original_file_name}")
                os.remove(temp_path)
                return {"status": "skipped", "reason": "db_duplicate", "file_name": original_file_name}

        os.rename(temp_path, saved_path)
        _write_log(f"[SAVED] {original_file_name} ({file_hash}) → {saved_path}")

        # Create the DB record FIRST so the Celery task can update it by file_id.
        file_id = str(uuid.uuid4())

        async with async_session() as db:
            await db.execute(
                insert(IngestedFileV2).values(
                    id=file_id,
                    business_id=storage_uid,
                    file_name=original_file_name,
                    file_type=file_ext.replace(".", ""),
                    file_path=saved_path,
                    source_type=file_ext.replace(".", ""),
                    meta_data={"file_hash": file_hash},
                    parser_used=parser_func.__name__,
                    status="uploaded",
                    total_chunks=0,
                    unique_chunks=0,
                    duplicate_chunks=0,
                    dedup_ratio=0.0,
                    error_message=None,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
            )
            await db.commit()

        # ── Celery: offload parse + embed to a background worker ──────────────
        # Only attempted when CELERY_ENABLED=true in .env.
        # Falls back transparently to inline (blocking) processing when Celery /
        # broker is not running, so the server continues to work without a broker.
        try:
            from app.worker.broker_config import is_celery_enabled
            if is_celery_enabled():
                from app.core.config.client_config_resolver import (
                    get_celery_ingestion_enqueue_kwargs,
                )
                from app.worker.tasks import run_ingestion_pipeline
                task = run_ingestion_pipeline.apply_async(
                    args=[file_id, saved_path, file_ext, tenant_ctx.tenant_id],
                    **get_celery_ingestion_enqueue_kwargs(tenant_ctx.tenant_id),
                )
                _write_log(f"[QUEUED] {original_file_name} → task_id={task.id}")
                log_info(f"[file_router_v2] Queued Celery task task_id={task.id} for {original_file_name}")
                return {
                    "file_id": file_id,
                    "status": "queued",
                    "task_id": task.id,
                    "path": saved_path,
                    "hash": file_hash,
                }
        except Exception:
            pass  # Celery / broker unavailable — fall through to inline

        # ── Inline fallback (no Celery / broker not running) ──────────────────
        parsed_output = await parser_func(saved_path)
        _write_log(f"[PARSED] {original_file_name} using {parser_func.__name__}")
        await IngestionOrchestrator().ingest_parsed_output(
            file_id=file_id, parsed=parsed_output,
            client_id=tenant_ctx.tenant_id,
        )
        _write_log(f"[INGESTED] {original_file_name} successfully processed.")
        log_info(f"[file_router_v2] ✅ Ingestion complete (inline) for {original_file_name}")
        log_ingestion_telemetry(
            tenant_ctx=tenant_ctx,
            event="INGESTION_FILE_COMPLETE",
            extra={"final_file_id": file_id},
        )

        return {"file_id": file_id, "status": "ingested", "path": saved_path, "hash": file_hash}

    except HTTPException:
        raise
    except Exception as e:
        log_warning(f"[file_router_v2] Ingestion failed: {e}")
        _write_log(f"[FAILED] {getattr(file, 'filename', 'unknown')}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------
# MEDIA FILE ROUTING (image / audio / video → MediaIngestionHookV1)
# -----------------------------------------------------------
async def _route_media_ingestion(
    file: UploadFile,
    file_ext: str,
    storage_business_id: uuid.UUID,
):
    """Route image/audio/video uploads through the media ingestion pipeline."""
    original_file_name = file.filename
    media_kind = MEDIA_EXT_MAP[file_ext]

    safe_name = Path(original_file_name).name
    safe_stem = Path(safe_name).stem
    safe_suffix = Path(safe_name).suffix or file_ext
    unique_suffix = uuid.uuid4().hex
    saved_file_name = f"{safe_stem}_{unique_suffix}{safe_suffix}"
    saved_path = os.path.join(MEDIA_UPLOAD_DIR, saved_file_name)

    # Resolve to absolute and verify stays inside upload dir
    abs_upload_dir = os.path.realpath(MEDIA_UPLOAD_DIR)
    abs_file_path = os.path.realpath(saved_path)
    if not abs_file_path.startswith(abs_upload_dir + os.sep):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    content = await file.read()
    async with aiofiles.open(abs_file_path, "wb") as f:
        await f.write(content)

    file_id = str(uuid.uuid4())
    # Persistable UUID scoped to validated tenant — matches document upload rows.
    business_id_str = str(storage_business_id)

    log_info(f"[file_router_v2] Routing {media_kind} file: {original_file_name}")
    _write_log(f"[MEDIA_ROUTE] {original_file_name} → {media_kind}")

    result = await MediaIngestionHookV1().handle(
        file_id=file_id,
        file_path=abs_file_path,
        file_type=media_kind,
        parsed_output={},
        business_id=business_id_str,
        media_kind=media_kind,
    )

    status = result.get("status", "success")
    _write_log(f"[MEDIA_INGESTED] {original_file_name} → {status}")
    log_info(f"[file_router_v2] ✅ Media ingestion complete for {original_file_name} ({status})")

    return {
        "file_id": file_id,
        "file_name": original_file_name,
        "media_kind": media_kind,
        "status": status,
        "path": abs_file_path,
        **{k: v for k, v in result.items() if k != "status"},
    }


# -----------------------------------------------------------
# EXTERNAL INGESTION ROUTES (RSS / API / WEB / + future sources)
# -----------------------------------------------------------
async def route_external_ingestion(source_type: str, source_url: str, business_id: str = None):
    try:
        # ── Tenant enforcement: validate and lock before any processing ──
        tenant_ctx = resolve_ingestion_tenant(
            business_id=business_id,
            file_id="",
            source=f"external/{source_type}",
            allow_default=True,
        )

        log_info(f"[file_router_v2] Routing {source_type.upper()} source: {source_url}")
        _write_log(f"[ROUTING] {source_type.upper()} → {source_url}")

        async with async_session() as db:

            # ✅ Gap-2: registry lookup replaces hardwired if/elif
            # Adding a new source type = register it in plugin_registry.py only.
            # This function never needs to change again.
            # ✅ REPLACE with this block:
            try:
                # ── Layer-4 Security: sanitize business_id before using in file path ──
                # Prevents path traversal attacks e.g. business_id = "../../etc/passwd"
                auth = None
                if business_id:
                    import re
                    safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "", str(business_id))
                    if not safe_id:
                        raise HTTPException(
                            status_code=400,
                            detail="Invalid business_id: contains no valid characters."
                        )
                
                    cfg_path = os.path.join("app", "core", "configs", f"client_{safe_id}.json")
                
                    # ── Extra guard: confirm resolved path stays inside configs dir ──
                    configs_dir   = os.path.realpath(os.path.join("app", "core", "configs"))
                    resolved_path = os.path.realpath(cfg_path)
                    if not resolved_path.startswith(configs_dir + os.sep):
                        raise HTTPException(
                            status_code=400,
                            detail="Invalid business_id: path escapes config directory."
                        )
                
                    if os.path.exists(resolved_path):
                        from app.core.config.client_config_schema import ClientConfig
                        from app.core.connectors.auth.resolver import resolve_auth
                        client_cfg = ClientConfig.from_json_file(resolved_path)
                        if client_cfg.connector:
                            auth = resolve_auth(client_cfg.connector.auth)
                
                connector = ingestor_registry.build(source_type, auth=auth)

            except PluginNotFoundError:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unsupported source type: '{source_type}'. "
                        f"Available: {list(ingestor_registry.list().keys())}"
                    ),
                )


            result = await connector.fetch(source_url, db_session=db)
            parsed_output = result.to_dict()

            # ✅ Create DB entry for source — use stable UUID tied to validated tenant slug
            safe_business_id = storage_business_uuid_for_tenant(
                tenant_ctx.tenant_id,
                business_id,
            )
            source_id = str(uuid.uuid4())

            await db.execute(
                insert(IngestedFileV2).values(
                    id=source_id,
                    business_id=safe_business_id,
                    file_name=os.path.basename(source_url) or f"{source_type}_source",
                    file_type=source_type,
                    file_path=source_url,
                    source_type=source_type,
                    meta_data={"source_url": source_url},
                    parser_used=f"{source_type}_ingestor_v2",
                    status="uploaded",
                    total_chunks=0,
                    unique_chunks=0,
                    duplicate_chunks=0,
                    dedup_ratio=0.0,
                    error_message=None,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
            )
            await db.commit()

        # ✅ Direct ingestion for pre-parsed payload
        if parsed_output and isinstance(parsed_output, dict):
            log_info(f"[file_router_v2] Passing parsed {source_type.upper()} output directly to ingestion pipeline...")
            await IngestionOrchestrator().ingest_parsed_output(
                file_id=source_id, parsed=parsed_output,
                client_id=tenant_ctx.tenant_id,
            )
        else:
            log_warning(f"[file_router_v2] No valid chunks found in parsed {source_type.upper()} output. Skipping ingestion.")
            return {"status": "skipped", "reason": "no_chunks"}

        _write_log(f"[INGESTED] {source_type.upper()} source processed: {source_url}")
        log_info(f"[file_router_v2] ✅ {source_type.upper()} ingestion complete for {source_url}")

        return {"source_type": source_type, "source_url": source_url, "status": "processed"}

    except Exception as e:
        log_warning(f"[file_router_v2] External ingestion failed: {e}")
        _write_log(f"[FAILED] {source_type.upper()} {source_url}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
