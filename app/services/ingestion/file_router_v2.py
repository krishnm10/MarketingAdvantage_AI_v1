# file_router_v2.py — Production-Safe Unified Version with DB-Based Deduplication
# Updated to integrate with enhanced ingestion_service_v2 (GlobalContentIndex-ready)
# (PATCH: filename sanitization + unique saved filename to prevent path traversal/overwrite)
# Gap-2: replaced hardwired if/elif connector routing with ingestor_registry lookup
import os
import uuid
import hashlib
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
    ".json": parse_json,
    ".xml":  parse_xml,
}

# -----------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------
def _safe_uuid(value):
    try:
        return uuid.UUID(str(value))
    except Exception:
        return None


def _write_log(message: str):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {message}\n")


def _validate_file_extension(file_name: str):
    _, ext = os.path.splitext(file_name.lower())
    if ext not in PARSER_MAP:
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
        with open(temp_path, "wb") as tmpf:
            tmpf.write(content)

        file_hash = _compute_file_hash(temp_path)
        _write_log(f"[HASH] {original_file_name} → {file_hash}")

        # ✅ DB deduplication check
        async with async_session() as db:
            existing = await db.scalar(
                select(IngestedFileV2).where(
                    (IngestedFileV2.meta_data["file_hash"].as_string() == file_hash)
                    | (IngestedFileV2.file_name == original_file_name)
                )
            )
            if existing:
                log_warning(f"[file_router_v2] DB duplicate detected: {original_file_name}")
                _write_log(f"[SKIPPED_DB_DUPLICATE] {original_file_name}")
                os.remove(temp_path)
                return {"status": "skipped", "reason": "db_duplicate", "file_name": original_file_name}

        os.rename(temp_path, saved_path)
        _write_log(f"[SAVED] {original_file_name} ({file_hash}) → {saved_path}")

        parsed_output = await parser_func(saved_path)
        _write_log(f"[PARSED] {original_file_name} using {parser_func.__name__}")

        file_id = str(uuid.uuid4())

        async with async_session() as db:
            await db.execute(
                insert(IngestedFileV2).values(
                    id=file_id,
                    business_id=_safe_uuid(business_id),
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

        await IngestionServiceV2.ingest_parsed_output(file_id, parsed_output)
        _write_log(f"[INGESTED] {original_file_name} successfully processed.")
        log_info(f"[file_router_v2] ✅ Ingestion complete for {original_file_name}")

        return {"file_id": file_id, "status": "ingested", "path": saved_path, "hash": file_hash}

    except Exception as e:
        log_warning(f"[file_router_v2] Ingestion failed: {e}")
        _write_log(f"[FAILED] {getattr(file, 'filename', 'unknown')}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------
# EXTERNAL INGESTION ROUTES (RSS / API / WEB / + future sources)
# -----------------------------------------------------------
async def route_external_ingestion(source_type: str, source_url: str, business_id: str = None):
    try:
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

            # ✅ Create DB entry for source
            safe_business_id = _safe_uuid(business_id)
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
            await IngestionServiceV2.ingest_parsed_output(source_id, parsed_output)
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
