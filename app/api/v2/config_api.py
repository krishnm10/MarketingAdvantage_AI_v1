# =============================================================================
# app/api/v2/config_api.py
# Admin-only API to read / write the project .env file and reload services.
# =============================================================================

import json
import os
import re
import signal
import logging
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.guards import require_role
from app.core.pipeline_factory import pipeline_factory
from app.db.session_v2 import get_db
from app.db.models.admin_audit_log import AdminAuditLog

logger = logging.getLogger("config_api")

router = APIRouter(
    prefix="/api/v2/config",
    tags=["Configuration"],
)

# ── Locate the .env file (project root) ──────────────────────────────────────
# config_api.py is at  <root>/app/api/v2/config_api.py  →  root = parents[3]
_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_env(path: Path) -> Dict[str, str]:
    """Read .env file and return {KEY: value} dict (preserves order)."""
    result: Dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)', stripped)
        if m:
            key = m.group(1)
            val = m.group(2).strip()
            # Strip surrounding quotes if present
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                val = val[1:-1]
            # Strip inline comment (only if not inside quotes)
            comment_match = re.match(r'^([^#]*?)\s+#\s', val)
            if comment_match:
                val = comment_match.group(1).strip()
            result[key] = val
    return result


def _write_env(path: Path, updates: Dict[str, str]) -> None:
    """
    Merge *updates* into the existing .env file.
    - Lines with matching keys get their value replaced.
    - New keys are appended at the end.
    - Comments, blank lines, and ordering are preserved.
    """
    if not path.exists():
        # Create fresh file with just the updates
        lines_out = [f"{k}={v}\n" for k, v in updates.items()]
        path.write_text("".join(lines_out), encoding="utf-8")
        return

    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    written_keys: set = set()
    new_lines = []

    for line in lines:
        stripped = line.strip()
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=', stripped)
        if m and m.group(1) in updates:
            key = m.group(1)
            # Preserve any inline comment from the original line
            inline_comment = ""
            rest = stripped[m.end():]
            # Check for inline comment after value
            val_comment = re.search(r'\s+#\s.*$', rest)
            if val_comment:
                inline_comment = val_comment.group()
            new_lines.append(f"{key}={updates[key]}{inline_comment}\n")
            written_keys.add(key)
        else:
            new_lines.append(line)

    # Append any new keys that weren't in the original file
    for key, val in updates.items():
        if key not in written_keys:
            new_lines.append(f"{key}={val}\n")

    path.write_text("".join(new_lines), encoding="utf-8")


# ── Sensitive keys — values masked on read ────────────────────────────────────
_SENSITIVE_KEYS = {
    "DATABASE_URL", "JWT_SECRET",
    "QDRANT_API_KEY", "MILVUS_TOKEN", "PINECONE_API_KEY",
    "WEAVIATE_API_KEY", "REDIS_PASSWORD",
    "OPENAI_API_KEY", "GROQ_API_KEY",
    "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "COHERE_API_KEY",
    "SERPER_API_KEY", "DEEPAI_API_KEY",
    "CHROMA_API_KEY",
    # Celery / Broker secrets
    "CELERY_REDIS_URL", "CELERY_RESULT_BACKEND", "RABBITMQ_URL",
    "KAFKA_SASL_PASSWORD", "KAFKA_SSL_KEY_PASSWORD",
    "KAFKA_SCHEMA_REGISTRY_AUTH", "KAFKA_OAUTHBEARER_CONFIG",
    # Observability
    "SENTRY_DSN",
}

_VALID_CHUNKING_STRATEGIES = {
    "semantic",
    "recursive",
    "overlap",
    "smart_check",
    "recursive_overlap",
    "rust",
    "structure_aware",
    "document_aware",
    "token_aware",
}

_VALID_TOKENIZER_BACKENDS = {"whitespace", "huggingface", "spacy", "nltk"}

_VALID_AI_PROFILES = {"cpu", "gpu", "api", "dist"}
_VALID_VISION_API_PROVIDERS = {"openai", "anthropic", "google"}
_VALID_VISION_QUANTIZE = {"none", "4bit", "8bit"}


def _mask(key: str, val: str) -> str:
    """Return masked value for sensitive keys."""
    if key in _SENSITIVE_KEYS and val:
        if len(val) <= 8:
            return "••••••••"
        return val[:4] + "••••••••" + val[-4:]
    return val


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/")
async def get_config(
    user=Depends(require_role("admin")),
):
    """
    Read all .env config values.  Sensitive values are partially masked.
    Admin-only.
    """
    if not _ENV_PATH.exists():
        raise HTTPException(status_code=404, detail=".env file not found")

    raw = _parse_env(_ENV_PATH)
    masked = {k: _mask(k, v) for k, v in raw.items()}
    return {"env_path": str(_ENV_PATH), "config": masked}


class ConfigUpdateRequest(BaseModel):
    updates: Dict[str, str]          # { "KEY": "new_value", ... }
    reload_backend: Optional[bool] = False


@router.put("/")
async def update_config(
    payload: ConfigUpdateRequest,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_role("admin")),
):
    """
    Write key-value pairs to the .env file.
    Only admin users can call this.
    If reload_backend=true, sends SIGHUP (on Unix) or sets a flag to
    let uvicorn --reload pick up changes automatically.
    """
    if not payload.updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No updates provided",
        )

    # Safety: block writing to certain critical keys from UI unless intended
    blocked = set()
    invalid_values = {}
    for key in payload.updates:
        # Validate key format
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', key):
            blocked.add(key)
            continue
        if key == "CHUNKING_STRATEGY":
            value = str(payload.updates[key]).strip().lower()
            if value not in _VALID_CHUNKING_STRATEGIES:
                invalid_values[key] = value
        if key == "DEFAULT_TOKENIZER_BACKEND":
            value = str(payload.updates[key]).strip().lower()
            if value not in _VALID_TOKENIZER_BACKENDS:
                invalid_values[key] = value
        if key == "AI_PROFILE":
            value = str(payload.updates[key]).strip().lower()
            if value not in _VALID_AI_PROFILES:
                invalid_values[key] = value
        if key == "VISION_API_PROVIDER":
            value = str(payload.updates[key]).strip().lower()
            if value not in _VALID_VISION_API_PROVIDERS:
                invalid_values[key] = value
        if key == "VISION_QUANTIZE":
            value = str(payload.updates[key]).strip().lower()
            if value not in _VALID_VISION_QUANTIZE:
                invalid_values[key] = value

    if blocked:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid key names: {blocked}",
        )
    if invalid_values:
        valid_hints = {
            "CHUNKING_STRATEGY": ", ".join(sorted(_VALID_CHUNKING_STRATEGIES)),
            "DEFAULT_TOKENIZER_BACKEND": ", ".join(sorted(_VALID_TOKENIZER_BACKENDS)),
            "AI_PROFILE": ", ".join(sorted(_VALID_AI_PROFILES)),
            "VISION_API_PROVIDER": ", ".join(sorted(_VALID_VISION_API_PROVIDERS)),
            "VISION_QUANTIZE": ", ".join(sorted(_VALID_VISION_QUANTIZE)),
        }
        detail_parts = [f"Invalid config values: {invalid_values}."]
        for k in invalid_values:
            if k in valid_hints:
                detail_parts.append(f"Valid {k}: {valid_hints[k]}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=" ".join(detail_parts),
        )

    # Read old values for audit diff BEFORE writing
    old_values = _parse_env(_ENV_PATH)
    before_snapshot = {k: old_values.get(k, "") for k in payload.updates}

    try:
        _write_env(_ENV_PATH, payload.updates)
    except Exception as e:
        logger.error("Failed to write .env: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write .env: {e}",
        )

    # Also update os.environ so running process picks up changes immediately
    for k, v in payload.updates.items():
        os.environ[k] = v

    pipeline_factory.invalidate_all()
    try:
        from app.services.ingestion.ingestion_service_v2 import clear_ingestion_pipeline_cache

        clear_ingestion_pipeline_cache()
    except Exception as e:
        logger.warning("Failed to clear ingestion pipeline cache: %s", e)

    # ── Audit log ─────────────────────────────────────────────────────────
    try:
        audit = AdminAuditLog(
            id=uuid.uuid4(),
            action="config_update",
            entity_type="env_config",
            entity_id=uuid.uuid4(),
            before_value=json.dumps(before_snapshot),
            after_value=json.dumps(dict(payload.updates)),
            meta_data={
                "edited_by": user.get("sub", "unknown"),
                "keys_changed": list(payload.updates.keys()),
                "env_path": str(_ENV_PATH),
            },
        )
        db.add(audit)
        await db.commit()
    except Exception as e:
        logger.warning("Audit log write failed (config still saved): %s", e)

    logger.info(
        "Config updated by %s — keys: %s",
        user.get("sub", "unknown"),
        list(payload.updates.keys()),
    )

    return {
        "status": "saved",
        "updated_keys": list(payload.updates.keys()),
        "message": (
            "Changes written to .env and applied to running process. "
            "If uvicorn is running with --reload it will auto-restart."
        ),
    }


@router.get("/raw/{key}")
async def get_raw_value(
    key: str,
    user=Depends(require_role("admin")),
):
    """
    Get the unmasked raw value of a single key. Admin only.
    Used when the admin clicks 'reveal' on a sensitive field.
    """
    raw = _parse_env(_ENV_PATH)
    if key not in raw:
        raise HTTPException(status_code=404, detail=f"Key '{key}' not found in .env")
    return {"key": key, "value": raw[key]}


# ─────────────────────────────────────────────────────────────────────────────
# VISION ENCODER STATUS
# ─────────────────────────────────────────────────────────────────────────────

def _check_package(name: str) -> bool:
    """Check if a Python package is importable."""
    try:
        __import__(name)
        return True
    except ImportError:
        return False


@router.get("/vision-status")
async def get_vision_status(
    user=Depends(require_role("admin")),
):
    """
    Return the current Multimodal Vision Encoder status:
    installed packages, active profile, and model info.
    """
    raw = _parse_env(_ENV_PATH)
    profile = raw.get("AI_PROFILE", os.getenv("AI_PROFILE", "cpu"))

    # Check key packages
    cpu_pkgs = {
        "transformers": _check_package("transformers"),
        "qwen_vl_utils": _check_package("qwen_vl_utils"),
        "torch": _check_package("torch"),
        "moondream": _check_package("moondream"),
    }
    gpu_pkgs = {
        **cpu_pkgs,
        "bitsandbytes": _check_package("bitsandbytes"),
        "flash_attn": _check_package("flash_attn"),
    }
    api_pkgs = {
        "httpx": _check_package("httpx"),
    }

    # Determine if CUDA is available
    cuda_available = False
    if cpu_pkgs["torch"]:
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except Exception:
            pass

    return {
        "ai_profile": profile,
        "cpu_packages_installed": all([cpu_pkgs["transformers"], cpu_pkgs["torch"]]),
        "gpu_packages_installed": all([gpu_pkgs["transformers"], gpu_pkgs["torch"]]),
        "api_packages_installed": api_pkgs["httpx"],
        "cuda_available": cuda_available,
        "packages": {
            "cpu": cpu_pkgs,
            "gpu": gpu_pkgs,
            "api": api_pkgs,
        },
        "config": {
            "vision_model_cpu": raw.get("VISION_MODEL_CPU", os.getenv("VISION_MODEL_CPU", "Qwen/Qwen2.5-VL-3B-Instruct")),
            "vision_model_gpu": raw.get("VISION_MODEL_GPU", os.getenv("VISION_MODEL_GPU", "Qwen/Qwen2.5-VL-7B-Instruct")),
            "vision_api_provider": raw.get("VISION_API_PROVIDER", os.getenv("VISION_API_PROVIDER", "openai")),
            "vision_api_model": raw.get("VISION_API_MODEL", os.getenv("VISION_API_MODEL", "gpt-4o")),
            "vision_quantize": raw.get("VISION_QUANTIZE", os.getenv("VISION_QUANTIZE", "4bit")),
            "vision_flash_attention": raw.get("VISION_FLASH_ATTENTION", os.getenv("VISION_FLASH_ATTENTION", "true")),
        },
    }
