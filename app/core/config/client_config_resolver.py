"""
================================================================================
Client Config Resolver — Authoritative Runtime Config Source of Truth.

STATUS: PRODUCTION. This module is the SINGLE entry point for loading,
        merging, validating, and returning ClientConfig instances for all
        pipelines (RAG, ingestion, retrieval).

RESPONSIBILITIES:
    1. Load the default config (base layer — always present).
    2. Load a client-specific override config (if one exists).
    3. Deep-merge the client overrides onto the default base.
    4. Optionally apply legacy environment-variable overlays (currently none for pipeline).
    5. Parse the merged dict through Pydantic (schema validation).
    6. Run compatibility validation (embedder ↔ vectordb, API keys, etc.).
    7. Emit structured log with config fingerprint.
    8. Return the FINAL, validated, merged ClientConfig.

USAGE:
    from app.core.config.client_config_resolver import get_client_config

    config = get_client_config("acme_corp")

    # Config is ready for pipeline construction:
    pipeline = pipeline_factory.build(config)

ERRORS:
    - FileNotFoundError: default config missing.
    - ConfigValidationError: incompatible or invalid configuration.
================================================================================
"""

from __future__ import annotations

import hashlib
import json as _json_module
import logging
import os
import uuid as _uuid_module
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config.client_config_schema import (
    ClientConfig,
    EmbedderType,
    RerankerType,
    VectorDBType,
)
from app.utils.path_sanitizer import sanitize_client_id
from app.utils.tenant_storage_uuid import storage_uuid_str_for_vectordb_metadata
from app.core.runtime.errors import ConfigResolutionError
from app.core.config.pipeline_runtime import warn_if_deprecated_pipeline_env_set

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class ConfigValidationError(ConfigResolutionError):
    """Raised when a resolved ClientConfig fails compatibility checks."""

    def __init__(self, client_id: str, issues: List["ConfigIssue"]) -> None:
        self.issues = issues
        details = "; ".join(
            f"[{i.severity.value}][{i.component}] {i.message}" for i in issues
        )
        super().__init__(
            f"Config validation failed for client_id='{client_id}': {details}",
            tenant_id=client_id,
            details={"issues": [
                {"severity": i.severity.value, "component": i.component, "message": i.message}
                for i in issues
            ]},
        )
        self.client_id = client_id


# ─────────────────────────────────────────────────────────────────────────────
# Config file discovery
# ─────────────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIRS: List[Path] = [
    _REPO_ROOT / "app" / "core" / "configs",
    _REPO_ROOT / "configs",
]
_DEFAULT_CONFIG_ID = "default"


def _find_config_path(client_id: str) -> Optional[Path]:
    """
    Search known config directories for a client config file.
    Returns the first match or None. Blocks path traversal.
    """
    safe_id = sanitize_client_id(client_id)
    for base in _CONFIG_DIRS:
        for ext in ("json", "yaml", "yml"):
            candidate = (base / f"{safe_id}.{ext}").resolve()
            if not str(candidate).startswith(str(base.resolve())):
                logger.warning(
                    "[ConfigResolver] Path traversal blocked for client_id=%.30s",
                    client_id[:30],
                )
                continue
            if candidate.exists():
                return candidate
    return None


def _resolve_client_id_for_config_lookup(client_id: str) -> str:
    """
    Map a canonical tenant storage UUID to its slug when a matching client JSON exists.

    Upload/API callers sometimes send ``business_id`` / ``client_id`` as the derived
    storage UUID (uuid5 namespace). That string does not match ``matha.json`` etc.,
    so config falls back to default-only — wrong embedder and Chroma path.
    """
    if _find_config_path(client_id):
        return client_id
    try:
        target_uuid = _uuid_module.UUID(str(client_id).strip())
    except (ValueError, AttributeError, TypeError):
        return client_id
    target = str(target_uuid)
    for base in _CONFIG_DIRS:
        if not base.is_dir():
            continue
        for p in sorted(base.glob("*.json")):
            if p.name == "default.json":
                continue
            stem = p.stem
            if stem.startswith(".") or stem.endswith(".template"):
                continue
            try:
                if storage_uuid_str_for_vectordb_metadata(stem) == target:
                    logger.info(
                        "[ConfigResolver] client_id storage UUID matched tenant slug '%s' "
                        "for config overlay",
                        stem,
                    )
                    return stem
            except Exception:
                continue
    return client_id


# ─────────────────────────────────────────────────────────────────────────────
# Environment overrides
# ─────────────────────────────────────────────────────────────────────────────

# Pipeline semantics (vectordb, embedder, collection, retrieval) are JSON-only —
# never merge MAI_EMBEDDER / MAI_VECTORDB / etc. into ClientConfig dicts here.
_ENV_OVERRIDES: Dict[str, str] = {}


def _apply_env_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay sparse env-derived fields onto merged JSON (currently none reserved)."""
    for env_var, dotted_path in _ENV_OVERRIDES.items():
        value = os.getenv(env_var)
        if value is None:
            continue
        keys = dotted_path.split(".")
        target = data
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value
        logger.debug(
            "[ConfigResolver] Env override applied: %s → %s",
            env_var, dotted_path,
        )

    return data


def load_default_client_raw_dict() -> Dict[str, Any]:
    """
    Loads app/core/configs/default.json (or YAML) as a plain dict — no env overlay.
    Used when creating tenant config files seeded from canonical defaults.
    """
    path = _find_config_path(_DEFAULT_CONFIG_ID)
    if path is None:
        raise FileNotFoundError(
            f"Default config not found. Searched: {[str(d) for d in _CONFIG_DIRS]}"
        )
    return deepcopy(_load_raw(path))


# ─────────────────────────────────────────────────────────────────────────────
# Deep merge
# ─────────────────────────────────────────────────────────────────────────────

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override dict onto base dict.

    Rules:
        - Keys only in base → kept as-is.
        - Keys only in override → added.
        - Keys in both, both dicts → recurse.
        - Keys in both, non-dict → override wins.
    """
    merged = deepcopy(base)
    for key, override_val in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(override_val, dict)
        ):
            merged[key] = _deep_merge(merged[key], override_val)
        else:
            merged[key] = deepcopy(override_val)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Config fingerprint — PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def _compute_fingerprint(data: Dict[str, Any]) -> str:
    """SHA-256 fingerprint of the canonical JSON representation (raw dict)."""
    canonical = _json_module.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def get_config_fingerprint(config: ClientConfig) -> str:
    """
    Deterministic fingerprint of a fully resolved ClientConfig.

    Produces a stable 16-char hex SHA-256 digest from the canonical
    JSON dump of the entire config.  The fingerprint changes when
    ANY field changes (embedder, vectordb, reranker, security flags,
    retrieval thresholds, etc.).

    Call this AFTER merge + env overrides + validation — on the final
    config object that will drive the pipeline.

    Usage:
        config = get_client_config("acme")
        fp = get_config_fingerprint(config)
        logger.info("config_fingerprint=%s", fp)
    """
    try:
        raw = config.model_dump(mode="json")
        return _compute_fingerprint(raw)
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_raw(path: Path) -> Dict[str, Any]:
    """Load a config file as a raw dict. Supports JSON and YAML."""
    import json

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required for YAML configs: pip install pyyaml")
        return yaml.safe_load(text) or {}

    return json.loads(text)


def _load_config_from_path(
    client_id: str, config_path: Path, *, apply_env: bool = True,
) -> ClientConfig:
    """Parse a config file into a validated ClientConfig."""
    raw = _load_raw(config_path)
    if raw.get("client_id") != client_id and client_id != _DEFAULT_CONFIG_ID:
        raw["client_id"] = client_id
    if apply_env:
        raw = _apply_env_overrides(raw)
    return ClientConfig.from_dict(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Structured logging helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_embedder_model(config: ClientConfig) -> Optional[str]:
    """Extract the active embedder model name for logging."""
    sub = getattr(config.embedder, config.embedder.type.value, None)
    return getattr(sub, "model", None) if sub else None


def _resolve_llm_model(config: ClientConfig) -> Optional[str]:
    """Extract the active LLM model name for logging."""
    if config.llm and config.llm.single:
        return config.llm.single.model
    if config.llm and config.llm.chain:
        return config.llm.chain[0].model if config.llm.chain else None
    return None


def _log_resolved_config(
    config: ClientConfig,
    *,
    config_source: str,
    fingerprint: str,
) -> None:
    """Emit a single structured JSON log for the resolved config."""
    try:
        logger.info(
            "%s",
            _json_module.dumps({
                "event": "CONFIG_RESOLVED",
                "client_id": config.client_id,
                "config_source": config_source,
                "config_fingerprint": fingerprint,
                "vectordb_backend": config.vectordb.type.value,
                "vectordb_collection": config.vectordb.collection,
                "embedder_type": config.embedder.type.value,
                "embedder_model": _resolve_embedder_model(config),
                "llm_model": _resolve_llm_model(config),
                "reranker_model": config.reranker.model if config.reranker else None,
                "search_mode": config.retrieval.search_mode.value,
                "data_sensitivity": config.security.data_sensitivity,
                "pii_enabled": config.security.pii_middleware.enabled,
            }, default=str),
        )
    except Exception:
        pass


# ── Tenant blueprint drift (ingestion / tokenization / Celery dispatch) ─────
_BLUEPRINT_TUNING_KEYS: tuple[str, ...] = ("tokenization", "celery_dispatch", "ingestion")
_tenant_blueprint_warned: set[str] = set()


def _blueprint_tuning_blob(raw: Dict[str, Any]) -> str:
    blob = {k: raw.get(k) for k in _BLUEPRINT_TUNING_KEYS}
    return _json_module.dumps(blob, sort_keys=True, default=str)


def _maybe_warn_tenant_blueprint_drift(
    client_id: str,
    base_raw: Dict[str, Any],
    merged_raw: Dict[str, Any],
    had_client_file: bool,
) -> None:
    if client_id == _DEFAULT_CONFIG_ID or not had_client_file:
        return
    if _blueprint_tuning_blob(base_raw) == _blueprint_tuning_blob(merged_raw):
        return
    if client_id in _tenant_blueprint_warned:
        return
    _tenant_blueprint_warned.add(client_id)
    logger.warning(
        "[ConfigResolver] client_id=%s: merged Client JSON changes default blueprint "
        "for tokenization, ingestion, and/or celery_dispatch — confirm intentional.",
        client_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API — get_client_config
# ─────────────────────────────────────────────────────────────────────────────

def get_client_config_for_ingestion(client_id: str) -> ClientConfig:
    """Same as get_client_config but disables sparse env overlays (JSON-authoritative)."""
    return get_client_config(client_id, apply_env=False)


def get_celery_ingestion_enqueue_kwargs(client_id: str) -> Dict[str, Any]:
    """
    Keyword args for Celery apply_async when enqueueing ingestion tasks for a tenant.
    Uses celery_dispatch from merged Client JSON (queue + optional time limits).
    """
    cfg = get_client_config_for_ingestion(client_id)
    d = cfg.celery_dispatch
    out: Dict[str, Any] = {"queue": d.ingestion_queue}
    if d.soft_time_limit and int(d.soft_time_limit) > 0:
        out["soft_time_limit"] = int(d.soft_time_limit)
    if d.hard_time_limit and int(d.hard_time_limit) > 0:
        out["time_limit"] = int(d.hard_time_limit)
    return out


def get_client_config(
    client_id: str,
    *,
    apply_env: bool = True,
) -> ClientConfig:
    """
    Load, merge, validate, and return the FINAL ClientConfig for a client.

    Resolution flow:
        1. Load default config (base layer).
        2. Load client-specific config (override layer, if exists).
        3. Deep-merge client overrides onto the default base.
        4. Sparse env overlays (none for pipeline semantics; deprecated MAI_* logged if set).
        5. Stamp client_id onto the merged dict.
        6. Parse through Pydantic (schema validation).
        7. Run compatibility validation — raise ConfigValidationError on ERROR.
        8. Emit structured log with config fingerprint.
        9. Return the FINAL merged ClientConfig.

    Args:
        client_id: Client/tenant identifier.
        apply_env: If True, apply any registered non-pipeline env overlays (currently empty).

    Returns:
        Validated, merged ClientConfig — ready for pipeline construction.

    Raises:
        FileNotFoundError:       Default config not found.
        ConfigValidationError:   Compatibility check found ERROR-severity issues.
        ValidationError:         Pydantic schema validation failed.
    """
    client_id = _resolve_client_id_for_config_lookup(client_id)

    # ── Step 1: Load default config (base) ────────────────────────────────
    default_path = _find_config_path(_DEFAULT_CONFIG_ID)
    if default_path is None:
        raise FileNotFoundError(
            f"Default config not found. Searched: {[str(d) for d in _CONFIG_DIRS]}"
        )
    base_raw = _load_raw(default_path)
    config_source = "default"

    # ── Step 2: Load client-specific config (override) ────────────────────
    client_path = _find_config_path(client_id)
    if client_path is not None and client_id != _DEFAULT_CONFIG_ID:
        client_raw = _load_raw(client_path)
        # ── Step 3: Deep merge ────────────────────────────────────────────
        merged_raw = _deep_merge(base_raw, client_raw)
        had_client_file = True
        config_source = f"default+{client_id}"
        logger.debug(
            "[ConfigResolver] Merged client config '%s' onto default", client_id,
        )
    else:
        merged_raw = base_raw
        had_client_file = False
        if client_id != _DEFAULT_CONFIG_ID:
            logger.debug(
                "[ConfigResolver] No override for '%s' — using default only",
                client_id,
            )

    _maybe_warn_tenant_blueprint_drift(client_id, base_raw, merged_raw, had_client_file)

    # ── Step 4: Apply sparse env overlays (pipeline MAI_* no longer merged) ─
    warn_if_deprecated_pipeline_env_set()
    if apply_env:
        merged_raw = _apply_env_overrides(merged_raw)

    # ── Step 5: Stamp client_id ───────────────────────────────────────────
    merged_raw["client_id"] = client_id

    # ── Step 6: Parse through Pydantic ────────────────────────────────────
    config = ClientConfig.from_dict(merged_raw)

    # ── Step 7: Compatibility validation ──────────────────────────────────
    issues = validate_config_compatibility(config)
    errors = [i for i in issues if i.severity == IssueSeverity.ERROR]

    for issue in issues:
        if issue.severity == IssueSeverity.WARNING:
            logger.warning(
                "[ConfigResolver][%s] %s — %s",
                client_id, issue.component, issue.message,
            )

    if errors:
        raise ConfigValidationError(client_id=client_id, issues=errors)

    # ── Step 8: Structured log ────────────────────────────────────────────
    fingerprint = _compute_fingerprint(merged_raw)
    _log_resolved_config(config, config_source=config_source, fingerprint=fingerprint)

    # ── Step 9: Return the FINAL config ───────────────────────────────────
    return config


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility validation
# ─────────────────────────────────────────────────────────────────────────────

class IssueSeverity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ConfigIssue:
    severity: IssueSeverity
    component: str
    message: str


_EMBEDDERS_REQUIRING_API_KEY: Dict[EmbedderType, str] = {
    EmbedderType.OPENAI: "OPENAI_API_KEY",
    EmbedderType.COHERE: "COHERE_API_KEY",
    EmbedderType.GEMINI: "GEMINI_API_KEY",
}

_VECTORDB_REQUIRING_API_KEY: Dict[VectorDBType, str] = {
    VectorDBType.PINECONE: "PINECONE_API_KEY",
    VectorDBType.WEAVIATE: "WEAVIATE_API_KEY",
}

_NORMALIZED_EMBEDDERS = frozenset({
    EmbedderType.HUGGINGFACE,
    EmbedderType.OPENAI,
    EmbedderType.COHERE,
})

_COSINE_ONLY_VECTORDBS = frozenset({
    VectorDBType.CHROMA,
})


def validate_config_compatibility(config: ClientConfig) -> List[ConfigIssue]:
    """
    Run post-schema compatibility checks that Pydantic alone cannot enforce.

    Returns a list of issues (empty = all clear). Does NOT raise — callers
    decide how to handle issues based on severity.
    """
    issues: List[ConfigIssue] = []

    _check_required_fields(config, issues)
    _check_provider_models(config, issues)
    _check_api_keys(config, issues)
    _check_embedder_vectordb_compat(config, issues)
    _check_embedding_dimension_compat(config, issues)
    _check_security_config(config, issues)
    _check_sensitivity_embedder_compat(config, issues)
    _check_reranker_compat(config, issues)
    _check_retrieval_config(config, issues)

    return issues


def _check_required_fields(config: ClientConfig, issues: List[ConfigIssue]) -> None:
    if not config.client_id or not config.client_id.strip():
        issues.append(ConfigIssue(
            severity=IssueSeverity.ERROR,
            component="identity",
            message="client_id is empty or blank — every config must have a unique identifier.",
        ))

    if config.vectordb.type and not config.vectordb.collection:
        issues.append(ConfigIssue(
            severity=IssueSeverity.ERROR,
            component="vectordb",
            message="vectordb.collection is empty — a collection name is required.",
        ))


def _check_provider_models(config: ClientConfig, issues: List[ConfigIssue]) -> None:
    """Require non-empty model ids in Client JSON (no silent .env fallback)."""
    sub = getattr(config.embedder, config.embedder.type.value, None)
    if sub is not None:
        model = getattr(sub, "model", None)
        if model is None or not str(model).strip():
            issues.append(ConfigIssue(
                severity=IssueSeverity.ERROR,
                component="embedder",
                message=(
                    f"Embedder type '{config.embedder.type.value}' requires a non-empty "
                    "'model' in Client JSON — configure embedder.<provider>.model."
                ),
            ))
    if config.llm and config.llm.single:
        m = config.llm.single.model
        if not m or not str(m).strip():
            issues.append(ConfigIssue(
                severity=IssueSeverity.ERROR,
                component="llm",
                message="llm.single.model is empty — set the LLM model id in Client JSON.",
            ))
    if config.llm and config.llm.chain:
        for i, step in enumerate(config.llm.chain):
            if not step.model or not str(step.model).strip():
                issues.append(ConfigIssue(
                    severity=IssueSeverity.ERROR,
                    component="llm",
                    message=(
                        f"llm.chain[{i}].model is empty — set each chain step model in Client JSON."
                    ),
                ))
    rr = config.reranker
    if rr is None:
        return
    if rr.type == RerankerType.LLM_JUDGE:
        if not rr.model or not str(rr.model).strip():
            issues.append(ConfigIssue(
                severity=IssueSeverity.ERROR,
                component="reranker",
                message="llm_judge reranker requires a non-empty 'model' in Client JSON.",
            ))
    elif rr.type in (
        RerankerType.CROSS_ENCODER,
        RerankerType.BGE_RERANKER,
        RerankerType.FLASHRANK,
        RerankerType.COHERE,
        RerankerType.COLBERT,
    ):
        if not rr.model or not str(rr.model).strip():
            issues.append(ConfigIssue(
                severity=IssueSeverity.ERROR,
                component="reranker",
                message=f"Reranker type '{rr.type.value}' requires a non-empty 'model' in Client JSON.",
            ))


def _check_api_keys(config: ClientConfig, issues: List[ConfigIssue]) -> None:
    embedder_key_env = _EMBEDDERS_REQUIRING_API_KEY.get(config.embedder.type)
    if embedder_key_env:
        sub = getattr(config.embedder, config.embedder.type.value, None)
        env_var = getattr(sub, "api_key_env", None) if sub else None
        actual_env = env_var or embedder_key_env
        if not os.getenv(actual_env):
            issues.append(ConfigIssue(
                severity=IssueSeverity.ERROR,
                component="embedder",
                message=(
                    f"Embedder '{config.embedder.type.value}' requires API key "
                    f"via env var '{actual_env}' — not set."
                ),
            ))

    vdb_key_env = _VECTORDB_REQUIRING_API_KEY.get(config.vectordb.type)
    if vdb_key_env and not os.getenv(vdb_key_env):
        issues.append(ConfigIssue(
            severity=IssueSeverity.WARNING,
            component="vectordb",
            message=(
                f"VectorDB '{config.vectordb.type.value}' may require API key "
                f"via '{vdb_key_env}' — not set. Will fail at connection time."
            ),
        ))

    if config.llm and config.llm.single:
        llm_key_env = config.llm.single.api_key_env
        if llm_key_env and not os.getenv(llm_key_env):
            issues.append(ConfigIssue(
                severity=IssueSeverity.WARNING,
                component="llm",
                message=(
                    f"LLM '{config.llm.single.type.value}' API key env var "
                    f"'{llm_key_env}' is not set."
                ),
            ))


def _check_embedder_vectordb_compat(
    config: ClientConfig, issues: List[ConfigIssue],
) -> None:
    if config.embedder.type == EmbedderType.OLLAMA:
        if config.vectordb.type not in _COSINE_ONLY_VECTORDBS:
            issues.append(ConfigIssue(
                severity=IssueSeverity.WARNING,
                component="embedder+vectordb",
                message=(
                    f"Ollama embedders may not produce normalized vectors. "
                    f"Ensure '{config.vectordb.type.value}' collection uses "
                    f"cosine distance metric to avoid ranking distortion."
                ),
            ))

    if (
        config.embedder.type in _NORMALIZED_EMBEDDERS
        and config.vectordb.type in _COSINE_ONLY_VECTORDBS
    ):
        issues.append(ConfigIssue(
            severity=IssueSeverity.INFO,
            component="embedder+vectordb",
            message=(
                f"'{config.embedder.type.value}' produces normalized vectors "
                f"with '{config.vectordb.type.value}' (cosine) — optimal alignment."
            ),
        ))

    if config.embedder.type == EmbedderType.GEMINI:
        if config.vectordb.type == VectorDBType.CHROMA:
            issues.append(ConfigIssue(
                severity=IssueSeverity.INFO,
                component="embedder+vectordb",
                message=(
                    "Gemini embeddings with Chroma: verify dimension compatibility. "
                    "Gemini embedding-001 outputs 768d by default."
                ),
            ))


def _check_embedding_dimension_compat(
    config: ClientConfig, issues: List[ConfigIssue],
) -> None:
    """
    Validate Pinecone embedding_dim matches known embedder output dimensions.
    Other VectorDBs auto-detect dimension at collection creation time, but
    Pinecone requires it upfront — mismatches cause silent data corruption.
    """
    if config.vectordb.type != VectorDBType.PINECONE or not config.vectordb.pinecone:
        return

    declared_dim = config.vectordb.pinecone.embedding_dim

    _KNOWN_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "gemini-embedding-001": 768,
        "BAAI/bge-large-en-v1.5": 1024,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-small-en-v1.5": 384,
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
    }

    sub = getattr(config.embedder, config.embedder.type.value, None)
    model_name = getattr(sub, "model", None) if sub else None

    if model_name and model_name in _KNOWN_DIMS:
        expected = _KNOWN_DIMS[model_name]
        if declared_dim != expected:
            issues.append(ConfigIssue(
                severity=IssueSeverity.ERROR,
                component="embedder+vectordb",
                message=(
                    f"Pinecone embedding_dim={declared_dim} does not match "
                    f"embedder '{model_name}' output dimension={expected}. "
                    f"This will cause upsert failures or silent data corruption."
                ),
            ))
    elif model_name:
        issues.append(ConfigIssue(
            severity=IssueSeverity.WARNING,
            component="embedder+vectordb",
            message=(
                f"Embedder model '{model_name}' is not in the known dimension "
                f"catalog. Declared Pinecone dim={declared_dim} — verify this "
                f"matches the actual embedder output."
            ),
        ))


_REMOTE_EMBEDDER_TYPES = frozenset({
    EmbedderType.OPENAI,
    EmbedderType.COHERE,
    EmbedderType.GEMINI,
})


def _check_sensitivity_embedder_compat(
    config: ClientConfig, issues: List[ConfigIssue],
) -> None:
    """
    High-sensitivity tenants must use local embedders only — data must
    never leave the service boundary.
    """
    sensitivity = config.security.data_sensitivity
    if sensitivity == "high" and config.embedder.type in _REMOTE_EMBEDDER_TYPES:
        issues.append(ConfigIssue(
            severity=IssueSeverity.ERROR,
            component="security+embedder",
            message=(
                f"data_sensitivity='high' but embedder type "
                f"'{config.embedder.type.value}' is a remote/cloud provider. "
                f"High-sensitivity tenants require local embedders "
                f"(huggingface, ollama) to prevent data exfiltration."
            ),
        ))


def _check_reranker_compat(
    config: ClientConfig, issues: List[ConfigIssue],
) -> None:
    """Validate reranker configuration for known incompatibilities."""
    rr = config.reranker
    if rr is None:
        return

    if rr.type == RerankerType.COHERE:
        env_var = rr.api_key_env or "COHERE_API_KEY"
        if not os.getenv(env_var):
            issues.append(ConfigIssue(
                severity=IssueSeverity.ERROR,
                component="reranker",
                message=(
                    f"Cohere reranker requires API key via env var '{env_var}' "
                    f"— not set. Pipeline build will fail."
                ),
            ))

    if rr.type == RerankerType.LLM_JUDGE:
        judge_provider = getattr(rr, "judge_provider", None) or "openai"
        if judge_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            issues.append(ConfigIssue(
                severity=IssueSeverity.ERROR,
                component="reranker",
                message=(
                    "LLM-Judge reranker with provider='openai' requires "
                    "OPENAI_API_KEY — not set."
                ),
            ))
        elif judge_provider == "gemini" and not os.getenv(rr.api_key_env or "GEMINI_API_KEY"):
            issues.append(ConfigIssue(
                severity=IssueSeverity.ERROR,
                component="reranker",
                message=(
                    "LLM-Judge reranker with provider='gemini' requires "
                    f"'{rr.api_key_env or 'GEMINI_API_KEY'}' — not set."
                ),
            ))

    if rr.top_k and config.retrieval.top_k_final:
        if rr.top_k < config.retrieval.top_k_final:
            issues.append(ConfigIssue(
                severity=IssueSeverity.WARNING,
                component="reranker+retrieval",
                message=(
                    f"reranker.top_k ({rr.top_k}) < retrieval.top_k_final "
                    f"({config.retrieval.top_k_final}). The reranker will "
                    f"truncate results before final trimming."
                ),
            ))


def _check_security_config(
    config: ClientConfig, issues: List[ConfigIssue],
) -> None:
    sensitivity = config.security.data_sensitivity
    if sensitivity == "high" and not config.security.pii_middleware.enabled:
        issues.append(ConfigIssue(
            severity=IssueSeverity.ERROR,
            component="security",
            message=(
                "data_sensitivity='high' but PII middleware is disabled — "
                "this combination is unsafe. Enable pii_middleware."
            ),
        ))

    if not config.security.pii_middleware.enabled:
        issues.append(ConfigIssue(
            severity=IssueSeverity.WARNING,
            component="security",
            message="PII middleware is disabled — PII may leak to external APIs.",
        ))


def _check_retrieval_config(
    config: ClientConfig, issues: List[ConfigIssue],
) -> None:
    r = config.retrieval
    if r.top_k_final > r.top_k_retrieval:
        issues.append(ConfigIssue(
            severity=IssueSeverity.ERROR,
            component="retrieval",
            message=(
                f"top_k_final ({r.top_k_final}) > top_k_retrieval ({r.top_k_retrieval}) — "
                f"cannot return more results than retrieved."
            ),
        ))

    if r.similarity_threshold > 0.9:
        issues.append(ConfigIssue(
            severity=IssueSeverity.WARNING,
            component="retrieval",
            message=(
                f"similarity_threshold={r.similarity_threshold} is very high — "
                f"may filter out most results. Consider lowering."
            ),
        ))
