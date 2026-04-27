"""
================================================================================
Client Config Resolver — Centralized config loading with validation.

STATUS: SHADOW-ONLY. Not wired into runtime. Safe to import and test
        without affecting production pipelines.

RESPONSIBILITIES:
    1. Locate client config files (JSON/YAML) across known directories.
    2. Load and parse into a validated ClientConfig via Pydantic.
    3. Apply environment-variable overrides for deployment flexibility.
    4. Fall back to the default config when a client-specific file is missing.
    5. Run post-load compatibility checks (embedder ↔ vectordb, required
       API keys, distance metric alignment) that Pydantic schema validation
       alone cannot catch.

USAGE (shadow / testing only):
    from app.core.config.client_config_resolver import get_client_config

    config = get_client_config("acme_corp")
    issues = validate_config_compatibility(config)
    if issues:
        for issue in issues:
            print(f"[{issue.severity}] {issue.message}")
================================================================================
"""

from __future__ import annotations

import json as _json_module
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.core.config.client_config_schema import (
    ClientConfig,
    EmbedderType,
    VectorDBType,
)
from app.utils.path_sanitizer import sanitize_client_id

logger = logging.getLogger(__name__)


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
    Returns the first match or None.
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


# ─────────────────────────────────────────────────────────────────────────────
# Environment overrides
# ─────────────────────────────────────────────────────────────────────────────

_ENV_OVERRIDES: Dict[str, str] = {
    # env var → dotted config path (applied post-load)
    "MAI_VECTORDB":     "vectordb.type",
    "MAI_EMBEDDER":     "embedder.type",
    "MAI_COLLECTION":   "vectordb.collection",
    "MAI_SEARCH_MODE":  "retrieval.search_mode",
}


def _apply_env_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Overlay environment variables onto the raw config dict before Pydantic
    parsing. Only applies overrides for env vars that are actually set.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Loader
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


def get_client_config(
    client_id: str,
    *,
    apply_env: bool = True,
) -> ClientConfig:
    """
    Load and validate a ClientConfig for the given client_id.

    SHADOW MODE (current):
        Loads the new (resolver) config AND the runtime (default) config,
        computes a structured diff, logs drift analysis, but ALWAYS returns
        the runtime/default config. The new config is never applied.

    Resolution order:
        1. Look for app/core/configs/{client_id}.json (or .yaml)
        2. Fall back to app/core/configs/default.json (or .yaml)
        3. Apply env-var overrides if apply_env=True
        4. Parse through Pydantic (schema validation)

    Args:
        client_id:  Client/tenant identifier.
        apply_env:  Apply MAI_* environment overrides (default True).

    Returns:
        Validated ClientConfig instance (ALWAYS the runtime/default config
        in shadow mode).

    Raises:
        FileNotFoundError: If neither client-specific nor default config exists.
        ValueError / ValidationError: If the config fails schema validation.
    """
    # ── Load the runtime (old) config — always the default ────────────────
    default_path = _find_config_path(_DEFAULT_CONFIG_ID)
    if default_path is None:
        raise FileNotFoundError(
            f"Default config not found. Searched: {[str(d) for d in _CONFIG_DIRS]}"
        )
    old_config = _load_config_from_path(
        _DEFAULT_CONFIG_ID, default_path, apply_env=apply_env,
    )

    # ── Load the new (resolver) config ────────────────────────────────────
    client_path = _find_config_path(client_id)
    if client_path is None:
        logger.info(
            "[ConfigResolver] No config for '%s' — falling back to default",
            client_id,
        )
        return old_config

    logger.info(
        "[ConfigResolver] Loading config for '%s' from %s",
        client_id, client_path,
    )
    new_config = _load_config_from_path(
        client_id, client_path, apply_env=apply_env,
    )

    # ── Shadow mode: compare, log, but NEVER return new_config ────────────
    _shadow_compare(old_config, new_config, client_id)

    return old_config


def _shadow_compare(
    old_config: ClientConfig,
    new_config: ClientConfig,
    client_id: str,
) -> None:
    """
    Shadow-mode drift analysis. Compares old vs new config, logs findings,
    but never affects the return value. Failures are swallowed.
    """
    try:
        from app.core.config.config_diff_logger import compare_configs
        from app.core.config.config_drift_gate import should_block_migration

        diff = compare_configs(old_config, new_config, client_id)
        should_block = should_block_migration(diff)

        drift_score = diff.get("drift_score", 0)
        has_critical = diff.get("has_critical", False)

        logger.info(
            "%s",
            _json_module.dumps({
                "event": "SHADOW_MODE_ACTIVE",
                "client_id": client_id,
                "drift_score": drift_score,
                "has_critical": has_critical,
                "total_changes": diff.get("total_changes", 0),
            }, default=str),
        )

        if should_block:
            logger.warning(
                "%s",
                _json_module.dumps({
                    "event": "SHADOW_MIGRATION_BLOCKED",
                    "message": "SHADOW MIGRATION BLOCKED - CRITICAL DRIFT DETECTED",
                    "client_id": client_id,
                    "drift_score": drift_score,
                }, default=str),
            )
        else:
            logger.info(
                "%s",
                _json_module.dumps({
                    "event": "SHADOW_CONFIG_VALIDATED",
                    "message": "SHADOW MODE - NEW CONFIG VALIDATED BUT NOT APPLIED",
                    "client_id": client_id,
                    "drift_score": drift_score,
                }, default=str),
            )
    except Exception as exc:
        logger.debug(
            "[ConfigResolver] Shadow comparison failed for '%s': %s",
            client_id, exc,
        )


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
    _check_api_keys(config, issues)
    _check_embedder_vectordb_compat(config, issues)
    _check_security_config(config, issues)
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
    """
    Embedder ↔ VectorDB compatibility:
      - Normalized embedders (HF, OpenAI, Cohere) produce unit vectors.
        Cosine and dot-product metrics work correctly.
      - Non-normalized embedders (Ollama) should use cosine distance,
        not dot-product, to avoid ranking distortion.
      - Chroma defaults to cosine — always compatible.
      - Qdrant/Pinecone/Milvus support configurable metrics — warn if
        the combination looks off.
    """
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
