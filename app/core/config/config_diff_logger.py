"""
================================================================================
Config Diff Logger — Structured comparison + logging for ClientConfig drift.

STATUS: SHADOW-ONLY. Safe to import without runtime side effects.

COMPONENTS:
    compare_configs()         — PURE function. Returns structured diff only.
                                No logging, no scoring, no side effects.
    log_config_drift_event()  — Fire-and-forget JSON log emitter.
                                No computation, no decision-making.

USAGE:
    from app.core.config.config_diff_logger import compare_configs, log_config_drift_event
    from app.core.config.config_drift_scorer import compute_drift_score

    diff = compare_configs(old_config, new_config, client_id="acme_corp")
    drift_score = compute_drift_score(diff)

    log_config_drift_event(
        client_id="acme_corp",
        event="CONFIG_DIFF_DETECTED",
        diff=diff,
        drift_score=drift_score,
        mode="SHADOW",
    )
================================================================================
"""

from __future__ import annotations

import json as _json_module
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Field definitions: (dotted_path, extractor, severity)
#
# Severity guide:
#   CRITICAL — change breaks index compatibility or security posture;
#              likely requires re-indexing or manual review.
#   WARNING  — change affects retrieval quality or cost; no re-index needed
#              but behaviour will shift.
#   INFO     — cosmetic or low-impact change.
# ─────────────────────────────────────────────────────────────────────────────

def _get_nested(obj: Any, dotted_path: str, default: Any = None) -> Any:
    """Safely traverse a dotted attribute path on an object or dict."""
    current = obj
    for key in dotted_path.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key, None)
        else:
            current = getattr(current, key, None)
    return current if current is not None else default


def _resolve_embedder_model(cfg: Any) -> Optional[str]:
    """Extract the embedder model name from the active sub-config."""
    emb = _get_nested(cfg, "embedder")
    if emb is None:
        return None
    emb_type = _get_nested(emb, "type")
    if emb_type is None:
        return None
    type_val = emb_type.value if hasattr(emb_type, "value") else str(emb_type)
    sub = _get_nested(emb, type_val)
    return _get_nested(sub, "model")


def _resolve_embedder_normalize(cfg: Any) -> Optional[bool]:
    emb = _get_nested(cfg, "embedder")
    if emb is None:
        return None
    emb_type = _get_nested(emb, "type")
    if emb_type is None:
        return None
    type_val = emb_type.value if hasattr(emb_type, "value") else str(emb_type)
    sub = _get_nested(emb, type_val)
    return _get_nested(sub, "normalize")


def _resolve_llm_model(cfg: Any) -> Optional[str]:
    single = _get_nested(cfg, "llm.single")
    if single is not None:
        return _get_nested(single, "model")
    chain = _get_nested(cfg, "llm.chain")
    if chain and isinstance(chain, list) and len(chain) > 0:
        return _get_nested(chain[0], "model")
    return None


def _resolve_reranker_model(cfg: Any) -> Optional[str]:
    return _get_nested(cfg, "reranker.model")


def _resolve_vectordb_backend(cfg: Any) -> Optional[str]:
    vdb_type = _get_nested(cfg, "vectordb.type")
    return vdb_type.value if hasattr(vdb_type, "value") else vdb_type


def _resolve_vectordb_metric(cfg: Any) -> Optional[str]:
    """Attempt to extract the distance metric from the active vectordb sub-config."""
    vdb = _get_nested(cfg, "vectordb")
    if vdb is None:
        return None
    vdb_type = _get_nested(vdb, "type")
    if vdb_type is None:
        return None
    type_val = vdb_type.value if hasattr(vdb_type, "value") else str(vdb_type)
    sub = _get_nested(vdb, type_val)
    return _get_nested(sub, "distance_metric") or _get_nested(sub, "metric")


_FIELD_SPECS: List[Dict[str, Any]] = [
    # Embedder fields — CRITICAL (re-index required on change)
    {
        "field": "embedder.model",
        "extractor": _resolve_embedder_model,
        "severity": "CRITICAL",
    },
    {
        "field": "embedder.type",
        "extractor": lambda cfg: (
            v.value if hasattr(v := _get_nested(cfg, "embedder.type"), "value") else v
        ),
        "severity": "CRITICAL",
    },
    {
        "field": "embedder.normalization",
        "extractor": _resolve_embedder_normalize,
        "severity": "CRITICAL",
    },
    # VectorDB fields — CRITICAL
    {
        "field": "vectordb.backend",
        "extractor": _resolve_vectordb_backend,
        "severity": "CRITICAL",
    },
    {
        "field": "vectordb.collection",
        "extractor": lambda cfg: _get_nested(cfg, "vectordb.collection"),
        "severity": "CRITICAL",
    },
    {
        "field": "vectordb.metric",
        "extractor": _resolve_vectordb_metric,
        "severity": "CRITICAL",
    },
    # LLM — WARNING (affects answer quality, not index)
    {
        "field": "llm.model",
        "extractor": _resolve_llm_model,
        "severity": "WARNING",
    },
    # Reranker — WARNING
    {
        "field": "reranker.model",
        "extractor": _resolve_reranker_model,
        "severity": "WARNING",
    },
    # Retrieval tuning — WARNING
    {
        "field": "retrieval.top_k",
        "extractor": lambda cfg: _get_nested(cfg, "retrieval.top_k_retrieval"),
        "severity": "WARNING",
    },
    {
        "field": "retrieval.top_k_final",
        "extractor": lambda cfg: _get_nested(cfg, "retrieval.top_k_final"),
        "severity": "WARNING",
    },
    {
        "field": "retrieval.similarity_threshold",
        "extractor": lambda cfg: _get_nested(cfg, "retrieval.similarity_threshold"),
        "severity": "WARNING",
    },
    # Security — CRITICAL
    {
        "field": "security.data_sensitivity",
        "extractor": lambda cfg: _get_nested(cfg, "security.data_sensitivity"),
        "severity": "CRITICAL",
    },
    {
        "field": "security.pii_enabled",
        "extractor": lambda cfg: _get_nested(cfg, "security.pii_middleware.enabled"),
        "severity": "CRITICAL",
    },
    # Tokenizer — INFO (rarely changes independently)
    {
        "field": "tokenizer.type",
        "extractor": lambda cfg: _get_nested(cfg, "ingestion.chunking.strategy"),
        "severity": "INFO",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compare_configs(
    old_config: Any,
    new_config: Any,
    client_id: str,
) -> Dict[str, Any]:
    """
    PURE function — compute a structured diff between two ClientConfig instances.

    No logging, no scoring, no side effects. Callers are responsible for
    scoring (config_drift_scorer) and logging (log_config_drift_event).

    Args:
        old_config: Previous ClientConfig (or None for first-time load).
        new_config: Current ClientConfig.
        client_id:  Client identifier (passthrough for correlation).

    Returns:
        {
            "client_id":      str,
            "critical_count": int,
            "warning_count":  int,
            "info_count":     int,
            "has_critical":   bool,
            "differences":    [
                {
                    "field":    "embedder.model",
                    "old":      "BAAI/bge-large-en-v1.5",
                    "new":      "text-embedding-3-small",
                    "severity": "CRITICAL"
                },
                ...
            ]
        }
    """
    differences: List[Dict[str, Any]] = []

    for spec in _FIELD_SPECS:
        field_name: str = spec["field"]
        extractor = spec["extractor"]
        severity: str = spec["severity"]

        old_val = _normalize(extractor(old_config) if old_config else None)
        new_val = _normalize(extractor(new_config) if new_config else None)

        if old_val != new_val:
            differences.append({
                "field": field_name,
                "old": old_val,
                "new": new_val,
                "severity": severity,
            })

    critical_count = sum(1 for d in differences if d["severity"] == "CRITICAL")
    warning_count = sum(1 for d in differences if d["severity"] == "WARNING")
    info_count = sum(1 for d in differences if d["severity"] == "INFO")

    return {
        "client_id": client_id,
        "critical_count": critical_count,
        "warning_count": warning_count,
        "info_count": info_count,
        "has_critical": critical_count > 0,
        "differences": differences,
    }


def log_config_drift_event(
    client_id: str,
    event: str,
    diff: Dict[str, Any],
    drift_score: int,
    mode: str,
) -> None:
    """
    Emit a single structured JSON log entry for a config drift event.

    This function performs NO computation and NO decision-making.
    It is a fire-and-forget log emitter wrapped in try/except so that
    logging failures never propagate to callers.

    Args:
        client_id:    Tenant/client identifier.
        event:        Event label (e.g. SHADOW_MODE_ACTIVE, MIGRATION_BLOCKED).
        diff:         The diff payload (output of compare_configs or config_diff_engine).
        drift_score:  Pre-computed drift score (from config_drift_scorer).
        mode:         Execution mode (e.g. SHADOW, CANARY, FULL_MIGRATION).
    """
    try:
        logger.info(
            "%s",
            _json_module.dumps(
                {
                    "event": event,
                    "client_id": client_id,
                    "mode": mode,
                    "drift_score": drift_score,
                    "diff": diff,
                },
                default=str,
            ),
        )
    except Exception:
        pass


def _normalize(value: Any) -> Any:
    """Normalize values for stable comparison."""
    if value is None:
        return None
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, float):
        return round(value, 6)
    return value
