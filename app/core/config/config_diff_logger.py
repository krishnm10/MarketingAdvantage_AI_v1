"""
================================================================================
Config Diff Logger — Structured comparison of ClientConfig snapshots.

STATUS: SHADOW-ONLY. Safe to import without runtime side effects.

PURPOSE:
    Detects configuration drift between two ClientConfig instances (e.g.
    before/after a hot-reload, migration, or env-var override). Produces a
    machine-readable diff with per-field severity for alerting and audit.

USAGE:
    from app.core.config.config_diff_logger import compare_configs

    diff = compare_configs(old_config, new_config, client_id="acme_corp")
    for d in diff["differences"]:
        print(f"[{d['severity']}] {d['field']}: {d['old']} → {d['new']}")
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
    Compute a structured diff between two ClientConfig instances.

    Args:
        old_config: Previous ClientConfig (or None for first-time load).
        new_config: Current ClientConfig.
        client_id:  Client identifier for the diff record.

    Returns:
        {
            "client_id": "...",
            "total_changes": int,
            "has_critical": bool,
            "differences": [
                {
                    "field": "embedder.model",
                    "old": "BAAI/bge-large-en-v1.5",
                    "new": "text-embedding-3-small",
                    "severity": "CRITICAL"
                },
                ...
            ]
        }

    Never raises exceptions — returns an empty diff on any internal error.
    """
    try:
        return _compare(old_config, new_config, client_id)
    except Exception as exc:
        logger.warning(
            "[ConfigDiff] Comparison failed for client_id='%s': %s",
            client_id, exc,
        )
        return {
            "client_id": client_id,
            "total_changes": 0,
            "has_critical": False,
            "differences": [],
            "error": str(exc)[:200],
        }


def _compare(
    old_config: Any,
    new_config: Any,
    client_id: str,
) -> Dict[str, Any]:
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

    has_critical = any(d["severity"] == "CRITICAL" for d in differences)

    critical_count = sum(1 for d in differences if d["severity"] == "CRITICAL")
    warning_count = sum(1 for d in differences if d["severity"] == "WARNING")
    info_count = sum(1 for d in differences if d["severity"] == "INFO")

    # Drift score: CRITICAL=10, WARNING=3, INFO=1
    drift_score = critical_count * 10 + warning_count * 3 + info_count

    if differences:
        try:
            logger.info(
                "%s",
                _json_module.dumps({
                    "event": "CONFIG_DIFF_DETECTED",
                    "client_id": client_id,
                    "diff_count": len(differences),
                    "critical_count": critical_count,
                    "warning_count": warning_count,
                    "info_count": info_count,
                    "drift_score": drift_score,
                    "summary": differences,
                }, default=str),
            )
        except Exception:
            pass

    return {
        "client_id": client_id,
        "total_changes": len(differences),
        "has_critical": has_critical,
        "drift_score": drift_score,
        "differences": differences,
    }


def _normalize(value: Any) -> Any:
    """Normalize values for stable comparison."""
    if value is None:
        return None
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, float):
        return round(value, 6)
    return value
