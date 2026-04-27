"""
================================================================================
Config Diff Engine — Pure, side-effect-free deep comparison of config dicts.

PURPOSE:
    Structural diff of two configuration dicts (or Pydantic models serialized
    to dict). Returns added/removed/modified/unchanged fields at every
    nesting level. No logging, no scoring, no I/O — deterministic pure
    function suitable for unit tests, snapshots, and higher-level analyzers.

USAGE:
    from app.core.config.config_diff_engine import compare_configs

    diff = compare_configs(old_dict, new_dict, client_id="acme")
    for m in diff["modified"]:
        print(f"{m['path']}: {m['old']} → {m['new']}")
================================================================================
"""

from __future__ import annotations

from typing import Any, Dict, List, Union


def compare_configs(
    old_config: Union[Dict[str, Any], Any],
    new_config: Union[Dict[str, Any], Any],
    client_id: str,
) -> Dict[str, Any]:
    """
    Deep-compare two config dicts and return a structured diff.

    Args:
        old_config: Previous config (dict or Pydantic model).
        new_config: Current config (dict or Pydantic model).
        client_id:  Client identifier (included in output for correlation).

    Returns:
        {
            "client_id": str,
            "added":     [{"path": "security.iban", "value": ...}],
            "removed":   [{"path": "legacy_flag",   "value": ...}],
            "modified":  [{"path": "embedder.type", "old": ..., "new": ...}],
            "unchanged": ["client_id", "vectordb.type", ...]
        }
    """
    old = _to_dict(old_config) if old_config is not None else {}
    new = _to_dict(new_config) if new_config is not None else {}

    added: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    modified: List[Dict[str, Any]] = []
    unchanged: List[str] = []

    _deep_compare(old, new, prefix="", added=added, removed=removed,
                  modified=modified, unchanged=unchanged)

    return {
        "client_id": client_id,
        "added": added,
        "removed": removed,
        "modified": modified,
        "unchanged": unchanged,
    }


def _to_dict(obj: Any) -> Dict[str, Any]:
    """Convert a Pydantic model or dict to a plain dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return {}


def _normalize_value(val: Any) -> Any:
    """Normalize a value for stable comparison."""
    if hasattr(val, "value"):
        return val.value
    if isinstance(val, float):
        return round(val, 6)
    return val


def _serialize_value(val: Any) -> Any:
    """Ensure a value is JSON-serializable."""
    if val is None or isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, (list, tuple)):
        return [_serialize_value(v) for v in val]
    if isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}
    if hasattr(val, "value"):
        return val.value
    return str(val)


def _deep_compare(
    old: Dict[str, Any],
    new: Dict[str, Any],
    prefix: str,
    added: List[Dict[str, Any]],
    removed: List[Dict[str, Any]],
    modified: List[Dict[str, Any]],
    unchanged: List[str],
) -> None:
    """Recursively compare two dicts, populating the diff lists."""
    all_keys = sorted(set(old.keys()) | set(new.keys()))

    for key in all_keys:
        path = f"{prefix}.{key}" if prefix else key
        in_old = key in old
        in_new = key in new

        if in_old and not in_new:
            removed.append({"path": path, "value": _serialize_value(old[key])})
            continue

        if in_new and not in_old:
            added.append({"path": path, "value": _serialize_value(new[key])})
            continue

        old_val = old[key]
        new_val = new[key]

        old_is_dict = isinstance(old_val, dict)
        new_is_dict = isinstance(new_val, dict)

        if old_is_dict and new_is_dict:
            _deep_compare(old_val, new_val, prefix=path, added=added,
                          removed=removed, modified=modified, unchanged=unchanged)
        elif _normalize_value(old_val) != _normalize_value(new_val):
            modified.append({
                "path": path,
                "old": _serialize_value(old_val),
                "new": _serialize_value(new_val),
            })
        else:
            unchanged.append(path)
