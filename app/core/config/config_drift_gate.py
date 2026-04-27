"""
================================================================================
Config Drift Gate — Pre-migration safety check.

STATUS: SHADOW-ONLY. Not wired into runtime migration paths yet.

PURPOSE:
    Consumes the structured diff from config_diff_logger.compare_configs()
    and decides whether a migration (re-index, pipeline swap, hot-reload)
    should proceed or be blocked due to critical configuration drift.

USAGE:
    from app.core.config.config_diff_logger import compare_configs
    from app.core.config.config_drift_gate import should_block_migration

    diff = compare_configs(old_cfg, new_cfg, client_id="acme")
    if should_block_migration(diff):
        # abort migration — requires manual review
        ...
================================================================================
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def should_block_migration(diff_result: Dict[str, Any]) -> bool:
    """
    Return True if the config diff contains any CRITICAL-severity change,
    indicating the migration is unsafe without manual review.

    Args:
        diff_result: Output of config_diff_logger.compare_configs().

    Returns:
        True  → block migration (critical drift detected).
        False → safe to proceed.
    """
    if not diff_result or not isinstance(diff_result, dict):
        return False

    differences: List[Dict[str, Any]] = diff_result.get("differences", [])
    client_id = diff_result.get("client_id", "unknown")

    critical_fields = [
        d["field"] for d in differences if d.get("severity") == "CRITICAL"
    ]

    if critical_fields:
        logger.error(
            "MIGRATION BLOCKED DUE TO CONFIG DRIFT — client_id='%s' | "
            "critical_fields=%s | total_changes=%d",
            client_id,
            critical_fields,
            diff_result.get("total_changes", 0),
        )
        return True

    return False
