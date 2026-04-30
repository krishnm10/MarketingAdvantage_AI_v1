"""
================================================================================
Config Drift Gate — Pre-migration safety check.

STATUS: SHADOW-ONLY. Not wired into runtime migration paths yet.

PURPOSE:
    PURE function. Consumes the structured diff from
    config_diff_logger.compare_configs() and returns a boolean decision
    on whether a migration should be blocked.

    No logging, no scoring, no side effects. Callers are responsible
    for logging via log_config_drift_event().

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

from typing import Any, Dict, List


def should_block_migration(
    diff_result: Dict[str, Any],
    drift_score: int = 0,
) -> bool:
    """
    PURE function — return True if the config diff contains any
    CRITICAL-severity change.

    No logging, no side effects. Callers handle observability via
    log_config_drift_event().

    Args:
        diff_result:  Output of config_diff_logger.compare_configs().
        drift_score:  Pre-computed drift score (accepted for interface
                      consistency; does not affect blocking decision).

    Returns:
        True  → block migration (critical drift detected).
        False → safe to proceed.
    """
    if not diff_result or not isinstance(diff_result, dict):
        return False

    differences: List[Dict[str, Any]] = diff_result.get("differences", [])

    return any(d.get("severity") == "CRITICAL" for d in differences)
