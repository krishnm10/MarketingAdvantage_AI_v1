from __future__ import annotations

from typing import Any, Dict


def compute_drift_score(diff: Dict[str, Any]) -> int:
    """
    Compute a numeric drift score from a diff summary.

    The diff dict is expected to contain:
      - critical_count
      - warning_count
      - info_count

    Missing fields default to 0. Returns an integer score only.
    This is a pure function with no logging or side effects.
    """
    critical_count_raw = diff.get("critical_count", 0)
    warning_count_raw = diff.get("warning_count", 0)
    info_count_raw = diff.get("info_count", 0)

    critical_count = int(critical_count_raw or 0)
    warning_count = int(warning_count_raw or 0)
    info_count = int(info_count_raw or 0)

    return (critical_count * 10) + (warning_count * 3) + info_count

