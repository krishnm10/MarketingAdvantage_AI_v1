# =============================================
# cost_tracker.py — Token and Request Cost Tracking
#
# Tracks per-tenant and per-file token usage across embedding, LLM, and
# vision API calls. Provides soft/hard budget enforcement.
#
# DESIGN:
#   - In-memory counters per (tenant, month) key — fast, zero dependencies
#   - Optional PostgreSQL persistence via flush_to_db()
#   - Soft limit: logs a WARNING when exceeded (continues processing)
#   - Hard limit: raises BudgetExceededError (stops processing)
#
# CONFIGURATION (via env vars):
#   COST_EMBED_TOKEN_SOFT_LIMIT   = 5000000  (5M tokens/tenant/month)
#   COST_EMBED_TOKEN_HARD_LIMIT   = 10000000 (10M tokens/tenant/month)
#   COST_LLM_TOKEN_SOFT_LIMIT     = 1000000  (1M tokens/tenant/month)
#   COST_LLM_TOKEN_HARD_LIMIT     = 2000000  (2M tokens/tenant/month)
# =============================================

from __future__ import annotations

import logging
import os
import threading
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# LIMITS
# ─────────────────────────────────────────────────────────────────────────────

_EMBED_SOFT_LIMIT: int = int(os.getenv("COST_EMBED_TOKEN_SOFT_LIMIT", "5000000"))
_EMBED_HARD_LIMIT: int = int(os.getenv("COST_EMBED_TOKEN_HARD_LIMIT", "10000000"))
_LLM_SOFT_LIMIT: int = int(os.getenv("COST_LLM_TOKEN_SOFT_LIMIT", "1000000"))
_LLM_HARD_LIMIT: int = int(os.getenv("COST_LLM_TOKEN_HARD_LIMIT", "2000000"))


class BudgetExceededError(RuntimeError):
    """Raised when a hard token budget limit is breached."""

    def __init__(self, tenant_id: str, call_type: str, used: int, limit: int):
        self.tenant_id = tenant_id
        self.call_type = call_type
        self.used = used
        self.limit = limit
        super().__init__(
            f"Hard token budget exceeded for tenant '{tenant_id}': "
            f"{call_type} used {used:,} / {limit:,} tokens this month"
        )


# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY STORE
# ─────────────────────────────────────────────────────────────────────────────

# Key: (tenant_id, year_month, call_type) → token count
# call_type: "embed" | "llm" | "vision"
_USAGE: Dict[Tuple[str, str, str], int] = defaultdict(int)
_USAGE_LOCK = threading.Lock()


def _month_key() -> str:
    return datetime.utcnow().strftime("%Y-%m")


def record_tokens(
    tenant_id: str,
    call_type: str,
    tokens: int,
    *,
    raise_on_hard_limit: bool = True,
) -> int:
    """
    Record token usage for a tenant/call_type and check limits.

    Args:
        tenant_id:             Business/tenant identifier
        call_type:             "embed", "llm", or "vision"
        tokens:                Number of tokens used in this call
        raise_on_hard_limit:   If True, raises BudgetExceededError on hard limit

    Returns:
        New running total for (tenant_id, current_month, call_type)
    """
    if tokens <= 0:
        return 0

    key = (str(tenant_id), _month_key(), call_type)
    with _USAGE_LOCK:
        _USAGE[key] += tokens
        new_total = _USAGE[key]

    soft_limit, hard_limit = _get_limits(call_type)

    if new_total >= hard_limit:
        logger.error(
            "Hard token budget exceeded: tenant=%s type=%s used=%d limit=%d",
            tenant_id, call_type, new_total, hard_limit,
        )
        if raise_on_hard_limit:
            raise BudgetExceededError(
                tenant_id=tenant_id,
                call_type=call_type,
                used=new_total,
                limit=hard_limit,
            )
    elif new_total >= soft_limit:
        logger.warning(
            "Soft token budget warning: tenant=%s type=%s used=%d soft_limit=%d",
            tenant_id, call_type, new_total, soft_limit,
        )

    return new_total


def get_usage(
    tenant_id: str,
    call_type: Optional[str] = None,
    *,
    month: Optional[str] = None,
) -> Dict[str, int]:
    """
    Get current token usage for a tenant.

    Args:
        tenant_id:  Business/tenant identifier
        call_type:  Filter by "embed", "llm", or "vision" (None = all types)
        month:      "YYYY-MM" string (None = current month)

    Returns:
        Dict of {call_type: token_count}
    """
    target_month = month or _month_key()
    result: Dict[str, int] = {}
    with _USAGE_LOCK:
        for (tid, mon, ctype), count in _USAGE.items():
            if tid != tenant_id or mon != target_month:
                continue
            if call_type and ctype != call_type:
                continue
            result[ctype] = result.get(ctype, 0) + count
    return result


def get_all_usage(*, month: Optional[str] = None) -> Dict[str, Dict[str, int]]:
    """
    Get token usage summary for all tenants.

    Returns:
        {tenant_id: {call_type: token_count}}
    """
    target_month = month or _month_key()
    result: Dict[str, Dict[str, int]] = defaultdict(dict)
    with _USAGE_LOCK:
        for (tid, mon, ctype), count in _USAGE.items():
            if mon != target_month:
                continue
            result[tid][ctype] = result[tid].get(ctype, 0) + count
    return dict(result)


def reset_usage(tenant_id: Optional[str] = None) -> None:
    """
    Reset usage counters. Useful in tests.
    If tenant_id is None, resets ALL counters.
    """
    with _USAGE_LOCK:
        if tenant_id is None:
            _USAGE.clear()
        else:
            keys_to_delete = [k for k in _USAGE if k[0] == tenant_id]
            for k in keys_to_delete:
                del _USAGE[k]


def _get_limits(call_type: str) -> Tuple[int, int]:
    """Return (soft_limit, hard_limit) for the given call type."""
    if call_type == "embed":
        return _EMBED_SOFT_LIMIT, _EMBED_HARD_LIMIT
    if call_type in ("llm", "vision"):
        return _LLM_SOFT_LIMIT, _LLM_HARD_LIMIT
    return _EMBED_SOFT_LIMIT, _EMBED_HARD_LIMIT
