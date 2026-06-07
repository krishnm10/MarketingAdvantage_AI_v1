from __future__ import annotations

"""
Phase 3B — Golden evaluation for document-set shadow analysis.

Constraints:
- Observational only: never mutates ranked_results, shadow metadata, or responses.
- Adapter- and domain-neutral: operates purely on file-id sets.
- Deterministic: identical inputs → identical outputs.
- No external calls, no DB writes, no persistent state.
"""

import time
from typing import Any, Dict, List, Optional, Set

from app.observability.metrics import record_docset_golden_eval

try:
    # Optional golden-set integration via tenant-scoped configuration.
    from app.ai.evaluation.golden_set_loader import load_golden_set
except Exception:  # pragma: no cover - defensive; loader is expected to exist
    load_golden_set = None  # type: ignore[assignment]


def _lookup_expected_file_ids(
    *,
    set_ref: Optional[str],
    client_id: str,
    raw_query: str,
    domain: Optional[str],
) -> Optional[Set[str]]:
    """
    Resolve expected file_ids for a given (client, query, domain) from golden sets.

    This implementation is intentionally conservative and tenant-scoped:
    - Uses a per-tenant golden set reference from runtime configuration.
    - Filters by tenant_id and (optionally) domain when present.
    - Matches cases by exact question text.

    Returns a set of expected file_ids, or None when no suitable golden exists.
    All failures are swallowed.
    """
    if not set_ref or load_golden_set is None:
        return None

    try:
        golden = load_golden_set(set_ref)
    except Exception:
        return None

    expected: Optional[Set[str]] = None
    raw_norm = raw_query.strip()

    for case in golden.cases:
        # Tenant-aware: prefer matching the configured tenant when available.
        if golden.tenant_id and golden.tenant_id != client_id:
            continue
        # Domain-aware: when domain is known, prefer matching golden.domain.
        if domain and golden.domain and golden.domain != domain:
            continue
        if case.question.strip() != raw_norm:
            continue
        if case.relevant_file_ids:
            expected = set(case.relevant_file_ids)
            break

    return expected


def evaluate_docset_golden(
    *,
    client_id: str,
    raw_query: str,
    domain: Optional[str],
    actual_file_ids: List[str],
    tenant_runtime: Any,
    trace: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """
    Adapter-neutral golden evaluation over document-set shadow matches.

    Inputs
    ------
    client_id:
        Tenant identifier for golden lookup and metrics labelling.
    raw_query:
        Original user query string (post-security-scan).
    domain:
        Optional coarse domain hint (e.g. "invoice") carried from shadow metadata.
    actual_file_ids:
        File IDs reported by shadow analysis as matched (order-insensitive).
    tenant_runtime:
        RuntimeComponents snapshot with feature flags.
    trace:
        Optional RAG chat trace collector for per-request evaluation events.

    Returns
    -------
    Optional[Dict[str, Any]]:
        Compact JSON-serializable evaluation payload suitable for attaching under
        debug_info["docset_golden_eval"], or None when:
        - Golden eval is disabled for this tenant.
        - No shadow matches are present.
        - No applicable golden is found.
        - Debug flag is off (metrics/trace still emitted when evaluation runs).

    This function must never raise; all failures are swallowed and reported
    only via degraded evaluation metadata and observability hooks.
    """
    if not getattr(tenant_runtime, "enable_docset_golden_eval", False):
        return None

    # No shadow matches → nothing to evaluate; treat as a clean skip.
    if not actual_file_ids:
        return None

    t0 = time.perf_counter()
    degraded = False
    degradation_reason: Optional[str] = None

    actual_ids: Set[str] = {str(fid).strip() for fid in actual_file_ids if str(fid).strip()}
    expected_ids: Set[str] = set()
    overlap: Set[str] = set()
    missing: Set[str] = set()
    unexpected: Set[str] = set()
    precision: Optional[float] = None
    recall: Optional[float] = None

    # Trace hook: evaluation started.
    if trace is not None:
        try:
            trace.add_event(
                "L2",
                "evaluation.started",
                0.0,
                {
                    "shadow_mode": True,
                    "golden_eval": True,
                    "domain": domain,
                },
            )
        except Exception:
            pass

    try:
        # docset_golden_set_ref is tenant-scoped runtime configuration loaded from
        # server-side ClientConfig.features; it is never derived from user/query/
        # request input and must not be overridden per request.
        set_ref = getattr(tenant_runtime, "docset_golden_set_ref", None)
        expected = _lookup_expected_file_ids(
            set_ref=set_ref,
            client_id=client_id,
            raw_query=raw_query,
            domain=domain,
        )
        if expected is not None:
            expected_ids = set(expected)

        overlap = actual_ids & expected_ids
        missing = expected_ids - actual_ids
        unexpected = actual_ids - expected_ids

        # Deterministic precision / recall over sets.
        if actual_ids:
            precision = len(overlap) / float(len(actual_ids))
        else:
            # No predictions: perfect precision only when nothing was expected.
            precision = 1.0 if not expected_ids else 0.0

        if expected_ids:
            recall = len(overlap) / float(len(expected_ids))
        else:
            # Nothing expected: define recall as 1.0 (trivial success).
            recall = 1.0
    except Exception as exc:
        degraded = True
        degradation_reason = str(exc)[:200]
        # Keep all sets empty and metrics nullable in degraded mode.

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    # Observability: metrics (best-effort, never raises).
    try:
        record_docset_golden_eval(
            route="chat",
            domain=domain or "unknown",
            duration_seconds=elapsed_ms / 1000.0,
            precision=precision,
            recall=recall,
            degraded=degraded,
        )
    except Exception:
        pass

    # Trace hook: completed or degraded.
    if trace is not None:
        try:
            payload: Dict[str, Any] = {
                "shadow_mode": True,
                "golden_eval": True,
                "domain": domain,
                "expected_count": len(expected_ids),
                "actual_count": len(actual_ids),
                "overlap_count": len(overlap),
                "precision": precision,
                "recall": recall,
                "degraded": degraded,
            }
            if degradation_reason:
                payload["degradation_reason"] = degradation_reason
            stage = "evaluation.degraded" if degraded else "evaluation.completed"
            trace.add_event(
                "L2",
                stage,
                elapsed_ms,
                payload,
            )
        except Exception:
            pass

    # When debug is disabled, we still ran evaluation + observability but do
    # not attach per-request evaluation metadata.
    if not getattr(tenant_runtime, "enable_docset_golden_debug", False):
        return None

    return {
        "golden_eval": True,
        "domain": domain,
        "expected_file_ids": sorted(expected_ids),
        "actual_file_ids": sorted(actual_ids),
        "overlap": sorted(overlap),
        "missing_matches": sorted(missing),
        "unexpected_matches": sorted(unexpected),
        "precision": precision,
        "recall": recall,
        "degraded": degraded,
        "degradation_reason": degradation_reason,
        "analysis_ms": elapsed_ms,
    }

