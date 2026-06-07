from __future__ import annotations

"""
Phase 3A — Shadow document-set analysis (observational only).

Constraints:
- Consumes the request-scoped structured AnalysisResult / DocumentMatch objects.
- Never re-runs analysis or adapter logic.
- Never mutates ranked_results, task_plan, tenant runtime, or shared analysis.
- Emits compact shadow metadata and observability only when enabled.
- Swallows all internal failures without affecting live behavior.
"""

import time
from typing import Any, Dict, List, Optional

from app.observability.metrics import record_docset_shadow
from app.retrieval.document_set_analysis import AnalysisResult
from app.retrieval.document_match import DocumentMatch
from app.retrieval.types_retrieve import RankedResult


def run_shadow_docset_analysis(
    *,
    raw_query: str,
    task_plan: Any,
    ranked_results: List[RankedResult],
    tenant_runtime: Any,
    trace: Optional[Any] = None,
    analysis: Optional[AnalysisResult] = None,
    matches: Optional[List[DocumentMatch]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Execute document-set shadow metadata from a request-scoped analysis result.

    Returns a compact, JSON-serializable metadata dict suitable for attaching
    under debug_info (e.g. debug_info["docset_shadow"]), or None when:
    - Shadow mode is disabled for the tenant.
    - Preconditions are not met (no task_plan, no ranked_results, etc.).
    - No structured analysis was produced for this request.

    This function must never raise; all exceptions are swallowed and reported
    only via degraded shadow metadata and observability hooks.
    """
    # Tenant-level gate: shadow mode must be explicitly enabled.
    if not getattr(tenant_runtime, "enable_docset_shadow_mode", False):
        return None

    # Shadow mode is only meaningful when we have both a task plan and results.
    if task_plan is None or not ranked_results:
        return None

    # No second analysis pass: consume the shared structured result only.
    if analysis is None:
        return None

    t0 = time.perf_counter()
    degraded = False
    degradation_reason: Optional[str] = None
    domain: Optional[str] = None
    docs_analyzed = 0
    docs_matched = 0
    matched_file_ids: List[str] = []

    # Trace hook: shadow analysis started.
    if trace is not None:
        try:
            trace.add_event(
                "L2",
                "analysis.started",
                0.0,
                {
                    "shadow_mode": True,
                },
            )
        except Exception:
            # Tracing must never affect behavior.
            pass

    try:
        domain = str(analysis.domain or "")
        docs_analyzed = int(analysis.docs_analyzed or 0)
        docs_matched = int(analysis.docs_matched or 0)
        for m in matches or []:
            fid = getattr(m, "file_id", None)
            if fid:
                matched_file_ids.append(str(fid))
        degraded = bool(analysis.degraded)
        if analysis.degradation_reason:
            degradation_reason = str(analysis.degradation_reason)[:200]
    except Exception as exc:
        degraded = True
        degradation_reason = str(exc)[:200]

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    # Observability: metrics (best-effort, never raises).
    try:
        record_docset_shadow(
            route="chat",
            domain=domain or "unknown",
            duration_seconds=elapsed_ms / 1000.0,
            docs_matched=docs_matched,
            degraded=degraded,
        )
    except Exception:
        pass

    # Trace hook: completed or degraded.
    if trace is not None:
        try:
            stage = "analysis.degraded" if degraded else "analysis.completed"
            payload: Dict[str, Any] = {
                "shadow_mode": True,
                "domain": domain,
                "docs_analyzed": docs_analyzed,
                "docs_matched": docs_matched,
                "matched_file_ids": matched_file_ids[:20],
                "degraded": degraded,
            }
            if degradation_reason:
                payload["degradation_reason"] = degradation_reason
            trace.add_event(
                "L2",
                stage,
                elapsed_ms,
                payload,
            )
        except Exception:
            pass

    # When shadow debug is disabled, we still ran observability but we do not
    # attach per-request metadata.
    if not getattr(tenant_runtime, "enable_docset_shadow_debug", False):
        return None

    return {
        "shadow_mode": True,
        "domain": domain,
        "docs_analyzed": docs_analyzed,
        "docs_matched": docs_matched,
        "matched_file_ids": matched_file_ids,
        "degraded": degraded,
        "degradation_reason": degradation_reason,
        "analysis_ms": elapsed_ms,
    }
