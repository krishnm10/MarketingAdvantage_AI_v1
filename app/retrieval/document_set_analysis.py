from __future__ import annotations

"""
Deterministic, debug-only document-set analysis core (Phase 2).

Constraints:
- Operates only on finalized RankedResult outputs.
- Domain-agnostic core; domain-specific logic lives in adapters.
- Never mutates ranked_results or affects retrieval/ranking/answers.
- Emits compact, JSON-serializable debug payloads only.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from app.retrieval.analysis_adapter_registry import get_adapter
from app.retrieval.document_match import DocumentMatch
from app.retrieval.types_retrieve import DomainType, RankedResult


@dataclass(frozen=True)
class AnalysisResult:
    """
    Compact debug-only analysis summary attached to debug_info["docset_analysis"].

    docs_matched semantics (Phase 2):
        Count of documents for which the domain adapter reported at least one
        positive signal (disjunctive / OR). It does NOT require full conjunctive
        satisfaction of every predicate implied by the user query. Per-document
        boolean flags in matches[] carry the detailed breakdown.
    """

    domain: str
    docs_analyzed: int
    docs_matched: int
    matches: List[Dict[str, Any]]
    degraded: bool = False
    degradation_reason: Optional[str] = None

    def to_debug_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "docs_analyzed": self.docs_analyzed,
            "docs_matched": self.docs_matched,
            "matches": list(self.matches),
            "degraded": self.degraded,
            "degradation_reason": self.degradation_reason,
        }


def _group_by_file_id(
    ranked_results: Iterable[RankedResult],
    max_docs: Optional[int],
) -> Mapping[Optional[str], List[RankedResult]]:
    grouped: Dict[Optional[str], List[RankedResult]] = {}

    for r in ranked_results:
        fid = getattr(r, "file_id", None)
        if fid not in grouped:
            if max_docs is not None and len(grouped) >= max_docs:
                # Respect debug-only doc cap deterministically.
                continue
            grouped[fid] = []
        grouped[fid].append(r)

    return grouped


def _is_docset_like_task(task_plan) -> bool:
    """Gate analysis to docset/aggregate-style tasks in Phase 2."""
    if task_plan is None:
        return False
    try:
        t = getattr(task_plan, "task_type", None)
        # Names come from TaskType enum; compare by value string to avoid imports.
        return str(getattr(t, "value", t)) in {
            "docset_filter",
            "aggregate",
            "discrepancy",
            "audit",
            "mixed",
        }
    except Exception:
        return False


def _run_docset_analysis_internal(
    *,
    raw_query: str,
    task_plan,
    ranked_results: List[RankedResult],
    tenant_runtime,
    trace=None,
) -> Optional[Tuple[AnalysisResult, List[DocumentMatch]]]:
    """
    Core analysis routine returning structured AnalysisResult and DocumentMatch list.

    All callers (debug serialization, shadow mode, shaping) should go through this
    function to ensure a single canonical implementation.
    """
    # Execution gate: analysis runs when enable_docset_analysis is on.
    # Debug emission is controlled separately by enable_docset_analysis_debug.
    if not getattr(tenant_runtime, "enable_docset_analysis", False):
        return None

    # Route/task gating.
    if not _is_docset_like_task(task_plan):
        return None

    domain = getattr(task_plan, "domain", None)
    if domain is None:
        return None
    try:
        domain_value = getattr(domain, "value", str(domain))
        domain_enum = DomainType(domain_value)
    except Exception:
        return None

    adapter = get_adapter(domain_enum)
    if adapter is None:
        return None

    # Generic per-domain adapter enablement: expect a boolean flag on the
    # runtime named enable_docset_<domain>_adapter (e.g. invoice).
    flag_attr = f"enable_docset_{domain_enum.value}_adapter"
    if not getattr(tenant_runtime, flag_attr, False):
        return None

    # Optional debug caps from runtime.
    # docset_max_docs_debug == 0 → use module default (20), never "cap at 0".
    raw_max_docs = getattr(tenant_runtime, "docset_max_docs_debug", 0)
    max_docs = 20 if raw_max_docs <= 0 else int(raw_max_docs)

    grouped = _group_by_file_id(ranked_results, max_docs=max_docs)
    if not grouped:
        return None

    if trace is not None:
        try:
            trace.add_event(
                "L2",
                "analysis.started",
                0.0,
                {"domain": domain_enum.value, "docs": len(grouped)},
            )
        except Exception:
            # Tracing must never break analysis.
            pass

    try:
        # For adapters that care about chunk caps, we can pass it via tenant_runtime
        # or interpret it inside the adapter; here we keep adapter signature simple
        # and rely on adapter constructor config (Phase 2 keeps this tight).
        raw_matches = adapter.analyze(
            grouped_chunks=grouped,
            raw_query=raw_query,
            task_plan=task_plan,
            tenant_runtime=tenant_runtime,
        )

        docs_analyzed = len(grouped)
        matches_objects: List[DocumentMatch] = []
        matches_dicts: List[Dict[str, Any]] = []
        for m in raw_matches or []:
            if isinstance(m, DocumentMatch):
                matches_objects.append(m)
                matches_dicts.append(m.to_debug_dict())
            else:
                # Fallback: assume mapping-like; adapter should normally normalize.
                m_dict = dict(m)  # type: ignore[arg-type]
                fid = str(m_dict.get("file_id") or "").strip()
                attrs: Dict[str, Any] = {
                    k: v
                    for k, v in m_dict.items()
                    if k not in ("file_id", "sample_chunk_ids")
                }
                sample_ids = list(m_dict.get("sample_chunk_ids") or [])
                dm = DocumentMatch(file_id=fid, attributes=attrs, sample_chunk_ids=sample_ids)
                matches_objects.append(dm)
                matches_dicts.append(dm.to_debug_dict())

        result = AnalysisResult(
            domain=domain_enum.value,
            docs_analyzed=docs_analyzed,
            docs_matched=len(matches_objects),
            matches=matches_dicts,
        )

        if trace is not None:
            try:
                trace.add_event(
                    "L2",
                    "analysis.completed",
                    0.0,
                    {
                        "domain": domain_enum.value,
                        "docs_analyzed": docs_analyzed,
                        "docs_matched": len(matches_objects),
                    },
                )
            except Exception:
                pass

        return result, matches_objects
    except Exception as exc:
        # Safe degradation: never fail the request or mutate results.
        if trace is not None:
            try:
                trace.add_event(
                    "L2",
                    "analysis.degraded",
                    0.0,
                    {
                        "domain": domain_enum.value,
                        "error": str(exc)[:200],
                    },
                )
            except Exception:
                pass

        degraded = AnalysisResult(
            domain=domain_enum.value,
            docs_analyzed=len(grouped),
            docs_matched=0,
            matches=[],
            degraded=True,
            degradation_reason=str(exc)[:200],
        )
        return degraded, []


def run_docset_analysis(
    *,
    raw_query: str,
    task_plan,
    ranked_results: List[RankedResult],
    tenant_runtime,
    trace=None,
) -> Optional[Dict[str, Any]]:
    """
    Backwards-compatible entry point for Phase 2 debug-only document-set analysis.

    Returns a JSON-serializable dict suitable for debug_info["docset_analysis"],
    or None when analysis is skipped. Never raises.
    """
    result = _run_docset_analysis_internal(
        raw_query=raw_query,
        task_plan=task_plan,
        ranked_results=ranked_results,
        tenant_runtime=tenant_runtime,
        trace=trace,
    )
    if result is None:
        return None
    analysis, _matches = result
    return analysis.to_debug_dict()


def run_docset_analysis_structured(
    *,
    raw_query: str,
    task_plan,
    ranked_results: List[RankedResult],
    tenant_runtime,
    trace=None,
) -> Optional[Tuple[AnalysisResult, List[DocumentMatch]]]:
    """
    Structured entry point for document-set analysis.

    Returns (AnalysisResult, List[DocumentMatch]) or None when analysis is skipped.
    Never raises.
    """
    return _run_docset_analysis_internal(
        raw_query=raw_query,
        task_plan=task_plan,
        ranked_results=ranked_results,
        tenant_runtime=tenant_runtime,
        trace=trace,
    )

