from __future__ import annotations

"""
Phase 5A — Deterministic document-set summary (observability only).

Summary Failure Types
1. Retrieval / upstream analysis unavailable
   - summary builder not called
2. Analysis degraded
   - deterministic degraded message returned
3. Summary construction failure
   - exception during deterministic summary generation
4. Future Phase 5B LLM polish failure
   - reserved for future use

Golden Evaluation Scope
- Goldens validate document selection behavior only
- Goldens validate eligibility / recall / precision only
- Goldens do NOT validate summary quality
- Goldens do NOT validate wording quality
- Goldens do NOT validate final answer quality

Phase 5A Boundary
- In Phase 5A, deterministic summaries are observability artifacts only.
- They must not be used as answer grounding.
- They must not replace raw chunk context.
- They must not be sent to the LLM.
- Raw chunk context remains authoritative for prompt construction, citations,
  grounding checks, and verify_or_refuse.

SUMMARY AUTHORITY RULE
Deterministic summaries are non-authoritative observability artifacts.
AnalysisResult and DocumentMatch remain the source of truth.
If any discrepancy is observed between AnalysisResult, DocumentMatch, and the
deterministic summary, the summary is incorrect and must never influence
retrieval, result shaping, trust gating, verification, or answer generation.
"""

from typing import Any, Dict, List, Optional, Tuple

from app.retrieval.document_set_analysis import AnalysisResult

SUMMARY_FORMAT_VERSION = "v1"

_MAX_SNIPPETS_PER_DOC = 3
_MAX_TOTAL_EVIDENCE = 10
_MAX_SNIPPET_LENGTH = 200

_RESERVED_MATCH_KEYS = frozenset(
    {
        "file_id",
        "sample_chunk_ids",
        "chunk_id",
        "score",
        "evidence_snippets",
        "aggregates",
        "evidence_unavailable",
    }
)

_DEGRADED_MESSAGE = (
    "Document-set analysis completed in degraded mode. "
    "Deterministic summary is limited."
)
_NO_MATCH_MESSAGE = "No documents matched the document-set criteria."


def _truncate_snippet(text: str, max_len: int = _MAX_SNIPPET_LENGTH) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip() + "…"


def _match_score(match: Dict[str, Any]) -> float:
    raw = match.get("score")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _ordered_matches(matches: List[Dict[str, Any]]) -> List[Tuple[int, Dict[str, Any]]]:
    indexed = list(enumerate(matches or []))
    indexed.sort(key=lambda item: (-_match_score(item[1]), item[0]))
    return indexed


def _extract_aggregates(analysis: AnalysisResult) -> Dict[str, Any]:
    raw = getattr(analysis, "aggregates", None)
    if isinstance(raw, dict):
        return dict(raw)
    for match in analysis.matches or []:
        if not isinstance(match, dict):
            continue
        nested = match.get("aggregates")
        if isinstance(nested, dict):
            return dict(nested)
    return {}


def _format_signal_lines(match: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for key, value in sorted((match or {}).items()):
        if key in _RESERVED_MATCH_KEYS:
            continue
        if isinstance(value, bool):
            lines.append(f"{key}: {'yes' if value else 'no'}")
        elif value is not None and not isinstance(value, (dict, list)):
            lines.append(f"{key}: {value}")
    return lines


def _collect_evidence_lines(match: Dict[str, Any]) -> Tuple[List[str], bool]:
    snippets = match.get("evidence_snippets") or []
    if not isinstance(snippets, list) or not snippets:
        return [], True

    lines: List[str] = []
    for snippet in snippets[:_MAX_SNIPPETS_PER_DOC]:
        if not isinstance(snippet, str):
            continue
        truncated = _truncate_snippet(snippet)
        if truncated:
            lines.append(truncated)
    if not lines:
        return [], True
    return lines, False


def build_docset_summary(analysis: AnalysisResult) -> str:
    """
    Build a deterministic, human-readable summary from structured analysis only.
    """
    if analysis.degraded:
        reason = (analysis.degradation_reason or "").strip()
        if reason:
            return f"{_DEGRADED_MESSAGE} ({reason[:200]})"
        return _DEGRADED_MESSAGE

    matched_count = int(analysis.docs_matched or 0)
    if matched_count <= 0:
        return _NO_MATCH_MESSAGE

    sections: List[str] = [f"Matched documents: {matched_count}"]
    aggregates = _extract_aggregates(analysis)
    if aggregates:
        agg_parts = [f"{key}={aggregates[key]}" for key in sorted(aggregates)]
        sections.append("Aggregates: " + ", ".join(agg_parts))

    total_evidence = 0
    doc_index = 0
    for _orig_idx, match in _ordered_matches(list(analysis.matches or [])):
        if not isinstance(match, dict):
            continue
        doc_index += 1
        doc_lines = [f"Document {doc_index}:"]

        signals = _format_signal_lines(match)
        if signals:
            doc_lines.extend(f"  - {line}" for line in signals)

        evidence_lines, unavailable = _collect_evidence_lines(match)
        remaining_budget = _MAX_TOTAL_EVIDENCE - total_evidence
        if remaining_budget <= 0:
            break

        if unavailable:
            doc_lines.append("  - evidence: unavailable")
        elif evidence_lines:
            for line in evidence_lines[:remaining_budget]:
                doc_lines.append(f"  - evidence: {line}")
                total_evidence += 1
                if total_evidence >= _MAX_TOTAL_EVIDENCE:
                    break
        else:
            doc_lines.append("  - evidence: unavailable")

        sections.append("\n".join(doc_lines))
        if total_evidence >= _MAX_TOTAL_EVIDENCE:
            break

    return "\n\n".join(sections)


def build_docset_summary_metadata(analysis: AnalysisResult) -> Dict[str, Any]:
    """
    Build compact deterministic metadata for debug/trace usage.
    """
    aggregates = _extract_aggregates(analysis)
    evidence_count = 0
    for match in analysis.matches or []:
        if not isinstance(match, dict):
            continue
        snippets, unavailable = _collect_evidence_lines(match)
        if unavailable:
            continue
        evidence_count += len(snippets)

    return {
        "summary_format_version": SUMMARY_FORMAT_VERSION,
        "domain": analysis.domain,
        "docs_analyzed": int(analysis.docs_analyzed or 0),
        "matched_count": int(analysis.docs_matched or 0),
        "degraded": bool(analysis.degraded),
        "aggregate_keys": sorted(aggregates.keys()),
        "aggregate_count": len(aggregates),
        "evidence_item_count": min(evidence_count, _MAX_TOTAL_EVIDENCE),
        "summary_used_llm": False,
        "golden_scope": "document_selection_only",
    }


def validate_llm_polish_output(
    *,
    deterministic_summary: str,
    polished_summary: str,
    analysis: AnalysisResult,
) -> bool:
    """
    Reserved for Phase 5B. Do not activate in Phase 5A.
    """
    raise NotImplementedError("validate_llm_polish_output is reserved for Phase 5B")
