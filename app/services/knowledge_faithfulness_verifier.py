from __future__ import annotations

"""
Phase 6A — KNOWLEDGE/docset claim-level faithfulness verifier.

Deterministic, citation-aware, refusal-only authority. No LLM extraction in v1.
"""

import json
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.retrieval.answer_integrity import (
    extract_citation_indices,
    extract_value_tokens,
)
from app.retrieval.verification_evidence import (
    VerificationEvidenceBundle,
    passage_by_index,
)

KNOWLEDGE_VERIFIER_VERSION = "v1"

MAX_TRACED_CLAIMS = 50
MAX_EVIDENCE_LINKS_PER_CLAIM = 3
MAX_SNIPPET_CHARS = 200
MAX_TRACE_CHARS = 16000

GATED_CATEGORIES = frozenset(
    {"numeric", "identifier", "date_time", "entity_presence"}
)
SHADOW_CATEGORIES = frozenset(
    {
        "entity_relationship",
        "aggregate",
        "comparative",
        "causal",
        "procedural",
        "speculative",
        "table_row",
    }
)

_RELATIONSHIP_RE = re.compile(
    r"\b(from|to|between|linked to|owed by|owes|associated with|related to)\b",
    re.IGNORECASE,
)
_TABLE_ROW_RE = re.compile(r"\|")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}")
_BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s+")
_ENTITY_STATUS_RE = re.compile(
    r"\b(overdue|paid|pending|disputed|approved|rejected|late fee|past due)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_AGGREGATE_RE = re.compile(
    r"\b(total|sum|average|avg|combined|across|overall)\b",
    re.IGNORECASE,
)
_COMPARATIVE_RE = re.compile(
    r"\b(more than|less than|higher|lower|greater|fewer|compared to|versus|vs\.?)\b",
    re.IGNORECASE,
)
_CAUSAL_RE = re.compile(
    r"\b(because|therefore|thus|due to|leads to|caused by|as a result)\b",
    re.IGNORECASE,
)
_PROCEDURAL_RE = re.compile(
    r"\b(should|must|recommend|you need to|please contact|try to)\b",
    re.IGNORECASE,
)
_SPECULATIVE_RE = re.compile(
    r"\b(might|may|could|likely|possibly|perhaps|unclear|unknown)\b",
    re.IGNORECASE,
)


class ClaimStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INDETERMINATE = "indeterminate"


class AnswerStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INDETERMINATE = "indeterminate"
    ERROR = "error"


@dataclass(frozen=True)
class ExtractedClaim:
    id: str
    text: str
    segment_index: int
    category: str
    gated: bool
    token: Optional[str] = None
    property_token: Optional[str] = None
    cited_sources: Tuple[int, ...] = ()


@dataclass(frozen=True)
class ClaimEvidenceLink:
    claim_id: str
    source_index: int
    chunk_id: str
    file_id: Optional[str]
    snippet: str


@dataclass(frozen=True)
class ClaimVerificationResult:
    claim_id: str
    category: str
    gated: bool
    status: ClaimStatus
    reason_code: str
    detail: str
    cited_sources: Tuple[int, ...]
    evidence_links: Tuple[ClaimEvidenceLink, ...] = ()


@dataclass(frozen=True)
class AnswerVerificationResult:
    status: AnswerStatus
    answer_out: str
    claim_results: Tuple[ClaimVerificationResult, ...]
    duration_ms: float
    indeterminate_policy_applied: bool = False
    alignment_valid: bool = True
    alignment_errors: Tuple[str, ...] = ()
    trace_truncated: bool = False
    error_detail: Optional[str] = None

    def to_debug_dict(
        self,
        *,
        include_claim_trace: bool = True,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "version": KNOWLEDGE_VERIFIER_VERSION,
            "status": self.status.value,
            "alignment_valid": self.alignment_valid,
            "alignment_errors": list(self.alignment_errors),
            "gated_claims_total": sum(1 for c in self.claim_results if c.gated),
            "gated_claims_failed": sum(
                1
                for c in self.claim_results
                if c.gated and c.status == ClaimStatus.FAIL
            ),
            "gated_claims_indeterminate": sum(
                1
                for c in self.claim_results
                if c.gated and c.status == ClaimStatus.INDETERMINATE
            ),
            "shadow_claims_total": sum(1 for c in self.claim_results if not c.gated),
            "indeterminate_policy_applied": self.indeterminate_policy_applied,
            "duration_ms": round(self.duration_ms, 3),
            "trace_truncated": self.trace_truncated,
        }
        if self.error_detail:
            payload["error_detail"] = self.error_detail
        if include_claim_trace:
            payload["claims"] = _build_claim_trace(self.claim_results)
            serialized = json.dumps(payload, ensure_ascii=False)
            if len(serialized) > MAX_TRACE_CHARS:
                payload["claims"] = payload["claims"][: max(1, MAX_TRACED_CLAIMS // 2)]
                payload["trace_truncated"] = True
        return payload


def _truncate_snippet(text: str) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= MAX_SNIPPET_CHARS:
        return cleaned
    return cleaned[: MAX_SNIPPET_CHARS - 1].rstrip() + "…"


def _build_claim_trace(
    claim_results: Sequence[ClaimVerificationResult],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for cr in claim_results[:MAX_TRACED_CLAIMS]:
        links = []
        for link in cr.evidence_links[:MAX_EVIDENCE_LINKS_PER_CLAIM]:
            links.append(
                {
                    "source_index": link.source_index,
                    "chunk_id": link.chunk_id,
                    "file_id": link.file_id,
                    "snippet": link.snippet,
                }
            )
        out.append(
            {
                "claim_id": cr.claim_id,
                "category": cr.category,
                "gated": cr.gated,
                "status": cr.status.value,
                "reason_code": cr.reason_code,
                "detail": cr.detail,
                "cited_sources": list(cr.cited_sources),
                "evidence_links": links,
            }
        )
    return out


def _split_segments(answer: str) -> List[Tuple[int, str, bool, bool]]:
    """Return (segment_index, text, is_bullet, is_table) in stable order."""
    segments: List[Tuple[int, str, bool, bool]] = []
    idx = 0
    for raw_line in (answer or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        is_table = bool(_TABLE_ROW_RE.search(line)) or bool(_TABLE_SEP_RE.match(line))
        is_bullet = bool(_BULLET_RE.match(raw_line)) and not is_table
        if is_bullet or is_table:
            segments.append((idx, line, is_bullet, is_table))
            idx += 1
        else:
            for sent in _SENTENCE_SPLIT_RE.split(line):
                s = sent.strip()
                if s:
                    segments.append((idx, s, False, False))
                    idx += 1
    if not segments and (answer or "").strip():
        segments.append((0, answer.strip(), False, False))
    return segments


def _classify_segment_category(segment_text: str, is_table: bool) -> Optional[str]:
    if is_table:
        return "table_row"
    if _AGGREGATE_RE.search(segment_text):
        return "aggregate"
    if _COMPARATIVE_RE.search(segment_text):
        return "comparative"
    if _CAUSAL_RE.search(segment_text):
        return "causal"
    if _PROCEDURAL_RE.search(segment_text):
        return "procedural"
    if _SPECULATIVE_RE.search(segment_text):
        return "speculative"
    return None


def _is_entity_relationship(segment_text: str, ids: Sequence[str]) -> bool:
    if len(ids) >= 2 and _RELATIONSHIP_RE.search(segment_text):
        return True
    if _RELATIONSHIP_RE.search(segment_text) and len(ids) >= 1:
        # Relational predicate with at least one ID and another capitalized entity hint.
        if re.search(r"\b[A-Z][a-z]+\b", segment_text):
            return True
    return False


def _extract_entity_property(segment_text: str) -> Optional[str]:
    """Return the first unary entity-status property token in segment text."""
    m = _ENTITY_STATUS_RE.search(segment_text or "")
    if not m:
        return None
    return m.group(0).strip().lower()


def extract_claims(answer: str) -> Tuple[ExtractedClaim, ...]:
    """Deterministic claim extraction for v1 gated and shadow categories."""
    claims: List[ExtractedClaim] = []
    claim_idx = 0

    for seg_idx, segment_text, _is_bullet, is_table in _split_segments(answer):
        segment_citations = tuple(extract_citation_indices(segment_text))
        shadow_cat = _classify_segment_category(segment_text, is_table)
        value_tokens = extract_value_tokens(segment_text)

        currencies = sorted(value_tokens.get("currency", set()))
        percentages = sorted(value_tokens.get("percentages", set()))
        dates = sorted(value_tokens.get("dates", set()))
        ids = sorted(value_tokens.get("structured_ids", set()))

        if shadow_cat == "table_row":
            claims.append(
                ExtractedClaim(
                    id=f"c{claim_idx}",
                    text=segment_text,
                    segment_index=seg_idx,
                    category="table_row",
                    gated=False,
                    cited_sources=segment_citations,
                )
            )
            claim_idx += 1
            continue

        if shadow_cat and shadow_cat not in ("table_row",):
            claims.append(
                ExtractedClaim(
                    id=f"c{claim_idx}",
                    text=segment_text,
                    segment_index=seg_idx,
                    category=shadow_cat,
                    gated=False,
                    cited_sources=segment_citations,
                )
            )
            claim_idx += 1

        if _is_entity_relationship(segment_text, ids):
            claims.append(
                ExtractedClaim(
                    id=f"c{claim_idx}",
                    text=segment_text,
                    segment_index=seg_idx,
                    category="entity_relationship",
                    gated=False,
                    cited_sources=segment_citations,
                )
            )
            claim_idx += 1

        for amt in currencies:
            claims.append(
                ExtractedClaim(
                    id=f"c{claim_idx}",
                    text=segment_text,
                    segment_index=seg_idx,
                    category="numeric",
                    gated=True,
                    token=amt,
                    cited_sources=segment_citations,
                )
            )
            claim_idx += 1

        for pct in percentages:
            claims.append(
                ExtractedClaim(
                    id=f"c{claim_idx}",
                    text=segment_text,
                    segment_index=seg_idx,
                    category="numeric",
                    gated=True,
                    token=pct,
                    cited_sources=segment_citations,
                )
            )
            claim_idx += 1

        for dt in dates:
            claims.append(
                ExtractedClaim(
                    id=f"c{claim_idx}",
                    text=segment_text,
                    segment_index=seg_idx,
                    category="date_time",
                    gated=True,
                    token=dt,
                    cited_sources=segment_citations,
                )
            )
            claim_idx += 1

        for sid in ids:
            cat = "identifier"
            gated = True
            if _is_entity_relationship(segment_text, ids):
                continue  # already captured as entity_relationship shadow claim
            claims.append(
                ExtractedClaim(
                    id=f"c{claim_idx}",
                    text=segment_text,
                    segment_index=seg_idx,
                    category=cat,
                    gated=gated,
                    token=sid,
                    cited_sources=segment_citations,
                )
            )
            claim_idx += 1

        entity_property = _extract_entity_property(segment_text)
        if (
            ids
            and entity_property
            and not _is_entity_relationship(segment_text, ids)
        ):
            claims.append(
                ExtractedClaim(
                    id=f"c{claim_idx}",
                    text=segment_text,
                    segment_index=seg_idx,
                    category="entity_presence",
                    gated=True,
                    token=ids[0],
                    property_token=entity_property,
                    cited_sources=segment_citations,
                )
            )
            claim_idx += 1

    return tuple(claims)


def _normalize_token_for_match(category: str, token: str) -> str:
    if category == "numeric":
        if token.endswith("%"):
            return token.strip().lower()
        return token.replace(",", "")
    if category == "identifier":
        return token.upper()
    if category == "date_time":
        return token.strip().lower()
    return token.strip()


def _token_in_text(category: str, token: str, text: str) -> bool:
    norm = _normalize_token_for_match(category, token)
    if category == "numeric":
        if token.endswith("%"):
            hay = text.lower()
            return norm in hay or token.lower() in hay
        flat = text.replace(",", "")
        return norm in flat or token in text
    if category == "identifier":
        return norm in text.upper()
    if category == "date_time":
        return norm in text.lower() or token.lower() in text.lower()
    if category == "entity_presence":
        return token.upper() in text.upper()
    return token in text


def _property_in_text(property_token: str, text: str) -> bool:
    return property_token.strip().lower() in (text or "").lower()


def _verify_gated_claim(
    claim: ExtractedClaim,
    bundle: VerificationEvidenceBundle,
) -> ClaimVerificationResult:
    if not bundle.alignment_valid:
        return ClaimVerificationResult(
            claim_id=claim.id,
            category=claim.category,
            gated=True,
            status=ClaimStatus.INDETERMINATE,
            reason_code="ALIGNMENT_INVALID",
            detail="evidence bundle alignment could not be established",
            cited_sources=claim.cited_sources,
        )

    if not claim.cited_sources:
        return ClaimVerificationResult(
            claim_id=claim.id,
            category=claim.category,
            gated=True,
            status=ClaimStatus.INDETERMINATE,
            reason_code="NO_CITATIONS",
            detail="gated claim has no [Source n] citations",
            cited_sources=(),
        )

    evidence_links: List[ClaimEvidenceLink] = []
    sources_checked: List[int] = []

    for src in claim.cited_sources:
        passage = passage_by_index(bundle, src)
        if passage is None:
            return ClaimVerificationResult(
                claim_id=claim.id,
                category=claim.category,
                gated=True,
                status=ClaimStatus.INDETERMINATE,
                reason_code="CITATION_ALIGNMENT_MISSING",
                detail=f"[Source {src}] not mapped to evidence passage",
                cited_sources=claim.cited_sources,
            )
        sources_checked.append(src)
        evidence_links.append(
            ClaimEvidenceLink(
                claim_id=claim.id,
                source_index=src,
                chunk_id=passage.chunk_id,
                file_id=passage.file_id,
                snippet=_truncate_snippet(passage.text),
            )
        )

    token = claim.token
    if claim.category == "entity_presence" and token:
        prop = claim.property_token
        for src in sources_checked:
            passage = passage_by_index(bundle, src)
            if not passage:
                continue
            entity_ok = _token_in_text("entity_presence", token, passage.text)
            if not entity_ok:
                continue
            if prop and not _property_in_text(prop, passage.text):
                continue
            detail = f"entity {token} found in cited passage"
            if prop:
                detail = f"entity {token} and property {prop!r} found in cited passage"
            return ClaimVerificationResult(
                claim_id=claim.id,
                category=claim.category,
                gated=True,
                status=ClaimStatus.PASS,
                reason_code="ENTITY_PROPERTY_PRESENT",
                detail=detail,
                cited_sources=tuple(sources_checked),
                evidence_links=tuple(evidence_links[:MAX_EVIDENCE_LINKS_PER_CLAIM]),
            )
        if prop:
            for src in sources_checked:
                passage = passage_by_index(bundle, src)
                if (
                    passage
                    and _token_in_text("entity_presence", token, passage.text)
                ):
                    return ClaimVerificationResult(
                        claim_id=claim.id,
                        category=claim.category,
                        gated=True,
                        status=ClaimStatus.FAIL,
                        reason_code="ENTITY_PROPERTY_UNSUPPORTED",
                        detail=(
                            f"entity {token} found but property {prop!r} "
                            "not supported in cited passages"
                        ),
                        cited_sources=tuple(sources_checked),
                        evidence_links=tuple(
                            evidence_links[:MAX_EVIDENCE_LINKS_PER_CLAIM]
                        ),
                    )
        return ClaimVerificationResult(
            claim_id=claim.id,
            category=claim.category,
            gated=True,
            status=ClaimStatus.FAIL,
            reason_code="ENTITY_NOT_FOUND",
            detail=f"entity {token} not found in cited passages",
            cited_sources=tuple(sources_checked),
            evidence_links=tuple(evidence_links[:MAX_EVIDENCE_LINKS_PER_CLAIM]),
        )

    if token:
        for src in sources_checked:
            passage = passage_by_index(bundle, src)
            if passage and _token_in_text(claim.category, token, passage.text):
                return ClaimVerificationResult(
                    claim_id=claim.id,
                    category=claim.category,
                    gated=True,
                    status=ClaimStatus.PASS,
                    reason_code="TOKEN_SUPPORTED",
                    detail=f"token {token!r} found in [Source {src}]",
                    cited_sources=tuple(sources_checked),
                    evidence_links=tuple(evidence_links[:MAX_EVIDENCE_LINKS_PER_CLAIM]),
                )
        return ClaimVerificationResult(
            claim_id=claim.id,
            category=claim.category,
            gated=True,
            status=ClaimStatus.FAIL,
            reason_code="TOKEN_UNSUPPORTED",
            detail=f"token {token!r} not found in cited passages",
            cited_sources=tuple(sources_checked),
            evidence_links=tuple(evidence_links[:MAX_EVIDENCE_LINKS_PER_CLAIM]),
        )

    return ClaimVerificationResult(
        claim_id=claim.id,
        category=claim.category,
        gated=True,
        status=ClaimStatus.INDETERMINATE,
        reason_code="NO_VERIFIABLE_TOKEN",
        detail="gated claim has no extractable verification token",
        cited_sources=claim.cited_sources,
        evidence_links=tuple(evidence_links[:MAX_EVIDENCE_LINKS_PER_CLAIM]),
    )


def _verify_shadow_claim(claim: ExtractedClaim) -> ClaimVerificationResult:
    return ClaimVerificationResult(
        claim_id=claim.id,
        category=claim.category,
        gated=False,
        status=ClaimStatus.INDETERMINATE,
        reason_code="SHADOW_ONLY",
        detail=f"category {claim.category} is shadow-only in v1",
        cited_sources=claim.cited_sources,
    )


def _rollup_answer_status(
    claim_results: Sequence[ClaimVerificationResult],
) -> AnswerStatus:
    gated = [c for c in claim_results if c.gated]
    if not gated:
        return AnswerStatus.PASS
    if any(c.status == ClaimStatus.FAIL for c in gated):
        return AnswerStatus.FAIL
    if any(c.status == ClaimStatus.INDETERMINATE for c in gated):
        return AnswerStatus.INDETERMINATE
    return AnswerStatus.PASS


def verify_knowledge_answer(
    *,
    answer: str,
    bundle: VerificationEvidenceBundle,
    refusal_text: str,
    gate_enabled: bool,
    indeterminate_policy: str = "pass_through",
    emit_debug: bool = True,
) -> AnswerVerificationResult:
    """
    Run Phase 6A KNOWLEDGE verifier.

    gate_enabled=False: observational only; answer_out is always the input answer.
    gate_enabled=True: may replace answer with refusal_text on FAIL (or INDETERMINATE
    when indeterminate_policy='treat_as_fail').
    """
    del emit_debug  # reserved for caller-side debug attachment
    t0 = time.perf_counter()
    try:
        claims = extract_claims(answer)
        claim_results: List[ClaimVerificationResult] = []
        for claim in claims:
            if claim.gated:
                claim_results.append(_verify_gated_claim(claim, bundle))
            else:
                claim_results.append(_verify_shadow_claim(claim))

        answer_status = _rollup_answer_status(claim_results)
        indeterminate_policy_applied = False
        answer_out = answer

        if gate_enabled:
            if answer_status == AnswerStatus.FAIL:
                answer_out = refusal_text
            elif (
                answer_status == AnswerStatus.INDETERMINATE
                and indeterminate_policy == "treat_as_fail"
            ):
                answer_out = refusal_text
                indeterminate_policy_applied = True

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        trace_truncated = len(claim_results) > MAX_TRACED_CLAIMS

        return AnswerVerificationResult(
            status=answer_status,
            answer_out=answer_out,
            claim_results=tuple(claim_results),
            duration_ms=elapsed_ms,
            indeterminate_policy_applied=indeterminate_policy_applied,
            alignment_valid=bundle.alignment_valid,
            alignment_errors=bundle.alignment_errors,
            trace_truncated=trace_truncated,
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return AnswerVerificationResult(
            status=AnswerStatus.INDETERMINATE,
            answer_out=answer,
            claim_results=(),
            duration_ms=elapsed_ms,
            alignment_valid=bundle.alignment_valid,
            alignment_errors=bundle.alignment_errors,
            error_detail=str(exc)[:300],
        )


def external_answer_status(result: AnswerVerificationResult) -> str:
    """Map internal status to trust-metric / user-facing outcome label."""
    if result.status == AnswerStatus.ERROR:
        return AnswerStatus.INDETERMINATE.value
    if result.indeterminate_policy_applied:
        return AnswerStatus.FAIL.value
    return result.status.value


def trust_gate_outcome(result: AnswerVerificationResult) -> str:
    """Outcome label for record_trust_gate when gating is active."""
    return external_answer_status(result)
