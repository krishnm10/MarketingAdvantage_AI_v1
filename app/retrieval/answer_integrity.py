from __future__ import annotations

"""
Answer integrity validators — Phase 5B prerequisite hardening (infrastructure only).

These helpers are pure, deterministic, and have no I/O. They do NOT alter runtime
answer behavior in the chat retrieval path unless explicitly invoked by future
Phase 5B code or observability-only debug emission.

KNOWN LIMITATIONS — NOT GUARANTEED at runtime today
----------------------------------------------------
- Faithfulness verification is SKIPPED for non-structured routes
  (see app.services.faithfulness_verifier.verify_or_refuse).
- The chat handler does NOT block or refuse answers based on citation or value
  validation results from this module.
- Post-verification semantic mutation is forbidden by contract but not enforced
  by runtime guards (see ANSWER_MUTATION_CONTRACT in retrieve_chat_api.py).
- _maybe_focus_fallback_answer and _build_focus_fallback_answer may replace the
  entire LLM answer with deterministic excerpts (documented exception to value
  and citation preservation relative to the pre-fallback answer).

ANSWER MUTATION CONTRACT (summary)
----------------------------------
1. All semantic answer mutations must occur before verify_or_refuse, or the
   mutated answer must be re-verified afterward.
2. Post-verification semantic mutation is forbidden unless verification is rerun.
3. Pre-verification mutators must preserve citation markers and value-critical
   tokens unless explicitly documented otherwise (focus fallback is the
   documented exception).
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Citation markers produced by chat_retrieve context assembly ([Source n]).
_CITATION_RE = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)

# Business-critical token patterns (deterministic extraction for comparison).
_CURRENCY_RE = re.compile(r"\$[\d,]+(?:\.\d{2})?")
_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?%")
_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
_STRUCTURED_ID_RE = re.compile(r"\b[A-Z]{2,}-[A-Z0-9]{3,}\b")
_COUNT_RE = re.compile(r"\b\d+\b")
# Heuristic named-entity tokens: capitalized multi-word sequences (observability only).
_NAMED_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")

# Categories exposed for value-preservation comparison.
VALUE_CATEGORIES = (
    "currency",
    "percentages",
    "dates",
    "structured_ids",
    "counts",
    "named_entities",
)


@dataclass(frozen=True)
class CitationIntegrityResult:
    valid: bool
    citation_indices: Tuple[int, ...]
    invalid_indices: Tuple[int, ...]
    violations: Tuple[str, ...]


@dataclass(frozen=True)
class ValuePreservationResult:
    valid: bool
    removed_tokens: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    added_tokens: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    violations: Tuple[str, ...] = ()


@dataclass(frozen=True)
class AnswerIntegritySnapshot:
    """Compact observability payload for debug/trace (non-blocking)."""

    citation_integrity_valid: bool
    citation_violations: Tuple[str, ...]
    value_preservation_valid: Optional[bool]
    value_violations: Tuple[str, ...]
    known_limitations: Tuple[str, ...]
    chunk_count: int
    citation_indices: Tuple[int, ...]
    # Derived Phase 5B observability fields (shadow-only; no runtime gating).
    integrity_findings: Tuple[str, ...]
    integrity_severity: str

    def to_debug_dict(self) -> Dict[str, Any]:
        return {
            "citation_integrity_valid": self.citation_integrity_valid,
            "citation_violations": list(self.citation_violations),
            "value_preservation_valid": self.value_preservation_valid,
            "value_violations": list(self.value_violations),
            "known_limitations": list(self.known_limitations),
            "chunk_count": self.chunk_count,
            "citation_indices": list(self.citation_indices),
            "integrity_findings": list(self.integrity_findings),
            "integrity_severity": self.integrity_severity,
        }


def extract_citation_indices(answer: str) -> List[int]:
    """Return 1-based [Source n] indices found in answer text (stable order)."""
    if not answer:
        return []
    seen: Set[int] = set()
    out: List[int] = []
    for m in _CITATION_RE.finditer(answer):
        idx = int(m.group(1))
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def validate_citation_integrity(
    answer: str,
    *,
    chunk_count: int,
) -> CitationIntegrityResult:
    """
    Validate that every [Source n] in answer maps to 1 <= n <= chunk_count.

    Does not validate that cited content matches chunk text (future Phase 5B).
    """
    indices = extract_citation_indices(answer)
    if chunk_count <= 0:
        if indices:
            return CitationIntegrityResult(
                valid=False,
                citation_indices=tuple(indices),
                invalid_indices=tuple(indices),
                violations=("no_chunks_available_but_citations_present",),
            )
        return CitationIntegrityResult(
            valid=True,
            citation_indices=(),
            invalid_indices=(),
            violations=(),
        )

    invalid = [i for i in indices if i < 1 or i > chunk_count]
    violations: List[str] = []
    for i in invalid:
        violations.append(f"citation_index_out_of_range:{i}:max={chunk_count}")
    return CitationIntegrityResult(
        valid=not invalid,
        citation_indices=tuple(indices),
        invalid_indices=tuple(invalid),
        violations=tuple(violations),
    )


def extract_value_tokens(text: str) -> Dict[str, Set[str]]:
    """Extract business-critical tokens by category from text."""
    blob = text or ""
    upper = blob.upper()
    structured = set(_STRUCTURED_ID_RE.findall(upper))
    return {
        "currency": set(_CURRENCY_RE.findall(blob)),
        "percentages": set(_PERCENT_RE.findall(blob)),
        "dates": set(_DATE_RE.findall(blob)),
        "structured_ids": structured,
        "counts": set(_COUNT_RE.findall(blob)),
        "named_entities": set(_NAMED_ENTITY_RE.findall(blob)),
    }


def validate_value_preservation(
    before: str,
    after: str,
    *,
    categories: Optional[Sequence[str]] = None,
    allow_removals: bool = False,
) -> ValuePreservationResult:
    """
    Compare value-critical tokens between two answer strings.

    By default (allow_removals=False), any token present in ``before`` but absent
    in ``after`` is a violation. Tokens newly introduced in ``after`` are always
    reported as violations unless they also appear in ``before``.

    Set allow_removals=True only for mutators documented to remove content
    (e.g. dedupe dropping duplicate lines).
    """
    cats = tuple(categories or VALUE_CATEGORIES)
    before_tokens = extract_value_tokens(before)
    after_tokens = extract_value_tokens(after)

    removed: Dict[str, Tuple[str, ...]] = {}
    added: Dict[str, Tuple[str, ...]] = {}
    violations: List[str] = []

    for cat in cats:
        b_set = before_tokens.get(cat, set())
        a_set = after_tokens.get(cat, set())
        rem = sorted(b_set - a_set)
        add = sorted(a_set - b_set)
        if rem:
            removed[cat] = tuple(rem)
            if not allow_removals:
                violations.append(f"removed_{cat}:{','.join(rem[:5])}")
        if add:
            added[cat] = tuple(add)
            violations.append(f"added_{cat}:{','.join(add[:5])}")

    return ValuePreservationResult(
        valid=not violations,
        removed_tokens=removed,
        added_tokens=added,
        violations=tuple(violations),
    )


def build_answer_integrity_snapshot(
    *,
    answer: str,
    chunk_count: int,
    pre_mutation_answer: Optional[str] = None,
    route: Optional[str] = None,
) -> AnswerIntegritySnapshot:
    """
    Build a compact observability snapshot (never raises).

    value_preservation_valid is None when pre_mutation_answer is not provided.
    """
    citation = validate_citation_integrity(answer, chunk_count=chunk_count)
    value_valid: Optional[bool] = None
    value_violations: Tuple[str, ...] = ()

    if pre_mutation_answer is not None:
        value = validate_value_preservation(pre_mutation_answer, answer)
        value_valid = value.valid
        value_violations = value.violations

    limitations: List[str] = [
        "runtime_does_not_block_on_integrity_failures",
        "post_verification_mutation_not_runtime_enforced",
    ]
    if route and route.lower() != "structured":
        limitations.append("faithfulness_verifier_skipped_for_route")

    # Derive high-level integrity findings / severity (observability-only).
    findings: List[str] = []
    severity: str = "none"

    if not citation.valid:
        findings.append("citation_invalid")
        if citation.violations:
            # Include first few violation codes for quick triage.
            findings.extend(list(citation.violations[:3]))
        severity = "warning"

    if value_valid is False:
        findings.append("value_drift_detected")
        if value_violations:
            findings.extend(list(value_violations[:3]))
        severity = "warning"

    # No baseline for value comparison: explicit informational-only signal.
    if value_valid is None:
        findings.append("no_pre_mutation_baseline")
        if severity == "none":
            severity = "info"

    if not findings:
        findings.append("no_issues_observed")

    return AnswerIntegritySnapshot(
        citation_integrity_valid=citation.valid,
        citation_violations=citation.violations,
        value_preservation_valid=value_valid,
        value_violations=value_violations,
        known_limitations=tuple(limitations),
        chunk_count=chunk_count,
        citation_indices=citation.citation_indices,
        integrity_findings=tuple(findings),
        integrity_severity=severity,
    )
