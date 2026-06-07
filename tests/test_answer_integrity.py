"""Tests for answer integrity validators (Phase 5B prerequisite)."""

from __future__ import annotations

from app.retrieval.answer_integrity import (
    build_answer_integrity_snapshot,
    extract_citation_indices,
    extract_value_tokens,
    validate_citation_integrity,
    validate_value_preservation,
)


def test_extract_citation_indices_stable():
    answer = "Fact A [Source 1]. Fact B [Source 2]. Repeat [Source 1]."
    assert extract_citation_indices(answer) == [1, 2]


def test_validate_citation_integrity_valid():
    result = validate_citation_integrity(
        "Overdue [Source 1] and fee [Source 2].",
        chunk_count=3,
    )
    assert result.valid is True
    assert result.invalid_indices == ()


def test_validate_citation_integrity_invalid_index():
    result = validate_citation_integrity(
        "Bad cite [Source 9].",
        chunk_count=2,
    )
    assert result.valid is False
    assert 9 in result.invalid_indices


def test_validate_value_preservation_detects_removed_currency():
    before = "Total $100.00 for INV-1 [Source 1]."
    after = "Total for INV-1 [Source 1]."
    result = validate_value_preservation(before, after)
    assert result.valid is False
    assert "currency" in result.removed_tokens


def test_validate_value_preservation_detects_added_structured_id():
    before = "Invoice INV-100 [Source 1]."
    after = "Invoice INV-100 and INV-200 [Source 1]."
    result = validate_value_preservation(before, after)
    assert result.valid is False
    assert "INV-200" in result.added_tokens.get("structured_ids", ())


def test_extract_value_tokens_categories():
    text = (
        "Policy POL-123 on 2024-01-15: $50.00 tax 12% "
        "for Acme Corporation account 98765."
    )
    tokens = extract_value_tokens(text)
    assert "$50.00" in tokens["currency"]
    assert "12%" in tokens["percentages"]
    assert "2024-01-15" in tokens["dates"]
    assert "POL-123" in tokens["structured_ids"]


def test_build_answer_integrity_snapshot_knowledge_route_limitation():
    snap = build_answer_integrity_snapshot(
        answer="INV-1 $10.00 [Source 1].",
        chunk_count=1,
        pre_mutation_answer="INV-1 $10.00 [Source 1].",
        route="knowledge",
    )
    assert snap.citation_integrity_valid is True
    assert snap.value_preservation_valid is True
    assert "faithfulness_verifier_skipped_for_route" in snap.known_limitations
    assert snap.integrity_severity == "none"
    assert "no_issues_observed" in snap.integrity_findings


def test_build_answer_integrity_snapshot_reports_citation_violation():
    snap = build_answer_integrity_snapshot(
        answer="Bad [Source 5].",
        chunk_count=2,
        route="structured",
    )
    assert snap.citation_integrity_valid is False
    assert snap.citation_violations
    assert snap.integrity_severity == "warning"
    assert "citation_invalid" in snap.integrity_findings


def test_known_limitations_include_non_blocking_runtime():
    snap = build_answer_integrity_snapshot(
        answer="x",
        chunk_count=0,
        route="knowledge",
    )
    assert "runtime_does_not_block_on_integrity_failures" in snap.known_limitations
    assert "no_pre_mutation_baseline" in snap.integrity_findings
    # Lack of baseline is informational-only, not a warning-level integrity failure.
    assert snap.integrity_severity in ("none", "info")
