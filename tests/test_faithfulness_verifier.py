"""Unit tests for app.services.faithfulness_verifier."""

from __future__ import annotations

from dataclasses import is_dataclass

from app.services.faithfulness_verifier import (
    FAIL_CLOSED_RESPONSE,
    CheckResult,
    VerificationResult,
    VerificationStatus,
    verify_or_refuse,
    verify_structured_answer,
)


def test_frozen_dataclasses() -> None:
    cr = CheckResult("C1", VerificationStatus.PASS, "ok")
    vr = VerificationResult(VerificationStatus.PASS, [cr])
    assert is_dataclass(cr) and cr.__dataclass_fields__
    assert is_dataclass(vr)
    try:
        cr.status = VerificationStatus.FAIL  # type: ignore[misc]
        raised = False
    except AttributeError:
        raised = True
    assert raised


def test_skipped_for_non_structured_route() -> None:
    answer, result = verify_or_refuse(
        answer="Hello!",
        source_texts=[],
        focus_id=None,
        route="chitchat",
    )
    assert answer == "Hello!"
    assert result.status == VerificationStatus.SKIPPED
    assert result.checks == []


def test_c5_fails_on_fabricated_echo() -> None:
    sources = ["INV-1101 vendor Acme total due 100.00"]
    answer = "Tax is 412.50 and total is 4,537.50 for INV-1101."
    result = verify_structured_answer(answer, sources, focus_id="INV-1101")
    assert result.status == VerificationStatus.FAIL
    c5 = next(c for c in result.checks if c.check_id == "C5")
    assert c5.status == VerificationStatus.FAIL


def test_pass_on_grounded_structured_answer() -> None:
    sources = [
        "Invoice INV-1101\nSubtotal $100.00\nTax $10.00\nTotal Due $110.00",
    ]
    answer = "Invoice INV-1101 total due is $110.00."
    result = verify_structured_answer(answer, sources, focus_id="INV-1101")
    assert result.status == VerificationStatus.PASS


def test_c3_warning_only() -> None:
    sources = [
        "INV-1101\nSubtotal $100.00\nTax $10.00\nTotal Due $120.00",
    ]
    answer = (
        "INV-1101\nSubtotal $100.00\nTax $10.00\nTotal Due $120.00"
    )
    result = verify_structured_answer(answer, sources, focus_id="INV-1101")
    assert result.status == VerificationStatus.WARNING
    c3 = next(c for c in result.checks if c.check_id == "C3")
    assert c3.status == VerificationStatus.WARNING


def test_verify_or_refuse_returns_closed_response_on_fail() -> None:
    answer, result = verify_or_refuse(
        answer="Routing 026015079",
        source_texts=["INV-1101 only"],
        focus_id="INV-1101",
        route="structured",
    )
    assert answer == FAIL_CLOSED_RESPONSE
    assert result.status == VerificationStatus.FAIL
