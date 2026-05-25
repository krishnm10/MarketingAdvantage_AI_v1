"""
Deterministic post-generation faithfulness checks for STRUCTURED RAG answers.

Regex-only — no LLM, no network. Target runtime <20ms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

FAIL_CLOSED_RESPONSE = (
    "I cannot verify this answer against the retrieved invoice passages. "
    "Please try again with a more specific query or check the source document."
)

_KNOWN_FABRICATED_AMOUNTS = frozenset({"412.50", "4537.50", "4,537.50"})
_KNOWN_FABRICATED_ROUTING = frozenset({"026015079"})
_KNOWN_FABRICATED_VENDORS = frozenset(
    {"corporate legal services", "corporate legal"}
)

_CURRENCY_RE = re.compile(r"\$[\d,]+(?:\.\d{2})?")
_ROUTING_RE = re.compile(r"\b\d{9}\b")
_ARITHMETIC_SUBTOTAL_RE = re.compile(
    r"subtotal[^\d$]*(?:\$?\s*)?([\d,]+(?:\.\d{2})?)",
    re.IGNORECASE,
)
_ARITHMETIC_TAX_RE = re.compile(
    r"tax(?:\s*\([^)]*\))?[^\d$]*(?:\$?\s*)?([\d,]+(?:\.\d{2})?)",
    re.IGNORECASE,
)
_ARITHMETIC_TOTAL_RE = re.compile(
    r"total(?:\s+due)?[^\d$]*(?:\$?\s*)?([\d,]+(?:\.\d{2})?)",
    re.IGNORECASE,
)


class VerificationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    status: VerificationStatus
    detail: str


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    checks: List[CheckResult]
    fail_reason: Optional[str] = None


def _normalize_focus(value: Optional[str]) -> str:
    if not value:
        return ""
    return value.replace("-", "").replace(" ", "").lower()


def _combined_sources(source_texts: List[str]) -> str:
    return "\n".join(t for t in source_texts if t)


def _parse_amount(value: str) -> Optional[float]:
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def _check_c1_currency_grounding(answer: str, sources: str) -> CheckResult:
    amounts = _CURRENCY_RE.findall(answer)
    if not amounts:
        return CheckResult(
            "C1",
            VerificationStatus.PASS,
            "no dollar amounts in answer",
        )
    sources_flat = sources.replace(",", "")
    missing = [
        amt
        for amt in amounts
        if amt not in sources and amt.replace(",", "") not in sources_flat
    ]
    if missing:
        return CheckResult(
            "C1",
            VerificationStatus.FAIL,
            f"ungrounded dollar amount(s): {', '.join(missing[:5])}",
        )
    return CheckResult(
        "C1",
        VerificationStatus.PASS,
        "all dollar amounts grounded in sources",
    )


def _check_c2_routing_grounding(answer: str, sources: str) -> CheckResult:
    routing_nums = _ROUTING_RE.findall(answer)
    if not routing_nums:
        return CheckResult(
            "C2",
            VerificationStatus.PASS,
            "no 9-digit routing numbers in answer",
        )
    missing = [num for num in routing_nums if num not in sources]
    if missing:
        return CheckResult(
            "C2",
            VerificationStatus.FAIL,
            f"ungrounded routing number(s): {', '.join(missing[:5])}",
        )
    return CheckResult(
        "C2",
        VerificationStatus.PASS,
        "all routing numbers grounded in sources",
    )


def _check_c3_arithmetic_consistency(answer: str) -> CheckResult:
    sub_m = _ARITHMETIC_SUBTOTAL_RE.search(answer)
    tax_m = _ARITHMETIC_TAX_RE.search(answer)
    total_m = _ARITHMETIC_TOTAL_RE.search(answer)
    if not (sub_m and tax_m and total_m):
        return CheckResult(
            "C3",
            VerificationStatus.PASS,
            "insufficient subtotal/tax/total fields for arithmetic check",
        )
    sub = _parse_amount(sub_m.group(1))
    tax = _parse_amount(tax_m.group(1))
    total = _parse_amount(total_m.group(1))
    if sub is None or tax is None or total is None:
        return CheckResult(
            "C3",
            VerificationStatus.PASS,
            "could not parse subtotal/tax/total for arithmetic check",
        )
    if abs((sub + tax) - total) > 0.01:
        return CheckResult(
            "C3",
            VerificationStatus.WARNING,
            f"subtotal ({sub}) + tax ({tax}) != total ({total})",
        )
    return CheckResult(
        "C3",
        VerificationStatus.PASS,
        "subtotal + tax equals total",
    )


def _check_c4_focus_entity_in_source(
    focus_id: Optional[str],
    source_texts: List[str],
) -> CheckResult:
    if not focus_id:
        return CheckResult(
            "C4",
            VerificationStatus.PASS,
            "no focus_id provided",
        )
    needle = _normalize_focus(focus_id)
    if not needle:
        return CheckResult(
            "C4",
            VerificationStatus.PASS,
            "empty focus_id after normalization",
        )
    for text in source_texts:
        if needle in _normalize_focus(text):
            return CheckResult(
                "C4",
                VerificationStatus.PASS,
                f"focus_id {focus_id!r} found in source passages",
            )
    return CheckResult(
        "C4",
        VerificationStatus.FAIL,
        f"focus_id {focus_id!r} not found in any source passage",
    )


def _check_c5_few_shot_echo_detection(answer: str) -> CheckResult:
    answer_lower = answer.lower()
    answer_flat = answer_lower.replace(",", "")
    hits: List[str] = []
    for amt in _KNOWN_FABRICATED_AMOUNTS:
        if amt.lower() in answer_lower or amt.replace(",", "") in answer_flat:
            hits.append(amt)
    for routing in _KNOWN_FABRICATED_ROUTING:
        if routing in answer:
            hits.append(routing)
    for vendor in _KNOWN_FABRICATED_VENDORS:
        if vendor in answer_lower:
            hits.append(vendor)
    if hits:
        return CheckResult(
            "C5",
            VerificationStatus.FAIL,
            f"known fabricated value(s) detected: {', '.join(hits[:5])}",
        )
    return CheckResult(
        "C5",
        VerificationStatus.PASS,
        "no known fabricated few-shot values detected",
    )


def verify_structured_answer(
    answer: str,
    source_texts: List[str],
    focus_id: Optional[str],
) -> VerificationResult:
    """Run C1–C5 on a STRUCTURED-route answer."""
    text = answer or ""
    sources = _combined_sources(source_texts)
    checks = [
        _check_c1_currency_grounding(text, sources),
        _check_c2_routing_grounding(text, sources),
        _check_c3_arithmetic_consistency(text),
        _check_c4_focus_entity_in_source(focus_id, source_texts),
        _check_c5_few_shot_echo_detection(text),
    ]

    fail_checks = [
        c
        for c in checks
        if c.status == VerificationStatus.FAIL
        and c.check_id in {"C1", "C2", "C4", "C5"}
    ]
    if fail_checks:
        fail_reason = "; ".join(f"{c.check_id}: {c.detail}" for c in fail_checks)
        return VerificationResult(
            status=VerificationStatus.FAIL,
            checks=checks,
            fail_reason=fail_reason,
        )

    warning_checks = [c for c in checks if c.status == VerificationStatus.WARNING]
    if warning_checks:
        return VerificationResult(
            status=VerificationStatus.WARNING,
            checks=checks,
            fail_reason=None,
        )

    return VerificationResult(
        status=VerificationStatus.PASS,
        checks=checks,
        fail_reason=None,
    )


def verify_or_refuse(
    answer: str,
    source_texts: List[str],
    focus_id: Optional[str],
    route: Optional[str],
) -> Tuple[str, VerificationResult]:
    """STRUCTURED-only gate; returns safe refusal text on fail-closed checks."""
    if route != "structured":
        return answer, VerificationResult(
            status=VerificationStatus.SKIPPED,
            checks=[],
            fail_reason=None,
        )

    result = verify_structured_answer(answer, source_texts, focus_id)
    if result.status == VerificationStatus.FAIL:
        return FAIL_CLOSED_RESPONSE, result
    return answer, result
