# =============================================
# security_middleware.py — Centralized Security Layer
#
# ARCHITECTURE:
#   All user inputs and document contents MUST pass through this module
#   before reaching embeddings, retrieval, or LLMs.
#
# Pipeline enforcement:
#   User Input → SecurityMiddleware → Processing → Storage → Retrieval → LLM
#
# Hard Failure Conditions (mark system "Security Unsafe" if absent):
#   - Raw user input sent directly to LLM or embedding model
#   - No PII masking before external API calls
#   - No prompt injection detection
#   - System prompts overrideable by user input
# =============================================

from __future__ import annotations

import re
import logging
import os
import uuid as _uuid_module
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PII PATTERNS — India-first with international fallback
#
# Two tiers:
#   _PII_PATTERNS        → regex-only, redact on match (low false-positive risk)
#   _VALIDATED_PII       → regex candidates that MUST pass a validator before
#                           redaction (credit_card → Luhn, iban → checksum)
# ─────────────────────────────────────────────────────────────────────────────

_PII_PATTERNS: Dict[str, re.Pattern] = {
    # Indian identifiers
    "aadhaar":     re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
    "pan":         re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    "phone_in":    re.compile(r"\b(?:\+91[- ]?|0)?[6-9]\d{9}\b"),
    "ifsc":        re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
    "passport_in": re.compile(r"\b[A-Z][0-9]{7}\b"),
    # International
    "email":       re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "ssn_us":      re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "ipv4":        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

# High-risk patterns: regex finds candidates, validator confirms before redaction.
_VALIDATED_PII: Dict[str, re.Pattern] = {
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "iban":        re.compile(r"\b[A-Z]{2}\d{2}[ ]?[\dA-Z]{4}(?:[ ]?[\dA-Z]{4}){1,7}(?:[ ]?[\dA-Z]{1,4})?\b"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Validators — only invoked on regex-matched candidates to cut false positives
# ─────────────────────────────────────────────────────────────────────────────

def is_valid_credit_card(number: str) -> bool:
    """Validate a credit card candidate using the Luhn algorithm."""
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def is_valid_iban(value: str) -> bool:
    """Validate an IBAN candidate using the ISO 13616 mod-97 checksum."""
    compact = value.replace(" ", "").replace("-", "").upper()
    if len(compact) < 15 or len(compact) > 34:
        return False
    if not compact[:2].isalpha() or not compact[2:4].isdigit():
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = ""
    for ch in rearranged:
        if ch.isdigit():
            numeric += ch
        elif ch.isalpha():
            numeric += str(ord(ch) - ord("A") + 10)
        else:
            return False
    try:
        return int(numeric) % 97 == 1
    except (ValueError, OverflowError):
        return False


_PII_VALIDATORS: Dict[str, Any] = {
    "credit_card": is_valid_credit_card,
    "iban":        is_valid_iban,
}

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT INJECTION PATTERNS
# Phrases commonly used in jailbreak / override attempts.
# Patterns are intentionally broad — false positives are low-risk in this context
# because the sanitizer preserves the text while tagging it.
# ─────────────────────────────────────────────────────────────────────────────

_INJECTION_PATTERNS: List[re.Pattern] = [
    # Instruction override attempts
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+(instructions?|prompts?)", re.IGNORECASE),
    re.compile(r"disregard\s+(the\s+)?(previous|prior|above)\s+(instructions?|context)", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?(previous|prior)\s+(instructions?|rules?)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a\s+)?(new|different)\s+(ai|assistant|model)", re.IGNORECASE),
    # System prompt extraction
    re.compile(r"(print|output|reveal|show|display|tell me)\s+(your\s+)?(system\s+prompt|instructions?|training)", re.IGNORECASE),
    re.compile(r"what\s+(are|is)\s+your\s+(system\s+prompt|original\s+instructions?)", re.IGNORECASE),
    # Role override
    re.compile(r"act\s+as\s+(if\s+)?(you\s+are\s+)?(?:an?\s+)?(?:evil|uncensored|jailbroken|dan|developer mode)", re.IGNORECASE),
    re.compile(r"jailbreak|developer\s+mode|DAN\s+mode", re.IGNORECASE),
    # Data exfiltration
    re.compile(r"(base64|hex)\s*(encode|decode)\s*(everything|all|the)", re.IGNORECASE),
    re.compile(r"<\|?(system|user|assistant|im_start|im_end)\|?>", re.IGNORECASE),
]

_INJECTION_THRESHOLD: int = int(os.getenv("SECURITY_INJECTION_THRESHOLD", "1"))
_PII_MIN_LENGTH_TO_SCAN: int = int(os.getenv("SECURITY_PII_MIN_LENGTH", "10"))


@dataclass
class SecurityScanResult:
    """Result of a security scan on a text payload."""
    redacted_text: str
    pii_types_found: List[str] = field(default_factory=list)
    injection_detected: bool = False
    injection_patterns_hit: List[str] = field(default_factory=list)
    is_safe: bool = True
    scan_id: str = field(default_factory=lambda: str(_uuid_module.uuid4()))

    @property
    def has_pii(self) -> bool:
        return bool(self.pii_types_found)


def redact_pii(text: str) -> Tuple[str, List[str]]:
    """
    Scan and redact all known PII patterns.

    Two-tier approach:
      1. Simple patterns (_PII_PATTERNS) → redact every regex match.
      2. Validated patterns (_VALIDATED_PII) → regex finds candidates,
         validator confirms, only confirmed matches are redacted.
         This eliminates false positives for high-ambiguity formats
         like credit card numbers and IBANs.

    Returns:
        (redacted_text, list_of_pii_type_names_found)
    """
    if not text or len(text) < _PII_MIN_LENGTH_TO_SCAN:
        return text, []

    found: List[str] = []

    # Tier 1 (FIRST): validated patterns — must run before simple patterns
    # so that e.g. a valid credit card is redacted before the Aadhaar regex
    # can false-match on its 12-digit substring.
    for pii_type, pattern in _VALIDATED_PII.items():
        validator = _PII_VALIDATORS.get(pii_type)

        def _make_replacer(
            _pii_type: str, _validator: Any, _found: List[str],
        ):
            def _replacer(m: re.Match) -> str:
                raw = m.group(0)
                if _validator is None or _validator(raw):
                    if _pii_type not in _found:
                        _found.append(_pii_type)
                    return f"[{_pii_type.upper()}_REDACTED]"
                return raw
            return _replacer

        text = pattern.sub(_make_replacer(pii_type, validator, found), text)

    # Tier 2: simple regex patterns (low false-positive risk)
    for pii_type, pattern in _PII_PATTERNS.items():
        if pattern.search(text):
            found.append(pii_type)
            text = pattern.sub(f"[{pii_type.upper()}_REDACTED]", text)

    return text, found


def detect_prompt_injection(text: str) -> Tuple[bool, List[str]]:
    """
    Detect prompt injection and jailbreak attempts.

    Returns:
        (is_injected: bool, list of matched pattern names)
    """
    if not text:
        return False, []

    hits: List[str] = []
    for pattern in _INJECTION_PATTERNS:
        m = pattern.search(text)
        if m:
            hits.append(m.group(0)[:80])  # capture first 80 chars — no full leak

    detected = len(hits) >= _INJECTION_THRESHOLD
    return detected, hits


def scan_text(
    text: str,
    *,
    redact_pii_data: bool = True,
    check_injection: bool = True,
    log_pii: bool = True,
    log_injection: bool = True,
    context: Optional[str] = None,
) -> SecurityScanResult:
    """
    Full security scan: PII redaction + injection detection.

    Args:
        text:             Raw input text
        redact_pii_data:  Whether to redact PII in the returned text
        check_injection:  Whether to check for prompt injection patterns
        log_pii:          Whether to log PII findings (never logs raw PII)
        log_injection:    Whether to log injection findings
        context:          Optional context string for log messages (e.g. file_id)

    Returns:
        SecurityScanResult with redacted_text and flags
    """
    ctx = f" [{context}]" if context else ""

    redacted = text
    pii_types: List[str] = []
    injection_detected = False
    injection_hits: List[str] = []

    if redact_pii_data:
        redacted, pii_types = redact_pii(text)
        if pii_types and log_pii:
            logger.warning(
                "PII detected and redacted%s: types=%s scan_length=%d",
                ctx, pii_types, len(text),
            )

    if check_injection:
        injection_detected, injection_hits = detect_prompt_injection(text)
        if injection_detected and log_injection:
            logger.warning(
                "Prompt injection detected%s: hits=%d patterns=%s",
                ctx, len(injection_hits),
                [h[:40] for h in injection_hits],
            )

    is_safe = not injection_detected  # PII-containing text is redacted but still safe to process

    return SecurityScanResult(
        redacted_text=redacted,
        pii_types_found=pii_types,
        injection_detected=injection_detected,
        injection_patterns_hit=injection_hits,
        is_safe=is_safe,
    )


def scan_batch(
    texts: List[str],
    *,
    context: Optional[str] = None,
    **kwargs: Any,
) -> List[SecurityScanResult]:
    """
    Scan a batch of texts for PII and injection in one call.
    Designed for pre-embedding sweeps (redact before calling external APIs).
    """
    return [scan_text(t, context=context, **kwargs) for t in texts]


def mask_api_key(key: str) -> str:
    """
    Return a masked representation of an API key safe for logging.

    Examples:
        "sk-abc123def456" → "sk-abc1***f456"
        "short"           → "***"
    """
    if not key or len(key) <= 10:
        return "***"
    return key[:6] + "***" + key[-4:]


def validate_business_id(business_id: Optional[Any]) -> str:
    """
    Validate and normalize business_id to UUID or safe slug format.

    Prevents env-var injection attacks where a malicious client_id like
    "GLOBAL" could be used to read `MAI_GLOBAL_VECTORDB` from environment.

    Accepts:
        - UUID strings (normalized to lowercase with dashes)
        - UUID objects
        - Safe slugs: [a-z0-9_-], max 64 chars
        - None → "default"

    Rejects:
        - Strings with special chars that could form env-var names
    """
    if business_id is None:
        return "default"

    raw = str(business_id).strip()
    if not raw:
        return "default"

    # Try UUID format — most common case in production
    try:
        return str(_uuid_module.UUID(raw))
    except (ValueError, AttributeError):
        pass

    # Safe slug: alphanumeric + hyphen/underscore only, max 64 chars
    safe = re.sub(r"[^a-z0-9_\-]", "", raw.lower())[:64]
    if not safe:
        logger.warning(
            "business_id '%s' could not be normalized to safe slug — using 'default'",
            raw[:20],
        )
        return "default"

    return safe
