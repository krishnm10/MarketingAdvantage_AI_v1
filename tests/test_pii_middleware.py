"""
Tests for PII Middleware (app/core/pipeline_nodes/pii_middleware.py)
Covers: regex PII detection, Luhn validation, actions, audit logging, alignment.
"""
from __future__ import annotations

import pytest


def _make_mw(**overrides):
    from app.core.pipeline_nodes.pii_middleware import RegexPIIMiddleware
    defaults = {
        "position": ["pre_embedding", "pre_llm"],
        "action": "REDACT",
        "block_on_severity": "CRITICAL",
        "trust_score_penalty": 0.15,
        "custom_patterns": [],
        "audit_log_enabled": False,
    }
    defaults.update(overrides)
    return RegexPIIMiddleware(config=defaults)


class TestRegexPIIDetection:
    """Test built-in regex PII pattern detection."""

    def test_email_detected(self):
        mw = _make_mw()
        result = mw.scan_text("Contact john@example.com for details")
        assert len(result.entities_found) > 0

    def test_phone_detected(self):
        mw = _make_mw()
        result = mw.scan_text("Call me at 9876543210")
        assert len(result.entities_found) > 0

    def test_aadhaar_detected(self):
        mw = _make_mw()
        result = mw.scan_text("Aadhaar: 1234 5678 9012")
        assert len(result.entities_found) > 0

    def test_pan_detected(self):
        mw = _make_mw()
        result = mw.scan_text("PAN: ABCDE1234F")
        assert len(result.entities_found) > 0

    def test_clean_text_no_pii(self):
        mw = _make_mw()
        result = mw.scan_text("The sky is blue and grass is green")
        assert result.entities_found == []

    def test_redact_action(self):
        mw = _make_mw(action="REDACT")
        result = mw.scan_text("Email: test@example.com")
        assert "test@example.com" not in result.redacted_text
        assert "REDACTED" in result.redacted_text.upper() or "[" in result.redacted_text

    def test_block_on_critical(self):
        mw = _make_mw(action="BLOCK", block_on_severity="LOW")
        result = mw.scan_text("Email: test@example.com")
        assert result.blocked is True

    def test_custom_pattern(self):
        mw = _make_mw(
            custom_patterns=[{"name": "employee_id", "pattern": "EMP-\\d{6}", "severity": "HIGH"}]
        )
        result = mw.scan_text("Employee EMP-123456 is active")
        found = [e.lower() for e in result.entities_found]
        assert any("employee" in e or "custom" in e for e in found) or len(found) > 0


class TestAlignmentStatus:
    """Test PII middleware alignment reporting."""

    def test_aligned_all_positions(self):
        mw = _make_mw(position=["pre_embedding", "pre_llm", "post_llm"])
        assert mw.alignment_status() == "aligned"

    def test_partial_missing_post_llm(self):
        mw = _make_mw(position=["pre_embedding", "pre_llm"])
        assert mw.alignment_status() == "partial"

    def test_partial_only_pre_llm(self):
        mw = _make_mw(position=["pre_llm"])
        assert mw.alignment_status() == "partial"


class TestScanChunks:
    """Test batch chunk scanning."""

    def test_scan_multiple_chunks(self):
        mw = _make_mw(position=["pre_llm"])
        chunks = [
            {"id": "1", "text": "Contact john@example.com", "score": 0.9},
            {"id": "2", "text": "The weather is nice", "score": 0.8},
        ]
        redacted_chunks, result = mw.scan_chunks(chunks, position="pre_llm")
        assert len(redacted_chunks) == 2
        assert "john@example.com" not in redacted_chunks[0].get("text", "")
