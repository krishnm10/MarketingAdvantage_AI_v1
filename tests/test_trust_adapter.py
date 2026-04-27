"""
Tests for TrustAdapter (app/core/trust_adapter.py)
Covers: base scoring, PII penalty, reranker confidence blending, bounds.
"""
from __future__ import annotations

import pytest


class TestTrustAdapter:
    """Test trust scoring adapter."""

    def _adapter(self):
        from app.core.trust_adapter import TrustAdapter
        return TrustAdapter()

    def test_empty_chunks_zero(self):
        adapter = self._adapter()
        assert adapter.calculate([]) == 0.0

    def test_normal_chunks_positive(self):
        adapter = self._adapter()
        chunks = [
            {"score": 0.9, "text": "relevant"},
            {"score": 0.8, "text": "also relevant"},
        ]
        score = adapter.calculate(chunks)
        assert 0.0 < score <= 1.0

    def test_pii_penalty_applied(self):
        adapter = self._adapter()
        chunks = [{"score": 0.5, "text": "data"}]
        normal = adapter.calculate(chunks)
        penalized = adapter.calculate(chunks, pii_redacted=True, pii_penalty=0.15)
        assert penalized < normal

    def test_pii_penalty_cannot_go_negative(self):
        adapter = self._adapter()
        chunks = [{"score": 0.1, "text": "data"}]
        score = adapter.calculate(chunks, pii_redacted=True, pii_penalty=0.99)
        assert score >= 0.0

    def test_reranker_confidence_blending(self):
        adapter = self._adapter()
        chunks = [{"score": 0.5, "text": "data"}]
        base = adapter.calculate(chunks)
        boosted = adapter.calculate(chunks, reranker_confidence=1.0)
        assert boosted >= base

    def test_score_capped_at_one(self):
        adapter = self._adapter()
        chunks = [{"score": 0.99, "text": "data"}]
        score = adapter.calculate(chunks, reranker_confidence=1.0)
        assert score <= 1.0


class TestTrustAdapterFallback:
    """Test fallback behavior when trust calculator module is unavailable."""

    def test_uses_average_scores(self):
        from app.core.trust_adapter import TrustAdapter
        adapter = TrustAdapter()
        chunks = [
            {"score": 0.8, "text": "a"},
            {"score": 0.6, "text": "b"},
        ]
        score = adapter.calculate(chunks)
        assert 0.5 < score < 1.0
