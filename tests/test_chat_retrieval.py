"""
Tests for Chat Retrieval pipeline:
  - Trust fallback (unvalidated content provisional scoring)
  - Policy decisions for unvalidated vs validated content
  - Query rewriter (unit-level)
  - Model discovery endpoints (schema validation)
"""

import pytest
from app.retrieval.types_retrieve import TrustSignals, SemanticSignal, RetrievalCandidate
from app.retrieval.trust_calculator import compute_ranking_score, compute_policy_trust_score
from app.retrieval.policy import RetrievalPolicy, TrustDecision


# ─────────────────────────────────────────────────────────────────────────────
# Trust Fallback Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTrustFallback:
    """Verify that unvalidated content does not get zeroed out."""

    def test_unvalidated_uses_semantic_score(self):
        """Unvalidated content should rank by semantic * conflict * temporal."""
        signals = TrustSignals(
            tap_trust_score=0.0,
            agentic_validation_score=0.0,
            reasoning_quality_score=0.0,
            conflict_modifier=1.0,
            temporal_decay=1.0,
            trust_state="unvalidated",
        )
        score = compute_ranking_score(0.82, signals)
        assert score > 0.8, f"Expected > 0.8, got {score}"

    def test_validated_zero_trust_still_zeros(self):
        """Validated content with zero trust should remain 0."""
        signals = TrustSignals(
            tap_trust_score=0.0,
            agentic_validation_score=0.0,
            reasoning_quality_score=0.0,
            conflict_modifier=1.0,
            temporal_decay=1.0,
            trust_state="validated",
        )
        score = compute_ranking_score(0.82, signals)
        assert score == 0.0

    def test_unvalidated_with_temporal_decay(self):
        """Unvalidated content should still respect temporal decay."""
        signals = TrustSignals(
            tap_trust_score=0.0,
            agentic_validation_score=0.0,
            reasoning_quality_score=0.0,
            conflict_modifier=1.0,
            temporal_decay=0.5,
            trust_state="unvalidated",
        )
        score = compute_ranking_score(0.8, signals)
        assert 0.35 < score < 0.45, f"Expected ~0.4, got {score}"

    def test_trust_state_defaults_to_validated(self):
        """Backwards compat: default trust_state is 'validated'."""
        signals = TrustSignals(
            tap_trust_score=0.9,
            agentic_validation_score=0.8,
            reasoning_quality_score=0.7,
            conflict_modifier=1.0,
            temporal_decay=1.0,
        )
        assert signals.trust_state == "validated"
        assert not signals.is_unvalidated

    def test_is_unvalidated_property(self):
        signals = TrustSignals(
            tap_trust_score=0.0,
            agentic_validation_score=0.0,
            reasoning_quality_score=0.0,
            conflict_modifier=1.0,
            temporal_decay=1.0,
            trust_state="unvalidated",
        )
        assert signals.is_unvalidated


# ─────────────────────────────────────────────────────────────────────────────
# Policy Decision Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPolicyDecisions:
    """Verify policy correctly classifies unvalidated content."""

    def setup_method(self):
        self.policy = RetrievalPolicy()

    def test_unvalidated_high_semantic_is_provisional(self):
        """High semantic + unvalidated → PROVISIONAL (not REJECTED)."""
        candidate = RetrievalCandidate(
            chunk_id="test-1",
            text="Test content",
            semantic=SemanticSignal(score=0.81),
            trust=TrustSignals(
                tap_trust_score=0.0,
                agentic_validation_score=0.0,
                reasoning_quality_score=0.0,
                conflict_modifier=1.0,
                temporal_decay=1.0,
                trust_state="unvalidated",
            ),
        )
        decision = self.policy.decide(candidate)
        assert decision == TrustDecision.PROVISIONAL

    def test_unvalidated_low_semantic_is_rejected(self):
        """Low semantic + unvalidated → REJECTED."""
        candidate = RetrievalCandidate(
            chunk_id="test-2",
            text="Test content",
            semantic=SemanticSignal(score=0.25),
            trust=TrustSignals(
                tap_trust_score=0.0,
                agentic_validation_score=0.0,
                reasoning_quality_score=0.0,
                conflict_modifier=1.0,
                temporal_decay=1.0,
                trust_state="unvalidated",
            ),
        )
        decision = self.policy.decide(candidate)
        assert decision == TrustDecision.REJECTED

    def test_validated_high_trust_is_trusted(self):
        """High trust + high semantic → TRUSTED."""
        candidate = RetrievalCandidate(
            chunk_id="test-3",
            text="Test content",
            semantic=SemanticSignal(score=0.8),
            trust=TrustSignals(
                tap_trust_score=0.9,
                agentic_validation_score=0.85,
                reasoning_quality_score=0.88,
                conflict_modifier=1.0,
                temporal_decay=0.95,
                trust_state="validated",
            ),
        )
        decision = self.policy.decide(candidate)
        assert decision == TrustDecision.TRUSTED


# ─────────────────────────────────────────────────────────────────────────────
# Query Rewriter Tests (unit)
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryRewriter:
    """Verify query rewriter logic without real LLM calls."""

    def test_single_message_returns_raw(self):
        from app.api.v2.retrieve_chat_api import _rewrite_query, ChatMessage
        msgs = [ChatMessage(role="user", content="What is marketing?")]
        result = _rewrite_query(msgs, llm=None, llm_model="test")
        assert result == "What is marketing?"

    def test_multi_turn_with_no_llm_falls_back(self):
        """When LLM is None, multi-turn should still return last user message."""
        from app.api.v2.retrieve_chat_api import _rewrite_query, ChatMessage

        class FakeLLM:
            def generate(self, *a, **kw):
                raise RuntimeError("LLM unavailable")

        msgs = [
            ChatMessage(role="user", content="Tell me about AI"),
            ChatMessage(role="assistant", content="AI is..."),
            ChatMessage(role="user", content="What are the risks?"),
        ]
        result = _rewrite_query(msgs, llm=FakeLLM(), llm_model="test")
        assert result == "What are the risks?"


# ─────────────────────────────────────────────────────────────────────────────
# Model Discovery Validation
# ─────────────────────────────────────────────────────────────────────────────

class TestModelDiscovery:
    """Validate model discovery data structures."""

    def test_llm_provider_definitions_complete(self):
        from app.api.v2.model_discovery_api import _LLM_PROVIDER_DEFS
        providers = {d["provider"] for d in _LLM_PROVIDER_DEFS}
        assert "openai" in providers
        assert "gemini" in providers
        assert "anthropic" in providers
        assert "groq" in providers
        assert "ollama" in providers

    def test_all_providers_have_required_fields(self):
        from app.api.v2.model_discovery_api import _LLM_PROVIDER_DEFS
        required = {"provider", "display_name", "default_model", "key_env"}
        for d in _LLM_PROVIDER_DEFS:
            missing = required - set(d.keys())
            assert not missing, f"Provider {d.get('provider', '?')} missing: {missing}"
