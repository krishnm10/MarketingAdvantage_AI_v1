"""
Tests for pipeline node contracts (app/ai/contracts/).
Verifies ABCs can be imported and basic contract shapes.
"""
from __future__ import annotations

import pytest


class TestContractImports:
    """Verify all new contracts are importable from app.ai.contracts."""

    def test_pipeline_node_contract(self):
        from app.ai.contracts import PipelineNodeContract, NodePosition, TrustSignal, NodeResult
        assert PipelineNodeContract is not None
        assert NodePosition.PRE_LLM.value == "pre_llm"

    def test_prompt_node_contract(self):
        from app.ai.contracts import PromptNodeContract, PromptRenderResult, TemplateValidationResult
        assert PromptNodeContract is not None

    def test_output_formatter_contract(self):
        from app.ai.contracts import OutputFormatterContract, FormatResult
        assert OutputFormatterContract is not None

    def test_context_window_contract(self):
        from app.ai.contracts import ContextWindowContract, ContextBudgetResult
        assert ContextWindowContract is not None

    def test_pii_middleware_contract(self):
        from app.ai.contracts import PIIMiddlewareContract, PIIAction, PIISeverity, PIIScanResult
        assert PIIMiddlewareContract is not None
        assert PIIAction.REDACT.value == "REDACT"
        assert PIISeverity.CRITICAL.value == "CRITICAL"


class TestTrustSignal:
    """Test TrustSignal dataclass."""

    def test_creation(self):
        from app.ai.contracts import TrustSignal
        signal = TrustSignal(source_node="pii_middleware", delta=-0.15, reason="PII redacted")
        assert signal.delta == -0.15
        assert signal.source_node == "pii_middleware"


class TestNodeResult:
    """Test NodeResult dataclass."""

    def test_defaults(self):
        from app.ai.contracts import NodeResult
        result = NodeResult()
        assert result.output is None
        assert result.trust_signals == []
        assert result.latency_ms == 0.0
