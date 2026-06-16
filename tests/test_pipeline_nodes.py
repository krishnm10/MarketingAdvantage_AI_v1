"""
Tests for pipeline node infrastructure.
Covers: registry, node set, output formatter, context window manager.
"""
from __future__ import annotations

import pytest


class TestPipelineNodeRegistry:
    """Test pipeline node registration and lookup."""

    def test_registry_exists(self):
        from app.core.plugin_registry import pipeline_node_registry
        assert pipeline_node_registry is not None
        assert pipeline_node_registry.domain == "pipeline_node"

    def test_pii_middleware_registered(self):
        try:
            import app.core.pipeline_nodes.register  # noqa: F401
        except ImportError:
            pytest.skip("Pipeline nodes not yet registered")
        from app.core.plugin_registry import pipeline_node_registry
        assert pipeline_node_registry.has("pii_middleware")

    def test_prompt_node_registered(self):
        try:
            import app.core.pipeline_nodes.register  # noqa: F401
        except ImportError:
            pytest.skip("Pipeline nodes not yet registered")
        from app.core.plugin_registry import pipeline_node_registry
        assert pipeline_node_registry.has("prompt_node")

    def test_build_prompt_node(self):
        try:
            import app.core.pipeline_nodes.register  # noqa: F401
        except ImportError:
            pytest.skip("Pipeline nodes not yet registered")
        from app.core.plugin_registry import pipeline_node_registry
        node = pipeline_node_registry.build("prompt_node", prompt_type="rag_context")
        assert node is not None
        assert node.node_type() == "prompt_node"


class TestPipelineNodeSet:
    """Test PipelineNodeSet container."""

    def test_empty_node_set(self):
        from app.core.pipeline_nodes.node_set import PipelineNodeSet
        ns = PipelineNodeSet()
        assert not ns.has_pii_middleware()
        assert not ns.has_prompt_node()
        assert not ns.has_output_formatter()
        assert not ns.has_context_window()


class TestOutputFormatter:
    """Test output formatter behavior."""

    def _formatter(self, **kwargs):
        from app.core.pipeline_nodes.output_formatter import OutputFormatter
        return OutputFormatter(**kwargs)

    def test_plain_text_passthrough(self):
        fmt = self._formatter()
        result = fmt.format_output("Hello world")
        assert result["formatted_output"] == "Hello world"
        assert result["blocked"] is False

    def test_trust_gate_blocks(self):
        fmt = self._formatter(min_trust_score=0.8, block_on_low_trust=True)
        result = fmt.format_output("Answer", trust_score=0.3)
        assert result["blocked"] is True
        assert "trust" in result["block_reason"].lower()

    def test_trust_gate_passes(self):
        fmt = self._formatter(min_trust_score=0.3, block_on_low_trust=True)
        result = fmt.format_output("Answer", trust_score=0.9)
        assert result["blocked"] is False

    def test_strip_boilerplate(self):
        fmt = self._formatter(strip_boilerplate=True)
        result = fmt.format_output("Sure, here is the answer.\nThe actual content.")
        assert "Sure," not in result["formatted_output"]


class TestContextWindowManager:
    """Test context window budget management."""

    def _manager(self, **kwargs):
        from app.core.pipeline_nodes.context_window_manager import ContextWindowManager
        return ContextWindowManager(**kwargs)

    def test_no_truncation_when_fits(self):
        mgr = self._manager()
        chunks = [{"text": "short text", "score": 0.9}]
        result = mgr.apply_budget(chunks, max_context_tokens=4096)
        assert result["chunks_used"] == 1
        assert result["chunks_dropped"] == 0

    def test_truncation_when_exceeds(self):
        mgr = self._manager(token_budget_context_fraction=0.1)
        chunks = [{"text": "x" * 2000, "score": 0.9} for _ in range(10)]
        result = mgr.apply_budget(chunks, max_context_tokens=1000)
        assert result["chunks_dropped"] > 0
        assert result["truncation_applied"] is True

    def test_least_relevant_strategy(self):
        mgr = self._manager(
            truncation_strategy="least_relevant",
            token_budget_context_fraction=0.3,
        )
        chunks = [
            {"text": "high relevance " * 50, "score": 0.95},
            {"text": "low relevance " * 50, "score": 0.1},
        ]
        result = mgr.apply_budget(chunks, max_context_tokens=1700)
        assert result["chunks_dropped"] >= 1
        remaining_texts = [c["text"] for c in result["trimmed_chunks"]]
        assert any("high relevance" in t for t in remaining_texts)
        assert not any("low relevance" in t for t in remaining_texts)
