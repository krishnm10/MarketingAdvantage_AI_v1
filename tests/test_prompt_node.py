"""
Tests for PromptNode (app/core/pipeline_nodes/prompt_node.py)
Covers: template rendering, placeholder detection, PII pre-save validation,
        built-in templates, custom templates.
"""
from __future__ import annotations

import pytest


class TestPromptNodeRendering:
    """Test prompt template rendering."""

    def _node(self, **kwargs):
        from app.core.pipeline_nodes.prompt_node import PromptNode
        return PromptNode(**kwargs)

    def test_default_rag_context(self):
        node = self._node()
        result = node.render(query="What is AI?", context="AI is artificial intelligence.")
        assert "What is AI?" in result["rendered_prompt"]
        assert "AI is artificial intelligence" in result["rendered_prompt"]

    def test_cot_template(self):
        node = self._node(prompt_type="cot")
        result = node.render(query="Explain RAG", context="RAG combines retrieval and generation.")
        assert "step by step" in result["rendered_prompt"].lower()

    def test_custom_template(self):
        node = self._node(
            prompt_type="custom",
            template="Q: {query}\nA: Based on {context}, the answer is:",
        )
        result = node.render(query="test?", context="some context")
        assert "Q: test?" in result["rendered_prompt"]
        assert "some context" in result["rendered_prompt"]

    def test_unresolved_placeholder_warned(self):
        node = self._node(
            prompt_type="custom",
            template="Hello {unknown_var}!",
        )
        result = node.render(query="q", context="c")
        assert any("unresolved" in w.lower() for w in result.get("warnings", []))

    def test_token_count_estimated(self):
        node = self._node(max_tokens_warning=10)
        result = node.render(query="q" * 100, context="c" * 100)
        assert result["token_count"] > 0

    def test_high_token_warning(self):
        node = self._node(max_tokens_warning=5)
        result = node.render(query="q" * 100, context="c" * 100)
        assert any("token" in w.lower() for w in result.get("warnings", []))


class TestPromptNodeValidation:
    """Test template validation."""

    def _node(self, **kwargs):
        from app.core.pipeline_nodes.prompt_node import PromptNode
        return PromptNode(**kwargs)

    def test_valid_template(self):
        node = self._node()
        result = node.validate_template(
            "Answer {query} using {context}",
            {"query": "user.input", "context": "retriever.output"},
        )
        assert result["valid"] is True
        assert "query" in result["placeholders_found"]
        assert "context" in result["placeholders_found"]

    def test_empty_template_invalid(self):
        node = self._node()
        result = node.validate_template("", {})
        assert result["valid"] is False

    def test_pii_in_template_warned(self):
        node = self._node()
        result = node.validate_template(
            "Contact john@example.com for {query}",
            {"query": "user.input"},
        )
        assert any("pii" in w.lower() for w in result.get("warnings", []))

    def test_unknown_prompt_type_error(self):
        node = self._node(prompt_type="nonexistent")
        errors = node.validate_config()
        assert len(errors) > 0
