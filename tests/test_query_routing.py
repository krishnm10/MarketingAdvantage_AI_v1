"""
Tests for L0 query routing (rule router, factories, orchestrator).
"""

import pytest

from app.services.query_routing.rule_router import RuleRouter
from app.services.query_routing.types import (
    QueryRoute,
    RouterLayer,
    RouteDecision,
    chitchat_decision,
    knowledge_decision,
)
from app.services.query_routing.orchestrator import QueryOrchestrator


class TestRuleRouter:
    def setup_method(self):
        self.router = RuleRouter()

    def test_hi_is_chitchat(self):
        d = self.router.route("Hi")
        assert d is not None
        assert d.route == QueryRoute.CHITCHAT
        assert not d.retrieval_allowed

    def test_thanks_is_chitchat(self):
        d = self.router.route("thanks")
        assert d is not None
        assert d.route == QueryRoute.CHITCHAT

    def test_meta_help(self):
        d = self.router.route("what can you do?")
        assert d is not None
        assert d.route == QueryRoute.META_HELP
        assert not d.retrieval_allowed

    def test_structured_id(self):
        d = self.router.route("details for INV-1234")
        assert d is not None
        assert d.route == QueryRoute.STRUCTURED
        assert d.retrieval_allowed
        assert d.rerank_allowed

    def test_bare_invoice_clarification_no_history(self):
        d = self.router.route("invoice", has_chat_history=False)
        assert d is not None
        assert d.route == QueryRoute.CLARIFICATION

    def test_knowledge_question_falls_through(self):
        d = self.router.route("What is the due date on this vendor invoice?")
        assert d is None


class TestRouteDecisionFactories:
    def test_knowledge_recall_limit(self):
        d = knowledge_decision(5)
        assert d.max_recall_candidates == 200
        assert d.rerank_allowed
        assert d.rewrite_allowed
        assert d.hyde_allowed

    def test_to_trace_dict_serializes_enums(self):
        d = chitchat_decision()
        td = d.to_trace_dict()
        assert td["route"] == "chitchat"
        assert td["layer_used"] == "rule"
        assert isinstance(td, dict)


class TestQueryOrchestrator:
    def test_rule_short_circuit_without_semantic_embed(self):
        import asyncio

        embed_calls = []

        async def mock_embed(text: str):
            embed_calls.append(text)
            return [1.0, 0.0, 0.0]

        async def _run():
            orch = QueryOrchestrator(embed_fn=mock_embed)
            return await orch.route("Hi", top_k=5)

        decision = asyncio.run(_run())
        assert decision.route == QueryRoute.CHITCHAT
        assert not decision.retrieval_allowed
        assert len(embed_calls) == 0

    def test_fallback_knowledge_when_semantic_not_ready(self):
        import asyncio

        async def mock_embed(text: str):
            return [0.0, 1.0, 0.0]

        async def _run():
            orch = QueryOrchestrator(embed_fn=mock_embed)
            # Semantic prototypes not built — rule miss falls through to KNOWLEDGE default.
            return await orch.route(
                "What is the due date on this vendor invoice?",
                top_k=5,
            )

        decision = asyncio.run(_run())
        assert decision.route == QueryRoute.KNOWLEDGE
        assert decision.retrieval_allowed
        assert decision.reason_code == "default_knowledge"
