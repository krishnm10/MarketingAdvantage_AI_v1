from __future__ import annotations

import pytest

from app.retrieval.task_classifier import TaskType, classify_task
from app.retrieval.types_retrieve import DomainType
from app.services.query_routing.types import knowledge_decision, structured_decision


class TestTaskClassifierInvoiceDomain:
    def test_invoice_docset_filter_with_late_payment_language(self):
        q = "show me invoices with late payment penalties"
        rd = knowledge_decision(top_k=5)

        plan = classify_task(raw_query=q, route_decision=rd)

        assert plan.domain == DomainType.INVOICE
        assert plan.task_type == TaskType.DOCSET_FILTER
        assert plan.predicates is not None
        assert "docset_filter" in plan.predicates.kinds

    def test_invoice_tax_threshold_parsing(self):
        q = "find invoices with tax above 10%"
        rd = knowledge_decision(top_k=5)

        plan = classify_task(raw_query=q, route_decision=rd)

        assert plan.domain == DomainType.INVOICE
        assert plan.task_type == TaskType.DOCSET_FILTER
        assert plan.predicates is not None
        kinds = set(plan.predicates.kinds)
        assert "docset_filter" in kinds
        # Numeric predicate should be parsed deterministically.
        assert plan.predicates.numeric_thresholds.get("tax_gt_percent") == pytest.approx(10.0)


class TestTaskClassifierGenericKnowledge:
    def test_generic_knowledge_default_open_qa(self):
        q = "how does hybrid search work in this platform?"
        rd = knowledge_decision(top_k=5)

        plan = classify_task(raw_query=q, route_decision=rd)

        assert plan.domain == DomainType.GENERIC
        assert plan.task_type == TaskType.OPEN_QA
        # No strong predicates expected.
        assert plan.predicates is None or not plan.predicates.kinds


class TestTaskClassifierStructuredAndOtherRoutes:
    def test_structured_route_maps_to_single_qa(self):
        q = "INV-1001"
        rd = structured_decision(top_k=1)

        plan = classify_task(raw_query=q, route_decision=rd)

        assert plan.task_type == TaskType.SINGLE_QA
        assert plan.domain in (DomainType.INVOICE, DomainType.GENERIC)
        # Structured lookups are treated as single-record queries in Phase 1.
        assert plan.confidence >= 0.0

