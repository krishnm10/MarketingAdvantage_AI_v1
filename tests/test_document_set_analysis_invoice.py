from __future__ import annotations

import json
from typing import List

import pytest

from app.retrieval.document_set_analysis import run_docset_analysis, _group_by_file_id
from app.retrieval.domain_adapters.invoice_adapter import InvoiceDocumentSetAdapter
from app.retrieval.types_retrieve import RankedResult


class DummyTaskPlan:
    def __init__(self, task_type: str = "docset_filter", domain: str = "invoice") -> None:
        self.task_type = type("TT", (), {"value": task_type})()
        self.domain = type("DT", (), {"value": domain})()


class DummyRuntime:
    def __init__(
        self,
        *,
        enable_docset_analysis: bool = True,
        enable_docset_analysis_debug: bool = True,
        enable_docset_invoice_adapter: bool = True,
        docset_max_docs_debug: int = 20,
        docset_max_chunks_per_doc_debug: int = 3,
    ) -> None:
        self.enable_docset_analysis = enable_docset_analysis
        self.enable_docset_analysis_debug = enable_docset_analysis_debug
        self.enable_docset_invoice_adapter = enable_docset_invoice_adapter
        self.docset_max_docs_debug = docset_max_docs_debug
        self.docset_max_chunks_per_doc_debug = docset_max_chunks_per_doc_debug


def _ranked(chunk_id: str, file_id: str, text: str) -> RankedResult:
    return RankedResult(
        chunk_id=chunk_id,
        text=text,
        score=0.9,
        explanation={},
        trust_decision=None,
        file_id=file_id,
    )


def test_group_by_file_id_deterministic():
    r1 = _ranked("c1", "f1", "first")
    r2 = _ranked("c2", "f2", "second")
    r3 = _ranked("c3", "f1", "third")

    grouped = _group_by_file_id([r2, r1, r3], max_docs=None)

    # Order of keys should be first-seen by file_id ("f2", then "f1").
    assert list(grouped.keys()) == ["f2", "f1"]
    assert [c.chunk_id for c in grouped["f2"]] == ["c2"]
    assert [c.chunk_id for c in grouped["f1"]] == ["c1", "c3"]


def test_invoice_adapter_detects_overdue_and_late_fee():
    adapter = InvoiceDocumentSetAdapter(max_chunks_per_doc=2)
    ranked: List[RankedResult] = [
        _ranked("c1", "inv-1", "This invoice is overdue and has a late fee of $10."),
        _ranked("c2", "inv-1", "Additional details."),
    ]
    grouped = {"inv-1": ranked}

    matches = adapter.analyze(
        grouped_chunks=grouped,
        raw_query="show me overdue invoices with late fees",
        task_plan=DummyTaskPlan(),
        tenant_runtime=DummyRuntime(),
    )

    assert len(matches) == 1
    payload = matches[0].to_debug_dict()
    assert payload["file_id"] == "inv-1"
    assert payload["overdue"] is True
    assert payload["has_late_fee"] is True
    assert payload["tax_gt_threshold"] is False
    assert payload["sample_chunk_ids"] == ["c1", "c2"]


def test_invoice_adapter_detects_tax_above_threshold():
    adapter = InvoiceDocumentSetAdapter(max_chunks_per_doc=1)
    ranked: List[RankedResult] = [
        _ranked("c1", "inv-2", "Tax 12% is applied on this invoice."),
    ]
    grouped = {"inv-2": ranked}

    matches = adapter.analyze(
        grouped_chunks=grouped,
        raw_query="find invoices with tax above 10%",
        task_plan=DummyTaskPlan(),
        tenant_runtime=DummyRuntime(),
    )

    assert len(matches) == 1
    payload = matches[0].to_debug_dict()
    assert payload["tax_gt_threshold"] is True
    assert payload["sample_chunk_ids"] == ["c1"]


def test_non_matching_docs_do_not_produce_false_positives():
    adapter = InvoiceDocumentSetAdapter(max_chunks_per_doc=2)
    ranked: List[RankedResult] = [
        _ranked("c1", "inv-3", "This invoice is fully paid."),
        _ranked("c2", "inv-4", "Standard invoice with no penalties."),
    ]
    grouped = {"inv-3": [ranked[0]], "inv-4": [ranked[1]]}

    matches = adapter.analyze(
        grouped_chunks=grouped,
        raw_query="show me invoices",
        task_plan=DummyTaskPlan(),
        tenant_runtime=DummyRuntime(),
    )

    assert matches == []


def test_run_docset_analysis_json_serializable_and_safe_empty():
    runtime = DummyRuntime()
    task_plan = DummyTaskPlan()
    ranked: List[RankedResult] = []

    result = run_docset_analysis(
        raw_query="show me overdue invoices",
        task_plan=task_plan,
        ranked_results=ranked,
        tenant_runtime=runtime,
        trace=None,
    )

    # Empty input should simply skip analysis (None).
    assert result is None

    # Now with a minimal matching document.
    ranked = [
        _ranked("c1", "inv-5", "This invoice is overdue with a late payment penalty."),
    ]
    result = run_docset_analysis(
        raw_query="show me overdue invoices",
        task_plan=task_plan,
        ranked_results=ranked,
        tenant_runtime=runtime,
        trace=None,
    )

    assert isinstance(result, dict)
    json.dumps(result)  # must be JSON-serializable
    assert result.get("domain") == "invoice"
    assert result.get("docs_analyzed") >= 1


def test_run_docset_analysis_malformed_input_degrades_safely(monkeypatch):
    runtime = DummyRuntime()
    task_plan = DummyTaskPlan()
    ranked = [
        _ranked("c1", "inv-6", "Overdue with late fee."),
    ]

    # Force adapter.analyze to raise.
    from app.retrieval import document_set_analysis as dsa

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(dsa, "get_adapter", lambda domain: type("X", (), {"analyze": _boom})())

    result = run_docset_analysis(
        raw_query="show me overdue invoices",
        task_plan=task_plan,
        ranked_results=ranked,
        tenant_runtime=runtime,
        trace=None,
    )

    assert isinstance(result, dict)
    assert result.get("degraded") is True
    assert "boom" in (result.get("degradation_reason") or "")

