"""
Golden-set loader, metric math, and optional live integration tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.ai.evaluation.golden_set_loader import load_golden_set
from app.ai.evaluation.rag_evaluator import (
    GoldenExample,
    RAGEvaluator,
    RetrievalResult,
    _precision_at_k,
    _recall_at_k,
    _reciprocal_rank,
)
from app.retrieval.golden_evaluation import evaluate_docset_golden


GOLDEN_ROOT = Path(__file__).resolve().parent / "golden_sets"


class TestGoldenSetLoader:
    def test_load_fixture(self) -> None:
        gs = load_golden_set("fixtures/mini_smoke.json", root=GOLDEN_ROOT)
        assert gs.set_id == "mini_smoke_fixture"
        assert len(gs.cases) == 1
        assert gs.cases[0].question_id == "q1"

    def test_load_vaidyanad_invoice(self) -> None:
        gs = load_golden_set("invoice/vaidyanad_inv_1101.json", root=GOLDEN_ROOT)
        assert gs.tenant_id == "vaidyanad"
        assert len(gs.cases) >= 2

    def test_to_golden_examples_skips_empty_when_flagged(self) -> None:
        gs = load_golden_set("invoice/vaidyanad_inv_1101.json", root=GOLDEN_ROOT)
        examples = gs.to_golden_examples()
        assert examples == []


class TestRetrievalMetrics:
    def test_precision_recall_mrr(self) -> None:
        ranked = ["a", "b", "c", "d"]
        relevant = {"b", "d"}
        assert _precision_at_k(ranked, relevant, 3) == pytest.approx(1 / 3)
        assert _recall_at_k(ranked, relevant, 3) == pytest.approx(0.5)
        assert _reciprocal_rank(ranked, relevant) == pytest.approx(0.5)

    def test_run_retrieval_eval_fixture(self) -> None:
        gs = load_golden_set("fixtures/mini_smoke.json", root=GOLDEN_ROOT)
        case = gs.cases[0]
        evaluator = RAGEvaluator(judge_model=None, k_values=[1, 3])
        golden = GoldenExample(
            question_id=case.question_id,
            question=case.question,
            relevant_chunk_ids=set(case.relevant_chunk_ids),
        )
        result = evaluator.run_retrieval_eval(
            [golden],
            [
                RetrievalResult(
                    question_id="q1",
                    ranked_ids=["chunk-a", "noise", "chunk-b"],
                    scores=[0.9, 0.8, 0.7],
                )
            ],
        )
        assert result["n_questions"] == 1
        assert result["aggregate"]["mrr"] == 1.0
        assert result["aggregate"]["recall@3"] == 1.0


class TestRagApiRetrievedChunks:
    def test_build_retrieved_chunks_prefers_reranked(self) -> None:
        from types import SimpleNamespace

        from app.api.v2.rag_api import _build_retrieved_chunks_for_eval

        result = SimpleNamespace(
            reranked=True,
            reranked_chunks=[
                {"id": "r1", "text": "alpha", "score": 0.95},
                {"id": "r2", "text": "beta", "score": 0.80},
            ],
            retrieved_chunks=[
                {"id": "v9", "text": "other", "score": 0.99},
            ],
        )
        items = _build_retrieved_chunks_for_eval(result)
        assert len(items) == 2
        assert items[0].chunk_id == "r1"
        assert items[0].stage == "reranked"


class TestRouteCheck:
    def test_extracts_route_from_route_decision_dict(self) -> None:
        from app.ai.evaluation.golden_set_loader import GoldenCaseSpec
        from app.ai.evaluation.golden_set_runner import GoldenSetRunner

        case = GoldenCaseSpec(
            question_id="q1",
            question="INV-1101",
            relevant_chunk_ids=set(),
            relevant_file_ids=set(),
            expected_answer=None,
            expected_answer_substrings=[],
            forbidden_substrings=[],
            route_hint="structured",
            metadata={},
        )
        debug = {
            "route_decision": {
                "route": "structured",
                "layer_used": "rule",
            }
        }
        result = GoldenSetRunner._check_route(case, debug)
        assert result.passed is True
        assert result.actual == "structured"


class TestChunkIdNormalize:
    def test_rag_response_shape(self) -> None:
        from app.ai.evaluation.chunk_id_normalize import ranked_lists_from_rag_response

        body = {
            "retrieved_chunks": [
                {"rank": 1, "chunk_id": "vec-1", "score": 0.9, "stage": "reranked"},
                {"rank": 2, "chunk_id": "vec-2", "score": 0.8, "stage": "reranked"},
            ]
        }
        ids, scores, stage = ranked_lists_from_rag_response(body)
        assert ids == ["vec-1", "vec-2"]
        assert stage == "reranked"
        assert len(scores) == 2


class TestDocsetGoldenEvaluation:
    def test_order_independent_precision_recall(self, monkeypatch) -> None:
        """Golden evaluator must be set-based, not order-based."""
        from types import SimpleNamespace
        from app.retrieval import golden_evaluation as ge

        def _fake_lookup_expected_file_ids(
            *, set_ref: str | None, client_id: str, raw_query: str, domain: str | None
        ):
            return {"A", "B", "C"}

        monkeypatch.setattr(ge, "_lookup_expected_file_ids", _fake_lookup_expected_file_ids)

        runtime = SimpleNamespace(
            enable_docset_golden_eval=True,
            enable_docset_golden_debug=True,
            docset_golden_set_ref="ignored",
        )

        result = evaluate_docset_golden(
            client_id="tenant-1",
            raw_query="any question",
            domain="invoice",
            actual_file_ids=["C", "B", "A"],
            tenant_runtime=runtime,
            trace=None,
        )

        assert isinstance(result, dict)
        assert set(result["expected_file_ids"]) == {"A", "B", "C"}
        assert set(result["actual_file_ids"]) == {"A", "B", "C"}
        assert set(result["overlap"]) == {"A", "B", "C"}
        assert result["missing_matches"] == []
        assert result["unexpected_matches"] == []
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)

    def test_tenant_isolation_never_mixes_golden_sets(self, monkeypatch) -> None:
        """Golden evaluator must not apply one tenant's golden to another tenant."""
        from types import SimpleNamespace
        from pathlib import Path

        from app.ai.evaluation.golden_set_loader import GoldenCaseSpec, GoldenSetFile
        from app.retrieval import golden_evaluation as ge

        case = GoldenCaseSpec(
            question_id="q1",
            question="Q",
            relevant_chunk_ids=set(),
            relevant_file_ids={"X"},
            expected_answer=None,
            expected_answer_substrings=[],
            forbidden_substrings=[],
            route_hint="knowledge",
            metadata={},
        )
        golden = GoldenSetFile(
            schema_version="1.0",
            set_id="test",
            domain="invoice",
            tenant_id="tenant-A",
            description="",
            corpus_fingerprint="",
            k_values=[5],
            cases=[case],
            source_path=Path("dummy"),
        )

        def _fake_load_golden_set(set_ref: str):
            return golden

        monkeypatch.setattr(ge, "load_golden_set", _fake_load_golden_set)

        runtime = SimpleNamespace(
            enable_docset_golden_eval=True,
            enable_docset_golden_debug=True,
            docset_golden_set_ref="any",
        )

        # Tenant A: golden applies.
        res_a = evaluate_docset_golden(
            client_id="tenant-A",
            raw_query="Q",
            domain="invoice",
            actual_file_ids=["X"],
            tenant_runtime=runtime,
            trace=None,
        )
        assert isinstance(res_a, dict)
        assert set(res_a["expected_file_ids"]) == {"X"}
        assert set(res_a["actual_file_ids"]) == {"X"}
        assert set(res_a["overlap"]) == {"X"}
        assert res_a["degraded"] is False

        # Tenant B: same golden file must not apply (tenant_id mismatch).
        res_b = evaluate_docset_golden(
            client_id="tenant-B",
            raw_query="Q",
            domain="invoice",
            actual_file_ids=["X"],
            tenant_runtime=runtime,
            trace=None,
        )
        assert isinstance(res_b, dict)
        # No expected_ids for tenant-B; evaluator treats this as "no expectations".
        assert res_b["expected_file_ids"] == []
        assert set(res_b["actual_file_ids"]) == {"X"}
        assert res_b["overlap"] == []
        assert res_b["degraded"] is False

    def test_docset_golden_set_ref_is_tenant_config(self, monkeypatch) -> None:
        """
        Golden evaluator must read docset_golden_set_ref from tenant runtime config,
        not from user or request input.
        """
        from types import SimpleNamespace
        from app.retrieval import golden_evaluation as ge

        captured: dict = {}

        def _fake_lookup_expected_file_ids(
            *, set_ref: str | None, client_id: str, raw_query: str, domain: str | None
        ):
            captured["set_ref"] = set_ref
            captured["client_id"] = client_id
            captured["raw_query"] = raw_query
            captured["domain"] = domain
            return set()  # empty but valid expectation set

        monkeypatch.setattr(ge, "_lookup_expected_file_ids", _fake_lookup_expected_file_ids)

        runtime = SimpleNamespace(
            enable_docset_golden_eval=True,
            enable_docset_golden_debug=True,
            docset_golden_set_ref="tenant-config-ref",
        )

        _ = evaluate_docset_golden(
            client_id="tenant-123",
            raw_query="any user question",
            domain="invoice",
            actual_file_ids=["doc-1"],
            tenant_runtime=runtime,
            trace=None,
        )

        assert captured["set_ref"] == "tenant-config-ref"
        assert captured["client_id"] == "tenant-123"
        assert captured["raw_query"] == "any user question"
        assert captured["domain"] == "invoice"


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("GOLDEN_SET_LIVE") != "1",
    reason="Set GOLDEN_SET_LIVE=1 and run uvicorn to enable live golden-set tests",
)
class TestGoldenSetLive:
    def test_vaidyanad_substring_smoke(self) -> None:
        from app.ai.evaluation.golden_set_runner import GoldenSetRunner

        runner = GoldenSetRunner()
        report = runner.run(
            "invoice/vaidyanad_inv_1101.json",
            paths=["chat"],
            run_faithfulness=False,
        )
        assert report.aggregate["n_cases"] >= 2
        for case in report.case_results:
            assert not case.errors, case.errors
