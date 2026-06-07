"""
Live golden-set evaluation harness for chat and RAG pipeline paths.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set

import httpx

from app.ai.evaluation.chunk_id_normalize import (
    build_context_text_from_chunks,
    ranked_lists_from_chat_response,
    ranked_lists_from_rag_response,
)
from app.ai.evaluation.golden_set_loader import GoldenCaseSpec, GoldenSetFile, load_golden_set
from app.ai.evaluation.rag_evaluator import GoldenExample, RAGEvaluator, RetrievalResult

logger = logging.getLogger(__name__)

EvalPath = Literal["chat", "rag"]


@dataclass
class SubstringCheckResult:
    question_id: str
    passed: bool
    missing_expected: List[str]
    forbidden_found: List[str]


@dataclass
class RouteCheckResult:
    question_id: str
    passed: bool
    expected: str
    actual: Optional[str]


@dataclass
class CaseEvalResult:
    question_id: str
    chat: Optional[Dict[str, Any]] = None
    rag_pipeline: Optional[Dict[str, Any]] = None
    substring_checks: Optional[SubstringCheckResult] = None
    route_check: Optional[RouteCheckResult] = None
    errors: List[str] = field(default_factory=list)


@dataclass
class GoldenSetReport:
    set_id: str
    tenant_id: str
    paths: List[str]
    case_results: List[CaseEvalResult]
    aggregate: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "set_id": self.set_id,
            "tenant_id": self.tenant_id,
            "paths": self.paths,
            "aggregate": self.aggregate,
            "cases": [
                {
                    "question_id": c.question_id,
                    "errors": c.errors,
                    "substring_checks": (
                        {
                            "passed": c.substring_checks.passed,
                            "missing_expected": c.substring_checks.missing_expected,
                            "forbidden_found": c.substring_checks.forbidden_found,
                        }
                        if c.substring_checks
                        else None
                    ),
                    "route_check": (
                        {
                            "passed": c.route_check.passed,
                            "expected": c.route_check.expected,
                            "actual": c.route_check.actual,
                        }
                        if c.route_check
                        else None
                    ),
                    "chat": c.chat,
                    "rag_pipeline": c.rag_pipeline,
                }
                for c in self.case_results
            ],
        }


class GoldenSetRunner:
    """HTTP-driven evaluator against a running MAI API."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        admin_user: Optional[str] = None,
        admin_pass: Optional[str] = None,
        judge_model: Optional[str] = "gpt-4o-mini",
        judge_provider: str = "openai",
        timeout_s: float = 180.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("GOLDEN_SET_BASE_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.admin_user = admin_user or os.getenv("GOLDEN_SET_ADMIN_USER", "admin")
        self.admin_pass = admin_pass or os.getenv("GOLDEN_SET_ADMIN_PASS", "admin")
        self.timeout_s = timeout_s
        self._evaluator = RAGEvaluator(
            judge_model=judge_model,
            judge_provider=judge_provider,
        )

    def _auth_headers(self, client: httpx.Client) -> Dict[str, str]:
        r = client.post(
            f"{self.base_url}/api/v2/auth/token",
            data={"username": self.admin_user, "password": self.admin_pass},
        )
        if r.status_code != 200:
            raise RuntimeError(f"Auth failed: {r.status_code} {r.text[:300]}")
        token = r.json().get("access_token")
        if not token:
            raise RuntimeError("Auth response missing access_token")
        return {"Authorization": f"Bearer {token}"}

    def _call_chat(
        self,
        client: httpx.Client,
        headers: Dict[str, str],
        *,
        tenant_id: str,
        question: str,
        top_k: int,
    ) -> Dict[str, Any]:
        body = {
            "session_id": str(uuid.uuid4()),
            "messages": [{"role": "user", "content": question}],
            "client_id": tenant_id,
            "top_k": top_k,
            "generate_answer": True,
        }
        r = client.post(
            f"{self.base_url}/api/v2/retrieve/chat",
            json=body,
            headers=headers,
            timeout=self.timeout_s,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Chat failed {r.status_code}: {r.text[:500]}")
        return r.json()

    def _call_rag(
        self,
        client: httpx.Client,
        headers: Dict[str, str],
        *,
        tenant_id: str,
        question: str,
        top_k_final: Optional[int],
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "client_id": tenant_id,
            "query": question,
        }
        if top_k_final is not None:
            body["top_k_final"] = top_k_final
            body["top_k_retrieval"] = max(top_k_final * 4, 20)

        r = client.post(
            f"{self.base_url}/api/v2/rag/query",
            json=body,
            headers=headers,
            timeout=self.timeout_s,
        )
        if r.status_code != 200:
            raise RuntimeError(f"RAG query failed {r.status_code}: {r.text[:500]}")
        return r.json()

    @staticmethod
    def _check_substrings(case: GoldenCaseSpec, answer: Optional[str]) -> SubstringCheckResult:
        text = answer or ""
        missing = [s for s in case.expected_answer_substrings if s not in text]
        forbidden = [s for s in case.forbidden_substrings if s in text]
        return SubstringCheckResult(
            question_id=case.question_id,
            passed=not missing and not forbidden,
            missing_expected=missing,
            forbidden_found=forbidden,
        )

    @staticmethod
    def _check_route(case: GoldenCaseSpec, debug_info: Optional[Dict[str, Any]]) -> RouteCheckResult:
        hint = (case.route_hint or "any").strip().lower()
        actual: Optional[str] = None
        if debug_info:
            route_decision = debug_info.get("route_decision")
            if isinstance(route_decision, dict):
                actual = str(route_decision.get("route") or "").strip().lower()
            if not actual:
                raw = debug_info.get("query_route") or debug_info.get("route")
                if raw is not None and not isinstance(raw, dict):
                    actual = str(raw).strip().lower()

        if hint == "any":
            return RouteCheckResult(case.question_id, True, hint, actual)

        passed = actual == hint
        return RouteCheckResult(case.question_id, passed, hint, actual)

    def _retrieval_metrics(
        self,
        golden: GoldenExample,
        ranked_ids: List[str],
        scores: List[float],
        stage: str,
        k_values: List[int],
    ) -> Dict[str, Any]:
        if len(scores) < len(ranked_ids):
            scores = scores + [0.0] * (len(ranked_ids) - len(scores))
        elif len(scores) > len(ranked_ids):
            scores = scores[: len(ranked_ids)]

        evaluator = RAGEvaluator(judge_model=None, k_values=k_values)
        result = evaluator.run_retrieval_eval(
            [golden],
            [
                RetrievalResult(
                    question_id=golden.question_id,
                    ranked_ids=ranked_ids,
                    scores=scores,
                    stage=stage,
                )
            ],
        )
        return result

    def _faithfulness_metrics(
        self,
        question: str,
        answer: Optional[str],
        context_text: str,
    ) -> Dict[str, Any]:
        if not answer or not context_text.strip():
            return {"skipped": True, "reason": "empty answer or context"}
        return self._evaluator.evaluate_faithfulness_batch(
            [question],
            [context_text],
            [answer],
        )

    def run(
        self,
        set_ref: str,
        *,
        paths: Optional[List[EvalPath]] = None,
        run_faithfulness: bool = True,
    ) -> GoldenSetReport:
        golden_file = load_golden_set(set_ref)
        paths = paths or ["chat", "rag"]
        max_k = max(golden_file.k_values) if golden_file.k_values else 10

        case_results: List[CaseEvalResult] = []

        with httpx.Client() as client:
            headers = self._auth_headers(client)

            for case in golden_file.cases:
                cer = CaseEvalResult(question_id=case.question_id)
                golden_ex: Optional[GoldenExample] = None
                if not case.skip_retrieval_metrics_if_empty():
                    golden_ex = GoldenExample(
                        question_id=case.question_id,
                        question=case.question,
                        relevant_chunk_ids=set(case.relevant_chunk_ids),
                        expected_answer=case.expected_answer,
                        metadata=dict(case.metadata),
                    )

                if "chat" in paths:
                    try:
                        chat_body = self._call_chat(
                            client,
                            headers,
                            tenant_id=golden_file.tenant_id,
                            question=case.question,
                            top_k=max_k,
                        )
                        ranked_ids, scores, stage = ranked_lists_from_chat_response(chat_body)
                        chat_out: Dict[str, Any] = {
                            "ranked_count": len(ranked_ids),
                            "stage": stage,
                        }
                        if golden_ex and ranked_ids:
                            chat_out["retrieval"] = self._retrieval_metrics(
                                golden_ex,
                                ranked_ids,
                                scores,
                                stage,
                                golden_file.k_values,
                            )
                        if run_faithfulness:
                            ctx = build_context_text_from_chunks(
                                chat_body.get("results") or []
                            )
                            chat_out["faithfulness"] = self._faithfulness_metrics(
                                case.question,
                                chat_body.get("answer"),
                                ctx,
                            )
                        chat_out["debug_info"] = chat_body.get("debug_info")
                        cer.chat = chat_out
                        cer.route_check = self._check_route(
                            case, chat_body.get("debug_info")
                        )
                        cer.substring_checks = self._check_substrings(
                            case, chat_body.get("answer")
                        )
                    except Exception as exc:
                        cer.errors.append(f"chat: {exc}")
                        logger.exception("[GoldenSetRunner] chat failed | %s", case.question_id)

                if "rag" in paths:
                    try:
                        rag_body = self._call_rag(
                            client,
                            headers,
                            tenant_id=golden_file.tenant_id,
                            question=case.question,
                            top_k_final=max_k,
                        )
                        ranked_ids, scores, stage = ranked_lists_from_rag_response(rag_body)
                        rag_out: Dict[str, Any] = {
                            "ranked_count": len(ranked_ids),
                            "stage": stage,
                        }
                        if golden_ex and ranked_ids:
                            rag_out["retrieval"] = self._retrieval_metrics(
                                golden_ex,
                                ranked_ids,
                                scores,
                                stage,
                                golden_file.k_values,
                            )
                        if run_faithfulness:
                            chunks = rag_body.get("retrieved_chunks") or []
                            ctx = build_context_text_from_chunks(chunks)
                            rag_out["faithfulness"] = self._faithfulness_metrics(
                                case.question,
                                rag_body.get("final_answer"),
                                ctx,
                            )
                        cer.rag_pipeline = rag_out
                        if cer.substring_checks is None:
                            cer.substring_checks = self._check_substrings(
                                case, rag_body.get("final_answer")
                            )
                    except Exception as exc:
                        cer.errors.append(f"rag: {exc}")
                        logger.exception("[GoldenSetRunner] rag failed | %s", case.question_id)

                case_results.append(cer)

        aggregate = _aggregate_report(case_results, golden_file)
        return GoldenSetReport(
            set_id=golden_file.set_id,
            tenant_id=golden_file.tenant_id,
            paths=list(paths),
            case_results=case_results,
            aggregate=aggregate,
        )


def _aggregate_report(
    case_results: List[CaseEvalResult],
    golden_file: GoldenSetFile,
) -> Dict[str, Any]:
    substring_pass = sum(
        1 for c in case_results if c.substring_checks and c.substring_checks.passed
    )
    route_pass = sum(1 for c in case_results if c.route_check and c.route_check.passed)
    errors = sum(1 for c in case_results if c.errors)

    def _mean_metric(path: str, key: str) -> Optional[float]:
        vals: List[float] = []
        for c in case_results:
            block = getattr(c, path, None)
            if not block:
                continue
            retrieval = block.get("retrieval") if isinstance(block, dict) else None
            if not retrieval:
                continue
            agg = retrieval.get("aggregate") or {}
            v = agg.get(key)
            if v is not None:
                vals.append(float(v))
        if not vals:
            return None
        return round(sum(vals) / len(vals), 4)

    return {
        "n_cases": len(case_results),
        "substring_checks_passed": substring_pass,
        "route_checks_passed": route_pass,
        "cases_with_errors": errors,
        "chat_mean_mrr": _mean_metric("chat", "mrr"),
        "chat_mean_recall_at_5": _mean_metric("chat", "recall@5"),
        "rag_mean_mrr": _mean_metric("rag_pipeline", "mrr"),
        "rag_mean_recall_at_5": _mean_metric("rag_pipeline", "recall@5"),
        "corpus_fingerprint": golden_file.corpus_fingerprint or None,
    }


def check_thresholds(report: GoldenSetReport) -> List[str]:
    """Return list of threshold violation messages (empty if all pass)."""
    violations: List[str] = []

    min_recall = os.getenv("GOLDEN_MIN_RECALL_AT_5")
    if min_recall:
        target = float(min_recall)
        for key in ("chat_mean_recall_at_5", "rag_mean_recall_at_5"):
            val = report.aggregate.get(key)
            if val is not None and val < target:
                violations.append(f"{key}={val} < {target}")

    min_mrr = os.getenv("GOLDEN_MIN_MRR")
    if min_mrr:
        target = float(min_mrr)
        for key in ("chat_mean_mrr", "rag_mean_mrr"):
            val = report.aggregate.get(key)
            if val is not None and val < target:
                violations.append(f"{key}={val} < {target}")

    return violations
