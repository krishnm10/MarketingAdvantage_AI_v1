# =============================================================================
# app/ai/evaluation/rag_evaluator.py
#
# RAGEvaluator — Phase 2 evaluation harness
#
# PURPOSE:
#   Provides a structured evaluation framework for RAG pipeline quality:
#     1. Retrieval metrics  — precision@k, recall@k, MRR, NDCG@k
#     2. Reranking metrics  — how much reranking improves the retrieval list
#     3. Generation quality — faithfulness, answer relevance (LLM-as-judge)
#     4. Threshold calibration — find optimal threshold_min_score from labeled data
#
# USAGE:
#   from app.ai.evaluation.rag_evaluator import RAGEvaluator, GoldenExample
#
#   evaluator = RAGEvaluator(judge_api_key_env="OPENAI_API_KEY")
#   results = evaluator.run_retrieval_eval(golden_set, retrieved_results)
#   calibrated_threshold = evaluator.calibrate_threshold(
#       golden_set, reranked_results, target_recall=0.9
#   )
#
# DESIGN:
#   - All external LLM calls are optional (set judge_model=None to skip).
#   - Metrics are computed in pure Python (no ML framework required).
#   - Results are serialisable dicts suitable for JSON logging.
#   - All eval loops are logged at INFO level for observability.
# =============================================================================

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Golden set data structures
# ---------------------------------------------------------------------------

@dataclass
class GoldenExample:
    """
    One labelled question for evaluation.

    Attributes
    ----------
    question_id   : Unique identifier (for report correlation).
    question      : Natural language query.
    relevant_chunk_ids : Set of chunk IDs that are relevant for this question.
                         Used to compute precision/recall.
    expected_answer   : Optional reference answer for faithfulness evaluation.
    metadata          : Optional extra context (e.g., difficulty, topic).
    """
    question_id:        str
    question:           str
    relevant_chunk_ids: Set[str]
    expected_answer:    Optional[str] = None
    metadata:           Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """
    Retrieved/reranked results for one question.

    Attributes
    ----------
    question_id   : Must match a GoldenExample.question_id.
    ranked_ids    : Ordered list of chunk IDs (most relevant first).
    scores        : Corresponding relevance scores.
    stage         : "retrieval" or "reranked" — identifies which pipeline stage.
    """
    question_id: str
    ranked_ids:  List[str]
    scores:      List[float]
    stage:       str = "retrieval"


# ---------------------------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------------------------

def _precision_at_k(ranked_ids: List[str], relevant: Set[str], k: int) -> float:
    top_k = ranked_ids[:k]
    hits  = sum(1 for cid in top_k if cid in relevant)
    return hits / k if k else 0.0


def _recall_at_k(ranked_ids: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 1.0
    top_k = ranked_ids[:k]
    hits  = sum(1 for cid in top_k if cid in relevant)
    return hits / len(relevant)


def _reciprocal_rank(ranked_ids: List[str], relevant: Set[str]) -> float:
    for i, cid in enumerate(ranked_ids, 1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def _ndcg_at_k(ranked_ids: List[str], relevant: Set[str], k: int) -> float:
    """Compute NDCG@k with binary relevance (1 = relevant, 0 = not)."""
    def _dcg(ids: List[str]) -> float:
        return sum(
            (1.0 if cid in relevant else 0.0) / math.log2(i + 2)
            for i, cid in enumerate(ids[:k])
        )

    ideal_ids = [cid for cid in ranked_ids if cid in relevant]
    ideal_ids += [cid for cid in ranked_ids if cid not in relevant]
    dcg  = _dcg(ranked_ids)
    idcg = _dcg(ideal_ids)
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# RAGEvaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """
    Structured RAG pipeline evaluator.

    Supports OpenAI and Gemini as judge providers for faithfulness evaluation.

    Args:
        judge_model    : LLM model for faithfulness scoring (e.g. 'gpt-4o-mini',
                         'gemini-1.5-flash'). Set to None to skip generation eval.
        judge_provider : Provider for the judge LLM: 'openai' (default) or 'gemini'.
        judge_api_key_env : Env var name for the judge API key.
        k_values       : List of k values for @k metrics (e.g. [1, 3, 5, 10]).
    """

    def __init__(
        self,
        *,
        judge_model: Optional[str] = "gpt-4o-mini",
        judge_provider: str = "openai",
        judge_api_key_env: Optional[str] = None,
        k_values: List[int] = None,
    ) -> None:
        self._judge_model    = judge_model
        self._judge_provider = judge_provider.lower()
        self._k_values       = k_values or [1, 3, 5, 10]
        self._judge_client   = None  # Legacy OpenAI SDK client (openai provider only)
        self._judge_llm      = None  # BaseLLM instance (any provider)

        # Determine API key env var based on provider
        if judge_api_key_env is None:
            _KEY_MAP = {
                "openai":    "OPENAI_API_KEY",
                "gemini":    "GOOGLE_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "groq":      "GROQ_API_KEY",
            }
            judge_api_key_env = _KEY_MAP.get(self._judge_provider, "OPENAI_API_KEY")

        if judge_model:
            api_key = os.environ.get(judge_api_key_env, "").strip()
            if api_key:
                try:
                    if self._judge_provider == "openai":
                        from openai import OpenAI
                        self._judge_client = OpenAI(api_key=api_key)
                    else:
                        # Use the shared LLM registry for non-OpenAI providers
                        from app.core.plugin_registry import llm_registry
                        kwargs: dict = {"model": judge_model, "api_key": api_key}
                        if self._judge_provider == "gemini":
                            pass  # GeminiLLM doesn't need base_url
                        elif self._judge_provider in ("ollama",):
                            kwargs["base_url"] = os.getenv(
                                "OLLAMA_BASE_URL", "http://localhost:11434"
                            )
                        self._judge_llm = llm_registry.build(
                            self._judge_provider, **kwargs
                        )
                    logger.info(
                        "[RAGEvaluator] LLM judge ready | provider=%s | model=%s",
                        self._judge_provider, judge_model,
                    )
                except Exception as e:
                    logger.warning(
                        "[RAGEvaluator] LLM judge init failed (provider=%s model=%s): %s",
                        self._judge_provider, judge_model, e,
                    )
            else:
                logger.warning(
                    "[RAGEvaluator] LLM judge skipped — env var '%s' not set.",
                    judge_api_key_env,
                )

    # ── Evaluation configuration matrix ─────────────────────────────────────

    @staticmethod
    def gemini_pipeline_configs() -> List[Dict[str, Any]]:
        """
        Return recommended evaluation configurations for Gemini-based pipelines.
        Each entry describes a combination of embedder, reranker, and LLM settings
        to benchmark retrieval and generation quality.

        Use this to drive systematic evaluation sweeps across Gemini variants:
          configs = RAGEvaluator.gemini_pipeline_configs()
          for cfg in configs:
              evaluator = RAGEvaluator(
                  judge_model=cfg["judge_model"],
                  judge_provider=cfg["judge_provider"],
              )
              ...
        """
        return [
            {
                "name":           "gemini-embed-flash-judge",
                "description":    "Gemini embedder + Gemini 1.5 Flash as judge",
                "embedder":       "google/text-embedding-004",
                "reranker":       "llm-judge/gemini-1.5-flash",
                "judge_model":    "gemini-1.5-flash",
                "judge_provider": "gemini",
                "judge_key_env":  "GOOGLE_API_KEY",
                "expected_dim":   768,
                "notes":          "Best cost/quality for multilingual marketing content.",
            },
            {
                "name":           "gemini-embed-pro-judge",
                "description":    "Gemini embedder + Gemini 1.5 Pro as judge",
                "embedder":       "google/text-embedding-004",
                "reranker":       "llm-judge/gemini-1.5-pro",
                "judge_model":    "gemini-1.5-pro",
                "judge_provider": "gemini",
                "judge_key_env":  "GOOGLE_API_KEY",
                "expected_dim":   768,
                "notes":          "Highest quality; use for golden-set calibration.",
            },
            {
                "name":           "gemini-embed-openai-judge",
                "description":    "Gemini embedder + OpenAI GPT-4o-mini as cross-provider judge",
                "embedder":       "google/text-embedding-004",
                "reranker":       "llm-judge/gpt-4o-mini",
                "judge_model":    "gpt-4o-mini",
                "judge_provider": "openai",
                "judge_key_env":  "OPENAI_API_KEY",
                "expected_dim":   768,
                "notes":          "Cross-provider validation to avoid provider-specific bias.",
            },
            {
                "name":           "openai-embed-gemini-judge",
                "description":    "OpenAI embedder + Gemini as judge (cross-provider baseline)",
                "embedder":       "openai/text-embedding-3-small",
                "reranker":       "llm-judge/gemini-1.5-flash",
                "judge_model":    "gemini-1.5-flash",
                "judge_provider": "gemini",
                "judge_key_env":  "GOOGLE_API_KEY",
                "expected_dim":   1536,
                "notes":          "Validates retrieval quality with Gemini judge on OpenAI embeddings.",
            },
        ]

    # ── Retrieval evaluation ────────────────────────────────────────────────

    def run_retrieval_eval(
        self,
        golden_set: List[GoldenExample],
        results: List[RetrievalResult],
    ) -> Dict[str, Any]:
        """
        Compute precision@k, recall@k, MRR, NDCG@k across the golden set.

        Parameters
        ----------
        golden_set : list[GoldenExample]
        results    : list[RetrievalResult] — one per golden question.

        Returns
        -------
        dict with aggregate metrics and per-question details.
        """
        result_map = {r.question_id: r for r in results}
        per_question: List[Dict[str, Any]] = []

        agg: Dict[str, List[float]] = {
            f"precision@{k}": [] for k in self._k_values
        }
        agg.update({f"recall@{k}": [] for k in self._k_values})
        agg.update({f"ndcg@{k}": [] for k in self._k_values})
        agg["mrr"] = []

        for golden in golden_set:
            res = result_map.get(golden.question_id)
            if res is None:
                logger.warning(
                    "[RAGEvaluator] No result for question_id=%r; skipping.",
                    golden.question_id,
                )
                continue

            ranked = res.ranked_ids
            relevant = golden.relevant_chunk_ids
            q_metrics: Dict[str, float] = {}

            for k in self._k_values:
                p = _precision_at_k(ranked, relevant, k)
                r = _recall_at_k(ranked, relevant, k)
                n = _ndcg_at_k(ranked, relevant, k)
                q_metrics[f"precision@{k}"] = round(p, 4)
                q_metrics[f"recall@{k}"]    = round(r, 4)
                q_metrics[f"ndcg@{k}"]      = round(n, 4)
                agg[f"precision@{k}"].append(p)
                agg[f"recall@{k}"].append(r)
                agg[f"ndcg@{k}"].append(n)

            mrr = _reciprocal_rank(ranked, relevant)
            q_metrics["mrr"] = round(mrr, 4)
            agg["mrr"].append(mrr)

            per_question.append({
                "question_id": golden.question_id,
                "question":    golden.question[:80],
                "n_relevant":  len(relevant),
                "n_retrieved": len(ranked),
                "stage":       res.stage,
                **q_metrics,
            })

        aggregate = {
            k: round(sum(v) / len(v), 4) if v else 0.0
            for k, v in agg.items()
        }

        logger.info(
            "[RAGEvaluator] Retrieval eval | n=%d | MRR=%.3f | recall@5=%.3f | ndcg@5=%.3f",
            len(per_question),
            aggregate.get("mrr", 0.0),
            aggregate.get("recall@5", 0.0),
            aggregate.get("ndcg@5", 0.0),
        )

        return {
            "n_questions":   len(per_question),
            "aggregate":     aggregate,
            "per_question":  per_question,
        }

    # ── Threshold calibration ───────────────────────────────────────────────

    def calibrate_threshold(
        self,
        golden_set: List[GoldenExample],
        reranked_results: List[RetrievalResult],
        *,
        target_recall: float = 0.9,
        score_steps: int = 20,
    ) -> Dict[str, Any]:
        """
        Find the minimum rerank_score threshold that preserves target_recall@5.

        Iterates from the highest score downwards, finding the lowest threshold
        where recall@5 ≥ target_recall across the golden set.

        Parameters
        ----------
        target_recall : minimum acceptable recall@5 at the calibrated threshold.
        score_steps   : number of threshold candidates to evaluate (resolution).

        Returns
        -------
        dict with ``recommended_threshold``, ``recall_at_threshold``,
        ``precision_at_threshold``, and the full sweep results.
        """
        result_map = {r.question_id: r for r in reranked_results}

        # Collect all scores to determine the sweep range
        all_scores: List[float] = []
        for r in reranked_results:
            all_scores.extend(r.scores)

        if not all_scores:
            return {"error": "No reranked results provided."}

        min_s = min(all_scores)
        max_s = max(all_scores)
        step  = (max_s - min_s) / score_steps if score_steps > 1 else 0.1

        sweep_results: List[Dict[str, float]] = []
        recommended_threshold = min_s

        for i in range(score_steps + 1):
            threshold = min_s + i * step

            # Simulate threshold gate: retain candidates with score >= threshold (min 1)
            gated_results = []
            for r in reranked_results:
                filtered_ids = [
                    cid for cid, s in zip(r.ranked_ids, r.scores)
                    if s >= threshold
                ]
                if not filtered_ids:
                    filtered_ids = r.ranked_ids[:1]  # safety floor
                gated_results.append(
                    RetrievalResult(
                        question_id=r.question_id,
                        ranked_ids=filtered_ids,
                        scores=[s for s in r.scores if s >= threshold] or r.scores[:1],
                        stage="gated",
                    )
                )

            eval_result = self.run_retrieval_eval(golden_set, gated_results)
            r5 = eval_result["aggregate"].get("recall@5", 0.0)
            p5 = eval_result["aggregate"].get("precision@5", 0.0)

            sweep_results.append({
                "threshold":   round(threshold, 4),
                "recall@5":    r5,
                "precision@5": p5,
            })

            # Track highest threshold that still meets target_recall
            if r5 >= target_recall:
                recommended_threshold = threshold

        logger.info(
            "[RAGEvaluator] Threshold calibration | target_recall=%.2f | "
            "recommended=%.4f",
            target_recall, recommended_threshold,
        )

        best = next(
            (s for s in reversed(sweep_results) if s["threshold"] == round(recommended_threshold, 4)),
            sweep_results[0],
        )

        return {
            "recommended_threshold":  round(recommended_threshold, 4),
            "recall_at_threshold":    best.get("recall@5", 0.0),
            "precision_at_threshold": best.get("precision@5", 0.0),
            "target_recall":          target_recall,
            "score_range":            {"min": round(min_s, 4), "max": round(max_s, 4)},
            "sweep_results":          sweep_results,
        }

    # ── Generation faithfulness ─────────────────────────────────────────────

    def evaluate_faithfulness_batch(
        self,
        questions: List[str],
        context_texts: List[str],
        answers: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate faithfulness of LLM answers against retrieved context.

        Uses the configured judge provider (OpenAI or Gemini) for LLM-as-judge
        faithfulness scoring. Returns null results if no judge is configured.

        Returns
        -------
        dict with aggregate faithfulness score and per-question results.
        """
        has_judge = (self._judge_client is not None) or (self._judge_llm is not None)
        if not has_judge:
            return {
                "error":       "LLM judge not configured.",
                "n_questions": 0,
                "per_question": [],
            }

        _FAITHFULNESS_PROMPT = """\
Evaluate whether the ANSWER is fully supported by the CONTEXT.

CONTEXT:
{context}

QUESTION: {question}
ANSWER: {answer}

Respond ONLY with JSON: {{"faithful": true/false, "score": <0.0-1.0>, "reason": "<one line>"}}"""

        per_q: List[Dict[str, Any]] = []
        total_score = 0.0

        for i, (q, ctx, ans) in enumerate(zip(questions, context_texts, answers)):
            prompt = _FAITHFULNESS_PROMPT.format(
                context=ctx[:6000], question=q, answer=ans[:2000]
            )
            try:
                if self._judge_client is not None:
                    # OpenAI SDK path
                    resp = self._judge_client.chat.completions.create(
                        model=self._judge_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=80,
                    )
                    raw = resp.choices[0].message.content.strip()
                else:
                    # Generic BaseLLM path (Gemini, Anthropic, etc.)
                    response = self._judge_llm.generate(
                        prompt, temperature=0.0, max_tokens=80
                    )
                    raw = response.text.strip()

                result = json.loads(raw)
                score  = float(result.get("score", 0.5))

            except Exception as e:
                logger.warning(
                    "[RAGEvaluator] Faithfulness eval failed q=%d provider=%s: %s",
                    i, self._judge_provider, e,
                )
                result = {"faithful": None, "score": None, "reason": str(e)}
                score  = 0.0

            per_q.append({"question": q[:80], **result})
            total_score += score

        avg_score = total_score / len(questions) if questions else 0.0

        logger.info(
            "[RAGEvaluator] Faithfulness | provider=%s | n=%d | avg_score=%.3f",
            self._judge_provider, len(questions), avg_score,
        )

        return {
            "n_questions":      len(questions),
            "avg_faithfulness": round(avg_score, 4),
            "judge_provider":   self._judge_provider,
            "judge_model":      self._judge_model,
            "per_question":     per_q,
        }
