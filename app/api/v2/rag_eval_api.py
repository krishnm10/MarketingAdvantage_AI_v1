"""
================================================================================
Marketing Advantage AI — RAG Evaluation API
File: app/api/v2/rag_eval_api.py

Endpoints:
  POST /api/v2/rag-eval/retrieval         → Evaluate retrieval quality
  POST /api/v2/rag-eval/calibrate         → Calibrate threshold from labeled data
  POST /api/v2/rag-eval/faithfulness      → Evaluate answer faithfulness
  POST /api/v2/rag-eval/pipeline-run      → Run a full golden-set evaluation

Design:
  - All endpoints are write-protected (admin role).
  - Results include structured metrics suitable for logging to dashboards.
  - LLM judge calls are gated on OPENAI_API_KEY being set.
================================================================================
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.auth.guards import require_role

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v2/rag-eval",
    tags=["RAG Evaluation"],
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class GoldenExampleSchema(BaseModel):
    question_id:         str
    question:            str
    relevant_chunk_ids:  List[str]
    expected_answer:     Optional[str] = None


class RetrievalResultSchema(BaseModel):
    question_id:  str
    ranked_ids:   List[str]
    scores:       List[float]
    stage:        str = "retrieval"


class RetrievalEvalRequest(BaseModel):
    golden_set: List[GoldenExampleSchema]
    results:    List[RetrievalResultSchema]


class CalibrateRequest(BaseModel):
    golden_set:       List[GoldenExampleSchema]
    reranked_results: List[RetrievalResultSchema]
    target_recall:    float = Field(0.9, ge=0.5, le=1.0)
    score_steps:      int   = Field(20, ge=5, le=100)


class FaithfulnessRequest(BaseModel):
    questions:      List[str]
    context_texts:  List[str]
    answers:        List[str]
    judge_model:    str = "gpt-4o-mini"
    judge_provider: str = Field("openai", description="LLM provider for judging: 'openai' | 'gemini'.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/retrieval", response_model=dict)
async def eval_retrieval(req: RetrievalEvalRequest):
    """
    Compute precision@k, recall@k, MRR, NDCG@k against a labeled golden set.
    """
    try:
        from app.ai.evaluation.rag_evaluator import (
            RAGEvaluator, GoldenExample, RetrievalResult,
        )

        golden_set = [
            GoldenExample(
                question_id=g.question_id,
                question=g.question,
                relevant_chunk_ids=set(g.relevant_chunk_ids),
                expected_answer=g.expected_answer,
            )
            for g in req.golden_set
        ]
        results = [
            RetrievalResult(
                question_id=r.question_id,
                ranked_ids=r.ranked_ids,
                scores=r.scores,
                stage=r.stage,
            )
            for r in req.results
        ]

        evaluator = RAGEvaluator(judge_model=None)
        return evaluator.run_retrieval_eval(golden_set, results)

    except Exception as e:
        logger.error("[rag_eval_api] retrieval eval failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calibrate", response_model=dict)
async def calibrate_threshold(req: CalibrateRequest):
    """
    Find the optimal threshold_min_score that preserves target_recall@5.
    Returns a recommended threshold value ready to set in pipeline config.
    """
    try:
        from app.ai.evaluation.rag_evaluator import (
            RAGEvaluator, GoldenExample, RetrievalResult,
        )

        golden_set = [
            GoldenExample(
                question_id=g.question_id,
                question=g.question,
                relevant_chunk_ids=set(g.relevant_chunk_ids),
            )
            for g in req.golden_set
        ]
        reranked = [
            RetrievalResult(
                question_id=r.question_id,
                ranked_ids=r.ranked_ids,
                scores=r.scores,
                stage=r.stage,
            )
            for r in req.reranked_results
        ]

        evaluator = RAGEvaluator(judge_model=None)
        result = evaluator.calibrate_threshold(
            golden_set, reranked,
            target_recall=req.target_recall,
            score_steps=req.score_steps,
        )
        return result

    except Exception as e:
        logger.error("[rag_eval_api] calibration failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/faithfulness", response_model=dict)
async def eval_faithfulness(req: FaithfulnessRequest):
    """
    Evaluate answer faithfulness using LLM-as-judge.
    Requires OPENAI_API_KEY to be set.
    """
    if len(req.questions) != len(req.context_texts) or len(req.questions) != len(req.answers):
        raise HTTPException(
            status_code=422,
            detail="questions, context_texts, and answers must have the same length.",
        )
    if len(req.questions) > 50:
        raise HTTPException(
            status_code=422,
            detail="Maximum 50 questions per faithfulness evaluation call.",
        )

    try:
        from app.ai.evaluation.rag_evaluator import RAGEvaluator
        evaluator = RAGEvaluator(
            judge_model=req.judge_model,
            judge_provider=req.judge_provider,
        )
        return evaluator.evaluate_faithfulness_batch(
            req.questions,
            req.context_texts,
            req.answers,
        )
    except Exception as e:
        logger.error("[rag_eval_api] faithfulness eval failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


class PipelineRunRequest(BaseModel):
    """Run a golden-set file against live chat and/or RAG paths."""

    set_ref: str = Field(..., description="Golden set path under tests/golden_sets/")
    paths: List[str] = Field(
        default_factory=lambda: ["chat", "rag"],
        description="Paths to evaluate: chat, rag",
    )
    run_faithfulness: bool = Field(True, description="Run LLM-as-judge when API keys are set.")


@router.post("/pipeline-run", response_model=dict)
async def pipeline_run_golden_set(
    req: PipelineRunRequest,
    _user=Depends(require_role("admin")),
):
    """
    Execute a golden-set evaluation harness server-side (admin only).
    """
    allowed = {"chat", "rag"}
    paths = [p.strip().lower() for p in req.paths if p.strip().lower() in allowed]
    if not paths:
        raise HTTPException(status_code=422, detail="paths must include chat and/or rag")

    try:
        from app.ai.evaluation.golden_set_runner import GoldenSetRunner, check_thresholds

        runner = GoldenSetRunner()
        report = runner.run(
            req.set_ref,
            paths=paths,  # type: ignore[arg-type]
            run_faithfulness=req.run_faithfulness,
        )
        payload = report.to_dict()
        payload["threshold_violations"] = check_thresholds(report)
        return payload
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error("[rag_eval_api] pipeline-run failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/evaluation-matrix", response_model=List[dict])
async def get_evaluation_matrix():
    """
    Return the recommended evaluation configurations for Gemini-based pipelines.
    Use these as templates when running systematic eval sweeps.
    """
    try:
        from app.ai.evaluation.rag_evaluator import RAGEvaluator
        return RAGEvaluator.gemini_pipeline_configs()
    except Exception as e:
        logger.error("[rag_eval_api] evaluation matrix failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
