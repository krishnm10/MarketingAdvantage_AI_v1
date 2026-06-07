# app/ai/evaluation — RAG quality evaluation package.

from app.ai.evaluation.golden_set_loader import (
    GoldenCaseSpec,
    GoldenSetFile,
    load_golden_set,
    resolve_golden_set_path,
)
from app.ai.evaluation.golden_set_runner import GoldenSetRunner, GoldenSetReport
from app.ai.evaluation.rag_evaluator import GoldenExample, RAGEvaluator, RetrievalResult

__all__ = [
    "GoldenCaseSpec",
    "GoldenSetFile",
    "GoldenSetRunner",
    "GoldenSetReport",
    "GoldenExample",
    "RAGEvaluator",
    "RetrievalResult",
    "load_golden_set",
    "resolve_golden_set_path",
]
