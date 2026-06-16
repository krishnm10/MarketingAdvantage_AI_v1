from __future__ import annotations

import logging
from typing import Optional

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# HTTP REQUEST METRICS
# ──────────────────────────────────────────────────────────────────────────────

_REQUEST_TOTAL = Counter(
    "mai_request_total",
    "Total HTTP requests processed by the MAI API.",
    ("route", "method", "status_class"),
)

_REQUEST_LATENCY_SECONDS = Histogram(
    "mai_request_latency_seconds",
    "HTTP request latency in seconds.",
    ("route", "method"),
)


def observe_http_request(
    route: str,
    method: str,
    status_code: int,
    duration_seconds: float,
) -> None:
    """Record HTTP request count and latency. Best-effort, never raises."""
    try:
        status_class = f"{int(status_code) // 100}xx"
        _REQUEST_TOTAL.labels(route=route, method=method, status_class=status_class).inc()
        _REQUEST_LATENCY_SECONDS.labels(route=route, method=method).observe(
            float(duration_seconds)
        )
    except Exception:  # pragma: no cover - metrics must never break requests
        logger.debug("Failed to record HTTP metrics", exc_info=True)


# ──────────────────────────────────────────────────────────────────────────────
# INGESTION / WORKER METRICS
# ──────────────────────────────────────────────────────────────────────────────

_INGESTION_TOTAL = Counter(
    "mai_ingestion_total",
    "Total ingestion operations by kind and result.",
    ("kind", "result"),
)

_INGESTION_DURATION_SECONDS = Histogram(
    "mai_ingestion_duration_seconds",
    "Ingestion duration in seconds by kind.",
    ("kind",),
)

_INGESTION_FAILURES_TOTAL = Counter(
    "mai_ingestion_failures_total",
    "Total terminal ingestion failures by stage.",
    ("stage",),
)

_DLQ_WRITES_TOTAL = Counter(
    "mai_dlq_writes_total",
    "Total durable DLQ tombstones written by stage.",
    ("stage",),
)

_WORKER_ACTIVE_JOBS = Gauge(
    "mai_worker_active_jobs",
    "Number of ingestion commands actively being processed by workers.",
)

_WORKER_QUEUE_DEPTH = Gauge(
    "mai_worker_queue_depth",
    "Current depth of the in-memory ingestion queue (pending commands).",
)

_WORKER_DLQ_TOTAL = Gauge(
    "mai_worker_dlq_total",
    "Total number of DLQ rows (all tenants).",
)


def record_ingestion_started(kind: str) -> None:
    try:
        _INGESTION_TOTAL.labels(kind=kind, result="started").inc()
    except Exception:  # pragma: no cover
        logger.debug("Failed to record ingestion start metric", exc_info=True)


def record_ingestion_completed(kind: str) -> None:
    try:
        _INGESTION_TOTAL.labels(kind=kind, result="completed").inc()
    except Exception:  # pragma: no cover
        logger.debug("Failed to record ingestion completion metric", exc_info=True)


def observe_ingestion_duration(kind: str, duration_seconds: float) -> None:
    try:
        _INGESTION_DURATION_SECONDS.labels(kind=kind).observe(float(duration_seconds))
    except Exception:  # pragma: no cover
        logger.debug("Failed to record ingestion duration metric", exc_info=True)


def record_ingestion_failure(stage: str) -> None:
    try:
        _INGESTION_FAILURES_TOTAL.labels(stage=stage).inc()
    except Exception:  # pragma: no cover
        logger.debug("Failed to record ingestion failure metric", exc_info=True)


def record_dlq_write(stage: str) -> None:
    try:
        _DLQ_WRITES_TOTAL.labels(stage=stage).inc()
    except Exception:  # pragma: no cover
        logger.debug("Failed to record DLQ write metric", exc_info=True)


def inc_worker_active_jobs() -> None:
    try:
        _WORKER_ACTIVE_JOBS.inc()
    except Exception:  # pragma: no cover
        logger.debug("Failed to increment worker_active_jobs", exc_info=True)


def dec_worker_active_jobs() -> None:
    try:
        _WORKER_ACTIVE_JOBS.dec()
    except Exception:  # pragma: no cover
        logger.debug("Failed to decrement worker_active_jobs", exc_info=True)


def set_worker_queue_depth(pending: int) -> None:
    try:
        _WORKER_QUEUE_DEPTH.set(float(pending))
    except Exception:  # pragma: no cover
        logger.debug("Failed to set worker_queue_depth", exc_info=True)


def set_worker_dlq_total(count: int) -> None:
    try:
        _WORKER_DLQ_TOTAL.set(float(count))
    except Exception:  # pragma: no cover
        logger.debug("Failed to set worker_dlq_total", exc_info=True)


# ──────────────────────────────────────────────────────────────────────────────
# RETRIEVAL / TRUST METRICS
# ──────────────────────────────────────────────────────────────────────────────

_RETRIEVAL_TOTAL = Counter(
    "mai_retrieval_total",
    "Total retrieval executions by route and search_mode.",
    ("route", "search_mode"),
)

_RETRIEVAL_LATENCY_SECONDS = Histogram(
    "mai_retrieval_latency_seconds",
    "Retrieval latency in seconds by route and search_mode.",
    ("route", "search_mode"),
)

_LLM_JUDGE_FALLBACK_TOTAL = Counter(
    "mai_llm_judge_fallback_total",
    "Times reranker.type=llm_judge fell back to another plugin.",
    ("fallback_reranker",),
)


def record_llm_judge_fallback(fallback_reranker: str) -> None:
    try:
        _LLM_JUDGE_FALLBACK_TOTAL.labels(fallback_reranker=fallback_reranker).inc()
    except Exception:  # pragma: no cover
        logger.debug("Failed to record llm_judge fallback metric", exc_info=True)


_TRUST_GATE_TOTAL = Counter(
    "mai_trust_gate_total",
    "Total trust/faithfulness gate outcomes by route and outcome.",
    ("route", "outcome"),
)


def record_retrieval(
    route: str,
    search_mode: str,
    duration_seconds: Optional[float] = None,
) -> None:
    try:
        _RETRIEVAL_TOTAL.labels(route=route, search_mode=search_mode).inc()
        if duration_seconds is not None:
            _RETRIEVAL_LATENCY_SECONDS.labels(
                route=route,
                search_mode=search_mode,
            ).observe(float(duration_seconds))
    except Exception:  # pragma: no cover
        logger.debug("Failed to record retrieval metrics", exc_info=True)


_DOCSET_SHADOW_RUN_TOTAL = Counter(
    "mai_docset_shadow_run_total",
    "Total Phase 3A docset shadow analysis executions by route and domain.",
    ("route", "domain"),
)

_DOCSET_SHADOW_DEGRADED_TOTAL = Counter(
    "mai_docset_shadow_degraded_total",
    "Total Phase 3A docset shadow analysis executions that degraded or failed.",
    ("route", "domain"),
)

_DOCSET_SHADOW_LATENCY_SECONDS = Histogram(
    "mai_docset_shadow_latency_seconds",
    "Phase 3A docset shadow analysis latency in seconds by route and domain.",
    ("route", "domain"),
)

_DOCSET_SHADOW_MATCHED_DOCS = Histogram(
    "mai_docset_shadow_matched_docs",
    "Distribution of docs_matched per Phase 3A docset shadow analysis run.",
    ("route", "domain"),
)

_DOCSET_GOLDEN_RUN_TOTAL = Counter(
    "mai_docset_golden_run_total",
    "Total Phase 3B docset golden evaluations by route and domain.",
    ("route", "domain"),
)

_DOCSET_GOLDEN_DEGRADED_TOTAL = Counter(
    "mai_docset_golden_degraded_total",
    "Total Phase 3B docset golden evaluations that degraded or failed.",
    ("route", "domain"),
)

_DOCSET_GOLDEN_LATENCY_SECONDS = Histogram(
    "mai_docset_golden_latency_seconds",
    "Phase 3B docset golden evaluation latency in seconds by route and domain.",
    ("route", "domain"),
)

_DOCSET_GOLDEN_PRECISION = Histogram(
    "mai_docset_golden_precision",
    "Distribution of precision for Phase 3B docset golden evaluations.",
    ("route", "domain"),
)

_DOCSET_GOLDEN_RECALL = Histogram(
    "mai_docset_golden_recall",
    "Distribution of recall for Phase 3B docset golden evaluations.",
    ("route", "domain"),
)

_DOCSET_SUMMARY_GENERATED_TOTAL = Counter(
    "retrieve_chat_docset_summary_generated_total",
    "Total Phase 5A deterministic docset summaries generated by route and domain.",
    ("route", "domain"),
)

_DOCSET_SUMMARY_FAILURES_TOTAL = Counter(
    "retrieve_chat_docset_summary_failures_total",
    "Total Phase 5A deterministic docset summary construction failures by route and domain.",
    ("route", "domain"),
)

_CHAT_ANSWER_POLISH_SCAFFOLD_TOTAL = Counter(
    "retrieve_chat_answer_polish_scaffold_total",
    "Phase 5B prerequisite scaffold observability emissions (polish not active).",
    ("route",),
)

_CHAT_ANSWER_POLISH_SCAFFOLD_FAILURES_TOTAL = Counter(
    "retrieve_chat_answer_polish_scaffold_failures_total",
    "Phase 5B prerequisite scaffold observability failures (best-effort).",
    ("route",),
)

_KNOWLEDGE_VERIFIER_TOTAL = Counter(
    "mai_knowledge_verifier_total",
    "Phase 6A KNOWLEDGE verifier runs by route and status.",
    ("route", "status"),
)

_KNOWLEDGE_VERIFIER_CLAIMS_TOTAL = Counter(
    "mai_knowledge_verifier_claims_total",
    "Phase 6A KNOWLEDGE verifier claim outcomes by category and status.",
    ("category", "status"),
)

_KNOWLEDGE_VERIFIER_LATENCY_SECONDS = Histogram(
    "mai_knowledge_verifier_latency_seconds",
    "Phase 6A KNOWLEDGE verifier latency in seconds.",
    ("route",),
)

_KNOWLEDGE_VERIFIER_ERRORS_TOTAL = Counter(
    "mai_knowledge_verifier_errors_total",
    "Phase 6A KNOWLEDGE verifier internal errors by stage.",
    ("stage",),
)


def record_trust_gate(route: str, outcome: str) -> None:
    try:
        _TRUST_GATE_TOTAL.labels(route=route, outcome=outcome).inc()
    except Exception:  # pragma: no cover
        logger.debug("Failed to record trust gate metric", exc_info=True)


def record_docset_shadow(
    route: str,
    domain: str,
    duration_seconds: Optional[float],
    docs_matched: Optional[int],
    degraded: bool,
) -> None:
    """
    Record Phase 3A docset shadow mode metrics.

    Best-effort only; all failures are swallowed.
    """
    try:
        _DOCSET_SHADOW_RUN_TOTAL.labels(route=route, domain=domain).inc()
        if degraded:
            _DOCSET_SHADOW_DEGRADED_TOTAL.labels(route=route, domain=domain).inc()
        if duration_seconds is not None:
            _DOCSET_SHADOW_LATENCY_SECONDS.labels(
                route=route,
                domain=domain,
            ).observe(float(duration_seconds))
        if docs_matched is not None:
            _DOCSET_SHADOW_MATCHED_DOCS.labels(
                route=route,
                domain=domain,
            ).observe(float(docs_matched))
    except Exception:  # pragma: no cover
        logger.debug("Failed to record docset shadow metrics", exc_info=True)


def record_docset_golden_eval(
    route: str,
    domain: str,
    duration_seconds: Optional[float],
    precision: Optional[float],
    recall: Optional[float],
    degraded: bool,
) -> None:
    """
    Record Phase 3B docset golden evaluation metrics.

    Best-effort only; all failures are swallowed.
    """
    try:
        _DOCSET_GOLDEN_RUN_TOTAL.labels(route=route, domain=domain).inc()
        if degraded:
            _DOCSET_GOLDEN_DEGRADED_TOTAL.labels(route=route, domain=domain).inc()
        if duration_seconds is not None:
            _DOCSET_GOLDEN_LATENCY_SECONDS.labels(
                route=route,
                domain=domain,
            ).observe(float(duration_seconds))
        if precision is not None:
            _DOCSET_GOLDEN_PRECISION.labels(
                route=route,
                domain=domain,
            ).observe(float(precision))
        if recall is not None:
            _DOCSET_GOLDEN_RECALL.labels(
                route=route,
                domain=domain,
            ).observe(float(recall))
    except Exception:  # pragma: no cover
        logger.debug("Failed to record docset golden metrics", exc_info=True)


def record_docset_summary_generated(
    route: str,
    domain: str,
    *,
    summary_mode: str,
    matched_count: int,
    degraded: bool,
) -> None:
    """
    Record Phase 5A docset deterministic summary generation metrics.

    Best-effort only; all failures are swallowed.
    """
    try:
        _DOCSET_SUMMARY_GENERATED_TOTAL.labels(route=route, domain=domain).inc()
    except Exception:  # pragma: no cover
        logger.debug("Failed to record docset summary generated metric", exc_info=True)


def record_docset_summary_failure(
    route: str,
    domain: str,
    *,
    failure_type: str,
) -> None:
    """
    Record Phase 5A docset deterministic summary failure metrics.

    Best-effort only; all failures are swallowed.
    """
    try:
        _DOCSET_SUMMARY_FAILURES_TOTAL.labels(route=route, domain=domain).inc()
    except Exception:  # pragma: no cover
        logger.debug("Failed to record docset summary failure metric", exc_info=True)


def record_chat_answer_polish_scaffold(
    route: str,
    *,
    citation_integrity_valid: Optional[bool] = None,
) -> None:
    """
    Record Phase 5B answer-polish scaffold observability (no polish active).

    Best-effort only; all failures are swallowed.
    """
    try:
        _CHAT_ANSWER_POLISH_SCAFFOLD_TOTAL.labels(route=route).inc()
    except Exception:  # pragma: no cover
        logger.debug(
            "Failed to record chat answer polish scaffold metric", exc_info=True
        )


def record_chat_answer_polish_scaffold_failure(route: str) -> None:
    """Record scaffold observability emission failure (best-effort)."""
    try:
        _CHAT_ANSWER_POLISH_SCAFFOLD_FAILURES_TOTAL.labels(route=route).inc()
    except Exception:  # pragma: no cover
        logger.debug(
            "Failed to record chat answer polish scaffold failure metric",
            exc_info=True,
        )


def record_knowledge_verifier(
    route: str,
    status: str,
    duration_seconds: Optional[float] = None,
    *,
    claim_results: Optional[list] = None,
) -> None:
    """Record Phase 6A KNOWLEDGE verifier outcome and per-claim counts."""
    try:
        _KNOWLEDGE_VERIFIER_TOTAL.labels(route=route, status=status).inc()
        if duration_seconds is not None:
            _KNOWLEDGE_VERIFIER_LATENCY_SECONDS.labels(route=route).observe(
                float(duration_seconds)
            )
        if claim_results:
            for cr in claim_results:
                cat = getattr(cr, "category", "unknown")
                st_attr = getattr(cr, "status", None)
                st = getattr(st_attr, "value", str(st_attr))
                _KNOWLEDGE_VERIFIER_CLAIMS_TOTAL.labels(
                    category=str(cat),
                    status=str(st),
                ).inc()
    except Exception:  # pragma: no cover
        logger.debug("Failed to record knowledge verifier metrics", exc_info=True)


def record_knowledge_verifier_error(stage: str) -> None:
    """Record Phase 6A KNOWLEDGE verifier internal error (best-effort)."""
    try:
        _KNOWLEDGE_VERIFIER_ERRORS_TOTAL.labels(stage=stage).inc()
    except Exception:  # pragma: no cover
        logger.debug(
            "Failed to record knowledge verifier error metric", exc_info=True
        )

