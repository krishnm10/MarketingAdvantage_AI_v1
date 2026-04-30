"""
================================================================================
Dual-Run Comparison Harness — Legacy vs. Unified Runtime Parity Test

Runs the SAME query through both execution paths using identical:
  - tenant_id
  - ClientConfig
  - components (vectordb, embedder, reranker, llm, nodes)

Captures and compares:
  - outputs       (final_answer, context_chunks, retrieved_chunks)
  - scores        (trust_score, chunk scores, reranked flag)
  - timing        (per-stage latency breakdown)
  - telemetry     (captured log records from mai.runtime.telemetry)

Usage:
    from app.core.runtime.dual_run_harness import DualRunHarness

    harness = DualRunHarness(config=my_config, components=my_components)
    report = harness.run("What was Q3 marketing ROI?")
    report.print_summary()
    report.assert_parity()  # raises if outputs diverge
================================================================================
"""

from __future__ import annotations

import copy
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Dict, List, Optional, Union

from app.core.config.client_config_schema import ClientConfig
from app.core.rag_pipeline import RAGPipeline, RAGResult
from app.core.runtime.runtime_context import RAGRuntimeContext
from app.core.runtime.runtime_constants import AUTHORITATIVE_RUNTIME, LEGACY_RUNTIME


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry capture
# ─────────────────────────────────────────────────────────────────────────────

class _TelemetryCapture(logging.Handler):
    """
    Logging handler that captures all mai.runtime.telemetry records
    as parsed JSON events for post-run comparison.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: List[Dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = json.loads(record.getMessage())
            payload["_ts"] = record.created
            payload["_level"] = record.levelname
            self.records.append(payload)
        except (json.JSONDecodeError, TypeError):
            self.records.append({
                "event": "RAW_LOG",
                "message": record.getMessage(),
                "_ts": record.created,
                "_level": record.levelname,
            })

    def drain(self) -> List[Dict[str, Any]]:
        captured = list(self.records)
        self.records.clear()
        return captured


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline components bundle
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineComponents:
    """All components needed to construct both runtime paths."""

    vectordb: Any
    embedder: Any
    reranker: Optional[Any] = None
    llm: Optional[Any] = None
    nodes: Optional[Any] = None


# ─────────────────────────────────────────────────────────────────────────────
# Run snapshot — captures everything from one execution
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunSnapshot:
    """Complete capture of one runtime execution."""

    label: str
    query: str
    tenant_id: str

    # Outputs
    final_answer: Optional[str] = None
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    reranked_chunks: Optional[List[Dict[str, Any]]] = None
    context_chunks: List[Dict[str, Any]] = field(default_factory=list)
    reranked: bool = False

    # Scores
    trust_score: Optional[float] = None
    chunk_scores: List[float] = field(default_factory=list)

    # Security
    pii_redacted: bool = False
    pii_entities_found: List[str] = field(default_factory=list)
    response_blocked: bool = False
    block_reason: Optional[str] = None

    # Timing
    latency: Dict[str, Any] = field(default_factory=dict)
    wall_ms: float = 0.0

    # Telemetry
    telemetry_events: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ─────────────────────────────────────────────────────────────────────────────
# Comparison report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComparisonReport:
    """Side-by-side comparison of legacy and unified runs."""

    query: str
    tenant_id: str
    legacy: RunSnapshot
    unified: RunSnapshot
    diffs: List[str] = field(default_factory=list)

    @property
    def parity(self) -> bool:
        return len(self.diffs) == 0

    def print_summary(self) -> str:
        buf = StringIO()
        _w = buf.write

        _w("=" * 78 + "\n")
        _w("DUAL-RUN COMPARISON REPORT\n")
        _w("=" * 78 + "\n")
        _w(f"Query:     {self.query[:80]}\n")
        _w(f"Tenant:    {self.tenant_id}\n")
        _w(f"Parity:    {'PASS' if self.parity else 'FAIL'}\n")
        _w("-" * 78 + "\n")

        _w("\n-- OUTPUTS -----------------------------------------\n")
        _w(f"  {'':30s} {'Legacy':>20s}  {'Unified':>20s}\n")
        _w(f"  {'Answer length':30s} {_safe_len(self.legacy.final_answer):>20s}  "
           f"{_safe_len(self.unified.final_answer):>20s}\n")
        _w(f"  {'Retrieved count':30s} {len(self.legacy.retrieved_chunks):>20d}  "
           f"{len(self.unified.retrieved_chunks):>20d}\n")
        _w(f"  {'Reranked':30s} {str(self.legacy.reranked):>20s}  "
           f"{str(self.unified.reranked):>20s}\n")
        _w(f"  {'Context count':30s} {len(self.legacy.context_chunks):>20d}  "
           f"{len(self.unified.context_chunks):>20d}\n")
        _w(f"  {'Answers match':30s} "
           f"{'YES' if self.legacy.final_answer == self.unified.final_answer else 'NO':>20s}\n")

        _w("\n-- SCORES ------------------------------------------\n")
        _w(f"  {'Trust score':30s} {_fmt_score(self.legacy.trust_score):>20s}  "
           f"{_fmt_score(self.unified.trust_score):>20s}\n")
        _w(f"  {'PII redacted':30s} {str(self.legacy.pii_redacted):>20s}  "
           f"{str(self.unified.pii_redacted):>20s}\n")
        _w(f"  {'Response blocked':30s} {str(self.legacy.response_blocked):>20s}  "
           f"{str(self.unified.response_blocked):>20s}\n")

        _w("\n-- TIMING (ms) -------------------------------------\n")
        all_keys = sorted(
            set(list(self.legacy.latency.keys()) + list(self.unified.latency.keys()))
        )
        for k in all_keys:
            lv = self.legacy.latency.get(k)
            uv = self.unified.latency.get(k)
            _w(f"  {k:30s} {_fmt_timing(lv):>20s}  {_fmt_timing(uv):>20s}\n")
        _w(f"  {'wall_ms':30s} {self.legacy.wall_ms:>20.2f}  "
           f"{self.unified.wall_ms:>20.2f}\n")

        _w("\n-- TELEMETRY ---------------------------------------\n")
        _w(f"  {'Events captured':30s} {len(self.legacy.telemetry_events):>20d}  "
           f"{len(self.unified.telemetry_events):>20d}\n")
        leg_events = [e.get("event", "?") for e in self.legacy.telemetry_events]
        uni_events = [e.get("event", "?") for e in self.unified.telemetry_events]
        _w(f"  Legacy events:  {leg_events}\n")
        _w(f"  Unified events: {uni_events}\n")

        if self.diffs:
            _w("\n-- DIFFS -------------------------------------------\n")
            for d in self.diffs:
                _w(f"  [!] {d}\n")

        _w("=" * 78 + "\n")

        text = buf.getvalue()
        print(text)
        return text

    def assert_parity(self) -> None:
        """Raise AssertionError with diff details if parity fails."""
        if not self.parity:
            details = "\n".join(f"  - {d}" for d in self.diffs)
            raise AssertionError(
                f"Dual-run parity FAILED with {len(self.diffs)} difference(s):\n"
                f"{details}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serializable representation for logging or storage."""
        return {
            "query": self.query,
            "tenant_id": self.tenant_id,
            "parity": self.parity,
            "diffs": self.diffs,
            "legacy": {
                "answer_len": len(self.legacy.final_answer or ""),
                "retrieved": len(self.legacy.retrieved_chunks),
                "context": len(self.legacy.context_chunks),
                "reranked": self.legacy.reranked,
                "trust_score": self.legacy.trust_score,
                "pii_redacted": self.legacy.pii_redacted,
                "blocked": self.legacy.response_blocked,
                "latency": self.legacy.latency,
                "wall_ms": self.legacy.wall_ms,
                "telemetry_count": len(self.legacy.telemetry_events),
                "error": self.legacy.error,
            },
            "unified": {
                "answer_len": len(self.unified.final_answer or ""),
                "retrieved": len(self.unified.retrieved_chunks),
                "context": len(self.unified.context_chunks),
                "reranked": self.unified.reranked,
                "trust_score": self.unified.trust_score,
                "pii_redacted": self.unified.pii_redacted,
                "blocked": self.unified.response_blocked,
                "latency": self.unified.latency,
                "wall_ms": self.unified.wall_ms,
                "telemetry_count": len(self.unified.telemetry_events),
                "error": self.unified.error,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Harness
# ─────────────────────────────────────────────────────────────────────────────

class DualRunHarness:
    """
    Runs the same query through legacy and unified paths, captures everything,
    and produces a structured ComparisonReport.

    Both paths are constructed from the SAME config and components.
    The harness temporarily patches feature flags to force each path,
    then restores them after each run.
    """

    def __init__(
        self,
        *,
        config: ClientConfig,
        components: PipelineComponents,
    ) -> None:
        self._config = config
        self._components = components
        self._telemetry_logger = logging.getLogger("mai.runtime.telemetry")

    def run(
        self,
        query: str,
        *,
        metadata_filters: Optional[Dict[str, Any]] = None,
        top_k_retrieval: Optional[int] = None,
        top_k_final: Optional[int] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ComparisonReport:
        """
        Execute the dual-run comparison.

        1. Run with flags OFF  → legacy path (_legacy_retrieve + _legacy_generate)
        2. Run with flags ON   → unified path (SharedRetrievalExecutor + SharedGenerationExecutor)
        3. Compare outputs, scores, timing, telemetry
        """
        tenant_id = self._config.client_id
        request_id = str(uuid.uuid4())

        query_params = dict(
            metadata_filters=metadata_filters,
            top_k_retrieval=top_k_retrieval,
            top_k_final=top_k_final,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        legacy_snap = self._run_path(
            label="legacy",
            query=query,
            tenant_id=tenant_id,
            request_id=f"{request_id}::legacy",
            enable_shared_retrieval=False,
            enable_shared_generation=False,
            query_params=query_params,
        )

        unified_snap = self._run_path(
            label="unified",
            query=query,
            tenant_id=tenant_id,
            request_id=f"{request_id}::unified",
            enable_shared_retrieval=True,
            enable_shared_generation=True,
            query_params=query_params,
        )

        diffs = self._compare(legacy_snap, unified_snap)

        return ComparisonReport(
            query=query,
            tenant_id=tenant_id,
            legacy=legacy_snap,
            unified=unified_snap,
            diffs=diffs,
        )

    def run_batch(
        self,
        queries: List[str],
        **kwargs: Any,
    ) -> List[ComparisonReport]:
        """Run multiple queries and return all reports."""
        return [self.run(q, **kwargs) for q in queries]

    # ── Internal ──────────────────────────────────────────────────────────

    def _run_path(
        self,
        *,
        label: str,
        query: str,
        tenant_id: str,
        request_id: str,
        enable_shared_retrieval: bool,
        enable_shared_generation: bool,
        query_params: Dict[str, Any],
    ) -> RunSnapshot:
        """Execute one path with controlled feature flags."""
        import app.core.runtime.runtime_flags as _flags

        original_retrieval = _flags.ENABLE_SHARED_RETRIEVAL
        original_generation = _flags.ENABLE_SHARED_GENERATION

        capture = _TelemetryCapture()
        self._telemetry_logger.addHandler(capture)

        snap = RunSnapshot(label=label, query=query, tenant_id=tenant_id)

        try:
            _flags.ENABLE_SHARED_RETRIEVAL = enable_shared_retrieval
            _flags.ENABLE_SHARED_GENERATION = enable_shared_generation

            pipeline = RAGPipeline(
                vectordb=self._components.vectordb,
                embedder=self._components.embedder,
                llm=self._components.llm,
                reranker=self._components.reranker,
                config=self._config,
                nodes=self._components.nodes,
            )

            ctx = RAGRuntimeContext(
                request_id=request_id,
                trace_id=request_id,
                tenant_id=tenant_id,
                client_id=self._config.client_id,
                pipeline_id=f"dual_run_harness::{label}",
                user_query=query,
                feature_flags={
                    "ENABLE_SHARED_RETRIEVAL": enable_shared_retrieval,
                    "ENABLE_SHARED_GENERATION": enable_shared_generation,
                },
                stack_used="rag_pipeline",
                runtime_authority=AUTHORITATIVE_RUNTIME,
            )

            t0 = time.perf_counter()
            result: RAGResult = pipeline.query(
                query,
                runtime_context=ctx,
                **{k: v for k, v in query_params.items() if v is not None},
            )
            snap.wall_ms = round((time.perf_counter() - t0) * 1000, 2)

            snap.final_answer = result.final_answer
            snap.retrieved_chunks = result.retrieved_chunks
            snap.reranked_chunks = result.reranked_chunks
            snap.context_chunks = result.context_chunks
            snap.reranked = result.reranked
            snap.trust_score = result.trust_score
            snap.chunk_scores = [
                c.get("score", 0.0) for c in result.context_chunks
            ]
            snap.pii_redacted = result.pii_redacted
            snap.pii_entities_found = result.pii_entities_found
            snap.response_blocked = result.response_blocked
            snap.block_reason = result.block_reason
            snap.latency = dict(result.latency)
            snap.metadata = dict(result.metadata)

        except Exception as exc:
            snap.error = f"{type(exc).__name__}: {exc}"
            snap.wall_ms = round((time.perf_counter() - t0) * 1000, 2)

        finally:
            _flags.ENABLE_SHARED_RETRIEVAL = original_retrieval
            _flags.ENABLE_SHARED_GENERATION = original_generation
            self._telemetry_logger.removeHandler(capture)
            snap.telemetry_events = capture.drain()

        return snap

    @staticmethod
    def _compare(legacy: RunSnapshot, unified: RunSnapshot) -> List[str]:
        """Produce a list of human-readable diff strings."""
        diffs: List[str] = []

        if legacy.error and not unified.error:
            diffs.append(f"Legacy errored ({legacy.error}) but unified succeeded")
        elif unified.error and not legacy.error:
            diffs.append(f"Unified errored ({unified.error}) but legacy succeeded")
        elif legacy.error and unified.error:
            diffs.append(f"Both errored: legacy={legacy.error}, unified={unified.error}")
            return diffs

        if not legacy.ok or not unified.ok:
            return diffs

        if legacy.final_answer != unified.final_answer:
            la = (legacy.final_answer or "")[:120]
            ua = (unified.final_answer or "")[:120]
            diffs.append(f"final_answer differs: legacy={la!r} vs unified={ua!r}")

        if len(legacy.retrieved_chunks) != len(unified.retrieved_chunks):
            diffs.append(
                f"retrieved_chunks count: {len(legacy.retrieved_chunks)} vs "
                f"{len(unified.retrieved_chunks)}"
            )

        if len(legacy.context_chunks) != len(unified.context_chunks):
            diffs.append(
                f"context_chunks count: {len(legacy.context_chunks)} vs "
                f"{len(unified.context_chunks)}"
            )

        if legacy.reranked != unified.reranked:
            diffs.append(
                f"reranked flag: {legacy.reranked} vs {unified.reranked}"
            )

        leg_ids = [c.get("id") for c in legacy.context_chunks]
        uni_ids = [c.get("id") for c in unified.context_chunks]
        if leg_ids != uni_ids:
            diffs.append(f"context chunk IDs differ: {leg_ids} vs {uni_ids}")

        leg_scores = [round(c.get("score", 0.0), 6) for c in legacy.context_chunks]
        uni_scores = [round(c.get("score", 0.0), 6) for c in unified.context_chunks]
        if leg_scores != uni_scores:
            diffs.append(f"context chunk scores differ: {leg_scores} vs {uni_scores}")

        if _score_diff(legacy.trust_score, unified.trust_score) > 0.001:
            diffs.append(
                f"trust_score: {legacy.trust_score} vs {unified.trust_score}"
            )

        if legacy.pii_redacted != unified.pii_redacted:
            diffs.append(
                f"pii_redacted: {legacy.pii_redacted} vs {unified.pii_redacted}"
            )

        if legacy.response_blocked != unified.response_blocked:
            diffs.append(
                f"response_blocked: {legacy.response_blocked} vs "
                f"{unified.response_blocked}"
            )

        if set(legacy.pii_entities_found) != set(unified.pii_entities_found):
            diffs.append(
                f"pii_entities differ: {legacy.pii_entities_found} vs "
                f"{unified.pii_entities_found}"
            )

        for key in ("embed_ms", "vectordb_ms", "rerank_ms", "llm_ms"):
            lv = legacy.latency.get(key)
            uv = unified.latency.get(key)
            if (lv is None) != (uv is None):
                diffs.append(f"latency key '{key}' present in one but not other")

        return diffs


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_len(s: Optional[str]) -> str:
    return str(len(s)) if s is not None else "None"


def _fmt_score(s: Optional[float]) -> str:
    return f"{s:.4f}" if s is not None else "None"


def _fmt_timing(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, (int, float)):
        return f"{v:.2f}"
    return str(v)[:20]


def _score_diff(a: Optional[float], b: Optional[float]) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    return abs(a - b)
