"""
Canonical telemetry field definitions for structured logging and tracing.

Every pipeline stage, API endpoint, and observability hook references
these constants to ensure consistent field names across all structured
logs, metrics, and traces. No runtime logic — definitions only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Field name constants — use these instead of raw strings in log emissions
# ─────────────────────────────────────────────────────────────────────────────

F_REQUEST_ID: str = "request_id"
F_TRACE_ID: str = "trace_id"
F_TENANT_ID: str = "tenant_id"
F_CLIENT_ID: str = "client_id"
F_PIPELINE_ID: str = "pipeline_id"

F_STACK_USED: str = "stack_used"
F_RUNTIME_AUTHORITY: str = "runtime_authority"

F_VECTORDB_BACKEND: str = "vectordb_backend"
F_EMBEDDER_PROVIDER: str = "embedder_provider"
F_EMBEDDER_MODEL: str = "embedder_model"
F_RERANKER_PROVIDER: str = "reranker_provider"
F_RERANKER_MODEL: str = "reranker_model"
F_LLM_PROVIDER: str = "llm_provider"
F_LLM_MODEL: str = "llm_model"

F_RETRIEVAL_MODE: str = "retrieval_mode"
F_SEARCH_MODE: str = "search_mode"

F_LATENCY_MS: str = "latency_ms"
F_TOKEN_USAGE: str = "token_usage"

F_RETRIEVED_COUNT: str = "retrieved_count"
F_RERANKED_COUNT: str = "reranked_count"
F_FINAL_COUNT: str = "final_count"

F_CACHE_HITS: str = "cache_hits"
F_CACHE_MISSES: str = "cache_misses"

F_THRESHOLD_GATE: str = "threshold_gate"
F_TOKEN_BUDGET: str = "token_budget"
F_SECURITY_EVENTS: str = "security_events"


# ─────────────────────────────────────────────────────────────────────────────
# Typed telemetry record — optional structured container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TelemetryRecord:
    """
    Typed container for one telemetry emission. Stages populate the
    fields they own; downstream collectors merge partial records by
    request_id / trace_id.
    """

    # Identity
    request_id: str = ""
    trace_id: str = ""
    tenant_id: str = ""
    client_id: str = ""
    pipeline_id: str = ""

    # Runtime
    stack_used: str = ""
    runtime_authority: str = ""

    # Components
    vectordb_backend: str = ""
    embedder_provider: str = ""
    embedder_model: str = ""
    reranker_provider: str = ""
    reranker_model: str = ""
    llm_provider: str = ""
    llm_model: str = ""

    # Retrieval
    retrieval_mode: str = ""
    search_mode: str = ""

    # Performance
    latency_ms: Dict[str, float] = field(default_factory=dict)
    token_usage: Dict[str, int] = field(default_factory=dict)

    # Counts
    retrieved_count: int = 0
    reranked_count: int = 0
    final_count: int = 0

    # Cache
    cache_hits: int = 0
    cache_misses: int = 0

    # Policy gates
    threshold_gate: Optional[Dict[str, Any]] = None
    token_budget: Optional[Dict[str, Any]] = None

    # Security
    security_events: List[str] = field(default_factory=list)
