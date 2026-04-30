"""
Unified runtime context carried through every stage of a RAG request.

This dataclass is the single request-scoped state object that flows from
API entry → pipeline → retrieval → generation → response. It replaces
ad-hoc parameter threading with one typed, inspectable structure.

No behavior methods — pure data carrier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RAGRuntimeContext:
    """
    Request-scoped context for a single RAG pipeline execution.

    Created at the API boundary, enriched as the request progresses,
    and available to every pipeline stage for consistent decision-making
    and telemetry emission.
    """

    # ── Request identity ─────────────────────────────────────────
    request_id: str
    trace_id: str
    tenant_id: str
    client_id: str
    pipeline_id: str

    # ── Configuration ────────────────────────────────────────────
    client_config: Optional[Any] = None

    # ── Query state ──────────────────────────────────────────────
    user_query: str = ""
    rewritten_queries: List[str] = field(default_factory=list)
    metadata_filters: Optional[Dict[str, Any]] = None

    # ── Security & feature gates ─────────────────────────────────
    security_flags: Dict[str, Any] = field(default_factory=dict)
    feature_flags: Dict[str, bool] = field(default_factory=dict)

    # ── Retrieval output ─────────────────────────────────────────
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Observability ────────────────────────────────────────────
    telemetry: Dict[str, Any] = field(default_factory=dict)

    # ── Cache control ────────────────────────────────────────────
    cache_policy: Dict[str, Any] = field(default_factory=dict)

    # ── Runtime identity ─────────────────────────────────────────
    stack_used: str = ""
    runtime_authority: str = ""
