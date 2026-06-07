from __future__ import annotations

"""
Lightweight registry for document-set analysis adapters.

Phase 2 scope:
- Domain-only mapping (DomainType -> adapter instance).
- No tenant-level flag policy here; gating is performed in the analysis core
  and/or handler path using RuntimeComponents / ClientConfig.
- No external IO or dynamic plugin loading.
"""

from typing import Dict, List, Optional, Protocol

from app.retrieval.types_retrieve import DomainType
from app.retrieval.document_match import DocumentMatch


class DocumentSetAdapter(Protocol):
    """
    Minimal protocol for domain-specific document-set analysis adapters.

    Adapters work solely on already-retrieved RankedResult content and must be
    deterministic and side-effect free.
    """

    def analyze(
        self,
        *,
        grouped_chunks,
        raw_query: str,
        task_plan,
        tenant_runtime,
    ) -> List[DocumentMatch]:
        """
        Perform domain-specific analysis on grouped chunks.

        Parameters
        ----------
        grouped_chunks:
            Mapping-like structure from file_id -> List[RankedResult] (or a small
            helper type), as produced by the analysis core.
        raw_query:
            Original user query string (post-security-scan).
        task_plan:
            L1 TaskPlan from app.retrieval.task_classifier.
        tenant_runtime:
            RuntimeComponents (or compatible snapshot) for this tenant.

        Returns
        -------
        List[DocumentMatch]:
            A list of normalized DocumentMatch instances ready for serialization.
        """
        ...


_ADAPTERS: Dict[DomainType, DocumentSetAdapter] = {}


def register_adapter(domain: DomainType, adapter: DocumentSetAdapter) -> None:
    """
    Register or override an adapter for a given domain.

    Phase 2 uses a static registration from the invoice adapter module;
    this function exists primarily for explicitness and tests.
    """
    _ADAPTERS[domain] = adapter


def get_adapter(domain: DomainType) -> Optional[DocumentSetAdapter]:
    """
    Resolve a document-set adapter for the given domain.

    Returns None when no adapter is registered. Tenant-scoped feature flags and
    policy decisions are handled by callers, not by this registry.
    """
    return _ADAPTERS.get(domain)


def _reset_adapters_for_test() -> None:
    """
    TEST-ONLY — do not call from production paths.

    Clears the in-memory adapter registry for pytest isolation. Leading underscore
    marks this as non-public; no app/ handler or analysis module imports this.
    """
    _ADAPTERS.clear()

