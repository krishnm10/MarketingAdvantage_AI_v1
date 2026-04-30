"""
Structured runtime error hierarchy for the RAG pipeline.

All errors carry optional request-scoped metadata (tenant_id, request_id,
pipeline_id) to enable correlated diagnostics without requiring callers
to re-attach context at every raise site.

Hierarchy:
    RAGError (base)
    ├── RetrievalError
    ├── GenerationError
    ├── SecurityError
    │   └── TenantIsolationError
    ├── ConfigResolutionError
    └── VectorDBError
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class RAGError(Exception):
    """
    Base exception for all RAG pipeline runtime errors.

    Carries structured metadata for observability and tracing.
    """

    def __init__(
        self,
        message: str,
        *,
        tenant_id: Optional[str] = None,
        request_id: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.tenant_id = tenant_id
        self.request_id = request_id
        self.pipeline_id = pipeline_id
        self.details = details or {}

    @property
    def metadata(self) -> Dict[str, Any]:
        """Structured metadata dict for logging and telemetry."""
        meta: Dict[str, Any] = {
            "error_type": type(self).__name__,
            "message": str(self),
        }
        if self.tenant_id:
            meta["tenant_id"] = self.tenant_id
        if self.request_id:
            meta["request_id"] = self.request_id
        if self.pipeline_id:
            meta["pipeline_id"] = self.pipeline_id
        if self.details:
            meta["details"] = self.details
        return meta


class RetrievalError(RAGError):
    """Raised when vector search or retrieval stage fails."""

    pass


class GenerationError(RAGError):
    """Raised when LLM generation fails or returns an unsafe response."""

    pass


class SecurityError(RAGError):
    """Raised for security policy violations (PII leakage, injection, etc.)."""

    pass


class TenantIsolationError(SecurityError):
    """Raised when tenant isolation is violated or cannot be enforced."""

    def __init__(
        self,
        message: str,
        *,
        tenant_id: Optional[str] = None,
        request_id: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        attempted_tenant: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            tenant_id=tenant_id,
            request_id=request_id,
            pipeline_id=pipeline_id,
            details=details,
        )
        self.attempted_tenant = attempted_tenant

    @property
    def metadata(self) -> Dict[str, Any]:
        meta = super().metadata
        if self.attempted_tenant:
            meta["attempted_tenant"] = self.attempted_tenant
        return meta


class ConfigResolutionError(RAGError):
    """Raised when client config resolution or validation fails."""

    pass


class VectorDBError(RAGError):
    """Raised for vector database operational failures."""

    def __init__(
        self,
        message: str,
        *,
        tenant_id: Optional[str] = None,
        request_id: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        backend: Optional[str] = None,
        collection: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            tenant_id=tenant_id,
            request_id=request_id,
            pipeline_id=pipeline_id,
            details=details,
        )
        self.backend = backend
        self.collection = collection

    @property
    def metadata(self) -> Dict[str, Any]:
        meta = super().metadata
        if self.backend:
            meta["backend"] = self.backend
        if self.collection:
            meta["collection"] = self.collection
        return meta
