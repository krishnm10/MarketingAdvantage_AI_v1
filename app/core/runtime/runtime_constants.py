"""
Runtime identity constants for pipeline selection.

These are the canonical identifiers used across config resolution,
pipeline factory, traffic routing, and telemetry to distinguish
which runtime is executing a request.
"""

AUTHORITATIVE_RUNTIME: str = "rag_pipeline"
LEGACY_RUNTIME: str = "retrieval_runtime"

LEGACY_RUNTIME_DEPRECATED: bool = True
