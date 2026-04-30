"""
================================================================================
Marketing Advantage AI — Runtime Module
File: app/core/runtime/__init__.py

The runtime module provides unified orchestration infrastructure for
RAG pipelines. All execution flows through config-driven executors.

Components:
  - SharedRetrievalExecutor: Unified retrieval (embed → search → rerank)
  - SharedGenerationExecutor: Unified generation (context → LLM → trust)
  - RAGRuntimeContext: Request-scoped context for telemetry and tracing
  - Runtime errors for structured error handling
================================================================================
"""

from app.core.runtime.runtime_flags import (
    ENABLE_SHARED_RETRIEVAL,
    ENABLE_SHARED_GENERATION,
    ENABLE_SHARED_CACHE,
    ENABLE_RUNTIME_CONTEXT,
)

from app.core.runtime.errors import (
    ConfigResolutionError,
    RetrievalError,
    GenerationError,
)

from app.core.runtime.runtime_context import RAGRuntimeContext
from app.core.runtime.runtime_constants import AUTHORITATIVE_RUNTIME

__all__ = [
    "ENABLE_SHARED_RETRIEVAL",
    "ENABLE_SHARED_GENERATION",
    "ENABLE_SHARED_CACHE",
    "ENABLE_RUNTIME_CONTEXT",
    "ConfigResolutionError",
    "RetrievalError",
    "GenerationError",
    "RAGRuntimeContext",
    "AUTHORITATIVE_RUNTIME",
]
