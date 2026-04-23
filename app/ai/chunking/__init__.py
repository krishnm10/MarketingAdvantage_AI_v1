"""
app/ai/chunking — Enterprise-grade chunking tokenization alignment layer.

Exposes:
  ChunkingTokenCounter — provider-agnostic token counter for all chunking strategies.
  AlignmentMetrics     — calibration results for a (model, factory-backend) pair.
"""
from app.ai.chunking.token_counter import AlignmentMetrics, ChunkingTokenCounter

__all__ = ["AlignmentMetrics", "ChunkingTokenCounter"]
