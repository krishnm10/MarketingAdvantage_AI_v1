# app/ai/validation/__init__.py
# Phase 1 validation layer — public API.

from app.ai.validation.exceptions import (
    TokenOverflowError,
    AlignmentError,
)
from app.ai.validation.chunk_sizer import (
    ValidationResult,
    ChunkSizerContract,
    DefaultChunkSizer,
)
from app.ai.validation.tokenizer_validator import (
    AlignmentReport,
    TokenizerValidator,
    EmbeddingSpaceMigrationGuard,
    tokenizer_validator,
    migration_guard,
)

__all__ = [
    # Exceptions
    "TokenOverflowError",
    "AlignmentError",
    # Chunk sizing
    "ValidationResult",
    "ChunkSizerContract",
    "DefaultChunkSizer",
    # Pipeline alignment
    "AlignmentReport",
    "TokenizerValidator",
    "EmbeddingSpaceMigrationGuard",
    "tokenizer_validator",
    "migration_guard",
]
