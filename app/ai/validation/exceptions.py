# =============================================================================
# app/ai/validation/exceptions.py
#
# All custom exceptions for the Phase 1 validation layer.
#
# Every exception carries explicit failure-mode codes (F-01…F-16) so that
# monitoring dashboards and alert rules can be mapped directly to root causes.
#
# RULE: No exception may be raised without a failure_mode code. This is the
# production contract — all callers can rely on it.
# =============================================================================

from __future__ import annotations

from typing import Any, Dict, List


class TokenOverflowError(ValueError):
    """
    Raised when a text chunk exceeds the safe token limit.

    Corresponds to failure modes: F-02, F-13, F-14, F-15.

    Fields
    ──────
    chunk_tokens  : Actual token count of the offending chunk.
    max_allowed   : Safe limit derived from ChunkSizerContract.safe_chunk_size().
    model_id      : Embedder model identifier.
    failure_mode  : Primary failure mode code from F-02/F-13/F-14/F-15.
    """

    def __init__(
        self,
        chunk_tokens: int,
        max_allowed: int,
        model_id: str,
        failure_mode: str,
    ) -> None:
        self.chunk_tokens = chunk_tokens
        self.max_allowed = max_allowed
        self.model_id = model_id
        self.failure_mode = failure_mode
        super().__init__(
            f"[TokenOverflowError {failure_mode}] "
            f"model_id={model_id!r}: "
            f"chunk_tokens={chunk_tokens} exceeds max_allowed={max_allowed}. "
            "Reduce chunk size or increase embed_max_tokens budget. "
            f"Failure mode: {failure_mode}."
        )


class AlignmentError(ValueError):
    """
    Raised when two or more pipeline components are misaligned in a way
    that would cause incorrect or silent retrieval failures.

    Raised by TokenizerValidator.validate_pipeline_alignment() for any
    BLOCKING misalignment.

    Fields
    ──────
    failure_modes      : List of F-codes, e.g. ["F-04", "F-08", "F-16"].
    embedder_model_id  : The embedder model_id being validated.
    vectordb_kind      : The VectorDB backend name.
    details            : Dict of field→error mappings with full context.
    """

    def __init__(
        self,
        failure_modes: List[str],
        embedder_model_id: str,
        vectordb_kind: str,
        details: Dict[str, Any],
    ) -> None:
        self.failure_modes = failure_modes
        self.embedder_model_id = embedder_model_id
        self.vectordb_kind = vectordb_kind
        self.details = details
        detail_str = "; ".join(f"{k}={v!r}" for k, v in details.items())
        modes_str = ", ".join(failure_modes)
        super().__init__(
            f"[AlignmentError {modes_str}] "
            f"embedder={embedder_model_id!r} vectordb={vectordb_kind!r}: "
            f"{detail_str}"
        )
