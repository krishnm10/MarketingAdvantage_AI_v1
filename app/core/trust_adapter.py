"""
Trust Scoring Adapter — Phase 1
Bridges the gap between RAGPipeline (which operates on chunk dicts) and the
existing trust_calculator module at app/retrieval/trust_calculator.py.

The real trust_calculator exposes formula-level functions that require
TrustSignals dataclass instances (tap_trust, agentic_validation, etc.).
When chunks carry those signals in metadata, the adapter constructs
TrustSignals and delegates to compute_policy_trust_score.

When TrustSignals metadata is absent, or the module cannot be imported,
the adapter falls back to average-similarity scoring and logs clearly.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TRUST_CALCULATOR_AVAILABLE = False
_compute_policy_trust_score = None
_TrustSignals = None

try:
    from app.retrieval.trust_calculator import compute_policy_trust_score as _cpts
    from app.retrieval.types_retrieve import TrustSignals as _TS

    _compute_policy_trust_score = _cpts
    _TrustSignals = _TS
    _TRUST_CALCULATOR_AVAILABLE = True
    logger.info("[TrustAdapter] Real trust_calculator loaded from app.retrieval.trust_calculator")
except ImportError:
    logger.warning(
        "[TrustAdapter] app.retrieval.trust_calculator not importable — "
        "will use fallback average-similarity scoring."
    )


class TrustAdapter:
    """Stable trust-scoring interface consumed by RAGPipeline."""

    def calculate(
        self,
        chunks: List[Dict[str, Any]],
        *,
        pii_redacted: bool = False,
        pii_penalty: float = 0.15,
        reranker_confidence: Optional[float] = None,
    ) -> float:
        """
        Compute a trust score in [0.0, 1.0] for a set of context chunks.

        Args:
            chunks: Context chunk dicts (must contain ``score`` or ``rerank_score``).
            pii_redacted: If True, subtract *pii_penalty* from the base score.
            pii_penalty: Penalty magnitude when PII was redacted (default 0.15).
            reranker_confidence: Optional reranker-level confidence to blend in.

        Returns:
            Clamped float in [0.0, 1.0].
        """
        base_score = self._base_score(chunks)

        if reranker_confidence is not None:
            base_score = 0.7 * base_score + 0.3 * reranker_confidence

        if pii_redacted:
            base_score -= pii_penalty

        return max(0.0, min(1.0, base_score))

    # ------------------------------------------------------------------
    # Internal scoring strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_has_trust_signals(chunk: Dict[str, Any]) -> bool:
        meta = chunk.get("metadata") or {}
        if not isinstance(meta, dict):
            return False
        return any(
            meta.get(key) is not None
            for key in (
                "tap_trust_score",
                "agentic_validation_score",
                "reasoning_quality_score",
            )
        )

    @staticmethod
    def _base_score(chunks: List[Dict[str, Any]]) -> float:
        if not chunks:
            return 0.0

        if _TRUST_CALCULATOR_AVAILABLE and any(
            TrustAdapter._chunk_has_trust_signals(c) for c in chunks
        ):
            try:
                scores: list[float] = []
                for chunk in chunks:
                    meta = chunk.get("metadata", {})
                    signals = _TrustSignals(
                        tap_trust_score=float(meta.get("tap_trust_score", 0.0)),
                        agentic_validation_score=float(meta.get("agentic_validation_score", 0.0)),
                        reasoning_quality_score=float(meta.get("reasoning_quality_score", 0.0)),
                        conflict_modifier=float(meta.get("conflict_modifier", 1.0)),
                        temporal_decay=float(meta.get("temporal_decay", 1.0)),
                    )
                    scores.append(_compute_policy_trust_score(signals))

                if scores:
                    avg = sum(scores) / len(scores)
                    logger.debug(
                        "[TrustAdapter] Real calculator — avg=%.4f over %d chunks",
                        avg, len(scores),
                    )
                    return avg
            except Exception as e:
                logger.warning(
                    "[TrustAdapter] Real calculator failed (%s), using fallback",
                    e,
                )

        return TrustAdapter._fallback_average(chunks)

    @staticmethod
    def _fallback_average(chunks: List[Dict[str, Any]]) -> float:
        """Last resort: average similarity / rerank score across chunks."""
        logger.debug("[TrustAdapter] Using fallback average-similarity scoring")
        scores = [
            c.get("rerank_score") or c.get("score") or 0.0
            for c in chunks
        ]
        valid = [float(s) for s in scores if isinstance(s, (int, float)) and s > 0]
        if not valid:
            return 0.0
        return sum(valid) / len(valid)
