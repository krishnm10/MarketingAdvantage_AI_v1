# app/retrieval/types.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List


# ---------------------------------------------------------
# Retrieval Intent (Governance-Level)
# ---------------------------------------------------------

class RetrievalIntent(str, Enum):
    """
    Defines WHY retrieval is happening.
    Controls policy strictness, not logic.
    """
    ANSWER = "answer"      # strict, high-trust
    EXPLORE = "explore"    # broader recall
    AUDIT = "audit"        # no filtering


class DomainType(str, Enum):
    """
    Coarse domain hint for L1 task classification (domain-agnostic platform).

    Phase 1 uses invoice as the first production-targeted domain; additional
    domains plug in via adapters in later phases without changing this enum's role.
    """

    INVOICE = "invoice"
    GENERIC = "generic"


# ---------------------------------------------------------
# Query Context (Immutable)
# ---------------------------------------------------------

@dataclass(frozen=True)
class QueryContext:
    """
    Immutable query context passed through retrieval.
    """
    query: str
    intent: RetrievalIntent
    business_id: Optional[str] = None
    requested_at: Optional[int] = None


# ---------------------------------------------------------
# Semantic Signal (Relevance)
# ---------------------------------------------------------

@dataclass(frozen=True)
class SemanticSignal:
    """
    Semantic relevance between query and content.
    """
    score: float  # normalized [0,1]


# ---------------------------------------------------------
# Trust & Governance Signals
# ---------------------------------------------------------

@dataclass(frozen=True)
class TrustSignals:
    """
    All governance-related signals.
    Produced BEFORE retrieval.
    """
    tap_trust_score: float
    agentic_validation_score: float
    reasoning_quality_score: float
    conflict_modifier: float
    temporal_decay: float
    # "validated" = real trust scores present; "unvalidated" = no validation_layer found
    trust_state: str = "validated"

    @property
    def is_unvalidated(self) -> bool:
        return self.trust_state == "unvalidated"


# ---------------------------------------------------------
# Retrieval Candidate (Internal Only)
# ---------------------------------------------------------

@dataclass
class RetrievalCandidate:
    """
    Internal retrieval unit.
    This object MAY be scored, but signals must never be mutated.
    """
    chunk_id: str
    text: str

    semantic: SemanticSignal
    trust: TrustSignals

    final_score: Optional[float] = None
    file_id: Optional[str] = None


# ---------------------------------------------------------
# Dropped Candidate (Transparency)
# ---------------------------------------------------------

@dataclass(frozen=True)
class DroppedCandidate:
    """
    Represents a candidate that was retrieved
    but rejected during scoring.
    """
    chunk_id: str
    reason: str


# ---------------------------------------------------------
# Final Ranked Result (External Contract)
# ---------------------------------------------------------

@dataclass(frozen=True)
class RankedResult:
    """
    Final ranked retrieval output.
    Safe to expose to API / CLI / LLM.
    """
    chunk_id: str
    text: str
    score: float
    explanation: Dict
    # 🔒 Governance decision (TRUSTED / PROVISIONAL / REJECTED)
    trust_decision: Optional[str] = None
    file_id: Optional[str] = None