# core/tap/types.py

from dataclasses import dataclass
from typing import List, Literal, Dict, Optional


# -------------------------------
# Enumerations
# -------------------------------

PillarType = Literal[
    "market",
    "finance",
    "execution",
    "risk",
    "compliance",
    "timing",
]

RiskLevel = Literal[
    "low",
    "medium",
    "high",
    "veto",
]

TapDecisionType = Literal[
    "CONFIDENT_RECOMMENDATION",
    "CAUTIOUS_RECOMMENDATION",
    "EXPLORATORY_INSIGHT",
    "REFUSE",
]


# -------------------------------
# TrustPacket (Input to TAP)
# -------------------------------

@dataclass(frozen=True)
class TrustPacket:
    """
    Atomic trust signal emitted by Step-2 agents.
    TAP must NEVER receive raw agent text.
    """

    agent_id: str
    pillar: PillarType

    # Core trust signals
    confidence: float              # 0.0 → 1.0
    evidence_strength: float       # 0.0 → 1.0
    freshness_days: int

    # Risk signaling
    risk_level: RiskLevel

    # Optional trace (NOT chain-of-thought)
    rationale: Optional[str] = None


# -------------------------------
# TAP Aggregation Output
# -------------------------------

@dataclass(frozen=True)
class TapDecision:
    """
    Final governed decision emitted by TAP.
    """

    trust_score: float
    decision_type: TapDecisionType

    # Explanation safe for user consumption
    summary: str

    # Risk flags surfaced to UI / logs
    risk_flags: List[str]

    # Audit metadata (internal)
    audit: Dict[str, object]
