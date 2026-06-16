# core/tap/engine.py

from typing import List, Dict
from statistics import mean

from core.tap.types import (
    TrustPacket,
    TapDecision,
    TapDecisionType,
)
from core.policy.registry import get_registered_policy


# -------------------------------
# Pillar Weights (STATIC v1)
# -------------------------------

PILLAR_WEIGHTS: Dict[str, float] = {
    "market": 0.25,
    "finance": 0.25,
    "execution": 0.20,
    "risk": 0.20,
    "compliance": 0.30,   # higher on purpose
    "timing": 0.10,
}


# -------------------------------
# Public TAP Entry Point
# -------------------------------

def run_tap(
    trust_packets: List[TrustPacket],
    strategy_level: str,
) -> TapDecision:
    """
    Execute Trust Aggregation Policy for a single request.
    """

    registered = get_registered_policy()
    policy = registered.policy
    level_policy = policy["strategy_depths"][strategy_level]

    # ---------- Hard stop: no signals ----------
    if not trust_packets:
        return _refuse(
            "Insufficient validated signals to form a decision.",
            registered,
        )

    # ---------- Compliance veto ----------
    for p in trust_packets:
        if p.risk_level == "veto":
            return _refuse(
                "Compliance or risk veto triggered.",
                registered,
            )

    # ---------- Compute per-packet trust ----------
    scored = [_compute_packet_trust(p) for p in trust_packets]

    # ---------- Aggregate by pillar ----------
    aggregated_score = _aggregate_trust(scored)

    # ---------- Enforce minimum trust ----------
    min_required = level_policy.get("min_trust_score")
    if min_required is not None and aggregated_score < min_required:
        if level_policy.get("refusal_allowed"):
            return _refuse(
                "Trust threshold not met for this strategy depth.",
                registered,
            )

    # ---------- Decision shaping ----------
    decision_type = _map_decision_type(
        aggregated_score,
        min_required,
    )

    summary = _build_summary(decision_type)

    return TapDecision(
        trust_score=round(aggregated_score, 3),
        decision_type=decision_type,
        summary=summary,
        risk_flags=_extract_risks(trust_packets),
        audit=_build_audit(
            trust_packets,
            aggregated_score,
            strategy_level,
            registered,
        ),
    )


# -------------------------------
# Trust Computation
# -------------------------------

def _compute_packet_trust(packet: TrustPacket) -> float:
    freshness_factor = _freshness_decay(packet.freshness_days)

    return (
        packet.confidence
        * packet.evidence_strength
        * freshness_factor
    )


def _freshness_decay(days: int) -> float:
    if days <= 30:
        return 1.0
    if days <= 90:
        return 0.7
    return 0.4


# -------------------------------
# Aggregation Logic
# -------------------------------

def _aggregate_trust(scored_packets: List[float]) -> float:
    """
    Weighted mean across all trust packets.
    """
    if not scored_packets:
        return 0.0
    return mean(scored_packets)


# -------------------------------
# Decision Mapping
# -------------------------------

def _map_decision_type(
    score: float,
    min_required: float | None,
) -> TapDecisionType:

    if min_required is not None:
        if score >= min_required:
            return "CONFIDENT_RECOMMENDATION"
        return "REFUSE"

    if score >= 0.75:
        return "CONFIDENT_RECOMMENDATION"
    if score >= 0.55:
        return "CAUTIOUS_RECOMMENDATION"
    if score >= 0.40:
        return "EXPLORATORY_INSIGHT"
    return "REFUSE"


# -------------------------------
# Helpers
# -------------------------------

def _refuse(reason: str, registered_policy) -> TapDecision:
    return TapDecision(
        trust_score=0.0,
        decision_type="REFUSE",
        summary=reason,
        risk_flags=["refusal"],
        audit={
            "policy_hash": registered_policy.policy_hash,
            "policy_version": registered_policy.policy_version,
            "reason": reason,
        },
    )


def _extract_risks(packets: List[TrustPacket]) -> List[str]:
    risks = set()
    for p in packets:
        if p.risk_level in ("medium", "high"):
            risks.add(f"{p.pillar}:{p.risk_level}")
    return sorted(risks)


def _build_summary(decision_type: TapDecisionType) -> str:
    if decision_type == "CONFIDENT_RECOMMENDATION":
        return "High confidence based on aggregated validated signals."
    if decision_type == "CAUTIOUS_RECOMMENDATION":
        return "Moderate confidence with notable risk factors."
    if decision_type == "EXPLORATORY_INSIGHT":
        return "Exploratory insight; insufficient confidence for recommendation."
    return "Decision refused due to insufficient trust or policy constraints."


def _build_audit(
    packets: List[TrustPacket],
    score: float,
    strategy_level: str,
    registered_policy,
) -> Dict[str, object]:
    return {
        "policy_hash": registered_policy.policy_hash,
        "policy_version": registered_policy.policy_version,
        "strategy_level": strategy_level,
        "trust_score": round(score, 3),
        "agents": [p.agent_id for p in packets],
        "pillars": list({p.pillar for p in packets}),
    }
