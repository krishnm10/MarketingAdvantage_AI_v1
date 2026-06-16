# core/tap/adapter.py

from typing import Any, Dict, Optional

from core.tap.types import TrustPacket, PillarType, RiskLevel


# -------------------------------
# Public Adapter API
# -------------------------------

def adapt_agent_output(agent_output: Dict[str, Any]) -> Optional[TrustPacket]:
    """
    Convert a Step-2 agent output into a TrustPacket.
    Returns None if the output is invalid or unsafe.
    """

    try:
        agent_id = _require_str(agent_output, "agent_id")
        pillar = _require_pillar(agent_output, "pillar")

        confidence = _clamp_float(
            agent_output.get("confidence"), default=0.0
        )
        evidence = _clamp_float(
            agent_output.get("evidence_strength"), default=0.0
        )
        freshness_days = _safe_int(
            agent_output.get("freshness_days"), default=999
        )
        risk_level = _map_risk_level(
            agent_output.get("risk_level")
        )

        rationale = _safe_rationale(agent_output.get("analysis"))

        return TrustPacket(
            agent_id=agent_id,
            pillar=pillar,
            confidence=confidence,
            evidence_strength=evidence,
            freshness_days=freshness_days,
            risk_level=risk_level,
            rationale=rationale,
        )

    except Exception:
        # Fail closed: invalid agent output is ignored
        return None


# -------------------------------
# Helpers (Strict & Defensive)
# -------------------------------

def _require_str(obj: Dict[str, Any], key: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid '{key}'")
    return value.strip()


def _require_pillar(obj: Dict[str, Any], key: str) -> PillarType:
    value = obj.get(key)
    allowed = {
        "market",
        "finance",
        "execution",
        "risk",
        "compliance",
        "timing",
    }
    if value not in allowed:
        raise ValueError(f"Invalid pillar: {value}")
    return value  # type: ignore


def _clamp_float(value: Any, default: float) -> float:
    try:
        v = float(value)
        return max(0.0, min(1.0, v))
    except Exception:
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        v = int(value)
        return max(0, v)
    except Exception:
        return default


def _map_risk_level(value: Any) -> RiskLevel:
    mapping = {
        "low": "low",
        "medium": "medium",
        "high": "high",
        "veto": "veto",
    }
    return mapping.get(str(value).lower(), "low")  # type: ignore


def _safe_rationale(value: Any) -> Optional[str]:
    """
    Rationale must be short, descriptive, and safe.
    NO chain-of-thought.
    """
    if not isinstance(value, str):
        return None

    text = value.strip()

    # Hard limit to prevent leakage
    if len(text) > 300:
        return text[:300] + "…"

    return text or None
