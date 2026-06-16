# core/tap/audit.py

from datetime import datetime
from typing import Dict, Any


class TapAuditSink:
    """
    Abstract audit sink.
    Implementations can write to DB, file, queue, or SIEM.
    """

    def write(self, record: Dict[str, Any]) -> None:
        raise NotImplementedError


class InMemoryAuditSink(TapAuditSink):
    """
    Default lightweight sink (dev / testing).
    """

    def __init__(self) -> None:
        self.records = []

    def write(self, record: Dict[str, Any]) -> None:
        self.records.append(record)


class StdoutAuditSink(TapAuditSink):
    """
    Safe fallback sink.
    """

    def write(self, record: Dict[str, Any]) -> None:
        print("[TAP_AUDIT]", record)


# -------------------------------
# Helper
# -------------------------------

def build_audit_record(
    tap_decision,
    request_id: str,
) -> Dict[str, Any]:
    """
    Normalize TAP decision into an audit-safe record.
    """

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id,
        "policy_hash": tap_decision.audit.get("policy_hash"),
        "policy_version": tap_decision.audit.get("policy_version"),
        "strategy_level": tap_decision.audit.get("strategy_level"),
        "decision_type": tap_decision.decision_type,
        "trust_score": tap_decision.trust_score,
        "risk_flags": tap_decision.risk_flags,
        "agents": tap_decision.audit.get("agents", []),
    }
