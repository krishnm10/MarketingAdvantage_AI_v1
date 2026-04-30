"""
================================================================================
Migration Controller — Stateless rollout state machine for per-client
config migration from env-driven to ClientConfig-driven pipelines.

STATUS: SHADOW-ONLY. Does NOT affect runtime pipeline construction,
        ingestion, or RAG API responses. Safe to import anywhere.

ARCHITECTURE:
    MigrationController reads:
        1. Per-client rollout stage from env/config (MIGRATION_STAGE_{CLIENT_ID})
        2. Config drift analysis from config_diff_logger + config_drift_gate
    and exposes deterministic answers to:
        - "What migration state is this client in?"
        - "Should this request use the new ClientConfig pipeline?"
        - "Should we dual-execute (shadow) for observability?"

INVARIANTS:
    - NEVER mutates state — reads env vars and config, returns decisions.
    - NEVER returns True for should_use_new_pipeline in SHADOW_ONLY.
    - ALWAYS respects drift gate: if critical drift is detected, the
      effective state is clamped to at most SHADOW_EXECUTION regardless
      of the configured stage.
    - ClientConfig is the ONLY target state. Env-driven config is the
      legacy fallback that remains active until FULL_MIGRATION.

STATES:
    SHADOW_ONLY         → Log diffs only. No execution change. (default)
    SHADOW_EXECUTION    → Dual-execute old + new pipeline, compare results,
                          but return old pipeline's output.
    CANARY_10           → 10% of requests use new pipeline.
    CANARY_25           → 25% of requests use new pipeline.
    CANARY_50           → 50% of requests use new pipeline.
    FULL_MIGRATION      → 100% new pipeline. Env fallback still available.
    ROLLBACK            → Emergency revert to env-driven pipeline.
================================================================================
"""

from __future__ import annotations

import hashlib
import json as _json_module
import logging
import os
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Migration states
# ─────────────────────────────────────────────────────────────────────────────

class MigrationState(str, Enum):
    SHADOW_ONLY       = "SHADOW_ONLY"
    SHADOW_EXECUTION  = "SHADOW_EXECUTION"
    CANARY_10         = "CANARY_10"
    CANARY_25         = "CANARY_25"
    CANARY_50         = "CANARY_50"
    FULL_MIGRATION    = "FULL_MIGRATION"
    ROLLBACK          = "ROLLBACK"


_STATE_ORDER = {
    MigrationState.ROLLBACK:          0,
    MigrationState.SHADOW_ONLY:       1,
    MigrationState.SHADOW_EXECUTION:  2,
    MigrationState.CANARY_10:         3,
    MigrationState.CANARY_25:         4,
    MigrationState.CANARY_50:         5,
    MigrationState.FULL_MIGRATION:    6,
}

_CANARY_PERCENTAGES: Dict[MigrationState, int] = {
    MigrationState.CANARY_10: 10,
    MigrationState.CANARY_25: 25,
    MigrationState.CANARY_50: 50,
}

_DEFAULT_STATE = MigrationState.SHADOW_ONLY


# ─────────────────────────────────────────────────────────────────────────────
# Controller
# ─────────────────────────────────────────────────────────────────────────────

class MigrationController:
    """
    Stateless migration state machine. Every call reads fresh env/config
    and drift-gate state — no mutable instance state, no side effects.

    Thread-safe: all methods are pure functions over env + config.
    """

    def get_state(self, client_id: str) -> MigrationState:
        """
        Return the current migration state for a client.

        Resolution:
            1. Read MIGRATION_STAGE_{CLIENT_ID} env var (uppercase, hyphens → underscores).
            2. Read global MIGRATION_STAGE env var as fallback.
            3. Default to SHADOW_ONLY.
            4. If drift gate blocks migration, clamp to max SHADOW_EXECUTION.
        """
        raw = self._read_configured_state(client_id)
        effective = self._clamp_for_drift(raw, client_id)

        if effective != raw:
            logger.warning(
                "%s",
                _json_module.dumps({
                    "event": "MIGRATION_STATE_CLAMPED",
                    "client_id": client_id,
                    "configured": raw.value,
                    "effective": effective.value,
                    "reason": "critical_config_drift",
                }, default=str),
            )

        return effective

    def should_use_new_pipeline(
        self,
        client_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Determine whether a specific request should use the new
        ClientConfig-driven pipeline.

        Returns True only for:
            - CANARY states (probabilistic, based on request hash)
            - FULL_MIGRATION (always)

        Returns False for:
            - SHADOW_ONLY, SHADOW_EXECUTION, ROLLBACK

        The context dict (e.g. request_id, query hash) is used as
        entropy for deterministic canary bucketing — same request
        always lands in the same bucket for debuggability.
        """
        state = self.get_state(client_id)

        if state == MigrationState.FULL_MIGRATION:
            return True

        if state == MigrationState.ROLLBACK:
            return False

        if state in _CANARY_PERCENTAGES:
            pct = _CANARY_PERCENTAGES[state]
            bucket = self._canary_bucket(client_id, context)
            return bucket < pct

        return False

    def should_run_shadow_dual_execution(self, client_id: str) -> bool:
        """
        Return True if this client should dual-execute both pipelines
        for observability (run new pipeline in background, compare results,
        but always return old pipeline's output).

        Active in SHADOW_EXECUTION and all CANARY states (for the
        requests that still use the old pipeline).
        """
        state = self.get_state(client_id)
        return state in (
            MigrationState.SHADOW_EXECUTION,
            MigrationState.CANARY_10,
            MigrationState.CANARY_25,
            MigrationState.CANARY_50,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _read_configured_state(client_id: str) -> MigrationState:
        """Read the configured migration stage from environment."""
        safe_id = client_id.upper().replace("-", "_").replace(" ", "_")

        raw = (
            os.getenv(f"MIGRATION_STAGE_{safe_id}")
            or os.getenv("MIGRATION_STAGE")
            or ""
        ).strip().upper()

        try:
            return MigrationState(raw) if raw else _DEFAULT_STATE
        except ValueError:
            logger.warning(
                "[MigrationController] Unknown MIGRATION_STAGE '%s' for "
                "client_id='%s' — defaulting to SHADOW_ONLY",
                raw, client_id,
            )
            return _DEFAULT_STATE

    @staticmethod
    def _clamp_for_drift(
        state: MigrationState, client_id: str,
    ) -> MigrationState:
        """
        If the drift gate detects critical config drift, clamp the
        effective state to at most SHADOW_EXECUTION. This prevents
        canary or full migration from proceeding with incompatible configs.
        """
        if _STATE_ORDER.get(state, 0) <= _STATE_ORDER[MigrationState.SHADOW_EXECUTION]:
            return state

        try:
            from app.core.config.client_config_resolver import (
                _find_config_path,
                _load_raw,
                _DEFAULT_CONFIG_ID,
            )
            from app.core.config.config_diff_logger import compare_configs
            from app.core.config.config_drift_gate import should_block_migration
            from app.core.config.config_drift_scorer import compute_drift_score

            default_path = _find_config_path(_DEFAULT_CONFIG_ID)
            client_path = _find_config_path(client_id)
            if default_path is None or client_path is None:
                return state

            from app.core.config.client_config_schema import ClientConfig
            baseline = ClientConfig.from_dict(_load_raw(default_path))
            merged = ClientConfig.from_dict(_load_raw(client_path))

            diff = compare_configs(baseline, merged, client_id)
            drift_score = compute_drift_score(diff)

            if should_block_migration(diff, drift_score):
                return MigrationState.SHADOW_EXECUTION

        except Exception as exc:
            logger.debug(
                "[MigrationController] Drift check failed for '%s': %s — "
                "not clamping",
                client_id, exc,
            )

        return state

    @staticmethod
    def _canary_bucket(
        client_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Deterministic canary bucketing (0-99) based on client_id + request
        context. Same input always produces the same bucket.
        """
        seed = client_id
        if context:
            request_id = (
                context.get("request_id")
                or context.get("query")
                or context.get("session_id")
                or ""
            )
            seed = f"{client_id}:{request_id}"

        digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % 100


# Module-level singleton for convenience
migration_controller = MigrationController()
