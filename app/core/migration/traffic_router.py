"""
================================================================================
Traffic Router — Deterministic pipeline selection for migration rollout.

STATUS: SHADOW-ONLY. Does NOT instantiate or call any pipeline.
        Returns a routing decision string that callers use to select
        which pipeline to invoke.

PURPOSE:
    Given a client_id and request context, decide whether the request
    should be served by the legacy env-driven pipeline (OLD_PIPELINE)
    or the new ClientConfig-driven pipeline (NEW_PIPELINE).

INVARIANTS:
    - NEVER returns NEW_PIPELINE in SHADOW_ONLY or ROLLBACK.
    - Deterministic: same (client_id, request_id) always produces the
      same routing decision — critical for debugging and replay.
    - Decision-only: does NOT construct, call, or import any pipeline.

USAGE:
    from app.core.migration.traffic_router import traffic_router

    decision = traffic_router.get_pipeline("acme_corp", {"request_id": "abc"})
    if decision == "NEW_PIPELINE":
        ...
    else:
        ...
================================================================================
"""

from __future__ import annotations

import json as _json_module
import logging
from typing import Any, Dict, Optional

from app.core.migration.migration_controller import (
    MigrationState,
    migration_controller,
)

logger = logging.getLogger(__name__)

OLD_PIPELINE = "OLD_PIPELINE"
NEW_PIPELINE = "NEW_PIPELINE"


class TrafficRouter:
    """
    Stateless traffic router. Delegates state resolution to
    MigrationController and translates it into a pipeline selection.
    """

    def get_pipeline(
        self,
        client_id: str,
        request_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Determine which pipeline should serve this request.

        Args:
            client_id:        Client/tenant identifier.
            request_context:  Optional dict with request_id, query, session_id
                              for deterministic canary bucketing.

        Returns:
            "OLD_PIPELINE" or "NEW_PIPELINE"
        """
        state = migration_controller.get_state(client_id)
        use_new = migration_controller.should_use_new_pipeline(
            client_id, request_context,
        )

        decision = NEW_PIPELINE if use_new else OLD_PIPELINE

        # Hard safety clamp — defence in depth
        if state in (MigrationState.SHADOW_ONLY, MigrationState.ROLLBACK):
            decision = OLD_PIPELINE

        if state == MigrationState.SHADOW_EXECUTION:
            decision = OLD_PIPELINE

        try:
            logger.debug(
                "%s",
                _json_module.dumps({
                    "event": "TRAFFIC_ROUTED",
                    "client_id": client_id,
                    "state": state.value,
                    "decision": decision,
                }, default=str),
            )
        except Exception:
            pass

        return decision


# Module-level singleton
traffic_router = TrafficRouter()
