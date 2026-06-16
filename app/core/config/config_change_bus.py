"""
Cross-process tenant config change notifications (Phase 2).

Production will use Redis pub/sub; dev uses a no-op bus that logs events.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

ConfigChangeHandler = Callable[[str, str], None]


@runtime_checkable
class ConfigChangeBus(Protocol):
    """Publish/subscribe channel for tenant config version changes."""

    def publish(self, client_id: str, version: str) -> None:
        ...

    def subscribe(self, handler: ConfigChangeHandler) -> None:
        ...


class NoOpConfigChangeBus:
    """Dev-only bus: logs publishes and invokes in-process subscribers."""

    def __init__(self) -> None:
        self._handlers: List[ConfigChangeHandler] = []

    def publish(self, client_id: str, version: str) -> None:
        logger.info(
            "[ConfigChangeBus] publish client_id=%r version=%s",
            client_id,
            version,
        )
        for handler in list(self._handlers):
            try:
                handler(client_id, version)
            except Exception:
                logger.exception(
                    "[ConfigChangeBus] subscriber failed for client_id=%r",
                    client_id,
                )

    def subscribe(self, handler: ConfigChangeHandler) -> None:
        if handler not in self._handlers:
            self._handlers.append(handler)


_default_bus: Optional[ConfigChangeBus] = None


def get_config_change_bus() -> ConfigChangeBus:
    global _default_bus
    if _default_bus is None:
        _default_bus = NoOpConfigChangeBus()
    return _default_bus


def set_config_change_bus(bus: Optional[ConfigChangeBus]) -> None:
    global _default_bus
    _default_bus = bus
