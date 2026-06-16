"""
Tenant-scoped secret resolver with LRU cache and config-change invalidation (Phase 3).
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from typing import Dict, Optional, Tuple

from app.core.config.config_change_bus import ConfigChangeBus, get_config_change_bus
from app.core.config.secret_ref import (
    SecretRef,
    SecretsBackendConfig,
    SecretsBackendProvider,
    resolve_vault_token,
)
from app.core.secrets.connectors.base import SecretResolutionError
from app.core.secrets.connectors.env_connector import EnvConnector
from app.core.secrets.connectors.vault_connector import VaultConnector
from app.utils.path_sanitizer import sanitize_client_id

logger = logging.getLogger(__name__)

CacheKey = Tuple[str, str]


class SecretResolver:
    """
    Resolve ``SecretRef`` values through the tenant-configured secrets backend.

    Caches resolved values in-memory keyed by ``(client_id, ref.uri)`` and
    clears tenant entries on ``ConfigChangeBus`` publishes.
    """

    def __init__(
        self,
        *,
        bus: Optional[ConfigChangeBus] = None,
        max_cache_size: int = 256,
        vault_token: Optional[str] = None,
    ) -> None:
        self._bus = bus or get_config_change_bus()
        self._max_cache_size = max(1, max_cache_size)
        self._vault_token = vault_token
        self._env_connector = EnvConnector()
        self._cache: OrderedDict[CacheKey, str] = OrderedDict()
        self._cache_lock = threading.RLock()
        self._bus_subscribed = False

    def _ensure_bus_subscribed(self) -> None:
        if self._bus_subscribed:
            return
        self._bus.subscribe(self._on_config_change)
        self._bus_subscribed = True

    def _on_config_change(self, client_id: str, version: str) -> None:
        del version
        self.clear_cache(client_id)
        logger.debug(
            "[SecretResolver] Cleared secret cache after config change client_id=%r",
            client_id,
        )

    def clear_cache(self, client_id: Optional[str] = None) -> None:
        with self._cache_lock:
            if client_id is None:
                self._cache.clear()
                return
            safe_id = sanitize_client_id(client_id)
            keys = [key for key in self._cache if key[0] == safe_id]
            for key in keys:
                del self._cache[key]

    def _cache_get(self, key: CacheKey) -> Optional[str]:
        with self._cache_lock:
            value = self._cache.get(key)
            if value is None:
                return None
            self._cache.move_to_end(key)
            return value

    def _cache_put(self, key: CacheKey, value: str) -> None:
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            while len(self._cache) > self._max_cache_size:
                self._cache.popitem(last=False)

    def _connector_for(
        self,
        backend: SecretsBackendConfig,
        ref: SecretRef,
    ):
        if ref.uri.startswith("env://"):
            if backend.provider != SecretsBackendProvider.ENV:
                raise SecretResolutionError(
                    "env:// secret references require secrets_backend.provider='env'."
                )
            return self._env_connector

        if ref.uri.startswith("vault://"):
            if backend.provider != SecretsBackendProvider.HASHICORP_VAULT:
                raise SecretResolutionError(
                    "vault:// secret references require "
                    "secrets_backend.provider='hashicorp_vault'."
                )
            return VaultConnector(
                backend,
                vault_token=resolve_vault_token(backend) or self._vault_token,
            )

        raise SecretResolutionError(
            f"Unsupported secret URI scheme for provider={backend.provider.value!r}."
        )

    async def resolve(
        self,
        ref: SecretRef,
        *,
        client_id: str,
        backend: SecretsBackendConfig,
        purpose: str,
    ) -> str:
        """
        Resolve *ref* to a plaintext secret for *client_id*.

        Uses LRU cache; invalidated when tenant config version changes.
        """
        self._ensure_bus_subscribed()
        safe_id = sanitize_client_id(client_id)
        cache_key: CacheKey = (safe_id, ref.uri)

        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        connector = self._connector_for(backend, ref)
        value = await connector.resolve(ref, client_id=safe_id, purpose=purpose)
        self._cache_put(cache_key, value)
        return value

    async def probe(
        self,
        ref: SecretRef,
        *,
        client_id: str,
        backend: SecretsBackendConfig,
        purpose: str,
    ) -> None:
        """Resolve without caching — for admin secret-existence checks only."""
        safe_id = sanitize_client_id(client_id)
        connector = self._connector_for(backend, ref)
        await connector.resolve(ref, client_id=safe_id, purpose=purpose)


_default_resolver: Optional[SecretResolver] = None
_resolver_lock = threading.Lock()


def get_secret_resolver() -> SecretResolver:
    global _default_resolver
    if _default_resolver is None:
        with _resolver_lock:
            if _default_resolver is None:
                _default_resolver = SecretResolver(
                    vault_token=os.environ.get("VAULT_TOKEN") or None,
                )
    return _default_resolver


def set_secret_resolver(resolver: Optional[SecretResolver]) -> None:
    global _default_resolver
    with _resolver_lock:
        _default_resolver = resolver
