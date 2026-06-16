"""
Secret backend connector protocol (Phase 3).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.core.config.secret_ref import SecretRef


class SecretResolutionError(RuntimeError):
    """Raised when a secret reference cannot be resolved."""


class SecretNamespaceViolationError(SecretResolutionError):
    """Raised when a secret URI is outside the requesting tenant namespace."""


@runtime_checkable
class SecretBackendConnector(Protocol):
    """Resolves a ``SecretRef`` to a plaintext secret value at runtime."""

    async def resolve(
        self,
        ref: SecretRef,
        *,
        client_id: str,
        purpose: str,
    ) -> str:
        """
        Resolve *ref* for *client_id*.

        ``purpose`` is a stable label for logs/metrics (e.g. ``embedder.openai``).
        Implementations MUST NOT log secret values or full URIs at info level.
        """
        ...
