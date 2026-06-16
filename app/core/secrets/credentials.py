"""Helpers for resolving tenant credentials during pipeline construction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from app.core.config.secret_ref import (
    SecretRef,
    SecretsBackendConfig,
    SecretsBackendProvider,
)
from app.core.secrets.connectors.base import SecretResolutionError
from app.core.secrets.resolver import SecretResolver, get_secret_resolver

if TYPE_CHECKING:
    from app.core.config.client_config_schema import ClientConfig


def effective_secrets_backend(config: "ClientConfig") -> SecretsBackendConfig:
    """Default to env backend when tenant JSON omits ``secrets_backend``."""
    if config.secrets_backend is not None:
        return config.secrets_backend
    return SecretsBackendConfig(provider=SecretsBackendProvider.ENV)


async def resolve_secret_optional(
    ref: Optional[SecretRef],
    *,
    config: "ClientConfig",
    purpose: str,
    resolver: Optional[SecretResolver] = None,
) -> Optional[str]:
    if ref is None:
        return None
    sr = resolver or get_secret_resolver()
    backend = effective_secrets_backend(config)
    return await sr.resolve(
        ref,
        client_id=config.client_id,
        backend=backend,
        purpose=purpose,
    )


async def resolve_secret_required(
    ref: SecretRef,
    *,
    config: "ClientConfig",
    purpose: str,
    resolver: Optional[SecretResolver] = None,
) -> str:
    value = await resolve_secret_optional(
        ref,
        config=config,
        purpose=purpose,
        resolver=resolver,
    )
    if not value:
        raise SecretResolutionError(
            f"Required secret could not be resolved for purpose={purpose!r}."
        )
    return value
