"""
resolve_auth() — builds the correct BaseAuthProvider from ConnectorAuthConfig.
File: app/core/connectors/auth/resolver.py

Reads actual secrets from environment variables at runtime.
Config only stores env var NAMES — never the secrets themselves.
"""
from __future__ import annotations
import os
from app.core.connectors.auth.providers import (
    NoAuth, APIKeyAuth, BasicAuth,
    OAuth2Auth, AzureADAuth, FormBasedAuth,
)
from app.core.config.client_config_schema import ConnectorAuthConfig, ConnectorAuthType


def resolve_auth(cfg: ConnectorAuthConfig):
    """
    Build a BaseAuthProvider from ConnectorAuthConfig.

    Called by pipeline_factory or file_router_v2 when
    constructing a connector for a specific client.

    Args:
        cfg: ConnectorAuthConfig from ClientConfig.connector.auth

    Returns:
        A ready-to-use BaseAuthProvider instance.

    Raises:
        EnvironmentError: if a required env var is not set.
        ValueError: if auth type is unknown.
    """
    t = cfg.type

    if t == ConnectorAuthType.NONE:
        return NoAuth()

    if t == ConnectorAuthType.APIKEY:
        return APIKeyAuth.from_env(cfg.api_key_env, header_name=cfg.header_name)

    if t == ConnectorAuthType.BASIC:
        return BasicAuth.from_env(cfg.username_env, cfg.password_env)

    if t == ConnectorAuthType.OAUTH2:
        return OAuth2Auth(
            token_url=cfg.token_url,
            client_id=_require_env(cfg.client_id_env),
            client_secret=_require_env(cfg.client_secret_env),
            scope=cfg.scope,
        )

    if t == ConnectorAuthType.AZUREAD:
        return AzureADAuth(
            tenant_id=_require_env(cfg.tenant_id_env),
            client_id=_require_env(cfg.client_id_env),
            client_secret=_require_env(cfg.client_secret_env),
            scope=cfg.scope,
        )

    if t == ConnectorAuthType.FORM:
        return FormBasedAuth(
            login_url=cfg.login_url,
            username=_require_env(cfg.username_env),
            password=_require_env(cfg.password_env),
            username_field=cfg.username_field,
            password_field=cfg.password_field,
        )

    raise ValueError(f"[resolve_auth] Unknown auth type: '{t}'")


def _require_env(env_var: str) -> str:
    """Read an env var, raise EnvironmentError if missing or empty."""
    val = os.environ.get(env_var, "").strip()
    if not val:
        raise EnvironmentError(
            f"[resolve_auth] Required environment variable '{env_var}' is not set."
        )
    return val
