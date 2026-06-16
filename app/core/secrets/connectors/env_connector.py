"""
Dev-only connector for ``env://VAR_NAME`` secret references.
"""

from __future__ import annotations

import os

from app.core.config.secret_ref import SecretRef
from app.core.secrets.connectors.base import SecretResolutionError


class EnvConnector:
    """Resolve ``env://`` URIs from the worker process environment."""

    async def resolve(
        self,
        ref: SecretRef,
        *,
        client_id: str,
        purpose: str,
    ) -> str:
        del client_id, purpose

        if not ref.uri.startswith("env://"):
            raise SecretResolutionError(
                f"EnvConnector received non-env URI scheme for purpose={purpose!r}."
            )

        var_name = ref.env_var_name()
        if not var_name:
            raise SecretResolutionError("env:// URI must include a non-empty variable name.")

        value = os.environ.get(var_name)
        if value is None or value == "":
            raise SecretResolutionError(
                f"Environment variable {var_name!r} is not set or empty."
            )
        return value
