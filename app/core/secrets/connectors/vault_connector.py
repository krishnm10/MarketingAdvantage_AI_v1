"""
HashiCorp Vault connector for ``vault://`` secret references (Phase 3 prod target).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import unquote

import httpx

from app.core.config.secret_ref import SecretRef, SecretsBackendConfig
from app.core.secrets.connectors.base import (
    SecretNamespaceViolationError,
    SecretResolutionError,
)
from app.utils.path_sanitizer import sanitize_client_id

logger = logging.getLogger(__name__)

VaultReadFn = Callable[[str], Dict[str, Any]]


def parse_vault_uri(uri: str) -> Tuple[str, str]:
    """
    Parse ``vault://path/to/secret#field`` into (secret_path, field_name).

    Default field name is ``value`` when ``#`` fragment is omitted.
    """
    if not uri.startswith("vault://"):
        raise SecretResolutionError("VaultConnector requires vault:// URIs.")
    remainder = uri[len("vault://") :].strip()
    if not remainder:
        raise SecretResolutionError("vault:// URI must include a secret path.")

    if "#" in remainder:
        path_part, field = remainder.split("#", 1)
        field = unquote(field.strip()) or "value"
    else:
        path_part, field = remainder, "value"

    path = unquote(path_part.strip().strip("/"))
    if not path:
        raise SecretResolutionError("vault:// URI secret path must be non-empty.")
    return path, field


def validate_vault_path_for_client(
    secret_path: str,
    client_id: str,
    *,
    namespace: Optional[str] = None,
    enforce_tenant_path: bool = True,
) -> None:
    """
    Ensure *secret_path* belongs to *client_id*.

    Rules:
      - Path segments must include the sanitized ``client_id``.
      - When ``namespace`` is configured on the backend, the path must start with
        ``{namespace}/{client_id}/`` (or equal ``{namespace}/{client_id}``).
      - Paths containing a different tenant segment immediately after the
        namespace prefix are rejected.
      - When *enforce_tenant_path* is false, skip validation (dev / flat paths).
    """
    if not enforce_tenant_path:
        return
    safe_id = sanitize_client_id(client_id)
    segments = [s for s in secret_path.split("/") if s]
    if safe_id not in segments:
        raise SecretNamespaceViolationError(
            f"Vault secret path does not include tenant namespace for client_id={safe_id!r}."
        )

    if namespace:
        ns_segments = [s for s in namespace.strip("/").split("/") if s]
        expected_prefix = ns_segments + [safe_id]
        if segments[: len(expected_prefix)] != expected_prefix:
            raise SecretNamespaceViolationError(
                f"Vault secret path must start with "
                f"{'/'.join(expected_prefix)!r} for client_id={safe_id!r}."
            )
        return

    if segments[0] != safe_id:
        raise SecretNamespaceViolationError(
            f"Vault secret path must start with client_id segment {safe_id!r}."
        )


class VaultConnector:
    """
    Resolve ``vault://`` URIs via HashiCorp Vault KV v2.

    Worker Vault token is supplied at construction time (from deployment IAM /
    bootstrap — never from tenant JSON plaintext).
    """

    def __init__(
        self,
        backend: SecretsBackendConfig,
        *,
        vault_token: Optional[str] = None,
        read_secret: Optional[VaultReadFn] = None,
        http_timeout: float = 10.0,
    ) -> None:
        if not backend.vault_addr:
            raise ValueError("VaultConnector requires secrets_backend.vault_addr.")
        self._backend = backend
        self._vault_token = vault_token
        self._read_secret = read_secret
        self._http_timeout = http_timeout

    async def resolve(
        self,
        ref: SecretRef,
        *,
        client_id: str,
        purpose: str,
    ) -> str:
        secret_path, field = parse_vault_uri(ref.uri)
        validate_vault_path_for_client(
            secret_path,
            client_id,
            namespace=self._backend.namespace,
            enforce_tenant_path=self._backend.enforce_tenant_path,
        )

        if self._read_secret is not None:
            payload = await asyncio.to_thread(self._read_secret, secret_path)
        else:
            payload = await asyncio.to_thread(self._fetch_kv2, secret_path)

        if field not in payload:
            raise SecretResolutionError(
                f"Vault secret field {field!r} not found for purpose={purpose!r}."
            )
        value = payload[field]
        if value is None or str(value) == "":
            raise SecretResolutionError(
                f"Vault secret field {field!r} is empty for purpose={purpose!r}."
            )
        return str(value)

    def _fetch_kv2(self, secret_path: str) -> Dict[str, Any]:
        token = self._vault_token
        if not token:
            raise SecretResolutionError(
                "Vault token is not configured for secret resolution."
            )

        mount = (self._backend.mount_path or "secret").strip("/")
        base = self._backend.vault_addr.rstrip("/")
        url = f"{base}/v1/{mount}/data/{secret_path}"

        headers = {"X-Vault-Token": token}
        if self._backend.namespace:
            headers["X-Vault-Namespace"] = self._backend.namespace

        try:
            with httpx.Client(timeout=self._http_timeout) as client:
                response = client.get(url, headers=headers)
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            logger.warning(
                "[VaultConnector] Vault read failed path=%r url=%r status=%s",
                secret_path,
                url,
                status,
            )
            if status == 404:
                raise SecretResolutionError(
                    f"Vault secret path {secret_path!r} not found (HTTP 404) at "
                    f"mount {mount!r}. Create the secret in Vault at KV path "
                    f"{secret_path!r} (API: /v1/{mount}/data/{secret_path}), or "
                    f"update the SecretRef URI to match an existing path."
                ) from exc
            if status in (401, 403):
                raise SecretResolutionError(
                    f"Vault denied access to {secret_path!r} (HTTP {status}). "
                    "Check the token policy allows read on this path."
                ) from exc
            raise SecretResolutionError(
                f"Vault secret read failed for {secret_path!r} (HTTP {status})."
            ) from exc
        except httpx.HTTPError as exc:
            logger.warning(
                "[VaultConnector] Vault read failed path=%r class=%s",
                secret_path,
                type(exc).__name__,
            )
            raise SecretResolutionError(
                f"Vault secret read failed for {secret_path!r} — could not reach "
                f"{self._backend.vault_addr!r}."
            ) from exc

        data = body.get("data", {}).get("data")
        if not isinstance(data, dict):
            raise SecretResolutionError("Vault response did not contain secret data.")
        return data
