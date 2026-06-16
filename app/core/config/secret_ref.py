"""
Tenant secret references and secrets-backend configuration (Phase 1).

Raw API keys MUST NOT appear in tenant JSON. Integrations reference secrets
via ``SecretRef`` URIs resolved at runtime against the tenant ``secrets_backend``.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

_ALLOWED_SECRET_SCHEMES: Tuple[str, ...] = (
    "vault://",
    "aws-sm://",
    "azure-kv://",
    "gcp-sm://",
    "env://",
)

_LEGACY_ENV_FIELD_NAMES: Tuple[str, ...] = (
    "api_key_env",
    "token_env",
    "password_env",
)


class SecretRef(BaseModel):
    """Pointer to a secret in the tenant-configured secrets backend."""

    uri: str = Field(
        ...,
        description=(
            "Secret locator URI. Allowed schemes: vault://, aws-sm://, "
            "azure-kv://, gcp-sm://, env:// (dev-only)."
        ),
    )

    @field_validator("uri")
    @classmethod
    def validate_uri(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("SecretRef.uri must be a string")
        uri = value.strip()
        if not uri:
            raise ValueError("SecretRef.uri must be non-empty")
        if uri.startswith("sk-"):
            raise ValueError(
                "Raw API keys are not allowed in tenant JSON; "
                "use a secret store URI (e.g. vault://tenant/openai-key)."
            )
        if not any(uri.startswith(scheme) for scheme in _ALLOWED_SECRET_SCHEMES):
            raise ValueError(
                "SecretRef.uri must start with one of: "
                + ", ".join(_ALLOWED_SECRET_SCHEMES)
            )
        return uri

    def env_var_name(self) -> Optional[str]:
        """When uri uses env://, return the env var name (dev backend only)."""
        if self.uri.startswith("env://"):
            name = self.uri[len("env://") :].strip()
            return name or None
        return None


def secret_ref_env_var_name(ref: Optional[SecretRef]) -> Optional[str]:
    """Resolve env:// secret refs to an env var name for legacy runtime paths."""
    if ref is None:
        return None
    return ref.env_var_name()


def secret_ref_uses_env_backend(ref: Optional[SecretRef]) -> bool:
    """True when the ref resolves from process environment (env:// only)."""
    return secret_ref_env_var_name(ref) is not None


def reject_legacy_secret_fields(
    data: Any,
    *,
    path: str = "",
    legacy_env_keys: Tuple[str, ...] = _LEGACY_ENV_FIELD_NAMES,
) -> Any:
    """
    Phase 8: reject legacy ``api_key_env`` / ``token_env`` / ``password_env`` keys.

    Use ``secret_ref: {"uri": "env://VAR"}`` (or vault/aws/azure/gcp URIs) instead.
    """
    if isinstance(data, dict):
        for key in legacy_env_keys:
            val = data.get(key)
            if val is not None and str(val).strip():
                loc = f"{path}.{key}" if path else key
                raise ValueError(
                    f"Legacy field '{loc}' was removed in tenant JSON v2. "
                    "Use secret_ref with a URI (e.g. env://OPENAI_API_KEY or vault://tenant/key)."
                )
        return {
            k: reject_legacy_secret_fields(v, path=f"{path}.{k}" if path else k, legacy_env_keys=legacy_env_keys)
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [
            reject_legacy_secret_fields(item, path=path, legacy_env_keys=legacy_env_keys)
            for item in data
        ]
    return data


def migrate_legacy_secret_fields(
    data: Any,
    *,
    env_fallback: Optional[str] = None,
    legacy_env_keys: Tuple[str, ...] = _LEGACY_ENV_FIELD_NAMES,
) -> Any:
    """
    Accept Phase 0 tenant JSON that still uses ``api_key_env`` / ``token_env`` /
    ``password_env`` or bare string ``secret_ref`` values.
    """
    if not isinstance(data, dict):
        return data

    out = dict(data)
    legacy_env: Optional[str] = None
    for key in legacy_env_keys:
        val = out.get(key)
        if isinstance(val, str) and val.strip():
            legacy_env = val.strip()
            break

    secret_ref_raw = out.get("secret_ref")
    if secret_ref_raw is None and legacy_env:
        out["secret_ref"] = {"uri": f"env://{legacy_env}"}
    elif isinstance(secret_ref_raw, str) and secret_ref_raw.strip():
        out["secret_ref"] = {"uri": secret_ref_raw.strip()}
    elif secret_ref_raw is None and env_fallback:
        out["secret_ref"] = {"uri": f"env://{env_fallback}"}

    for key in legacy_env_keys:
        out.pop(key, None)
    return out


class SecretsBackendProvider(str, Enum):
    HASHICORP_VAULT = "hashicorp_vault"
    AWS_SECRETS_MANAGER = "aws_secrets_manager"
    AZURE_KEY_VAULT = "azure_key_vault"
    GCP_SECRET_MANAGER = "gcp_secret_manager"
    ENV = "env"


class SecretsBackendConfig(BaseModel):
    """
    One secrets backend per tenant (BYOK boundary).

    Connection credentials for the backend itself are supplied via deployment
    IAM / worker env — not inline in tenant JSON.
    """

    provider: SecretsBackendProvider = Field(
        ...,
        description="Primary secret store for this tenant.",
    )

    # HashiCorp Vault
    vault_addr: Optional[str] = Field(
        None, description="Vault server URL (e.g. https://vault.example.com:8200)."
    )
    namespace: Optional[str] = Field(None, description="Vault enterprise namespace.")
    role_id: Optional[str] = Field(None, description="AppRole role_id for Vault auth.")
    secret_id_ref: Optional[SecretRef] = Field(
        None, description="SecretRef to AppRole secret_id (never inline)."
    )
    mount_path: Optional[str] = Field("secret", description="KV mount path.")
    vault_token: Optional[str] = Field(
        None,
        description=(
            "Vault client token for this tenant (write-only in admin API responses). "
            "Falls back to worker VAULT_TOKEN env when unset."
        ),
    )
    enforce_tenant_path: bool = Field(
        True,
        description=(
            "When true, vault:// paths must include the tenant client_id prefix "
            "(e.g. default/my-secret). Disable for single-tenant dev when secrets "
            "live at flat paths (e.g. mai_secrets)."
        ),
    )

    # AWS Secrets Manager
    aws_region: Optional[str] = Field(None, description="AWS region for Secrets Manager.")
    aws_role_arn: Optional[str] = Field(
        None, description="IAM role ARN workers assume to read tenant secrets."
    )

    # Azure Key Vault
    azure_vault_url: Optional[str] = Field(None, description="Azure Key Vault URL.")
    azure_tenant_id: Optional[str] = Field(None, description="Azure AD tenant id.")
    azure_client_id: Optional[str] = Field(None, description="Azure AD application id.")
    azure_client_secret: Optional[str] = Field(
        None,
        description="Azure AD client secret (write-only in admin API responses).",
    )

    # GCP Secret Manager
    gcp_project_id: Optional[str] = Field(None, description="GCP project id.")
    gcp_secret_prefix: Optional[str] = Field(
        None, description="Prefix for secret resource names in GCP SM."
    )

    @model_validator(mode="after")
    def validate_provider_fields(self) -> "SecretsBackendConfig":
        p = self.provider
        if p == SecretsBackendProvider.HASHICORP_VAULT and not self.vault_addr:
            raise ValueError(
                "SecretsBackendConfig: vault_addr is required when provider='hashicorp_vault'."
            )
        if p == SecretsBackendProvider.AWS_SECRETS_MANAGER and not self.aws_region:
            raise ValueError(
                "SecretsBackendConfig: aws_region is required when "
                "provider='aws_secrets_manager'."
            )
        if p == SecretsBackendProvider.AZURE_KEY_VAULT and not self.azure_vault_url:
            raise ValueError(
                "SecretsBackendConfig: azure_vault_url is required when "
                "provider='azure_key_vault'."
            )
        if p == SecretsBackendProvider.GCP_SECRET_MANAGER and not self.gcp_project_id:
            raise ValueError(
                "SecretsBackendConfig: gcp_project_id is required when "
                "provider='gcp_secret_manager'."
            )
        return self


SECRETS_BACKEND_WRITE_ONLY_FIELDS: Tuple[str, ...] = (
    "vault_token",
    "azure_client_secret",
)


def merge_secrets_backend_write_only(
    existing: Optional[Dict[str, Any]],
    incoming: Dict[str, Any],
) -> Dict[str, Any]:
    """Preserve stored write-only credentials when the client omits or clears them."""
    merged = dict(incoming)
    prev = existing if isinstance(existing, dict) else {}
    for key in SECRETS_BACKEND_WRITE_ONLY_FIELDS:
        raw = merged.get(key)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            if prev.get(key):
                merged[key] = prev[key]
        elif isinstance(raw, str):
            merged[key] = raw.strip()
    return merged


def secrets_backend_configured_flags(raw: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    data = raw if isinstance(raw, dict) else {}
    return {
        "vault_token_configured": bool(str(data.get("vault_token") or "").strip()),
        "azure_client_secret_configured": bool(
            str(data.get("azure_client_secret") or "").strip()
        ),
    }


def redact_secrets_backend_for_api(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Remove write-only credential values; expose configured booleans instead."""
    out = dict(raw)
    flags = secrets_backend_configured_flags(out)
    for key in SECRETS_BACKEND_WRITE_ONLY_FIELDS:
        out.pop(key, None)
    out.update(flags)
    return out


def resolve_vault_token(
    backend: SecretsBackendConfig,
    *,
    fallback_env: bool = True,
) -> Optional[str]:
    """Tenant-stored token first, then worker ``VAULT_TOKEN`` env."""
    stored = (backend.vault_token or "").strip()
    if stored:
        return stored
    if fallback_env:
        env_token = (os.environ.get("VAULT_TOKEN") or "").strip()
        return env_token or None
    return None
