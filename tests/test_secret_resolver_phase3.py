"""Phase 3: SecretResolver, EnvConnector, VaultConnector."""

from __future__ import annotations

import pytest

from app.core.config.config_change_bus import NoOpConfigChangeBus, set_config_change_bus
from app.core.config.secret_ref import SecretRef, SecretsBackendConfig, SecretsBackendProvider
from app.core.secrets.connectors.base import SecretNamespaceViolationError, SecretResolutionError
from app.core.secrets.connectors.env_connector import EnvConnector
from app.core.secrets.connectors.vault_connector import VaultConnector, validate_vault_path_for_client
from app.core.secrets.resolver import SecretResolver, set_secret_resolver


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def isolated_secret_stack():
    set_config_change_bus(NoOpConfigChangeBus())
    set_secret_resolver(None)
    yield
    set_config_change_bus(None)
    set_secret_resolver(None)


class TestEnvConnector:
    @pytest.mark.anyio
    async def test_resolves_env_uri(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-value")
        connector = EnvConnector()
        value = await connector.resolve(
            SecretRef(uri="env://OPENAI_API_KEY"),
            client_id="tenant-a",
            purpose="embedder.openai",
        )
        assert value == "test-key-value"

    @pytest.mark.anyio
    async def test_missing_env_var_fails(self) -> None:
        connector = EnvConnector()
        with pytest.raises(SecretResolutionError, match="not set or empty"):
            await connector.resolve(
                SecretRef(uri="env://MISSING_SECRET_VAR_XYZ"),
                client_id="tenant-a",
                purpose="llm.openai",
            )


class TestVaultConnector:
    def test_validate_rejects_cross_tenant_path(self) -> None:
        with pytest.raises(SecretNamespaceViolationError):
            validate_vault_path_for_client(
                "mai/tenants/tenant-b/openai/api-key",
                "tenant-a",
                namespace="mai/tenants",
            )

    @pytest.mark.anyio
    async def test_resolve_raises_before_vault_read_for_wrong_client(self) -> None:
        backend = SecretsBackendConfig(
            provider=SecretsBackendProvider.HASHICORP_VAULT,
            vault_addr="https://vault.example.com:8200",
            namespace="mai/tenants",
        )
        connector = VaultConnector(
            backend,
            read_secret=lambda _path: {"api_key": "secret"},
        )

        with pytest.raises(SecretNamespaceViolationError):
            await connector.resolve(
                SecretRef(uri="vault://mai/tenants/tenant-b/openai/api-key#api_key"),
                client_id="tenant-a",
                purpose="embedder.openai",
            )

    @pytest.mark.anyio
    async def test_resolve_flat_path_when_tenant_prefix_disabled(self) -> None:
        backend = SecretsBackendConfig(
            provider=SecretsBackendProvider.HASHICORP_VAULT,
            vault_addr="https://vault.example.com:8200",
            enforce_tenant_path=False,
        )
        connector = VaultConnector(
            backend,
            read_secret=lambda path: {"GOOGLE_API_KEY": "ok"} if path == "mai_secrets" else {},
        )
        value = await connector.resolve(
            SecretRef(uri="vault://mai_secrets#GOOGLE_API_KEY"),
            client_id="default",
            purpose="embedder.gemini",
        )
        assert value == "ok"

    @pytest.mark.anyio
    async def test_resolve_accepts_matching_tenant_path(self) -> None:
        backend = SecretsBackendConfig(
            provider=SecretsBackendProvider.HASHICORP_VAULT,
            vault_addr="https://vault.example.com:8200",
            namespace="mai/tenants",
        )
        connector = VaultConnector(
            backend,
            read_secret=lambda _path: {"api_key": "vault-secret"},
        )
        value = await connector.resolve(
            SecretRef(uri="vault://mai/tenants/tenant-a/openai/api-key#api_key"),
            client_id="tenant-a",
            purpose="embedder.openai",
        )
        assert value == "vault-secret"


class TestSecretResolver:
    @pytest.mark.anyio
    async def test_caches_resolved_secret(
        self,
        isolated_secret_stack,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "cached-value")
        backend = SecretsBackendConfig(provider=SecretsBackendProvider.ENV)
        bus = NoOpConfigChangeBus()
        resolver = SecretResolver(bus=bus, max_cache_size=8)
        ref = SecretRef(uri="env://OPENAI_API_KEY")

        v1 = await resolver.resolve(
            ref, client_id="tenant-a", backend=backend, purpose="embedder.openai"
        )
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        v2 = await resolver.resolve(
            ref, client_id="tenant-a", backend=backend, purpose="embedder.openai"
        )

        assert v1 == "cached-value"
        assert v2 == "cached-value"

    @pytest.mark.anyio
    async def test_bus_publish_clears_tenant_cache(
        self,
        isolated_secret_stack,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "first-value")
        backend = SecretsBackendConfig(provider=SecretsBackendProvider.ENV)
        bus = NoOpConfigChangeBus()
        resolver = SecretResolver(bus=bus, max_cache_size=8)
        ref = SecretRef(uri="env://OPENAI_API_KEY")

        first = await resolver.resolve(
            ref, client_id="tenant-a", backend=backend, purpose="embedder.openai"
        )
        assert first == "first-value"

        monkeypatch.setenv("OPENAI_API_KEY", "second-value")
        cached = await resolver.resolve(
            ref, client_id="tenant-a", backend=backend, purpose="embedder.openai"
        )
        assert cached == "first-value"

        bus.publish("tenant-a", "version-2")

        refreshed = await resolver.resolve(
            ref, client_id="tenant-a", backend=backend, purpose="embedder.openai"
        )
        assert refreshed == "second-value"
