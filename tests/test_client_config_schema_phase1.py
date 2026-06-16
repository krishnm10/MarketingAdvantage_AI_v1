"""Phase 1 tenant JSON schema: SecretRef, SecretsBackend, PublicTenantConfig."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.core.config.client_config_schema import ClientConfig
from app.core.config.public_tenant_config import PublicTenantConfig
from app.core.config.secret_ref import SecretRef, SecretsBackendConfig, SecretsBackendProvider


class TestSecretRef:
    def test_rejects_raw_sk_prefix_key(self) -> None:
        with pytest.raises(ValidationError) as exc:
            SecretRef(uri="sk-proj-abc123")
        assert "Raw API keys are not allowed" in str(exc.value)

    def test_rejects_string_without_allowed_scheme(self) -> None:
        with pytest.raises(ValidationError) as exc:
            SecretRef(uri="OPENAI_API_KEY")
        assert "must start with one of" in str(exc.value)

    def test_accepts_vault_uri(self) -> None:
        ref = SecretRef(uri="vault://acme/openai/api-key")
        assert ref.uri == "vault://acme/openai/api-key"

    def test_accepts_env_uri_for_dev(self) -> None:
        ref = SecretRef(uri="env://GOOGLE_API_KEY")
        assert ref.env_var_name() == "GOOGLE_API_KEY"


class TestLegacySecretFieldRejection:
    def test_rejects_api_key_env_in_embedder(self) -> None:
        with pytest.raises(ValidationError) as exc:
            ClientConfig.from_dict(
                {
                    "client_id": "test",
                    "vectordb": {
                        "type": "chroma",
                        "collection": "c",
                        "chroma": {"persist_directory": "./db"},
                    },
                    "embedder": {
                        "type": "gemini",
                        "gemini": {
                            "model": "gemini-embedding-2",
                            "api_key_env": "GOOGLE_API_KEY",
                        },
                    },
                    "llm": {
                        "single": {
                            "type": "ollama",
                            "model": "llama3.2",
                            "base_url": "http://localhost:11434",
                        }
                    },
                }
            )
        assert "secret_ref" in str(exc.value).lower()

    def test_accepts_secret_ref_instead_of_api_key_env(self) -> None:
        cfg = ClientConfig.from_dict(
            {
                "client_id": "test",
                "vectordb": {
                    "type": "chroma",
                    "collection": "c",
                    "chroma": {"persist_directory": "./db"},
                },
                "embedder": {
                    "type": "gemini",
                    "gemini": {
                        "model": "gemini-embedding-2",
                        "secret_ref": {"uri": "env://GOOGLE_API_KEY"},
                    },
                },
                "llm": {
                    "single": {
                        "type": "gemini",
                        "model": "gemini-2.5-flash",
                        "secret_ref": {"uri": "env://GOOGLE_API_KEY"},
                        "base_url": "https://generativelanguage.googleapis.com/v1",
                    }
                },
            }
        )
        assert cfg.embedder.gemini is not None
        assert cfg.embedder.gemini.secret_ref is not None
        assert cfg.embedder.gemini.secret_ref.uri == "env://GOOGLE_API_KEY"
        assert cfg.llm.single is not None
        assert cfg.llm.single.secret_ref is not None
        assert cfg.llm.single.secret_ref.uri == "env://GOOGLE_API_KEY"


class TestPublicTenantConfig:
    def _full_client_config(self) -> ClientConfig:
        return ClientConfig.from_dict(
            {
                "client_id": "acme",
                "client_name": "Acme Corp",
                "secrets_backend": {
                    "provider": "hashicorp_vault",
                    "vault_addr": "https://vault.example.com:8200",
                    "namespace": "acme",
                    "role_id": "role-123",
                    "secret_id_ref": {"uri": "vault://acme/worker/approle-secret-id"},
                },
                "vectordb": {
                    "type": "qdrant",
                    "collection": "docs",
                    "qdrant": {
                        "url": "https://xyz.qdrant.io",
                        "secret_ref": {"uri": "aws-sm://acme/qdrant-api-key"},
                    },
                },
                "embedder": {
                    "type": "openai",
                    "openai": {
                        "model": "text-embedding-3-small",
                        "secret_ref": {"uri": "vault://acme/openai/embed-key"},
                    },
                },
                "llm": {
                    "single": {
                        "type": "openai",
                        "model": "gpt-4o-mini",
                        "base_url": "https://api.openai.com/v1",
                        "secret_ref": {"uri": "vault://acme/openai/llm-key"},
                    }
                },
                "reranker": {
                    "type": "cohere",
                    "model": "rerank-english-v3.0",
                    "secret_ref": {"uri": "vault://acme/cohere/key"},
                },
                "ingestion": {
                    "vision": {
                        "ai_profile": "api",
                        "vision_api_secret_ref": {
                            "uri": "vault://acme/openai/vision-key"
                        },
                    }
                },
            }
        )

    def test_mapper_strips_all_secret_references(self) -> None:
        cfg = self._full_client_config()
        public = cfg.to_public()
        payload = public.model_dump(mode="json")
        serialized = json.dumps(payload)

        forbidden_substrings = (
            "secret_ref",
            "secrets_backend",
            "api_key_env",
            "password_env",
            "token_env",
            "vault://",
            "aws-sm://",
            "role_id",
            "vault_addr",
        )
        for needle in forbidden_substrings:
            assert needle not in serialized, f"leaked {needle!r} in public payload"

        assert public.client_id == "acme"
        assert public.providers.embedder_type == "openai"
        assert public.providers.embedder_configured is True
        assert public.providers.secret_store_provider == "hashicorp_vault"
        assert public.ingestion.vision.ai_profile == "api"
        assert public.ingestion.vision.vision_api_configured is True

    def test_from_client_config_classmethod(self) -> None:
        cfg = self._full_client_config()
        public = PublicTenantConfig.from_client_config(cfg)
        assert isinstance(public, PublicTenantConfig)
        assert public.providers.llm_provider == "openai"

    def test_secrets_backend_config_validates_vault_addr(self) -> None:
        with pytest.raises(ValidationError):
            SecretsBackendConfig(provider=SecretsBackendProvider.HASHICORP_VAULT)
