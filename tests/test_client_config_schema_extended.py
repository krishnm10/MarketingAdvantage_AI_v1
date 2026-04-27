"""
Tests for ClientConfig schema extensions.
Verifies backward compatibility and new optional sections.
"""
from __future__ import annotations

import pytest


class TestBackwardCompatibility:
    """Ensure existing configs without new fields still load."""

    def test_minimal_config_loads(self):
        from app.core.config.client_config_schema import ClientConfig
        cfg = ClientConfig.from_dict({
            "client_id": "test",
            "vectordb": {
                "type": "chroma",
                "collection": "test_collection",
                "chroma": {"persist_directory": "./test_db"},
            },
            "embedder": {
                "type": "huggingface",
                "huggingface": {"model": "BAAI/bge-large-en-v1.5"},
            },
        })
        assert cfg.client_id == "test"
        assert cfg.security is not None
        assert cfg.security.pii_middleware.enabled is False

    def test_security_defaults(self):
        from app.core.config.client_config_schema import ClientConfig
        cfg = ClientConfig.from_dict({
            "client_id": "test",
            "vectordb": {"type": "chroma", "collection": "c", "chroma": {"persist_directory": "."}},
            "embedder": {"type": "huggingface", "huggingface": {"model": "m"}},
        })
        assert cfg.security.pii_middleware.enabled is False
        assert cfg.security.pii_middleware.action == "REDACT"

    def test_prompt_defaults(self):
        from app.core.config.client_config_schema import ClientConfig
        cfg = ClientConfig.from_dict({
            "client_id": "test",
            "vectordb": {"type": "chroma", "collection": "c", "chroma": {"persist_directory": "."}},
            "embedder": {"type": "huggingface", "huggingface": {"model": "m"}},
        })
        assert cfg.prompt.enabled is False
        assert cfg.prompt.prompt_type == "rag_context"


class TestNewConfigSections:
    """Test new config sections with explicit values."""

    def test_pii_middleware_config(self):
        from app.core.config.client_config_schema import ClientConfig
        cfg = ClientConfig.from_dict({
            "client_id": "test",
            "vectordb": {"type": "chroma", "collection": "c", "chroma": {"persist_directory": "."}},
            "embedder": {"type": "huggingface", "huggingface": {"model": "m"}},
            "security": {
                "pii_middleware": {
                    "enabled": True,
                    "positions": ["pre_embedding", "pre_llm", "post_llm"],
                    "action": "REDACT",
                    "trust_score_penalty": 0.2,
                }
            },
        })
        assert cfg.security.pii_middleware.enabled is True
        assert "post_llm" in cfg.security.pii_middleware.positions
        assert cfg.security.pii_middleware.trust_score_penalty == 0.2

    def test_new_llm_types(self):
        from app.core.config.client_config_schema import LLMType
        assert LLMType.MISTRAL == "mistral"
        assert LLMType.AZURE_OPENAI == "azure_openai"
        assert LLMType.CUSTOM == "custom"

    def test_feature_flags_extended(self):
        from app.core.config.client_config_schema import FeatureFlags
        ff = FeatureFlags()
        assert ff.enable_bundle_validation is False
        assert ff.enable_pii_middleware is False
        assert ff.enable_advanced_nodes is False

    def test_fusion_method_in_retrieval(self):
        from app.core.config.client_config_schema import RetrievalConfig
        rc = RetrievalConfig()
        assert rc.fusion_method == "rrf"
