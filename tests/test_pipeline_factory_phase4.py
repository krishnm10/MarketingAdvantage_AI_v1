"""Phase 4: PipelineFactory wired to SecretResolver."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.config.client_config_schema import ClientConfig
from app.core.pipeline_factory import PipelineFactory
from app.core.secrets.connectors.base import SecretResolutionError
from app.core.secrets.resolver import SecretResolver


def _openai_embedder_config() -> ClientConfig:
    return ClientConfig.from_dict(
        {
            "client_id": "tenant-a",
            "secrets_backend": {"provider": "env"},
            "vectordb": {
                "type": "chroma",
                "collection": "docs",
                "chroma": {"persist_directory": "./test_db"},
            },
            "embedder": {
                "type": "openai",
                "openai": {
                    "model": "text-embedding-3-small",
                    "secret_ref": {"uri": "env://OPENAI_API_KEY"},
                },
            },
        }
    )


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_pipeline_factory_builds_embedder_with_resolved_secret() -> None:
    config = _openai_embedder_config()
    resolver = AsyncMock(spec=SecretResolver)
    resolver.resolve = AsyncMock(return_value="resolved-openai-key")

    factory = PipelineFactory(cache_pipelines=False, secret_resolver=resolver)

    mock_vdb = MagicMock()
    mock_vdb.kind = "chroma"
    mock_vdb.ensure_collection = MagicMock()

    mock_emb = MagicMock()
    mock_emb.info.model = "text-embedding-3-small"
    mock_emb.info.dim = 1536
    mock_emb.embedding_dim = 1536

    with patch("app.core.pipeline_factory.vectordb_registry.build", return_value=mock_vdb), patch(
        "app.core.pipeline_factory.embedder_registry.build", return_value=mock_emb
    ) as embedder_build, patch(
        "app.core.config.client_config_resolver.validate_config_compatibility",
        return_value=[],
    ):
        pipeline = await factory.build_async(config, skip_cache=True)

    assert pipeline.embedder is mock_emb
    embedder_build.assert_called_once()
    assert embedder_build.call_args.kwargs["api_key"] == "resolved-openai-key"
    resolver.resolve.assert_awaited()
    purpose = resolver.resolve.await_args.kwargs["purpose"]
    assert purpose == "embedder.openai"


@pytest.mark.anyio
async def test_pipeline_factory_fails_fast_when_secret_resolution_fails() -> None:
    config = _openai_embedder_config()
    resolver = AsyncMock(spec=SecretResolver)
    resolver.resolve = AsyncMock(
        side_effect=SecretResolutionError("Environment variable 'OPENAI_API_KEY' is not set.")
    )

    factory = PipelineFactory(cache_pipelines=False, secret_resolver=resolver)

    with patch(
        "app.core.config.client_config_resolver.validate_config_compatibility",
        return_value=[],
    ):
        with pytest.raises(SecretResolutionError, match="not set"):
            await factory.build_async(config, skip_cache=True)
