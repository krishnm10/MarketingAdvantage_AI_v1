"""PipelineFactory LRU cache eviction and staleness tests."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from app.core.config.client_config_schema import ClientConfig
from app.core.pipeline_factory import PipelineFactory, _MAX_CACHED_PIPELINES


def _test_config(client_id: str) -> ClientConfig:
    return ClientConfig.from_dict(
        {
            "client_id": client_id,
            "vectordb": {
                "type": "chroma",
                "collection": "docs",
                "chroma": {"persist_directory": "./test_db"},
            },
            "embedder": {
                "type": "huggingface",
                "huggingface": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            },
        }
    )


def _mock_pipeline(client_id: str) -> MagicMock:
    pipeline = MagicMock()
    pipeline.client_id = client_id
    pipeline.close = MagicMock()
    return pipeline


def test_lru_evicts_oldest_when_over_max(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.core.pipeline_factory._MAX_CACHED_PIPELINES", 2)

    factory = PipelineFactory(cache_pipelines=True)
    p1 = _mock_pipeline("tenant-a")
    p2 = _mock_pipeline("tenant-b")
    p3 = _mock_pipeline("tenant-c")

    with factory._lock:
        factory._cache["tenant-a"] = p1
        factory._cache["tenant-b"] = p2
        factory._cache_fingerprints["tenant-a"] = "fp-a"
        factory._cache_fingerprints["tenant-b"] = "fp-b"
        factory._cache_inserted_at["tenant-a"] = time.time()
        factory._cache_inserted_at["tenant-b"] = time.time()
        factory._cache["tenant-c"] = p3
        factory._cache_fingerprints["tenant-c"] = "fp-c"
        factory._cache_inserted_at["tenant-c"] = time.time()
        factory._evict_oldest_if_over_capacity()

    assert "tenant-a" not in factory._cache
    assert "tenant-b" in factory._cache
    assert "tenant-c" in factory._cache
    p1.close.assert_called_once()


def test_stale_cache_entry_is_removed_on_get(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.core.pipeline_factory._MAX_CACHE_AGE_S", 1)

    factory = PipelineFactory(cache_pipelines=True)
    pipeline = _mock_pipeline("tenant-stale")

    with factory._lock:
        factory._cache["tenant-stale"] = pipeline
        factory._cache_fingerprints["tenant-stale"] = "fp"
        factory._cache_inserted_at["tenant-stale"] = time.time() - 10

    assert factory.get_cached("tenant-stale") is None
    pipeline.close.assert_called_once()
