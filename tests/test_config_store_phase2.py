"""Phase 2: ConfigStore, ConfigChangeBus, and version-aware runtime cache."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict
from unittest.mock import patch

import pytest

from app.core.config.client_config_schema import EffectiveTenantRuntime
from app.core.config.config_change_bus import NoOpConfigChangeBus, set_config_change_bus
from app.core.config.config_store import (
    FileSystemConfigStore,
    compute_config_version,
    set_config_store,
)
import app.core.config.effective_tenant_runtime as effective_runtime_module
from app.core.config.effective_tenant_runtime import (
    get_effective_tenant_runtime,
    invalidate_effective_tenant_runtime_cache,
)


_MINIMAL_MERGED: Dict[str, Any] = {
    "client_id": "tenant-a",
    "vectordb": {
        "type": "chroma",
        "collection": "c",
        "chroma": {"persist_directory": "./db"},
    },
    "embedder": {
        "type": "huggingface",
        "huggingface": {"model": "BAAI/bge-large-en-v1.5"},
    },
}


class InMemoryConfigStore:
    """Test double with controllable version keys."""

    def __init__(
        self,
        *,
        merged_by_client: Dict[str, Dict[str, Any]],
        versions: Dict[str, str] | None = None,
    ) -> None:
        self._merged = {k: deepcopy(v) for k, v in merged_by_client.items()}
        self._versions = dict(versions or {})
        for cid, raw in self._merged.items():
            self._versions.setdefault(cid, compute_config_version(raw))
        self.load_raw_calls = 0
        self.get_version_calls = 0

    def load_raw(self, client_id: str) -> Dict[str, Any]:
        self.load_raw_calls += 1
        if client_id not in self._merged:
            raise KeyError(client_id)
        return deepcopy(self._merged[client_id])

    def get_version(self, client_id: str) -> str:
        self.get_version_calls += 1
        return self._versions[client_id]

    def save_raw(self, client_id: str, data: Dict[str, Any]) -> str:
        self._merged[client_id] = deepcopy(data)
        self._versions[client_id] = compute_config_version(data)
        return self._versions[client_id]

    def set_version(self, client_id: str, version: str) -> None:
        self._versions[client_id] = version


def _sample_runtime(client_id: str = "tenant-a") -> EffectiveTenantRuntime:
    return EffectiveTenantRuntime(
        client_id=client_id,
        fingerprint="abc123",
        runtime_mode="authoritative_config",
        stack_profile="cloud",
        embedder={"type": "huggingface", "model": "m", "locked": True},
        llm={
            "configured_provider": None,
            "configured_model": None,
            "effective_provider": "none",
            "effective_model": "none",
            "source": "tenant_json",
        },
        reranker={
            "configured_type": None,
            "configured_model": None,
            "effective_plugin": "none",
            "effective_model": None,
            "coercion_applied": False,
            "coercion_reason": None,
        },
        retrieval={
            "search_mode": "semantic",
            "top_k_retrieval": 20,
            "top_k_final": 5,
            "enable_hyde": False,
            "prompt_template_id": None,
            "prompt_ssot": {
                "effective_template_id": None,
                "source": "default_builtin",
                "configured_prompt_type": None,
                "library_found": False,
                "preview": None,
                "legacy_inline_detected": False,
            },
        },
        prompt_node={
            "enabled": False,
            "configured_prompt_type": None,
            "effective_template_id": None,
        },
        features={
            "enable_rag": True,
            "enable_reranking": True,
            "enable_hybrid_search": False,
            "enable_pii_middleware": False,
            "enable_advanced_nodes": False,
        },
        warnings=[],
    )


@pytest.fixture
def isolated_config_stack(monkeypatch: pytest.MonkeyPatch):
    """Reset store, bus, and runtime cache between tests."""
    invalidate_effective_tenant_runtime_cache()
    effective_runtime_module._BUS_SUBSCRIBED = False
    set_config_store(None)
    set_config_change_bus(NoOpConfigChangeBus())
    yield
    invalidate_effective_tenant_runtime_cache()
    effective_runtime_module._BUS_SUBSCRIBED = False
    set_config_store(None)
    set_config_change_bus(None)


class TestFileSystemConfigStore:
    def test_get_version_changes_when_file_content_changes(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg_dir = tmp_path / "configs"
        cfg_dir.mkdir()
        default_path = cfg_dir / "default.json"
        default_path.write_text(
            '{"client_id":"default","vectordb":{"type":"chroma","collection":"c",'
            '"chroma":{"persist_directory":"./db"}},"embedder":{"type":"huggingface",'
            '"huggingface":{"model":"m"}}}',
            encoding="utf-8",
        )
        tenant_path = cfg_dir / "tenant-a.json"
        tenant_path.write_text(
            '{"client_id":"tenant-a","features":{"enable_rag":true}}',
            encoding="utf-8",
        )

        store = FileSystemConfigStore(config_dirs=[cfg_dir])
        v1 = store.get_version("tenant-a")

        tenant_path.write_text(
            '{"client_id":"tenant-a","features":{"enable_rag":false}}',
            encoding="utf-8",
        )
        v2 = store.get_version("tenant-a")

        assert v1 != v2


class TestRuntimeVersionCache:
    def test_cache_hit_when_version_unchanged(self, isolated_config_stack) -> None:
        store = InMemoryConfigStore(merged_by_client={"tenant-a": _MINIMAL_MERGED})
        set_config_store(store)
        sample = _sample_runtime()

        with patch.object(
            effective_runtime_module,
            "_build_effective_tenant_runtime_uncached",
            return_value=sample,
        ) as mock_build:
            rt1 = get_effective_tenant_runtime("tenant-a")
            rt2 = get_effective_tenant_runtime("tenant-a")

        assert mock_build.call_count == 1
        assert rt1.fingerprint == rt2.fingerprint
        assert rt1.client_id == rt2.client_id

    def test_cache_miss_when_version_changes(self, isolated_config_stack) -> None:
        store = InMemoryConfigStore(merged_by_client={"tenant-a": _MINIMAL_MERGED})
        set_config_store(store)
        sample = _sample_runtime()

        with patch.object(
            effective_runtime_module,
            "_build_effective_tenant_runtime_uncached",
            return_value=sample,
        ) as mock_build:
            get_effective_tenant_runtime("tenant-a")
            store.set_version("tenant-a", "forced-new-version")
            get_effective_tenant_runtime("tenant-a")

        assert mock_build.call_count == 2

    def test_skip_cache_bypasses_version_lookup(self, isolated_config_stack) -> None:
        store = InMemoryConfigStore(merged_by_client={"tenant-a": _MINIMAL_MERGED})
        set_config_store(store)
        sample = _sample_runtime()

        with patch.object(
            effective_runtime_module,
            "_build_effective_tenant_runtime_uncached",
            return_value=sample,
        ) as mock_build:
            get_effective_tenant_runtime("tenant-a", skip_cache=True)
            get_effective_tenant_runtime("tenant-a", skip_cache=True)

        assert mock_build.call_count == 2


class TestConfigChangeBus:
    def test_publish_invalidates_runtime_cache(self, isolated_config_stack) -> None:
        store = InMemoryConfigStore(merged_by_client={"tenant-a": _MINIMAL_MERGED})
        bus = NoOpConfigChangeBus()
        set_config_store(store)
        set_config_change_bus(bus)
        sample = _sample_runtime()

        with patch.object(
            effective_runtime_module,
            "_build_effective_tenant_runtime_uncached",
            return_value=sample,
        ) as mock_build:
            get_effective_tenant_runtime("tenant-a")
            bus.publish("tenant-a", "remote-version-bump")
            get_effective_tenant_runtime("tenant-a")

        assert mock_build.call_count == 2
