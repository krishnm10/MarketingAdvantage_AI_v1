"""Tests for EffectiveTenantRuntime SSOT bridge."""

from __future__ import annotations

import pytest

from app.core.config.client_config_schema import PROMPT_PREVIEW_MAX_CHARS
from app.core.config.effective_tenant_runtime import build_effective_tenant_runtime
from app.core.config.reranker_config_coercion import is_local_ollama_stack
from app.core.config.client_config_resolver import get_client_config
from app.core.prompts.ssot import resolve_prompt_ssot


@pytest.fixture
def matha_runtime():
    return build_effective_tenant_runtime("matha")


def test_matha_runtime_loads():
    rt = build_effective_tenant_runtime("matha")
    assert rt.client_id == "matha"
    assert rt.fingerprint
    assert rt.runtime_mode == "authoritative_config"


def test_matha_ollama_stack_profile(matha_runtime):
    cfg = get_client_config("matha")
    if is_local_ollama_stack(cfg):
        assert matha_runtime.stack_profile == "local_ollama"
        assert matha_runtime.llm.effective_provider == "ollama"
        assert matha_runtime.reranker.effective_plugin == "flashrank"


def test_matha_prompt_library_ssot(matha_runtime):
    assert matha_runtime.retrieval.prompt_template_id == "querysystem"
    assert matha_runtime.retrieval.prompt_ssot.library_found
    assert matha_runtime.retrieval.prompt_ssot.source == "library"
    preview = matha_runtime.retrieval.prompt_ssot.preview
    assert preview is None or len(preview) <= PROMPT_PREVIEW_MAX_CHARS


def test_preview_truncation():
    from app.core.prompts.ssot import _truncate_preview

    long_text = "x" * 500
    out = _truncate_preview(long_text)
    assert out is not None
    assert len(out) <= PROMPT_PREVIEW_MAX_CHARS


def test_reranker_disabled_none():
    from app.core.config.client_config_schema import ClientConfig, FeatureFlags

    cfg = get_client_config("matha")
    patched = cfg.model_copy(
        update={"features": cfg.features.model_copy(update={"enable_reranking": False})}
    )
    from app.core.config.reranker_config_coercion import resolve_reranker_runtime

    rr = resolve_reranker_runtime(patched)
    assert rr.plugin_name == "none"


def test_resolve_prompt_ssot_library_priority():
    cfg = get_client_config("matha")
    res = resolve_prompt_ssot(cfg)
    assert res.source == "library"
    assert res.effective_template_id == "querysystem"
    assert res.instructions
