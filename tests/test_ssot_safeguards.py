"""Enterprise safeguards: emergency prompt fallback, runtime cache, coercion audit."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.core.config.effective_tenant_runtime import (
    build_effective_tenant_runtime,
    invalidate_effective_tenant_runtime_cache,
)
from app.core.config.client_config_resolver import get_client_config
from app.core.prompts.ssot import (
    emergency_fallback_instructions,
    resolve_prompt_ssot,
)


def test_emergency_fallback_when_library_missing():
    cfg = get_client_config("matha")
    broken = cfg.model_copy(
        update={
            "retrieval": cfg.retrieval.model_copy(
                update={"prompt_template_id": "nonexistent-template-xyz"}
            )
        }
    )
    res = resolve_prompt_ssot(broken)
    assert res.source == "emergency_fallback"
    assert res.instructions == emergency_fallback_instructions()
    assert res.library_found is False


def test_runtime_cache_hit_and_invalidation():
    invalidate_effective_tenant_runtime_cache()
    rt1 = build_effective_tenant_runtime("matha")
    rt2 = build_effective_tenant_runtime("matha")
    assert rt1.fingerprint == rt2.fingerprint
    assert rt1.client_id == rt2.client_id

    invalidate_effective_tenant_runtime_cache("matha")
    rt3 = build_effective_tenant_runtime("matha", skip_cache=False)
    assert rt3.fingerprint == rt1.fingerprint


def test_reranker_coercion_audit_emitted_for_matha():
    with patch(
        "app.core.config.reranker_config_coercion._audit_reranker_coercion"
    ) as mock_audit:
        from app.core.config.reranker_config_coercion import resolve_reranker_runtime

        cfg = get_client_config("matha")
        resolve_reranker_runtime(cfg)
        if cfg.reranker and cfg.is_reranking_enabled():
            # matha Ollama stack should coerce flashrank
            mock_audit.assert_called_once()
