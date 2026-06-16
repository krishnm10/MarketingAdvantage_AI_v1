"""Tests for stack-topology aware reranker resolution."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.config.client_config_schema import (
    ClientConfig,
    EmbedderConfig,
    EmbedderType,
    LLMConfig,
    LLMType,
    RerankerConfig,
    RerankerType,
    SingleLLMConfig,
    VectorDBConfig,
    VectorDBType,
)
from app.core.config.reranker_config_coercion import (
    coerce_reranker_config,
    is_local_ollama_stack,
    looks_like_llm_judge_model,
    normalize_model_for_plugin,
    resolve_reranker_runtime,
)
from app.retrieval.components import resolve_runtime_components
from app.retrieval.reranker_runtime import apply_reranker_with_fallback
from app.core.rerankers.base import RerankCandidate


def _minimal_config(**overrides) -> ClientConfig:
    base = {
        "client_id": "matha",
        "vectordb": {
            "type": "chroma",
            "collection": "ingested_content",
            "chroma": {"persist_directory": "./testing_Matha"},
        },
        "embedder": {
            "type": "ollama",
            "ollama": {"model": "nomic-embed-text", "base_url": "http://localhost:11434"},
        },
        "llm": {
            "single": {
                "type": "ollama",
                "model": "llama3.2:1b",
                "base_url": "http://localhost:11434",
            }
        },
        "features": {"enable_reranking": True},
    }
    base.update(overrides)
    return ClientConfig.from_dict(base)


def test_looks_like_llm_judge_model():
    assert looks_like_llm_judge_model("gpt-4o-mini")
    assert looks_like_llm_judge_model("llm-judge/gpt-4o-mini")
    assert not looks_like_llm_judge_model("ms-marco-MiniLM-L-12-v2")


def test_normalize_flashrank_strips_prefix():
    assert normalize_model_for_plugin(
        "flashrank", "flashrank/ms-marco-MiniLM-L-12-v2"
    ) == "ms-marco-MiniLM-L-12-v2"


def test_normalize_flashrank_rejects_llm_name():
    assert normalize_model_for_plugin("flashrank", "gpt-4o-mini") == (
        "ms-marco-MiniLM-L-12-v2"
    )


def test_ollama_stack_forces_local_flashrank():
    cfg = _minimal_config(
        reranker={
            "type": "crossencoder",
            "model": "gpt-4o-mini",
            "secret_ref": {"uri": "env://OPENAI_API_KEY"},
            "top_k": 5,
        }
    )
    assert is_local_ollama_stack(cfg)
    resolved = resolve_reranker_runtime(cfg)
    assert resolved.plugin_name == "flashrank"
    assert resolved.model_name == "ms-marco-MiniLM-L-12-v2"
    assert resolved.fallback_applied is True


def test_coerce_crossencoder_gpt_to_llm_judge_when_api_key_non_local():
    cfg = RerankerConfig(
        type=RerankerType.CROSS_ENCODER,
        model="gpt-4o-mini",
        secret_ref={"uri": "env://OPENAI_API_KEY"},
    )
    out = coerce_reranker_config(cfg, local_stack=False)
    assert out.type == RerankerType.LLM_JUDGE
    assert out.model == "gpt-4o-mini"


def test_resolve_runtime_components_matha_like():
    cfg = _minimal_config(
        reranker={
            "type": "crossencoder",
            "model": "gpt-4o-mini",
            "secret_ref": {"uri": "env://OPENAI_API_KEY"},
        }
    )
    rc = resolve_runtime_components(cfg)
    assert rc.reranker_name == "flashrank"
    assert "marco" in (rc.reranker_model or "").lower()


def test_circuit_breaker_falls_back_to_flashrank():
    cfg = ClientConfig.from_dict(
        {
            "client_id": "cloud",
            "vectordb": {
                "type": "chroma",
                "collection": "ingested_content",
                "chroma": {"persist_directory": "./pluggable_db"},
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
                    "base_url": "https://generativelanguage.googleapis.com",
                }
            },
            "features": {"enable_reranking": True},
            "reranker": {
                "type": "llm_judge",
                "model": "gpt-4o-mini",
                "secret_ref": {"uri": "env://OPENAI_API_KEY"},
                "judge_provider": "openai",
            },
        }
    )
    candidates = [
        RerankCandidate(id="a", text="hello", vector_score=0.9, metadata={}),
        RerankCandidate(id="b", text="world", vector_score=0.8, metadata={}),
    ]

    class _FailJudge:
        def rerank(self, query, cands, top_k=10):
            raise RuntimeError("401 Unauthorized invalid api key")

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = candidates

    with patch(
        "app.retrieval.reranker_runtime._build_reranker_instance",
        side_effect=[_FailJudge(), mock_reranker],
    ):
        scored, used, fb, reason = apply_reranker_with_fallback(
            config=cfg,
            query="test",
            candidates=candidates,
            top_k=2,
        )

    assert used == "flashrank"
    assert fb is True
    assert reason == "circuit_breaker_local_fallback"
    mock_reranker.rerank.assert_called_once()
