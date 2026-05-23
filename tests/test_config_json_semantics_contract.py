"""
Contract: pipeline semantics come from merged Client JSON / templates, not provider env.

With OLLAMA_* / HF_EMBED_* unset, default templates and resolve_runtime_components()
must still yield stable models and URLs from default.json + static fallbacks.
"""

from __future__ import annotations

import os

import pytest

from app.core.config.default_config_templates import (
    default_embedder_dict_for_type,
    default_llm_root_dict_for_provider,
)
from app.core.config.client_config_resolver import get_client_config
from app.retrieval.components import resolve_runtime_components


def test_default_embedder_template_no_ollama_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in list(os.environ.keys()):
        if k.startswith("OLLAMA_") or k.startswith("HF_EMBED"):
            monkeypatch.delenv(k, raising=False)
    d = default_embedder_dict_for_type("ollama", None)
    assert d["type"] == "ollama"
    assert d["ollama"]["base_url"] == "http://localhost:11434"
    assert d["ollama"]["model"] == "nomic-embed-text"


def test_default_llm_template_gemini_uses_google_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_LLM_MODEL", raising=False)
    d = default_llm_root_dict_for_provider("gemini", None)
    assert d["single"]["api_key_env"] == "GOOGLE_API_KEY"
    assert "generativelanguage.googleapis.com" in d["single"]["base_url"]


def test_default_llm_switching_from_gemini_to_ollama_replaces_provider_fields() -> None:
    """prev_single must not overwrite type/model/base_url when LLM provider changes."""
    prev = {
        "single": {
            "type": "gemini",
            "model": "gemini-2.5-flash",
            "api_key_env": "GOOGLE_API_KEY",
            "base_url": "https://generativelanguage.googleapis.com/v1",
            "temperature": 0.5,
            "max_tokens": 2048,
        }
    }
    d = default_llm_root_dict_for_provider("ollama", prev)
    assert d["single"]["type"] == "ollama"
    assert d["single"]["model"] == "llama3.2"
    assert d["single"]["base_url"] == "http://localhost:11434"
    assert d["single"]["temperature"] == 0.5
    assert d["single"]["max_tokens"] == 2048


def test_resolve_runtime_components_answer_min_score_from_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-for-contract-test")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-for-contract-test")
    monkeypatch.delenv("RAG_ANSWER_MIN_SCORE", raising=False)
    cfg = get_client_config("default")
    rc = resolve_runtime_components(cfg)
    assert rc.rag_min_score == float(cfg.retrieval.answer_min_score)
    assert rc.llm_base_url
