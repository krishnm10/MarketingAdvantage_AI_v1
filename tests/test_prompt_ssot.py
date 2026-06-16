"""Tests for prompt SSOT resolver."""

from __future__ import annotations

from app.core.config.client_config_resolver import get_client_config
from app.core.prompts.ssot import (
    PROMPT_PRESET_LIBRARY,
    enforce_library_first_prompt_persist,
    resolve_prompt_ssot,
    sync_prompt_ssot_to_config,
)


def test_preset_map_has_files():
    for preset_id in PROMPT_PRESET_LIBRARY.values():
        res_id = preset_id
        from app.core.prompts.library_loader import load_prompt_library_raw

        assert load_prompt_library_raw(res_id) is not None, f"missing preset {res_id}"


def test_prompt_template_id_wins_over_prompt_type():
    cfg = get_client_config("matha")
    res = resolve_prompt_ssot(cfg)
    assert res.effective_template_id == cfg.retrieval.prompt_template_id
    assert res.source == "library"


def test_sync_prompt_ssot_sets_library_id():
    cfg = get_client_config("matha")
    patched = cfg.model_copy(
        update={
            "retrieval": cfg.retrieval.model_copy(update={"prompt_template_id": None}),
            "prompt": cfg.prompt.model_copy(
                update={"enabled": True, "prompt_type": "cot"}
            ),
        }
    )
    synced = sync_prompt_ssot_to_config(patched)
    assert synced.retrieval.prompt_template_id == PROMPT_PRESET_LIBRARY["cot"]
    assert synced.prompt is None or synced.prompt.template in (None, "")


def test_enforce_library_first_strips_inline_template():
    from app.api.v2.rag_config_api import _build_default_config_dict
    from app.core.config.client_config_resolver import load_default_client_raw_dict

    seed = _build_default_config_dict("new_tenant_xyz")
    default_prompt_id = load_default_client_raw_dict()["retrieval"]["prompt_template_id"]
    assert seed["retrieval"]["prompt_template_id"] == default_prompt_id
    prompt = seed.get("prompt") or {}
    assert prompt.get("template") in (None, "")
    assert "Finance & Invoice Specialist" not in str(seed)


def test_enforce_clears_dual_source_dict():
    raw = {
        "client_id": "dual_test",
        "retrieval": {"prompt_template_id": "querysystem", "search_mode": "hybrid"},
        "prompt": {
            "enabled": True,
            "prompt_type": "rag_context",
            "template": "legacy inline text",
        },
        "vectordb": {"type": "chroma", "collection": "c"},
        "embedder": {"type": "ollama", "ollama": {"model": "nomic-embed-text"}},
        "llm": {
            "single": {
                "type": "ollama",
                "model": "llama3.2",
                "base_url": "http://localhost:11434",
            }
        },
    }
    out = enforce_library_first_prompt_persist(raw, client_id="dual_test")
    assert out["retrieval"]["prompt_template_id"] == "querysystem"
    assert out["prompt"]["template"] is None
