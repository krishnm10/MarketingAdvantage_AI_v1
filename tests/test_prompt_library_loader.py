"""Unit tests for app.core.prompts.library_loader."""

from __future__ import annotations

import json

from app.api.v2 import retrieve_chat_api as chat_api
from app.core.prompts import library_loader


def test_load_invalid_slug_returns_none(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(library_loader, "PROMPTS_DIR", tmp_path)
    assert library_loader.load_prompt_library_raw("../../etc") is None
    assert library_loader.load_prompt_library_raw("") is None


def test_load_missing_file_returns_none(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(library_loader, "PROMPTS_DIR", tmp_path)
    assert library_loader.load_prompt_library_raw("acme-rag-v1") is None


def test_load_and_system_instructions(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(library_loader, "PROMPTS_DIR", tmp_path)
    data = {
        "template_id": "acme-rag-v1",
        "name": "Acme",
        "system_instructions": "You are the Acme invoice assistant.",
        "version": "1.0.0",
    }
    (tmp_path / "acme-rag-v1.json").write_text(
        json.dumps(data), encoding="utf-8"
    )
    raw = library_loader.load_prompt_library_raw("acme-rag-v1")
    assert raw is not None
    assert library_loader.system_instructions_from_record(raw) == (
        "You are the Acme invoice assistant."
    )
    assert library_loader.get_system_instructions("acme-rag-v1") == (
        "You are the Acme invoice assistant."
    )


def test_content_fallback_field(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(library_loader, "PROMPTS_DIR", tmp_path)
    (tmp_path / "legacy.json").write_text(
        json.dumps({"template_id": "legacy", "content": "From content field"}),
        encoding="utf-8",
    )
    assert library_loader.get_system_instructions("legacy") == "From content field"


def test_chat_compliance_block_uses_grounding_phrase() -> None:
    blob = chat_api._chat_rag_compliance_block()
    assert chat_api._GROUNDING_REFUSAL_PHRASE in blob
    assert "Rules:" in blob and "  9. " in blob


def test_validate_all_templates_passes_on_repo_prompts() -> None:
    library_loader.validate_all_templates()


def test_load_template_ignores_examples_field(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(library_loader, "PROMPTS_DIR", tmp_path)
    (tmp_path / "with-examples.json").write_text(
        json.dumps(
            {
                "template_id": "with-examples",
                "system_instructions": "Answer using ONLY the provided context.",
                "examples": [
                    {
                        "label": "Example C",
                        "input": "total?",
                        "output": "Total Due: $4,537.50",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert library_loader.load_template("with-examples") == (
        "Answer using ONLY the provided context."
    )


def test_strip_is_noop_on_migrated_instructions() -> None:
    inst = library_loader.load_template("preset-chain-of-thought")
    assert inst is not None
    assert library_loader.strip_examples_from_instructions(inst) == inst.strip()
    assert "4537.50" not in inst
    assert "Example C" not in inst


def test_preset_rag_context_unchanged_instructions() -> None:
    inst = library_loader.load_template("preset-rag-context")
    assert inst is not None
    assert "ONLY the context provided" in inst
    raw = library_loader.load_prompt_library_raw("preset-rag-context")
    assert raw is not None
    assert raw.get("examples") == []
    assert raw.get("examples_note")


def test_cot_examples_accessible_for_editor() -> None:
    raw = library_loader.load_prompt_library_raw("preset-chain-of-thought")
    assert raw is not None
    examples = raw.get("examples") or []
    assert len(examples) == 3
    labels = [ex["label"] for ex in examples]
    assert any("Example C" in label for label in labels)
    assert any("4,537.50" in ex.get("output", "") for ex in examples)
