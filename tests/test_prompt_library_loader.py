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
