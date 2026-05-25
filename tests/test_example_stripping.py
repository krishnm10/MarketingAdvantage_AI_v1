"""Phase 5 — example stripping / library_loader safety checks."""

from __future__ import annotations

from app.core.prompts import library_loader


def test_strips_cot_dangerous_values_from_system_instructions() -> None:
    raw = library_loader.load_prompt_library_raw("preset-chain-of-thought")
    assert raw is not None
    inst = library_loader.system_instructions_from_record(raw)
    assert inst is not None
    result = library_loader.strip_and_verify(inst)
    assert "4,537.50" not in result
    assert "026015079" not in result
    assert "412.50" not in result


def test_preserves_rag_context_instructions_and_has_no_examples() -> None:
    raw = library_loader.load_prompt_library_raw("preset-rag-context")
    assert raw is not None
    inst = library_loader.system_instructions_from_record(raw)
    assert inst is not None
    result = library_loader.strip_and_verify(inst)
    assert result == inst
    examples = raw.get("examples")
    has_examples = isinstance(examples, list) and len(examples) > 0
    assert has_examples is False
