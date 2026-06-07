"""Unit tests for pre-verification answer mutators in retrieve_chat_api."""

from __future__ import annotations

from app.api.v2.retrieve_chat_api import (
    _GROUNDING_REFUSAL_PHRASE,
    _build_focus_fallback_answer,
    _dedupe_identifier_lines,
    _filter_to_multi_identifier_chunks,
    _maybe_focus_fallback_answer,
    _strip_contradictory_refusal,
)
from app.retrieval.types_retrieve import RankedResult
from app.retrieval.answer_integrity import (
    extract_citation_indices,
    validate_value_preservation,
)


def test_strip_contradictory_refusal_removes_only_refusal_line():
    refusal = _GROUNDING_REFUSAL_PHRASE
    raw = (
        f"Invoice INV-1 is overdue [Source 1].\n"
        f"{refusal}"
    )
    cleaned, changed = _strip_contradictory_refusal(raw)
    assert changed is True
    assert refusal not in cleaned
    assert "INV-1" in cleaned
    assert extract_citation_indices(cleaned) == [1]


def test_strip_contradictory_refusal_preserves_citations_and_currency():
    raw = "Total due is $110.00 for INV-2 [Source 2]."
    cleaned, changed = _strip_contradictory_refusal(raw)
    assert changed is False
    assert cleaned == raw
    value = validate_value_preservation(raw, cleaned)
    assert value.valid is True


def test_strip_contradictory_refusal_stable_output():
    raw = (
        "Line A $50.00 [Source 1].\n"
        f"{_GROUNDING_REFUSAL_PHRASE}"
    )
    first, _ = _strip_contradictory_refusal(raw)
    second, _ = _strip_contradictory_refusal(raw)
    assert first == second


def test_dedupe_identifier_lines_collapses_duplicate_ids():
    raw = (
        "- INV-100 — overdue [Source 1]\n"
        "- INV-100 — overdue (again) [Source 1]\n"
        "- INV-200 — paid [Source 2]\n"
    )
    merged, changed = _dedupe_identifier_lines(raw)
    assert changed is True
    assert merged.count("INV-100") == 1
    assert "INV-200" in merged
    assert extract_citation_indices(merged) == [1, 2]


def test_dedupe_identifier_lines_preserves_unique_id_values():
    raw = "- INV-100 [Source 1]\n- INV-200 [Source 2]\n"
    merged, changed = _dedupe_identifier_lines(raw)
    assert changed is False
    assert "INV-100" in merged
    assert "INV-200" in merged


def test_dedupe_identifier_lines_deterministic():
    raw = "- INV-A [Source 1]\n- INV-A again [Source 1]\n"
    a, _ = _dedupe_identifier_lines(raw)
    b, _ = _dedupe_identifier_lines(raw)
    assert a == b


def test_build_focus_fallback_answer_includes_context_excerpt():
    context = (
        "[Source 1] (score=0.900)\n"
        "Invoice INV-55 is overdue with $25.00 late fee.\n\n"
        "---\n\n"
        "[Source 2] (score=0.500)\n"
        "Unrelated chunk."
    )
    fb = _build_focus_fallback_answer(context, "INV-55")
    assert fb is not None
    assert "INV-55" in fb
    assert "$25.00" in fb
    assert "overdue" in fb


def test_maybe_focus_fallback_answer_documented_full_replacement():
    """
    DOCUMENTED EXCEPTION: focus fallback replaces bare refusal with excerpts.
    Value/citation preservation relative to pre-fallback answer is NOT GUARANTEED.
    """
    context = (
        "[Source 1] (score=0.900)\n"
        "Invoice INV-77 overdue $10.00.\n"
    )
    replaced, used = _maybe_focus_fallback_answer(
        _GROUNDING_REFUSAL_PHRASE,
        context,
        "INV-77",
    )
    assert used is True
    assert replaced is not None
    assert "INV-77" in replaced
    # Excerpt style copies context blocks including [Source n] headers from context_str.
    assert "[Source 1]" in replaced


def test_maybe_focus_fallback_answer_no_op_when_grounded_answer_exists():
    context = "[Source 1] (score=0.900)\nINV-88 text.\n"
    answer = "INV-88 is overdue [Source 1]."
    out, used = _maybe_focus_fallback_answer(answer, context, "INV-88")
    assert used is False
    assert out == answer


def test_mutators_do_not_introduce_invalid_citation_index_when_present():
    """When citations exist, mutators should not rewrite [Source n] indices."""
    raw = "Claim CLM-9001 amount $50.00 on 2024-06-01 [Source 1]."
    for fn in (
        lambda t: _strip_contradictory_refusal(t)[0],
        lambda t: _dedupe_identifier_lines(t)[0],
    ):
        out = fn(raw)
        assert extract_citation_indices(out) == [1]
        assert "CLM-9001" in out
        assert "$50.00" in out


def _chunk(cid: str, text: str, score: float, file_id: str) -> RankedResult:
    return RankedResult(
        chunk_id=cid,
        text=text,
        score=score,
        explanation={},
        file_id=file_id,
    )


def test_filter_to_multi_identifier_chunks_picks_one_chunk_per_id():
    chunks = [
        _chunk("a", "Invoice INV-6640 from NovaBuild total $9460", 0.80, "file-a"),
        _chunk("b", "Invoice INV-4427 from DataVault total $3696", 0.75, "file-b"),
        _chunk("c", "Unrelated policy text", 0.90, "file-c"),
    ]
    filtered = _filter_to_multi_identifier_chunks(
        chunks,
        target_ids=["INV-6640", "INV-4427"],
        max_chunks=2,
    )
    assert len(filtered) == 2
    texts = " ".join(c.text for c in filtered)
    assert "INV-6640" in texts
    assert "INV-4427" in texts
    assert filtered[0].chunk_id == "a"
    assert filtered[1].chunk_id == "b"


def test_filter_to_multi_identifier_chunks_respects_max_chunks_cap():
    chunks = [
        _chunk("a", "INV-6640 alpha", 0.9, "file-a"),
        _chunk("b", "INV-4427 beta", 0.8, "file-b"),
        _chunk("c", "INV-9999 gamma", 0.7, "file-c"),
    ]
    filtered = _filter_to_multi_identifier_chunks(
        chunks,
        target_ids=["INV-6640", "INV-4427", "INV-9999"],
        max_chunks=2,
    )
    assert len(filtered) == 2
