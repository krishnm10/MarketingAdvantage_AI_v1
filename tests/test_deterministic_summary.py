from __future__ import annotations

from copy import deepcopy

import pytest

from app.retrieval.deterministic_summary import (
    SUMMARY_FORMAT_VERSION,
    build_docset_summary,
    build_docset_summary_metadata,
    validate_llm_polish_output,
)
from app.retrieval.document_set_analysis import AnalysisResult


def _analysis(**kwargs) -> AnalysisResult:
    defaults = {
        "domain": "invoice",
        "docs_analyzed": 2,
        "docs_matched": 1,
        "matches": [],
        "degraded": False,
        "degradation_reason": None,
    }
    defaults.update(kwargs)
    return AnalysisResult(**defaults)


def test_build_docset_summary_stable_output():
    analysis = _analysis(
        docs_matched=2,
        matches=[
            {
                "file_id": "inv-1",
                "score": 0.8,
                "overdue": True,
                "evidence_snippets": ["Invoice overdue with penalty."],
            },
            {
                "file_id": "inv-2",
                "score": 0.6,
                "has_late_fee": True,
                "evidence_snippets": ["Late fee applied."],
            },
        ],
    )
    first = build_docset_summary(analysis)
    second = build_docset_summary(analysis)
    assert first == second
    assert first


def test_build_docset_summary_evidence_ordering():
    analysis = _analysis(
        docs_matched=3,
        matches=[
            {
                "file_id": "a",
                "score": 0.5,
                "overdue": True,
                "evidence_snippets": ["evidence A"],
            },
            {
                "file_id": "b",
                "score": 0.9,
                "overdue": True,
                "evidence_snippets": ["evidence B"],
            },
            {
                "file_id": "c",
                "score": 0.5,
                "overdue": True,
                "evidence_snippets": ["evidence C"],
            },
        ],
    )
    summary = build_docset_summary(analysis)
    pos_b = summary.index("evidence B")
    pos_a = summary.index("evidence A")
    pos_c = summary.index("evidence C")
    assert pos_b < pos_a < pos_c


def test_build_docset_summary_evidence_truncation():
    long_text = "x" * 250
    analysis = _analysis(
        docs_matched=1,
        matches=[
            {
                "file_id": "inv-1",
                "score": 1.0,
                "overdue": True,
                "evidence_snippets": [
                    long_text,
                    "snippet-2",
                    "snippet-3",
                    "snippet-4",
                ],
            },
        ],
    )
    summary = build_docset_summary(analysis)
    assert "snippet-4" not in summary
    assert "snippet-3" in summary
    assert "x" * 250 not in summary
    assert len([line for line in summary.splitlines() if "evidence:" in line]) <= 3


def test_build_docset_summary_total_evidence_cap():
    matches = []
    for idx in range(5):
        matches.append(
            {
                "file_id": f"inv-{idx}",
                "score": float(idx),
                "overdue": True,
                "evidence_snippets": [f"evidence-{idx}-1", f"evidence-{idx}-2"],
            }
        )
    analysis = _analysis(docs_matched=5, matches=matches)
    summary = build_docset_summary(analysis)
    evidence_lines = [
        line for line in summary.splitlines() if line.strip().startswith("- evidence:")
    ]
    assert len(evidence_lines) <= 10


def test_build_docset_summary_no_matches():
    analysis = _analysis(docs_matched=0, matches=[])
    summary = build_docset_summary(analysis)
    assert summary
    assert "No documents matched" in summary


def test_build_docset_summary_degraded():
    analysis = _analysis(
        degraded=True,
        degradation_reason="adapter unavailable",
        docs_matched=0,
        matches=[],
    )
    summary = build_docset_summary(analysis)
    assert "degraded" in summary.lower()
    assert "adapter unavailable" in summary


def test_build_docset_summary_aggregates_verbatim():
    analysis = _analysis(
        docs_matched=1,
        matches=[
            {
                "file_id": "inv-1",
                "overdue": True,
                "aggregates": {"total_overdue": 3, "ratio": 0.3333333333},
            }
        ],
    )
    summary = build_docset_summary(analysis)
    assert "total_overdue=3" in summary
    assert "ratio=0.3333333333" in summary


def test_build_docset_summary_does_not_mutate_analysis():
    matches = [
        {
            "file_id": "inv-1",
            "score": 0.7,
            "overdue": True,
            "evidence_snippets": ["stable evidence"],
        }
    ]
    analysis = _analysis(docs_matched=1, matches=matches)
    snapshot = deepcopy(analysis.to_debug_dict())
    build_docset_summary(analysis)
    build_docset_summary_metadata(analysis)
    assert analysis.to_debug_dict() == snapshot


@pytest.mark.skip(reason="Phase 5B reserved")
def test_docset_llm_polish_invalid_output_fallback():
    pass


@pytest.mark.skip(reason="Phase 5B reserved")
def test_docset_llm_polish_contradiction_fallback():
    pass


@pytest.mark.skip(reason="Phase 5B reserved")
def test_docset_llm_polish_empty_output_fallback():
    pass


def test_validate_llm_polish_output_reserved():
    with pytest.raises(NotImplementedError):
        validate_llm_polish_output(
            deterministic_summary="x",
            polished_summary="y",
            analysis=_analysis(),
        )


def test_build_docset_summary_metadata_format_version():
    meta = build_docset_summary_metadata(_analysis(docs_matched=0))
    assert meta["summary_format_version"] == SUMMARY_FORMAT_VERSION
    assert meta["summary_used_llm"] is False
    assert meta["golden_scope"] == "document_selection_only"
