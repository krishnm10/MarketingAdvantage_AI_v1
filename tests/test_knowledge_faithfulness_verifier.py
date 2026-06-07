"""Unit tests for Phase 6A KNOWLEDGE faithfulness verifier."""

from __future__ import annotations

from unittest.mock import patch

from app.retrieval.verification_evidence import (
    Passage,
    VerificationEvidenceBundle,
    build_verification_evidence_bundle,
)
from app.services.knowledge_faithfulness_verifier import (
    ClaimStatus,
    AnswerStatus,
    extract_claims,
    verify_knowledge_answer,
    KNOWLEDGE_VERIFIER_VERSION,
    MAX_TRACED_CLAIMS,
    MAX_SNIPPET_CHARS,
)
from app.api.v2.retrieve_chat_api import _GROUNDING_REFUSAL_PHRASE


def _bundle_from_context(
    context_str: str,
    chunk_metas: list[tuple[str, str | None]],
) -> VerificationEvidenceBundle:
    return build_verification_evidence_bundle(
        context_str=context_str,
        chunk_metas=chunk_metas,
        route="knowledge",
        raw_query="test query",
    )


def _sample_context(
    sources: list[tuple[str, str]],
) -> tuple[str, list[tuple[str, str | None]]]:
    """sources: [(chunk_id, body_text), ...]"""
    parts = []
    metas = []
    for i, (cid, body) in enumerate(sources, start=1):
        parts.append(f"[Source {i}] (score=0.900)\n{body}")
        metas.append((cid, f"file-{i}"))
    return "\n\n---\n\n".join(parts), metas


class TestEvidenceBundle:
    def test_alignment_from_sanitized_context(self):
        ctx, metas = _sample_context(
            [("c1", "Invoice INV-100 is overdue."), ("c2", "Total due $50.00.")]
        )
        bundle = _bundle_from_context(ctx, metas)
        assert bundle.alignment_valid is True
        assert len(bundle.passages) == 2
        assert bundle.passages[0].index == 1
        assert bundle.passages[0].chunk_id == "c1"
        assert bundle.passages[0].file_id == "file-1"
        assert "INV-100" in bundle.passages[0].text

    def test_mismatch_produces_alignment_errors(self):
        ctx, metas = _sample_context([("c1", "text only")])
        bundle = _bundle_from_context(ctx + "\n\n---\n\n[Source 2] (score=0.1)\nextra", metas)
        assert bundle.alignment_valid is False
        assert bundle.alignment_errors

    def test_bundle_ignores_top_chunks_text_if_context_differs(self):
        ctx, metas = _sample_context([("c1", "Sanitized visible text.")])
        bundle = _bundle_from_context(ctx, metas)
        assert bundle.passages[0].text == "Sanitized visible text."
        assert "RAW_SECRET" not in bundle.passages[0].text


class TestClaimExtraction:
    def test_numeric_and_identifier_gated(self):
        claims = extract_claims("Total is $110.00 for INV-200 [Source 2].")
        categories = {c.category for c in claims}
        assert "numeric" in categories
        assert "identifier" in categories
        assert all(c.gated for c in claims if c.category in ("numeric", "identifier"))

    def test_entity_presence_vs_relationship(self):
        presence = extract_claims("Invoice INV-100 is overdue [Source 1].")
        rel = extract_claims("Invoice INV-100 is from Vendor Acme [Source 1].")
        pres_cats = [c.category for c in presence]
        rel_cats = [c.category for c in rel]
        assert "entity_presence" in pres_cats
        assert "entity_relationship" in rel_cats
        rel_claim = next(c for c in rel if c.category == "entity_relationship")
        assert rel_claim.gated is False

    def test_bullet_line_gated(self):
        claims = extract_claims("- INV-100 — overdue [Source 1]\n")
        assert any(c.gated and c.category == "identifier" for c in claims)

    def test_table_row_shadow_only(self):
        claims = extract_claims("| INV-1 | $50.00 | overdue | [Source 1] |\n")
        table = [c for c in claims if c.category == "table_row"]
        assert table
        assert all(not c.gated for c in table)

    def test_extraction_deterministic(self):
        answer = "Invoice INV-100 is overdue [Source 1]. Total $50.00 [Source 1]."
        a = extract_claims(answer)
        b = extract_claims(answer)
        assert [(c.id, c.category, c.token) for c in a] == [
            (c.id, c.category, c.token) for c in b
        ]

    def test_percentage_numeric_extraction_25_and_12_5(self):
        claims = extract_claims("Tax rate is 25% and discount is 12.5% [Source 1].")
        pct_tokens = [
            c.token for c in claims if c.category == "numeric" and c.token and "%" in c.token
        ]
        assert "25%" in pct_tokens
        assert "12.5%" in pct_tokens
        assert all(c.gated for c in claims if c.token in ("25%", "12.5%"))


class TestEntityPresenceVerification:
    def test_entity_only_insufficient_when_property_asserted(self):
        ctx, metas = _sample_context([("c1", "Invoice INV-100 reference only.")])
        bundle = _bundle_from_context(ctx, metas)
        result = verify_knowledge_answer(
            answer="Invoice INV-100 is overdue [Source 1].",
            bundle=bundle,
            refusal_text=_GROUNDING_REFUSAL_PHRASE,
            gate_enabled=True,
        )
        ep = next(c for c in result.claim_results if c.category == "entity_presence")
        assert ep.status == ClaimStatus.FAIL
        assert ep.reason_code == "ENTITY_PROPERTY_UNSUPPORTED"
        assert result.status == AnswerStatus.FAIL

    def test_entity_and_property_pass_when_both_supported(self):
        ctx, metas = _sample_context([("c1", "Invoice INV-100 is overdue.")])
        bundle = _bundle_from_context(ctx, metas)
        result = verify_knowledge_answer(
            answer="Invoice INV-100 is overdue [Source 1].",
            bundle=bundle,
            refusal_text=_GROUNDING_REFUSAL_PHRASE,
            gate_enabled=False,
        )
        ep = next(c for c in result.claim_results if c.category == "entity_presence")
        assert ep.status == ClaimStatus.PASS
        assert ep.reason_code == "ENTITY_PROPERTY_PRESENT"
        presence_claim = next(
            c for c in extract_claims("Invoice INV-100 is overdue [Source 1].")
            if c.category == "entity_presence"
        )
        assert presence_claim.property_token == "overdue"


class TestPercentageVerification:
    def test_unsupported_percentage_triggers_gate_fail(self):
        ctx, metas = _sample_context([("c1", "Invoice INV-100 with 10% tax noted.")])
        bundle = _bundle_from_context(ctx, metas)
        result = verify_knowledge_answer(
            answer="Tax rate is 25% [Source 1].",
            bundle=bundle,
            refusal_text=_GROUNDING_REFUSAL_PHRASE,
            gate_enabled=True,
        )
        numeric_fails = [
            c
            for c in result.claim_results
            if c.category == "numeric" and c.status == ClaimStatus.FAIL
        ]
        assert numeric_fails
        assert result.status == AnswerStatus.FAIL
        assert result.answer_out == _GROUNDING_REFUSAL_PHRASE

    def test_supported_percentage_passes(self):
        ctx, metas = _sample_context([("c1", "Discount is 12.5% on invoice INV-100.")])
        bundle = _bundle_from_context(ctx, metas)
        result = verify_knowledge_answer(
            answer="Discount is 12.5% [Source 1].",
            bundle=bundle,
            refusal_text=_GROUNDING_REFUSAL_PHRASE,
            gate_enabled=True,
        )
        numeric_pass = [
            c
            for c in result.claim_results
            if c.category == "numeric" and c.status == ClaimStatus.PASS
        ]
        assert numeric_pass
        assert result.status == AnswerStatus.PASS
        assert result.answer_out.startswith("Discount")


class TestVerification:
    def test_pass_when_token_in_cited_passage(self):
        ctx, metas = _sample_context([("c1", "Invoice INV-100 is overdue with $50.00 fee.")])
        bundle = _bundle_from_context(ctx, metas)
        result = verify_knowledge_answer(
            answer="Invoice INV-100 is overdue [Source 1]. Fee is $50.00 [Source 1].",
            bundle=bundle,
            refusal_text=_GROUNDING_REFUSAL_PHRASE,
            gate_enabled=False,
        )
        gated = [c for c in result.claim_results if c.gated]
        assert all(c.status == ClaimStatus.PASS for c in gated)
        assert result.status == AnswerStatus.PASS

    def test_fail_unsupported_numeric_when_gate_on(self):
        ctx, metas = _sample_context([("c1", "Invoice INV-100 is overdue.")])
        bundle = _bundle_from_context(ctx, metas)
        result = verify_knowledge_answer(
            answer="Total due is $999.00 [Source 1].",
            bundle=bundle,
            refusal_text=_GROUNDING_REFUSAL_PHRASE,
            gate_enabled=True,
        )
        assert result.status == AnswerStatus.FAIL
        assert result.answer_out == _GROUNDING_REFUSAL_PHRASE

    def test_raw_vs_sanitized_mismatch_no_pass(self):
        ctx, metas = _sample_context([("c1", "Invoice summary only.")])
        bundle = _bundle_from_context(ctx, metas)
        result = verify_knowledge_answer(
            answer="Secret value $412.50 appears [Source 1].",
            bundle=bundle,
            refusal_text=_GROUNDING_REFUSAL_PHRASE,
            gate_enabled=True,
        )
        numeric = next(c for c in result.claim_results if c.category == "numeric")
        assert numeric.status in (ClaimStatus.FAIL, ClaimStatus.INDETERMINATE)
        assert result.status in (AnswerStatus.FAIL, AnswerStatus.INDETERMINATE)

    def test_alignment_failure_indeterminate_not_fail(self):
        bundle = VerificationEvidenceBundle(
            route="knowledge",
            raw_query="q",
            passages=(),
            alignment_valid=False,
            alignment_errors=("empty_context_str",),
        )
        result = verify_knowledge_answer(
            answer="Total $100.00 [Source 1].",
            bundle=bundle,
            refusal_text=_GROUNDING_REFUSAL_PHRASE,
            gate_enabled=True,
        )
        assert result.status == AnswerStatus.INDETERMINATE
        assert result.answer_out != _GROUNDING_REFUSAL_PHRASE

    def test_shadow_only_fail_does_not_gate(self):
        ctx, metas = _sample_context([("c1", "Data row.")])
        bundle = _bundle_from_context(ctx, metas)
        result = verify_knowledge_answer(
            answer="| INV-1 | $999.00 | [Source 1] |\n",
            bundle=bundle,
            refusal_text=_GROUNDING_REFUSAL_PHRASE,
            gate_enabled=True,
        )
        assert result.status == AnswerStatus.PASS
        assert result.answer_out.startswith("|")

    def test_indeterminate_policy_pass_through(self):
        bundle = VerificationEvidenceBundle(
            route="knowledge",
            raw_query="q",
            passages=(Passage(1, "c1", "f1", "text"),),
            alignment_valid=False,
            alignment_errors=("forced",),
        )
        answer = "Some text."
        result = verify_knowledge_answer(
            answer=answer,
            bundle=bundle,
            refusal_text=_GROUNDING_REFUSAL_PHRASE,
            gate_enabled=True,
            indeterminate_policy="pass_through",
        )
        assert result.answer_out == answer

    def test_indeterminate_policy_treat_as_fail(self):
        bundle = VerificationEvidenceBundle(
            route="knowledge",
            raw_query="q",
            passages=(Passage(1, "c1", "f1", "text"),),
            alignment_valid=False,
            alignment_errors=("forced",),
        )
        result = verify_knowledge_answer(
            answer="Total $50.00 [Source 1].",
            bundle=bundle,
            refusal_text=_GROUNDING_REFUSAL_PHRASE,
            gate_enabled=True,
            indeterminate_policy="treat_as_fail",
        )
        assert result.answer_out == _GROUNDING_REFUSAL_PHRASE
        assert result.indeterminate_policy_applied is True

    def test_one_fail_many_pass_rollup(self):
        ctx, metas = _sample_context(
            [("c1", "Invoice INV-100 is overdue."), ("c2", "Other doc.")]
        )
        bundle = _bundle_from_context(ctx, metas)
        result = verify_knowledge_answer(
            answer=(
                "Invoice INV-100 is overdue [Source 1]. "
                "Mystery amount $999.00 [Source 2]."
            ),
            bundle=bundle,
            refusal_text=_GROUNDING_REFUSAL_PHRASE,
            gate_enabled=True,
        )
        assert result.status == AnswerStatus.FAIL

    def test_debug_trace_caps(self):
        ctx, metas = _sample_context([("c1", "x" * 300 + " INV-1 $1.00")])
        bundle = _bundle_from_context(ctx, metas)
        many_claims = " ".join(
            f"${i}.00 [Source 1]." for i in range(MAX_TRACED_CLAIMS + 5)
        )
        result = verify_knowledge_answer(
            answer=many_claims,
            bundle=bundle,
            refusal_text=_GROUNDING_REFUSAL_PHRASE,
            gate_enabled=False,
        )
        dbg = result.to_debug_dict(include_claim_trace=True)
        assert dbg["version"] == KNOWLEDGE_VERIFIER_VERSION
        assert len(dbg["claims"]) <= MAX_TRACED_CLAIMS
        if result.trace_truncated:
            assert dbg.get("trace_truncated") is True
        if dbg["claims"] and dbg["claims"][0].get("evidence_links"):
            snippet = dbg["claims"][0]["evidence_links"][0]["snippet"]
            assert len(snippet) <= MAX_SNIPPET_CHARS + 1

    def test_verifier_internal_exception_returns_indeterminate(self):
        ctx, metas = _sample_context([("c1", "Invoice INV-100 is overdue.")])
        bundle = _bundle_from_context(ctx, metas)
        answer = "Invoice INV-100 is overdue [Source 1]."
        with patch(
            "app.services.knowledge_faithfulness_verifier.extract_claims",
            side_effect=RuntimeError("simulated verifier fault"),
        ):
            result = verify_knowledge_answer(
                answer=answer,
                bundle=bundle,
                refusal_text=_GROUNDING_REFUSAL_PHRASE,
                gate_enabled=True,
            )
        assert result.status == AnswerStatus.INDETERMINATE
        assert result.answer_out == answer
        assert result.error_detail is not None
        assert "simulated verifier fault" in result.error_detail
