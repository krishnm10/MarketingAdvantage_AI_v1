from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v2.retrieve_chat_api import ChatRetrieveResponse
from app.main import app
from app.retrieval.components import RuntimeComponents
from app.retrieval.types_retrieve import RankedResult
from app.retrieval.domain_adapters import invoice_adapter as _invoice_adapter  # ensure invoice adapter registration
from app.services.query_routing.types import (
    QueryRoute,
    knowledge_decision,
    RouteDecision,
    structured_decision,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def override_auth():
    from app.auth.deps import get_current_user

    app.dependency_overrides[get_current_user] = lambda: {
        "role": "admin",
        "sub": "test-admin",
    }
    yield
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture(autouse=True)
def override_db():
    from app.db.session_v2 import get_db

    mock_db = AsyncMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=None)

    async def _override_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture()
def minimal_runtime() -> RuntimeComponents:
    """Minimal RuntimeComponents for fields used in chat_retrieve."""
    return RuntimeComponents(
        runtime_mode="authoritative_config",
        config_fingerprint="test",
        client_id="pytest-tenant",
        embedder_type="huggingface",
        embedder_model="test-model",
        vectordb_type="chroma",
        collection="test_collection",
        llm_provider="none",
        llm_model="none",
        llm_api_key_env=None,
        llm_base_url=None,
        reranker_name="none",
        reranker_model="",
        search_mode="semantic",
        enable_hyde=False,
        enable_reranking=False,
        hybrid_alpha=0.7,
        similarity_threshold=0.0,
        top_k_retrieval=20,
        top_k_final=5,
        rag_min_score=0.25,
        chunking_strategy="unknown",
        enable_l1_task_classification=True,
        enable_docset_analysis=False,
        enable_docset_analysis_debug=False,
        enable_docset_invoice_adapter=False,
        enable_docset_result_shaping=False,
        docset_max_docs_returned=0,
        docset_max_chunks_per_doc_view=0,
        docset_max_docs_debug=20,
        docset_max_chunks_per_doc_debug=3,
        enable_docset_shadow_mode=False,
        enable_docset_shadow_debug=False,
        enable_docset_golden_eval=False,
        enable_docset_golden_debug=False,
    )


def _sample_ranked(file_id: str, chunk_id: str, text: str) -> RankedResult:
    return RankedResult(
        chunk_id=chunk_id,
        text=text,
        score=0.9,
        explanation={},
        trust_decision=None,
        file_id=file_id,
    )


async def _post_chat(client: AsyncClient, message: str) -> "httpx.Response":
    return await client.post(
        "/api/v2/retrieve/chat",
        json={
            "messages": [{"role": "user", "content": message}],
            "client_id": "pytest-tenant",
            "top_k": 5,
            "enable_hyde": False,
        },
    )


@contextmanager
def _tenant_patches(minimal_runtime: RuntimeComponents, route: RouteDecision):
    from app.utils.tenant_validator import validate_tenant_id_strict, get_storage_uuid_str
    from app.retrieval.components import resolve_config_or_fail, resolve_runtime_components
    from app.api.v2 import retrieve_chat_api as chat_mod

    mock_orch = MagicMock()
    mock_orch.route = AsyncMock(return_value=route)

    with (
        patch.object(
            chat_mod,
            "get_orchestrator",
            return_value=mock_orch,
        ),
        patch(
            "app.utils.tenant_validator.validate_tenant_id_strict",
            return_value=MagicMock(
                tenant_id="pytest-tenant",
                raw_input="pytest-tenant",
                source="body",
                endpoint="chat_retrieve",
            ),
        ),
        patch(
            "app.utils.tenant_validator.get_storage_uuid_str",
            return_value="00000000-0000-0000-0000-000000000001",
        ),
        patch(
            "app.retrieval.components.resolve_config_or_fail",
            return_value=(MagicMock(client_id="pytest-tenant"), "authoritative_config"),
        ),
        patch(
            "app.retrieval.components.resolve_runtime_components",
            return_value=minimal_runtime,
        ),
    ):
        yield


@pytest.mark.anyio
async def test_flag_off_no_docset_analysis(minimal_runtime):
    """Flag OFF: no docset_analysis, schema + behavior unchanged."""
    runtime_off = replace(minimal_runtime, enable_docset_analysis_debug=False)
    route = knowledge_decision(top_k=5)

    async def _fake_retrieve(*args, **kwargs):
        return (
            [
                _sample_ranked(
                    "inv-1",
                    "c1",
                    "This invoice is overdue with a late payment penalty.",
                )
            ],
            [],
        )

    from app.api.v2 import retrieve_chat_api as chat_mod

    with (
        _tenant_patches(runtime_off, route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(MagicMock(generate=MagicMock(return_value=MagicMock(text="answer"))), "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(
                client,
                "show me overdue invoices with late payment penalties",
            )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    ChatRetrieveResponse.model_validate(body)
    debug = body.get("debug_info") or {}

    assert "docset_analysis" not in debug
    assert debug.get("query_route") == QueryRoute.KNOWLEDGE.value


@pytest.mark.anyio
async def test_flag_on_knowledge_invoice_adds_docset_analysis(minimal_runtime):
    """Flag ON + KNOWLEDGE + invoice query: debug_info contains docset_analysis."""
    runtime_on = replace(
        minimal_runtime,
        enable_docset_analysis=True,
        enable_docset_analysis_debug=True,
        enable_docset_invoice_adapter=True,
    )
    route = knowledge_decision(top_k=5)

    async def _fake_retrieve(*args, **kwargs):
        return (
            [
                _sample_ranked(
                    "inv-2",
                    "c1",
                    "Invoice INV-2 is overdue and has a late fee. Tax 12% is applied.",
                )
            ],
            [],
        )

    from app.api.v2 import retrieve_chat_api as chat_mod
    from app.retrieval.types_retrieve import DomainType
    from app.retrieval.task_classifier import TaskPlan as _TaskPlan, TaskType

    def _invoice_task_plan(*args, **kwargs):
        return _TaskPlan(
            route=QueryRoute.KNOWLEDGE,
            task_type=TaskType.DOCSET_FILTER,
            domain=DomainType.INVOICE,
            predicates=None,
            confidence=0.8,
        )

    with (
        _tenant_patches(runtime_on, route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(MagicMock(generate=MagicMock(return_value=MagicMock(text="answer"))), "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
        patch("app.api.v2.retrieve_chat_api.classify_task", side_effect=_invoice_task_plan),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(
                client,
                "show me invoices with late payment penalties",
            )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    ChatRetrieveResponse.model_validate(body)

    debug = body.get("debug_info") or {}
    assert debug.get("query_route") == QueryRoute.KNOWLEDGE.value

    analysis = debug.get("docset_analysis")
    assert isinstance(analysis, dict)
    assert analysis.get("domain") == "invoice"
    # At least one match should be present for the invoice doc.
    assert (analysis.get("docs_matched") or 0) >= 1


@pytest.mark.anyio
async def test_complex_invoice_docset_query_and_structured_control(minimal_runtime):
    """Complex KNOWLEDGE docset query analysis and STRUCTURED control behavior."""
    runtime_on = replace(
        minimal_runtime,
        enable_docset_analysis=True,
        enable_docset_analysis_debug=True,
        enable_docset_invoice_adapter=True,
    )

    # KNOWLEDGE route with multi-document scenario.
    knowledge_route = knowledge_decision(top_k=5)

    async def _fake_retrieve_complex(*args, **kwargs):
        return (
            [
                _sample_ranked(  # matches all conditions
                    "inv-A",
                    "c1",
                    "Invoice A is overdue and has a late payment penalty. Tax 15% is applied.",
                ),
                _sample_ranked(  # matches only overdue
                    "inv-B",
                    "c2",
                    "Invoice B is overdue but has no penalty.",
                ),
                _sample_ranked(  # matches none
                    "inv-C",
                    "c3",
                    "Invoice C is fully paid.",
                ),
            ],
            [],
        )

    from app.api.v2 import retrieve_chat_api as chat_mod
    from app.retrieval.types_retrieve import DomainType
    from app.retrieval.task_classifier import TaskPlan as _TaskPlan, TaskType

    def _invoice_task_plan(*args, **kwargs):
        return _TaskPlan(
            route=QueryRoute.KNOWLEDGE,
            task_type=TaskType.DOCSET_FILTER,
            domain=DomainType.INVOICE,
            predicates=None,
            confidence=0.8,
        )

    # First run: analysis disabled (baseline).
    runtime_off = replace(
        minimal_runtime,
        enable_docset_analysis_debug=False,
        enable_docset_invoice_adapter=True,
    )
    with (
        _tenant_patches(runtime_off, knowledge_route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(MagicMock(generate=MagicMock(return_value=MagicMock(text="answer"))), "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve_complex,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
        patch("app.api.v2.retrieve_chat_api.classify_task", side_effect=_invoice_task_plan),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp_off = await _post_chat(
                client,
                "Which invoices are overdue, include a late payment penalty, and have tax above 10%?",
            )

    assert resp_off.status_code == 200, resp_off.text
    body_off = resp_off.json()
    ChatRetrieveResponse.model_validate(body_off)
    debug_off = body_off.get("debug_info") or {}
    assert "docset_analysis" not in debug_off

    # Second run: analysis enabled; retrieval mocked identically.
    with (
        _tenant_patches(runtime_on, knowledge_route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(MagicMock(generate=MagicMock(return_value=MagicMock(text="answer"))), "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve_complex,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp_on = await _post_chat(
                client,
                "Which invoices are overdue, include a late payment penalty, and have tax above 10%?",
            )

    assert resp_on.status_code == 200, resp_on.text
    body_on = resp_on.json()
    ChatRetrieveResponse.model_validate(body_on)

    # Schema + core behavior unchanged.
    assert body_on["results"] == body_off["results"]
    assert body_on.get("answer") == body_off.get("answer")

    debug_on = body_on.get("debug_info") or {}
    analysis = debug_on.get("docset_analysis")
    assert isinstance(analysis, dict)
    assert analysis.get("domain") == "invoice"
    assert analysis.get("docs_analyzed") == 3
    # docs_matched is disjunctive: any positive signal counts (inv-A + inv-B), not full query AND.
    assert analysis.get("docs_matched") == 2

    # Check per-document flags.
    matches = {m["file_id"]: m for m in analysis.get("matches", [])}
    assert matches["inv-A"]["overdue"] is True
    assert matches["inv-A"]["has_late_fee"] is True
    assert matches["inv-A"]["tax_gt_threshold"] is True

    assert matches["inv-B"]["overdue"] is True
    assert matches["inv-B"]["has_late_fee"] is False
    assert matches["inv-B"]["tax_gt_threshold"] is False
    assert "inv-C" not in matches

    # STRUCTURED route control: must not trigger analysis or shadow.
    structured_route = structured_decision(top_k=1)
    with (
        _tenant_patches(runtime_on, structured_route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(MagicMock(generate=MagicMock(return_value=MagicMock(text="answer"))), "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve_complex,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp_structured = await _post_chat(
                client,
                "What is the due date on invoice INV-1234?",
            )

    assert resp_structured.status_code == 200, resp_structured.text
    body_structured = resp_structured.json()
    debug_structured = body_structured.get("debug_info") or {}
    assert debug_structured.get("query_route") == QueryRoute.STRUCTURED.value
    assert "docset_analysis" not in debug_structured
    assert "docset_shadow" not in debug_structured


@pytest.mark.anyio
async def test_shadow_mode_does_not_change_results_or_answer(minimal_runtime):
    """
    Phase 3A: Shadow Mode must be observational only.
    Enabling shadow must not change results, answer, or schema.
    """
    # Baseline: no docset analysis, no shadow.
    runtime_baseline = replace(
        minimal_runtime,
        enable_docset_analysis_debug=False,
        enable_docset_invoice_adapter=True,
        enable_docset_shadow_mode=False,
        enable_docset_shadow_debug=False,
    )
    # Shadow enabled: still debug-only, observational.
    runtime_shadow = replace(
        minimal_runtime,
        enable_docset_analysis=True,
        enable_docset_analysis_debug=False,
        enable_docset_invoice_adapter=True,
        enable_docset_shadow_mode=True,
        enable_docset_shadow_debug=True,
        enable_docset_golden_eval=False,
        enable_docset_golden_debug=False,
    )
    route = knowledge_decision(top_k=5)

    async def _fake_retrieve(*args, **kwargs):
        return (
            [
                _sample_ranked(
                    "inv-A",
                    "c1",
                    "Invoice A is overdue and has a late payment penalty. Tax 15% is applied.",
                ),
                _sample_ranked(
                    "inv-B",
                    "c2",
                    "Invoice B is overdue but has no penalty.",
                ),
            ],
            [],
        )

    from app.api.v2 import retrieve_chat_api as chat_mod
    from app.retrieval.types_retrieve import DomainType
    from app.retrieval.task_classifier import TaskPlan as _TaskPlan, TaskType

    def _invoice_task_plan(*args, **kwargs):
        return _TaskPlan(
            route=QueryRoute.KNOWLEDGE,
            task_type=TaskType.DOCSET_FILTER,
            domain=DomainType.INVOICE,
            predicates=None,
            confidence=0.8,
        )

    # Baseline run.
    with (
        _tenant_patches(runtime_baseline, route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(MagicMock(generate=MagicMock(return_value=MagicMock(text="answer"))), "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
        patch("app.api.v2.retrieve_chat_api.classify_task", side_effect=_invoice_task_plan),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp_off = await _post_chat(
                client,
                "Which invoices are overdue, include a late payment penalty, and have tax above 10%?",
            )

    assert resp_off.status_code == 200, resp_off.text
    body_off = resp_off.json()
    ChatRetrieveResponse.model_validate(body_off)

    # Shadow-enabled run with identical retrieval.
    with (
        _tenant_patches(runtime_shadow, route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(MagicMock(generate=MagicMock(return_value=MagicMock(text="answer"))), "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
        patch("app.api.v2.retrieve_chat_api.classify_task", side_effect=_invoice_task_plan),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp_on = await _post_chat(
                client,
                "Which invoices are overdue, include a late payment penalty, and have tax above 10%?",
            )

    assert resp_on.status_code == 200, resp_on.text
    body_on = resp_on.json()
    ChatRetrieveResponse.model_validate(body_on)

    # Live behavior: results and answer must be identical.
    assert body_on["results"] == body_off["results"]
    assert body_on.get("answer") == body_off.get("answer")

    debug_off = body_off.get("debug_info") or {}
    debug_on = body_on.get("debug_info") or {}

    # Flag off: no shadow payload.
    assert "docset_shadow" not in debug_off
    # Flag on + KNOWLEDGE + invoice/docset query: shadow metadata appears.
    shadow = debug_on.get("docset_shadow")
    assert isinstance(shadow, dict)
    assert shadow.get("shadow_mode") is True
    assert shadow.get("domain") == "invoice"
    # At least one matched invoice should be surfaced.
    matched_ids = set(shadow.get("matched_file_ids") or [])
    assert "inv-A" in matched_ids or "inv-B" in matched_ids


@pytest.mark.anyio
async def test_golden_eval_flag_on_emits_metadata(minimal_runtime, monkeypatch):
    """
    Phase 3B: Golden evaluation attaches adapter-agnostic metrics over shadow matches.
    """
    from app.retrieval import golden_evaluation as ge

    # Configure runtime to enable both shadow and golden evaluation + debug.
    runtime_golden = replace(
        minimal_runtime,
        enable_docset_analysis=True,
        enable_docset_analysis_debug=False,
        enable_docset_invoice_adapter=True,
        enable_docset_shadow_mode=True,
        enable_docset_shadow_debug=True,
        enable_docset_golden_eval=True,
        enable_docset_golden_debug=True,
        docset_golden_set_ref="ignored",
    )
    route = knowledge_decision(top_k=5)

    async def _fake_retrieve(*args, **kwargs):
        return (
            [
                _sample_ranked(
                    "inv-A",
                    "c1",
                    "Invoice A is overdue and has a late payment penalty. Tax 15% is applied.",
                ),
                _sample_ranked(
                    "inv-B",
                    "c2",
                    "Invoice B is overdue but has no penalty.",
                ),
            ],
            [],
        )

    # For this test, define a synthetic golden expectation purely in terms of file_ids.
    def _fake_lookup_expected_file_ids(
        *, set_ref: str | None, client_id: str, raw_query: str, domain: str | None
    ):
        # Expect only inv-A as the canonical match.
        return {"inv-A"}

    monkeypatch.setattr(ge, "_lookup_expected_file_ids", _fake_lookup_expected_file_ids)

    from app.api.v2 import retrieve_chat_api as chat_mod
    from app.retrieval.types_retrieve import DomainType
    from app.retrieval.task_classifier import TaskPlan as _TaskPlan, TaskType

    def _invoice_task_plan(*args, **kwargs):
        return _TaskPlan(
            route=QueryRoute.KNOWLEDGE,
            task_type=TaskType.DOCSET_FILTER,
            domain=DomainType.INVOICE,
            predicates=None,
            confidence=0.8,
        )

    with (
        _tenant_patches(runtime_golden, route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(MagicMock(generate=MagicMock(return_value=MagicMock(text="answer"))), "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
        patch("app.api.v2.retrieve_chat_api.classify_task", side_effect=_invoice_task_plan),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(
                client,
                "Which invoices are overdue, include a late payment penalty, and have tax above 10%?",
            )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    ChatRetrieveResponse.model_validate(body)

    debug = body.get("debug_info") or {}
    # Shadow must have run.
    shadow = debug.get("docset_shadow")
    assert isinstance(shadow, dict)
    assert shadow.get("domain") == "invoice"

    # Golden evaluation metadata must be present and based only on file-id sets.
    golden = debug.get("docset_golden_eval")
    assert isinstance(golden, dict)
    assert golden.get("golden_eval") is True
    assert golden.get("domain") == "invoice"

    expected_ids = set(golden.get("expected_file_ids") or [])
    actual_ids = set(golden.get("actual_file_ids") or [])
    assert expected_ids == {"inv-A"}
    # Order-independence: same set regardless of ranking.
    assert actual_ids == {"inv-A", "inv-B"}

    # Metrics: precision/recall over sets.
    # inv-A expected, inv-A + inv-B returned → precision=0.5, recall=1.0
    assert golden.get("precision") == pytest.approx(0.5)
    assert golden.get("recall") == pytest.approx(1.0)


@pytest.mark.anyio
async def test_golden_eval_no_golden_found_is_not_degraded(minimal_runtime, monkeypatch):
    """
    Golden evaluation enabled + no matching golden must not degrade or change behavior.
    """
    from app.retrieval import golden_evaluation as ge

    # Baseline: shadow only.
    runtime_baseline = replace(
        minimal_runtime,
        enable_docset_analysis=True,
        enable_docset_analysis_debug=False,
        enable_docset_invoice_adapter=True,
        enable_docset_shadow_mode=True,
        enable_docset_shadow_debug=True,
        enable_docset_golden_eval=False,
        enable_docset_golden_debug=False,
        docset_golden_set_ref=None,
    )

    # Golden enabled but lookup returns no expectations.
    runtime_golden = replace(
        minimal_runtime,
        enable_docset_analysis=True,
        enable_docset_analysis_debug=False,
        enable_docset_invoice_adapter=True,
        enable_docset_shadow_mode=True,
        enable_docset_shadow_debug=True,
        enable_docset_golden_eval=True,
        enable_docset_golden_debug=True,
        docset_golden_set_ref="ignored",
    )
    route = knowledge_decision(top_k=5)

    async def _fake_retrieve(*args, **kwargs):
        return (
            [
                _sample_ranked(
                    "inv-A",
                    "c1",
                    "Invoice A is overdue and has a late payment penalty. Tax 15% is applied.",
                ),
                _sample_ranked(
                    "inv-B",
                    "c2",
                    "Invoice B is overdue but has no penalty.",
                ),
            ],
            [],
        )

    # Force golden lookup to report "no golden available".
    def _no_golden(*, set_ref: str | None, client_id: str, raw_query: str, domain: str | None):
        return None

    monkeypatch.setattr(ge, "_lookup_expected_file_ids", _no_golden)

    from app.api.v2 import retrieve_chat_api as chat_mod
    from app.retrieval.types_retrieve import DomainType
    from app.retrieval.task_classifier import TaskPlan as _TaskPlan, TaskType

    def _invoice_task_plan(*args, **kwargs):
        return _TaskPlan(
            route=QueryRoute.KNOWLEDGE,
            task_type=TaskType.DOCSET_FILTER,
            domain=DomainType.INVOICE,
            predicates=None,
            confidence=0.8,
        )

    # Baseline run (no golden).
    with (
        _tenant_patches(runtime_baseline, route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(MagicMock(generate=MagicMock(return_value=MagicMock(text="answer"))), "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
        patch("app.api.v2.retrieve_chat_api.classify_task", side_effect=_invoice_task_plan),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp_baseline = await _post_chat(
                client,
                "Which invoices are overdue, include a late payment penalty, and have tax above 10%?",
            )

    assert resp_baseline.status_code == 200, resp_baseline.text
    body_baseline = resp_baseline.json()
    ChatRetrieveResponse.model_validate(body_baseline)

    # Golden-enabled run with identical retrieval and no golden available.
    with (
        _tenant_patches(runtime_golden, route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(MagicMock(generate=MagicMock(return_value=MagicMock(text="answer"))), "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
        patch("app.api.v2.retrieve_chat_api.classify_task", side_effect=_invoice_task_plan),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp_golden = await _post_chat(
                client,
                "Which invoices are overdue, include a late payment penalty, and have tax above 10%?",
            )

    assert resp_golden.status_code == 200, resp_golden.text
    body_golden = resp_golden.json()
    ChatRetrieveResponse.model_validate(body_golden)

    # Live behavior unchanged.
    assert body_golden["results"] == body_baseline["results"]
    assert body_golden.get("answer") == body_baseline.get("answer")

    debug_golden = body_golden.get("debug_info") or {}
    golden_meta = debug_golden.get("docset_golden_eval")
    assert isinstance(golden_meta, dict)
    # Missing golden data must not be treated as degraded failure.
    assert golden_meta.get("degraded") is False


@pytest.mark.anyio
async def test_shadow_mode_unsupported_domain_is_safe(minimal_runtime):
    """
    Unsupported domain / missing adapter: shadow mode must not fail the request.
    """
    runtime_shadow = replace(
        minimal_runtime,
        enable_docset_analysis_debug=False,
        enable_docset_invoice_adapter=False,
        enable_docset_shadow_mode=True,
        enable_docset_shadow_debug=True,
        enable_docset_golden_eval=True,
        enable_docset_golden_debug=True,
        docset_golden_set_ref="ignored",
    )

    # Force a GENERIC domain task plan via L1 classifier patch.
    from app.api.v2 import retrieve_chat_api as chat_mod
    from app.retrieval.types_retrieve import DomainType
    from app.retrieval.task_classifier import TaskPlan as _TaskPlan

    def _generic_task_plan(*args, **kwargs):
        return _TaskPlan(
            route=QueryRoute.KNOWLEDGE,
            task_type=TaskType.DOCSET_FILTER,
            domain=DomainType.GENERIC,
            predicates=None,
            confidence=0.8,
        )

    route = knowledge_decision(top_k=5)

    async def _fake_retrieve(*args, **kwargs):
        return ([_sample_ranked("doc-1", "c1", "Some generic document.")], [])

    with (
        _tenant_patches(runtime_shadow, route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(MagicMock(generate=MagicMock(return_value=MagicMock(text="answer"))), "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
        patch("app.api.v2.retrieve_chat_api.classify_task", side_effect=_generic_task_plan),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(
                client,
                "Show me generic documents.",
            )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    ChatRetrieveResponse.model_validate(body)
    debug = body.get("debug_info") or {}
    # Shadow metadata may or may not appear, but request must succeed and schema must validate.
    assert debug.get("query_route") == QueryRoute.KNOWLEDGE.value


@pytest.mark.anyio
async def test_shadow_mode_failure_is_swallowed(minimal_runtime):
    """
    Degraded structured analysis must not break the chat response.
    """
    from app.retrieval.document_set_analysis import AnalysisResult

    runtime_shadow = replace(
        minimal_runtime,
        enable_docset_analysis=True,
        enable_docset_analysis_debug=False,
        enable_docset_invoice_adapter=True,
        enable_docset_shadow_mode=True,
        enable_docset_shadow_debug=True,
        enable_docset_golden_eval=True,
        enable_docset_golden_debug=True,
        docset_golden_set_ref="ignored",
    )
    route = knowledge_decision(top_k=5)

    async def _fake_retrieve(*args, **kwargs):
        return (
            [
                _sample_ranked(
                    "inv-err",
                    "c1",
                    "Invoice with text that will cause shadow failure.",
                )
            ],
            [],
        )

    def _degraded_structured(*args, **kwargs):
        return (
            AnalysisResult(
                domain="invoice",
                docs_analyzed=1,
                docs_matched=0,
                matches=[],
                degraded=True,
                degradation_reason="synthetic shadow failure",
            ),
            [],
        )

    from app.api.v2 import retrieve_chat_api as chat_mod

    with (
        _tenant_patches(runtime_shadow, route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(MagicMock(generate=MagicMock(return_value=MagicMock(text="answer"))), "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
        patch(
            "app.api.v2.retrieve_chat_api.run_docset_analysis_structured",
            side_effect=_degraded_structured,
        ),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(
                client,
                "show me overdue invoices with late payment penalties",
            )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    ChatRetrieveResponse.model_validate(body)
    # We only assert that the request succeeds and schema validates; the failure
    # is fully swallowed by shadow_analysis.


@pytest.mark.anyio
async def test_result_shaping_truncates_docs_and_chunks(minimal_runtime):
    """
    Phase 4: Active result shaping deterministically truncates eligible documents
    and chunks using existing retrieval order and DocumentMatch output.
    """
    # Baseline runtime: analysis + adapter enabled, shaping OFF.
    runtime_baseline = replace(
        minimal_runtime,
        enable_docset_analysis=True,
        enable_docset_analysis_debug=False,
        enable_docset_invoice_adapter=True,
        enable_docset_result_shaping=False,
        docset_max_docs_returned=1,
        docset_max_chunks_per_doc_view=1,
    )
    # Shaping runtime: same, but shaping flag ON.
    runtime_shaping = replace(
        minimal_runtime,
        enable_docset_analysis=True,
        enable_docset_analysis_debug=False,
        enable_docset_invoice_adapter=True,
        enable_docset_result_shaping=True,
        docset_max_docs_returned=1,
        docset_max_chunks_per_doc_view=1,
    )
    route = knowledge_decision(top_k=5)

    async def _fake_retrieve(*args, **kwargs):
        # Three invoices; A and B carry positive signals, C is a non-match.
        return (
            [
                _sample_ranked(
                    "inv-A",
                    "c1",
                    "Invoice A is overdue and has a late fee. Tax 12% is applied.",
                ),
                _sample_ranked(
                    "inv-B",
                    "c2",
                    "Invoice B is overdue and has a late fee. Tax 15% is applied.",
                ),
                _sample_ranked(
                    "inv-C",
                    "c3",
                    "Invoice C is fully paid.",
                ),
            ],
            [],
        )

    from app.api.v2 import retrieve_chat_api as chat_mod
    from app.retrieval.types_retrieve import DomainType
    from app.retrieval.task_classifier import TaskPlan as _TaskPlan, TaskType

    def _invoice_task_plan(*args, **kwargs):
        return _TaskPlan(
            route=QueryRoute.KNOWLEDGE,
            task_type=TaskType.DOCSET_FILTER,
            domain=DomainType.INVOICE,
            predicates=None,
            confidence=0.9,
        )

    # Baseline: shaping disabled, all retrieved chunks surface.
    with (
        _tenant_patches(runtime_baseline, route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(MagicMock(generate=MagicMock(return_value=MagicMock(text="answer"))), "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
        patch("app.api.v2.retrieve_chat_api.classify_task", side_effect=_invoice_task_plan),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp_baseline = await _post_chat(
                client,
                "Which invoices are overdue, include a late payment penalty, and have tax above 10%?",
            )

    assert resp_baseline.status_code == 200, resp_baseline.text
    body_baseline = resp_baseline.json()
    ChatRetrieveResponse.model_validate(body_baseline)
    results_baseline = body_baseline["results"]
    # All three retrieved chunks should be visible without shaping.
    assert [r["chunk_id"] for r in results_baseline] == ["c1", "c2", "c3"]

    # Shaping enabled: only the first eligible document (inv-A) should remain,
    # with at most one chunk due to docset_max_chunks_per_doc_view=1.
    with (
        _tenant_patches(runtime_shaping, route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(MagicMock(generate=MagicMock(return_value=MagicMock(text="answer"))), "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
        patch("app.api.v2.retrieve_chat_api.classify_task", side_effect=_invoice_task_plan),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp_shaped = await _post_chat(
                client,
                "Which invoices are overdue, include a late payment penalty, and have tax above 10%?",
            )

    assert resp_shaped.status_code == 200, resp_shaped.text
    body_shaped = resp_shaped.json()
    ChatRetrieveResponse.model_validate(body_shaped)
    results_shaped = body_shaped["results"]

    # Only one chunk from the first eligible document remains.
    assert [r["chunk_id"] for r in results_shaped] == ["c1"]
    assert all(r["chunk_id"] != "c2" for r in results_shaped)
    assert all(r["chunk_id"] != "c3" for r in results_shaped)

    debug_shaped = body_shaped.get("debug_info") or {}
    assert debug_shaped.get("query_route") == QueryRoute.KNOWLEDGE.value
    # Shaping should be enabled at the runtime level; per-request application
    # details are captured in tracing, but the flag state is visible here.
    assert debug_shaped.get("ranked_results_count") == len(results_shaped)


@pytest.mark.anyio
async def test_result_shaping_flag_on_without_analysis_fails_open(minimal_runtime):
    """
    Phase 4: When analysis is disabled, enabling result shaping must fail open
    and preserve Phase 3 behavior (no change to results/answer).
    """
    runtime_baseline = replace(
        minimal_runtime,
        enable_docset_analysis=False,
        enable_docset_analysis_debug=False,
        enable_docset_invoice_adapter=True,
        enable_docset_result_shaping=False,
    )
    runtime_shaping = replace(
        minimal_runtime,
        enable_docset_analysis=False,
        enable_docset_analysis_debug=False,
        enable_docset_invoice_adapter=True,
        enable_docset_result_shaping=True,
    )
    route = knowledge_decision(top_k=5)

    async def _fake_retrieve(*args, **kwargs):
        return (
            [
                _sample_ranked(
                    "inv-X",
                    "c1",
                    "Invoice X is overdue and has a late fee.",
                ),
                _sample_ranked(
                    "inv-Y",
                    "c2",
                    "Invoice Y is overdue and has a late fee.",
                ),
            ],
            [],
        )

    from app.api.v2 import retrieve_chat_api as chat_mod
    from app.retrieval.types_retrieve import DomainType
    from app.retrieval.task_classifier import TaskPlan as _TaskPlan, TaskType

    def _invoice_task_plan(*args, **kwargs):
        return _TaskPlan(
            route=QueryRoute.KNOWLEDGE,
            task_type=TaskType.DOCSET_FILTER,
            domain=DomainType.INVOICE,
            predicates=None,
            confidence=0.9,
        )

    # Baseline run (no shaping).
    with (
        _tenant_patches(runtime_baseline, route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(MagicMock(generate=MagicMock(return_value=MagicMock(text="answer"))), "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
        patch("app.api.v2.retrieve_chat_api.classify_task", side_effect=_invoice_task_plan),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp_baseline = await _post_chat(
                client,
                "Which invoices are overdue and include a late payment penalty?",
            )

    assert resp_baseline.status_code == 200, resp_baseline.text
    body_baseline = resp_baseline.json()
    ChatRetrieveResponse.model_validate(body_baseline)

    # Shaping-enabled run but with analysis disabled: behavior must be identical.
    with (
        _tenant_patches(runtime_shaping, route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(MagicMock(generate=MagicMock(return_value=MagicMock(text="answer"))), "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
        patch("app.api.v2.retrieve_chat_api.classify_task", side_effect=_invoice_task_plan),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp_shaping = await _post_chat(
                client,
                "Which invoices are overdue and include a late payment penalty?",
            )

    assert resp_shaping.status_code == 200, resp_shaping.text
    body_shaping = resp_shaping.json()
    ChatRetrieveResponse.model_validate(body_shaping)

    assert body_shaping["results"] == body_baseline["results"]
    assert body_shaping.get("answer") == body_baseline.get("answer")


@pytest.mark.anyio
async def test_analysis_debug_flag_decoupled_from_execution(minimal_runtime):
    """Analysis can run without emitting debug_info['docset_analysis']."""
    runtime = replace(
        minimal_runtime,
        enable_docset_analysis=True,
        enable_docset_analysis_debug=False,
        enable_docset_invoice_adapter=True,
    )
    route = knowledge_decision(top_k=5)

    async def _fake_retrieve(*args, **kwargs):
        return (
            [
                _sample_ranked(
                    "inv-2",
                    "c1",
                    "Invoice INV-2 is overdue and has a late fee. Tax 12% is applied.",
                )
            ],
            [],
        )

    from app.api.v2 import retrieve_chat_api as chat_mod
    from app.retrieval.types_retrieve import DomainType
    from app.retrieval.task_classifier import TaskPlan as _TaskPlan, TaskType

    def _invoice_task_plan(*args, **kwargs):
        return _TaskPlan(
            route=QueryRoute.KNOWLEDGE,
            task_type=TaskType.DOCSET_FILTER,
            domain=DomainType.INVOICE,
            predicates=None,
            confidence=0.8,
        )

    with (
        _tenant_patches(runtime, route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(MagicMock(generate=MagicMock(return_value=MagicMock(text="answer"))), "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
        patch("app.api.v2.retrieve_chat_api.classify_task", side_effect=_invoice_task_plan),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(client, "show me overdue invoices with late fees")

    assert resp.status_code == 200, resp.text
    debug = resp.json().get("debug_info") or {}
    assert "docset_analysis" not in debug


@pytest.mark.anyio
async def test_analysis_runs_once_per_request(minimal_runtime):
    """Structured analysis executes at most once per request."""
    runtime = replace(
        minimal_runtime,
        enable_docset_analysis=True,
        enable_docset_analysis_debug=True,
        enable_docset_invoice_adapter=True,
        enable_docset_shadow_mode=True,
        enable_docset_shadow_debug=True,
        enable_docset_golden_eval=True,
        enable_docset_golden_debug=True,
        enable_docset_result_shaping=True,
        docset_max_docs_returned=1,
        docset_golden_set_ref="ignored",
    )
    route = knowledge_decision(top_k=5)
    call_count = {"n": 0}
    real_structured = __import__(
        "app.retrieval.document_set_analysis", fromlist=["run_docset_analysis_structured"]
    ).run_docset_analysis_structured

    def _counting_structured(*args, **kwargs):
        call_count["n"] += 1
        return real_structured(*args, **kwargs)

    async def _fake_retrieve(*args, **kwargs):
        return (
            [
                _sample_ranked(
                    "inv-A",
                    "c1",
                    "Invoice A is overdue and has a late fee. Tax 12% is applied.",
                ),
                _sample_ranked(
                    "inv-B",
                    "c2",
                    "Invoice B is overdue and has a late fee. Tax 15% is applied.",
                ),
            ],
            [],
        )

    from app.api.v2 import retrieve_chat_api as chat_mod
    from app.retrieval.types_retrieve import DomainType
    from app.retrieval.task_classifier import TaskPlan as _TaskPlan, TaskType

    def _invoice_task_plan(*args, **kwargs):
        return _TaskPlan(
            route=QueryRoute.KNOWLEDGE,
            task_type=TaskType.DOCSET_FILTER,
            domain=DomainType.INVOICE,
            predicates=None,
            confidence=0.9,
        )

    with (
        _tenant_patches(runtime, route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(MagicMock(generate=MagicMock(return_value=MagicMock(text="answer"))), "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
        patch("app.api.v2.retrieve_chat_api.classify_task", side_effect=_invoice_task_plan),
        patch(
            "app.api.v2.retrieve_chat_api.run_docset_analysis_structured",
            side_effect=_counting_structured,
        ),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(
                client,
                "Which invoices are overdue and include a late payment penalty?",
            )

    assert resp.status_code == 200, resp.text
    assert call_count["n"] == 1


@pytest.mark.anyio
async def test_shared_analysis_instance_and_immutability(minimal_runtime):
    """
    Shaping, shadow, and golden consume the same AnalysisResult instance and
    must not mutate shared analysis state.
    """
    runtime = replace(
        minimal_runtime,
        enable_docset_analysis=True,
        enable_docset_analysis_debug=True,
        enable_docset_invoice_adapter=True,
        enable_docset_shadow_mode=True,
        enable_docset_shadow_debug=True,
        enable_docset_golden_eval=True,
        enable_docset_golden_debug=True,
        enable_docset_result_shaping=True,
        docset_max_docs_returned=1,
        docset_golden_set_ref="ignored",
    )
    route = knowledge_decision(top_k=5)

    captured: dict[str, object] = {}
    analysis_before: dict[str, object] = {}
    matches_before: list[dict[str, object]] = []

    real_shape = __import__(
        "app.retrieval.docset_result_shaper", fromlist=["shape_docset_results"]
    ).shape_docset_results
    real_shadow = __import__(
        "app.retrieval.shadow_analysis", fromlist=["run_shadow_docset_analysis"]
    ).run_shadow_docset_analysis
    real_golden = __import__(
        "app.retrieval.golden_evaluation", fromlist=["evaluate_docset_golden"]
    ).evaluate_docset_golden

    def _capture_shape(**kwargs):
        captured["shape_analysis_id"] = id(kwargs["analysis"])
        captured["shape_matches_id"] = id(kwargs["matches"])
        if "payload" not in analysis_before:
            analysis_before["payload"] = kwargs["analysis"].to_debug_dict()
            matches_before[:] = [m.to_debug_dict() for m in kwargs["matches"]]
            captured["analysis_id"] = id(kwargs["analysis"])
            captured["matches_id"] = id(kwargs["matches"])
        return real_shape(**kwargs)

    def _capture_shadow(**kwargs):
        captured["shadow_analysis_id"] = id(kwargs["analysis"])
        captured["shadow_matches_id"] = id(kwargs["matches"])
        return real_shadow(**kwargs)

    def _capture_golden(**kwargs):
        captured["golden_called"] = True
        return real_golden(**kwargs)

    async def _fake_retrieve(*args, **kwargs):
        return (
            [
                _sample_ranked(
                    "inv-A",
                    "c1",
                    "Invoice A is overdue and has a late fee. Tax 12% is applied.",
                ),
                _sample_ranked(
                    "inv-B",
                    "c2",
                    "Invoice B is overdue and has a late fee. Tax 15% is applied.",
                ),
            ],
            [],
        )

    from app.api.v2 import retrieve_chat_api as chat_mod
    from app.retrieval.types_retrieve import DomainType
    from app.retrieval.task_classifier import TaskPlan as _TaskPlan, TaskType

    def _invoice_task_plan(*args, **kwargs):
        return _TaskPlan(
            route=QueryRoute.KNOWLEDGE,
            task_type=TaskType.DOCSET_FILTER,
            domain=DomainType.INVOICE,
            predicates=None,
            confidence=0.9,
        )

    with (
        _tenant_patches(runtime, route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(MagicMock(generate=MagicMock(return_value=MagicMock(text="answer"))), "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
        patch("app.api.v2.retrieve_chat_api.classify_task", side_effect=_invoice_task_plan),
        patch(
            "app.api.v2.retrieve_chat_api.shape_docset_results",
            side_effect=_capture_shape,
        ),
        patch(
            "app.api.v2.retrieve_chat_api.run_shadow_docset_analysis",
            side_effect=_capture_shadow,
        ),
        patch(
            "app.api.v2.retrieve_chat_api.evaluate_docset_golden",
            side_effect=_capture_golden,
        ),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(
                client,
                "Which invoices are overdue and include a late payment penalty?",
            )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    analysis_after = (body.get("debug_info") or {}).get("docset_analysis")
    assert analysis_after == analysis_before["payload"]
    assert captured["shape_analysis_id"] == captured["analysis_id"]
    assert captured["shadow_analysis_id"] == captured["analysis_id"]
    assert captured["shape_matches_id"] == captured["matches_id"]
    assert captured["shadow_matches_id"] == captured["matches_id"]
    assert captured.get("golden_called") is True


@pytest.mark.anyio
async def test_trust_gate_uses_baseline_pre_shaping_option_a(minimal_runtime):
    """
    Trust gate Option A: rag_min_score is evaluated on the baseline retrieval
    set even when shaping narrows grounding to a lower-scoring eligible subset.
    """
    runtime = replace(
        minimal_runtime,
        enable_docset_analysis=True,
        enable_docset_invoice_adapter=True,
        enable_docset_result_shaping=True,
        docset_max_docs_returned=1,
        rag_min_score=0.5,
    )
    route = knowledge_decision(top_k=5)

    async def _fake_retrieve(*args, **kwargs):
        return (
            [
                RankedResult(
                    chunk_id="c-ineligible",
                    text="Invoice PAID is fully paid with no penalties.",
                    score=0.99,
                    explanation={},
                    trust_decision=None,
                    file_id="inv-PAID",
                ),
                RankedResult(
                    chunk_id="c-low",
                    text="Invoice LOW is overdue and has a late fee. Tax 12% is applied.",
                    score=0.3,
                    explanation={},
                    trust_decision=None,
                    file_id="inv-LOW",
                ),
            ],
            [],
        )

    from app.api.v2 import retrieve_chat_api as chat_mod
    from app.retrieval.types_retrieve import DomainType
    from app.retrieval.task_classifier import TaskPlan as _TaskPlan, TaskType

    def _invoice_task_plan(*args, **kwargs):
        return _TaskPlan(
            route=QueryRoute.KNOWLEDGE,
            task_type=TaskType.DOCSET_FILTER,
            domain=DomainType.INVOICE,
            predicates=None,
            confidence=0.9,
        )

    with (
        _tenant_patches(runtime, route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(MagicMock(generate=MagicMock(return_value=MagicMock(text="answer"))), "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
        patch("app.api.v2.retrieve_chat_api.classify_task", side_effect=_invoice_task_plan),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(
                client,
                "Which invoices are overdue and include a late payment penalty?",
            )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Shaping keeps only the eligible low-scoring document.
    assert [r["chunk_id"] for r in body["results"]] == ["c-low"]
    assert body["results"][0]["score"] == pytest.approx(0.3)
    # Baseline max (0.99 from ineligible inv-PAID) passes the gate even though
    # shaped grounding max is below rag_min_score.
    assert body.get("answer_error") is None


@pytest.mark.anyio
async def test_option_a_answer_grounding_excludes_baseline_only_documents(minimal_runtime):
    """
    Option A safety: baseline trust pass must not ground answers in excluded
    baseline-only documents; LLM context must come from shaped results only.
    """
    runtime = replace(
        minimal_runtime,
        enable_docset_analysis=True,
        enable_docset_invoice_adapter=True,
        enable_docset_result_shaping=True,
        docset_max_docs_returned=1,
        rag_min_score=0.5,
    )
    route = knowledge_decision(top_k=5)
    captured_prompts: list[str] = []

    async def _fake_retrieve(*args, **kwargs):
        return (
            [
                RankedResult(
                    chunk_id="c-ineligible",
                    text="Invoice PAID is fully paid with no penalties.",
                    score=0.99,
                    explanation={},
                    trust_decision=None,
                    file_id="inv-PAID",
                ),
                RankedResult(
                    chunk_id="c-low",
                    text="Invoice LOW is overdue and has a late fee. Tax 12% is applied.",
                    score=0.3,
                    explanation={},
                    trust_decision=None,
                    file_id="inv-LOW",
                ),
            ],
            [],
        )

    from app.api.v2 import retrieve_chat_api as chat_mod
    from app.retrieval.types_retrieve import DomainType
    from app.retrieval.task_classifier import TaskPlan as _TaskPlan, TaskType

    def _invoice_task_plan(*args, **kwargs):
        return _TaskPlan(
            route=QueryRoute.KNOWLEDGE,
            task_type=TaskType.DOCSET_FILTER,
            domain=DomainType.INVOICE,
            predicates=None,
            confidence=0.9,
        )

    mock_llm = MagicMock()

    def _capture_generate(prompt, **kwargs):
        captured_prompts.append(prompt)
        return MagicMock(text="Invoice LOW is overdue [Source 1].")

    mock_llm.generate = MagicMock(side_effect=_capture_generate)

    with (
        _tenant_patches(runtime, route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(mock_llm, "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
        patch("app.api.v2.retrieve_chat_api.classify_task", side_effect=_invoice_task_plan),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(
                client,
                "Which invoices are overdue and include a late payment penalty?",
            )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [r["chunk_id"] for r in body["results"]] == ["c-low"]
    assert body.get("answer_error") is None
    assert captured_prompts
    grounding_prompts = [
        p for p in captured_prompts if "diverse search queries" not in p.lower()
    ]
    assert grounding_prompts, "LLM should receive a grounded prompt"
    prompt = grounding_prompts[-1]
    assert "fully paid" not in prompt.lower()
    assert "inv-paid" not in prompt.lower()
    assert "invoice low" in prompt.lower()


@pytest.mark.anyio
async def test_analysis_exception_fail_open_consistency(minimal_runtime):
    """
    HR2B: analysis exception leaves shaping/shadow/golden unavailable and
    preserves Phase 3 behavior with a single analysis attempt.
    """
    runtime_baseline = replace(
        minimal_runtime,
        enable_docset_analysis=False,
        enable_docset_result_shaping=False,
        enable_docset_shadow_mode=False,
        enable_docset_golden_eval=False,
    )
    runtime_fail = replace(
        minimal_runtime,
        enable_docset_analysis=True,
        enable_docset_analysis_debug=True,
        enable_docset_invoice_adapter=True,
        enable_docset_result_shaping=True,
        enable_docset_shadow_mode=True,
        enable_docset_shadow_debug=True,
        enable_docset_golden_eval=True,
        enable_docset_golden_debug=True,
        docset_golden_set_ref="ignored",
    )
    route = knowledge_decision(top_k=5)
    call_count = {"n": 0}

    async def _fake_retrieve(*args, **kwargs):
        return (
            [
                _sample_ranked(
                    "inv-A",
                    "c1",
                    "Invoice A is overdue and has a late fee. Tax 12% is applied.",
                ),
                _sample_ranked(
                    "inv-B",
                    "c2",
                    "Invoice B is overdue and has a late fee. Tax 15% is applied.",
                ),
            ],
            [],
        )

    from app.api.v2 import retrieve_chat_api as chat_mod
    from app.retrieval.types_retrieve import DomainType
    from app.retrieval.task_classifier import TaskPlan as _TaskPlan, TaskType

    def _invoice_task_plan(*args, **kwargs):
        return _TaskPlan(
            route=QueryRoute.KNOWLEDGE,
            task_type=TaskType.DOCSET_FILTER,
            domain=DomainType.INVOICE,
            predicates=None,
            confidence=0.9,
        )

    def _raising_structured(*args, **kwargs):
        call_count["n"] += 1
        raise RuntimeError("synthetic analysis failure")

    with (
        _tenant_patches(runtime_baseline, route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(MagicMock(generate=MagicMock(return_value=MagicMock(text="answer"))), "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
        patch("app.api.v2.retrieve_chat_api.classify_task", side_effect=_invoice_task_plan),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp_baseline = await _post_chat(
                client,
                "Which invoices are overdue and include a late payment penalty?",
            )

    with (
        _tenant_patches(runtime_fail, route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(MagicMock(generate=MagicMock(return_value=MagicMock(text="answer"))), "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
        patch("app.api.v2.retrieve_chat_api.classify_task", side_effect=_invoice_task_plan),
        patch(
            "app.api.v2.retrieve_chat_api.run_docset_analysis_structured",
            side_effect=_raising_structured,
        ),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp_fail = await _post_chat(
                client,
                "Which invoices are overdue and include a late payment penalty?",
            )

    assert resp_baseline.status_code == 200
    assert resp_fail.status_code == 200
    body_baseline = resp_baseline.json()
    body_fail = resp_fail.json()
    ChatRetrieveResponse.model_validate(body_baseline)
    ChatRetrieveResponse.model_validate(body_fail)
    assert body_fail["results"] == body_baseline["results"]
    assert body_fail.get("answer") == body_baseline.get("answer")
    assert call_count["n"] == 1
    debug_fail = body_fail.get("debug_info") or {}
    assert "docset_analysis" not in debug_fail
    assert "docset_shadow" not in debug_fail
    assert "docset_golden_eval" not in debug_fail


def test_consumer_order_independence(minimal_runtime):
    """
    Shaping and shadow are read-only consumers; invocation order must not
    change outputs or mutate shared analysis state.
    """
    from copy import deepcopy

    from app.retrieval.document_set_analysis import AnalysisResult
    from app.retrieval.document_match import DocumentMatch
    from app.retrieval.docset_result_shaper import shape_docset_results
    from app.retrieval.shadow_analysis import run_shadow_docset_analysis
    from app.retrieval.task_classifier import TaskPlan, TaskType
    from app.retrieval.types_retrieve import DomainType

    ranked = [
        _sample_ranked(
            "inv-A",
            "c1",
            "Invoice A is overdue and has a late fee. Tax 12% is applied.",
        ),
        _sample_ranked(
            "inv-B",
            "c2",
            "Invoice B is overdue and has a late fee. Tax 15% is applied.",
        ),
    ]
    analysis = AnalysisResult(
        domain="invoice",
        docs_analyzed=2,
        docs_matched=2,
        matches=[
            {"file_id": "inv-A", "overdue": True},
            {"file_id": "inv-B", "overdue": True},
        ],
    )
    matches = [
        DocumentMatch(file_id="inv-A", attributes={"overdue": True}, sample_chunk_ids=["c1"]),
        DocumentMatch(file_id="inv-B", attributes={"overdue": True}, sample_chunk_ids=["c2"]),
    ]
    runtime = replace(
        minimal_runtime,
        enable_docset_result_shaping=True,
        docset_max_docs_returned=1,
        enable_docset_shadow_mode=True,
        enable_docset_shadow_debug=True,
    )
    task_plan = TaskPlan(
        route=QueryRoute.KNOWLEDGE,
        task_type=TaskType.DOCSET_FILTER,
        domain=DomainType.INVOICE,
        predicates=None,
        confidence=0.9,
    )
    analysis_snapshot = deepcopy(analysis.to_debug_dict())
    matches_snapshot = [m.to_debug_dict() for m in matches]

    def _run_shape_first():
        shaped = shape_docset_results(
            ranked_results=list(ranked),
            analysis=analysis,
            matches=list(matches),
            runtime=runtime,
        )
        shadow = run_shadow_docset_analysis(
            raw_query="overdue invoices",
            task_plan=task_plan,
            ranked_results=ranked,
            tenant_runtime=runtime,
            analysis=analysis,
            matches=matches,
        )
        return shaped, shadow

    def _run_shadow_first():
        shadow = run_shadow_docset_analysis(
            raw_query="overdue invoices",
            task_plan=task_plan,
            ranked_results=ranked,
            tenant_runtime=runtime,
            analysis=analysis,
            matches=matches,
        )
        shaped = shape_docset_results(
            ranked_results=list(ranked),
            analysis=analysis,
            matches=list(matches),
            runtime=runtime,
        )
        return shaped, shadow

    shaped_a, shadow_a = _run_shape_first()
    assert analysis.to_debug_dict() == analysis_snapshot
    assert [m.to_debug_dict() for m in matches] == matches_snapshot

    shaped_b, shadow_b = _run_shadow_first()
    assert analysis.to_debug_dict() == analysis_snapshot
    assert [m.to_debug_dict() for m in matches] == matches_snapshot

    assert [r.chunk_id for r in shaped_a or []] == [r.chunk_id for r in shaped_b or []]

    def _shadow_semantics(payload):
        if payload is None:
            return None
        return {k: v for k, v in payload.items() if k != "analysis_ms"}

    assert _shadow_semantics(shadow_a) == _shadow_semantics(shadow_b)


@pytest.mark.anyio
async def test_non_docset_trust_gate_unchanged_after_hardening(minimal_runtime):
    """Non-docset requests preserve trust-gate behavior when shaping flags are on."""
    runtime = replace(
        minimal_runtime,
        enable_docset_analysis=True,
        enable_docset_result_shaping=True,
        rag_min_score=0.5,
    )
    route = knowledge_decision(top_k=5)

    async def _fake_retrieve(*args, **kwargs):
        return (
            [
                _sample_ranked("doc-1", "c1", "Generic knowledge chunk."),
            ],
            [],
        )

    from app.api.v2 import retrieve_chat_api as chat_mod
    from app.retrieval.types_retrieve import DomainType
    from app.retrieval.task_classifier import TaskPlan as _TaskPlan, TaskType

    def _generic_task_plan(*args, **kwargs):
        return _TaskPlan(
            route=QueryRoute.KNOWLEDGE,
            task_type=TaskType.SINGLE_QA,
            domain=DomainType.GENERIC,
            predicates=None,
            confidence=0.9,
        )

    with (
        _tenant_patches(runtime, route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(MagicMock(generate=MagicMock(return_value=MagicMock(text="answer"))), "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
        patch("app.api.v2.retrieve_chat_api.classify_task", side_effect=_generic_task_plan),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(client, "What is in the knowledge base?")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("answer_error") is None
    assert len(body["results"]) == 1


@pytest.mark.anyio
async def test_summary_construction_failure_falls_back(minimal_runtime):
    """Summary builder failure must not affect answer generation or trust path."""
    runtime = replace(
        minimal_runtime,
        enable_docset_analysis=True,
        enable_docset_invoice_adapter=True,
        enable_docset_summary=True,
        enable_docset_summary_debug=True,
    )
    route = knowledge_decision(top_k=5)

    async def _fake_retrieve(*args, **kwargs):
        return (
            [
                _sample_ranked(
                    "inv-1",
                    "c1",
                    "This invoice is overdue with a late payment penalty.",
                )
            ],
            [],
        )

    from app.api.v2 import retrieve_chat_api as chat_mod
    from app.retrieval.types_retrieve import DomainType
    from app.retrieval.task_classifier import TaskPlan as _TaskPlan, TaskType

    def _invoice_task_plan(*args, **kwargs):
        return _TaskPlan(
            route=QueryRoute.KNOWLEDGE,
            task_type=TaskType.DOCSET_FILTER,
            domain=DomainType.INVOICE,
            predicates=None,
            confidence=0.9,
        )

    with (
        _tenant_patches(runtime, route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(MagicMock(generate=MagicMock(return_value=MagicMock(text="answer"))), "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
        patch("app.api.v2.retrieve_chat_api.classify_task", side_effect=_invoice_task_plan),
        patch(
            "app.api.v2.retrieve_chat_api.build_docset_summary",
            side_effect=RuntimeError("summary boom"),
        ),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(
                client,
                "Which invoices are overdue and include a late payment penalty?",
            )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    debug = body.get("debug_info") or {}
    assert debug.get("summary_mode") == "construction_failed"
    assert debug.get("summary_degraded") is True
    assert body.get("answer_error") is None


@pytest.mark.anyio
async def test_non_docset_path_unchanged_with_summary_flag(minimal_runtime):
    """Non-docset requests remain unchanged even when summary flags are on."""
    runtime = replace(
        minimal_runtime,
        enable_docset_analysis=True,
        enable_docset_summary=True,
        enable_docset_summary_debug=True,
    )
    route = knowledge_decision(top_k=5)

    async def _fake_retrieve(*args, **kwargs):
        return (
            [
                _sample_ranked("doc-1", "c1", "Generic knowledge chunk."),
            ],
            [],
        )

    from app.api.v2 import retrieve_chat_api as chat_mod
    from app.retrieval.types_retrieve import DomainType
    from app.retrieval.task_classifier import TaskPlan as _TaskPlan, TaskType

    def _generic_task_plan(*args, **kwargs):
        return _TaskPlan(
            route=QueryRoute.KNOWLEDGE,
            task_type=TaskType.SINGLE_QA,
            domain=DomainType.GENERIC,
            predicates=None,
            confidence=0.9,
        )

    with (
        _tenant_patches(runtime, route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(MagicMock(generate=MagicMock(return_value=MagicMock(text="answer"))), "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
        patch("app.api.v2.retrieve_chat_api.classify_task", side_effect=_generic_task_plan),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(client, "What is in the knowledge base?")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    debug = body.get("debug_info") or {}
    assert "summary_mode" not in debug
    assert "docset_summary" not in debug


@pytest.mark.anyio
async def test_docset_phase5a_is_observability_only(minimal_runtime):
    """Phase 5A summary is emitted for observability but never enters LLM prompt."""
    runtime = replace(
        minimal_runtime,
        enable_docset_analysis=True,
        enable_docset_invoice_adapter=True,
        enable_docset_summary=True,
        enable_docset_summary_debug=True,
    )
    route = knowledge_decision(top_k=5)
    captured_prompts: list[str] = []
    verifier_calls: list[list[str]] = []

    async def _fake_retrieve(*args, **kwargs):
        return (
            [
                _sample_ranked(
                    "inv-1",
                    "c1",
                    "This invoice is overdue with a late payment penalty.",
                )
            ],
            [],
        )

    from app.api.v2 import retrieve_chat_api as chat_mod
    from app.retrieval.types_retrieve import DomainType
    from app.retrieval.task_classifier import TaskPlan as _TaskPlan, TaskType

    def _invoice_task_plan(*args, **kwargs):
        return _TaskPlan(
            route=QueryRoute.KNOWLEDGE,
            task_type=TaskType.DOCSET_FILTER,
            domain=DomainType.INVOICE,
            predicates=None,
            confidence=0.9,
        )

    mock_llm = MagicMock()

    def _capture_generate(prompt, **kwargs):
        captured_prompts.append(prompt)
        return MagicMock(text="Invoice is overdue [Source 1].")

    mock_llm.generate = MagicMock(side_effect=_capture_generate)

    def _capture_verify(answer, source_texts, **kwargs):
        verifier_calls.append(list(source_texts))
        from app.services.faithfulness_verifier import VerificationResult, VerificationStatus

        return answer, VerificationResult(
            status=VerificationStatus.PASS, checks=[], fail_reason=None
        )

    with (
        _tenant_patches(runtime, route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(mock_llm, "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
        patch("app.api.v2.retrieve_chat_api.classify_task", side_effect=_invoice_task_plan),
        patch("app.api.v2.retrieve_chat_api.verify_or_refuse", side_effect=_capture_verify),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(
                client,
                "Which invoices are overdue and include a late payment penalty?",
            )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    debug = body.get("debug_info") or {}
    assert debug.get("summary_mode") == "deterministic"
    assert debug.get("summary_used_llm") is False
    assert debug.get("golden_scope") == "document_selection_only"
    assert debug.get("docset_summary") is not None

    assert captured_prompts
    grounding_prompts = [
        p for p in captured_prompts if "diverse search queries" not in p.lower()
    ]
    assert grounding_prompts
    prompt = grounding_prompts[-1]
    summary_text = (debug.get("docset_summary") or {}).get("text") or ""
    assert summary_text
    assert summary_text not in prompt
    assert "RETRIEVED PASSAGES" in prompt
    assert "overdue with a late payment penalty" in prompt

    assert verifier_calls
    assert any("overdue with a late payment penalty" in text for text in verifier_calls[0])


@pytest.mark.anyio
async def test_debug_info_includes_summary_version_when_enabled(minimal_runtime):
    runtime = replace(
        minimal_runtime,
        enable_docset_analysis=True,
        enable_docset_invoice_adapter=True,
        enable_docset_summary=True,
        enable_docset_summary_debug=False,
    )
    route = knowledge_decision(top_k=5)

    async def _fake_retrieve(*args, **kwargs):
        return (
            [
                _sample_ranked(
                    "inv-1",
                    "c1",
                    "This invoice is overdue with a late payment penalty.",
                )
            ],
            [],
        )

    from app.api.v2 import retrieve_chat_api as chat_mod
    from app.retrieval.deterministic_summary import SUMMARY_FORMAT_VERSION
    from app.retrieval.types_retrieve import DomainType
    from app.retrieval.task_classifier import TaskPlan as _TaskPlan, TaskType

    def _invoice_task_plan(*args, **kwargs):
        return _TaskPlan(
            route=QueryRoute.KNOWLEDGE,
            task_type=TaskType.DOCSET_FILTER,
            domain=DomainType.INVOICE,
            predicates=None,
            confidence=0.9,
        )

    with (
        _tenant_patches(runtime, route),
        patch.object(
            chat_mod,
            "_embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch.object(
            chat_mod,
            "_resolve_llm",
            return_value=(MagicMock(generate=MagicMock(return_value=MagicMock(text="answer"))), "model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_fake_retrieve,
        ),
        patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None),
        patch("app.api.v2.retrieve_chat_api.classify_task", side_effect=_invoice_task_plan),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(
                client,
                "Which invoices are overdue and include a late payment penalty?",
            )

    assert resp.status_code == 200, resp.text
    debug = (resp.json().get("debug_info") or {})
    assert debug.get("summary_format_version") == SUMMARY_FORMAT_VERSION
    assert "docset_summary" not in debug


# ── Phase 6A KNOWLEDGE verifier integration ─────────────────────────────────


@contextmanager
def _knowledge_chat_patches(chat_mod, minimal_runtime, route, retrieve_results, llm_text):
    async def _fake_retrieve(*args, **kwargs):
        return (retrieve_results, [])

    mock_llm = MagicMock()
    mock_llm.generate = MagicMock(return_value=MagicMock(text=llm_text))

    with ExitStack() as stack:
        stack.enter_context(_tenant_patches(minimal_runtime, route))
        stack.enter_context(
            patch.object(
                chat_mod,
                "_embed_in_thread",
                new_callable=AsyncMock,
                return_value=[0.1] * 8,
            )
        )
        stack.enter_context(
            patch.object(
                chat_mod,
                "_resolve_llm",
                return_value=(mock_llm, "model"),
            )
        )
        stack.enter_context(
            patch(
                "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
                return_value=MagicMock(vectordb=MagicMock()),
            )
        )
        stack.enter_context(
            patch(
                "app.retrieval.runtime.RetrievalRuntime.retrieve",
                new_callable=AsyncMock,
                side_effect=_fake_retrieve,
            )
        )
        stack.enter_context(
            patch.object(chat_mod, "maybe_start_rag_chat_trace", return_value=None)
        )
        yield


@pytest.mark.anyio
async def test_phase6a_flags_off_no_knowledge_verifier_debug(minimal_runtime):
    """All Phase 6A flags OFF: response unchanged; no knowledge_verifier block."""
    runtime = replace(
        minimal_runtime,
        enable_knowledge_faithfulness_shadow=False,
        enable_knowledge_faithfulness_gate=False,
        enable_knowledge_faithfulness_debug=False,
    )
    route = knowledge_decision(top_k=5)
    ranked = [
        _sample_ranked("inv-1", "c1", "Invoice INV-100 is overdue with $50.00 fee."),
    ]
    llm_answer = "Invoice INV-100 is overdue [Source 1]."

    from app.api.v2 import retrieve_chat_api as chat_mod

    with _knowledge_chat_patches(chat_mod, runtime, route, ranked, llm_answer):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(client, "Which invoices are overdue?")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("answer") == llm_answer
    debug = body.get("debug_info") or {}
    assert "knowledge_verifier" not in debug
    fv = debug.get("faithfulness_verifier") or {}
    assert fv.get("status") == "skipped"


@pytest.mark.anyio
async def test_phase6a_shadow_only_observational(minimal_runtime):
    """Shadow ON, gate OFF: answer unchanged; knowledge_verifier debug present."""
    runtime = replace(
        minimal_runtime,
        enable_knowledge_faithfulness_shadow=True,
        enable_knowledge_faithfulness_gate=False,
    )
    route = knowledge_decision(top_k=5)
    ranked = [
        _sample_ranked("inv-1", "c1", "Invoice INV-100 is overdue."),
    ]
    llm_answer = "Unsupported total $999.00 [Source 1]."

    from app.api.v2 import retrieve_chat_api as chat_mod

    with _knowledge_chat_patches(chat_mod, runtime, route, ranked, llm_answer):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(client, "Which invoices are overdue?")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("answer") == llm_answer
    kv = (body.get("debug_info") or {}).get("knowledge_verifier") or {}
    assert kv.get("version") == "v1"
    assert kv.get("status") in ("fail", "indeterminate")


@pytest.mark.anyio
async def test_phase6a_gate_refusal_on_unsupported_claim(minimal_runtime):
    """Gate ON: unsupported gated claim triggers KNOWLEDGE refusal text."""
    from app.api.v2.retrieve_chat_api import _GROUNDING_REFUSAL_PHRASE

    runtime = replace(
        minimal_runtime,
        enable_knowledge_faithfulness_shadow=True,
        enable_knowledge_faithfulness_gate=True,
    )
    route = knowledge_decision(top_k=5)
    ranked = [
        _sample_ranked("inv-1", "c1", "Invoice INV-100 is overdue."),
    ]
    llm_answer = "Total due is $999.00 [Source 1]."

    from app.api.v2 import retrieve_chat_api as chat_mod

    with _knowledge_chat_patches(chat_mod, runtime, route, ranked, llm_answer):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(client, "Which invoices are overdue?")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("answer") == _GROUNDING_REFUSAL_PHRASE
    kv = (body.get("debug_info") or {}).get("knowledge_verifier") or {}
    assert kv.get("status") == "fail"


@pytest.mark.anyio
async def test_phase6a_structured_route_flags_on_unchanged(minimal_runtime):
    """STRUCTURED route with Phase 6A flags ON: no knowledge_verifier block."""
    runtime = replace(
        minimal_runtime,
        enable_knowledge_faithfulness_shadow=True,
        enable_knowledge_faithfulness_gate=True,
        enable_knowledge_faithfulness_debug=True,
    )
    route = structured_decision(top_k=1)
    ranked = [
        _sample_ranked("inv-1", "c1", "Invoice INV-1234 due date is 2024-01-15."),
    ]
    llm_answer = "Due date is 2024-01-15 [Source 1]."

    from app.api.v2 import retrieve_chat_api as chat_mod

    with _knowledge_chat_patches(chat_mod, runtime, route, ranked, llm_answer):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(client, "What is the due date on invoice INV-1234?")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("answer") == llm_answer
    debug = body.get("debug_info") or {}
    assert debug.get("query_route") == QueryRoute.STRUCTURED.value
    assert "knowledge_verifier" not in debug


@pytest.mark.anyio
async def test_phase6a_gate_only_without_shadow(minimal_runtime):
    """gate=true, shadow=false: gating still applies; answer replaced on FAIL."""
    from app.api.v2.retrieve_chat_api import _GROUNDING_REFUSAL_PHRASE

    runtime = replace(
        minimal_runtime,
        enable_knowledge_faithfulness_shadow=False,
        enable_knowledge_faithfulness_gate=True,
    )
    route = knowledge_decision(top_k=5)
    ranked = [
        _sample_ranked("inv-1", "c1", "Invoice INV-100 is overdue."),
    ]
    llm_answer = "Total due is $999.00 [Source 1]."

    from app.api.v2 import retrieve_chat_api as chat_mod

    with _knowledge_chat_patches(chat_mod, runtime, route, ranked, llm_answer):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(client, "Which invoices are overdue?")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("answer") == _GROUNDING_REFUSAL_PHRASE
    kv = (body.get("debug_info") or {}).get("knowledge_verifier") or {}
    assert kv.get("status") == "fail"


@pytest.mark.anyio
async def test_phase6a_treat_as_fail_integration(minimal_runtime):
    """INDETERMINATE + treat_as_fail policy replaces answer with refusal."""
    from app.api.v2.retrieve_chat_api import _GROUNDING_REFUSAL_PHRASE

    runtime = replace(
        minimal_runtime,
        enable_knowledge_faithfulness_shadow=False,
        enable_knowledge_faithfulness_gate=True,
        knowledge_indeterminate_policy="treat_as_fail",
    )
    route = knowledge_decision(top_k=5)
    ranked = [
        _sample_ranked("inv-1", "c1", "Invoice INV-100 is overdue."),
    ]
    llm_answer = "Total due is $50.00."

    from app.api.v2 import retrieve_chat_api as chat_mod

    with _knowledge_chat_patches(chat_mod, runtime, route, ranked, llm_answer):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(client, "Which invoices are overdue?")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("answer") == _GROUNDING_REFUSAL_PHRASE
    kv = (body.get("debug_info") or {}).get("knowledge_verifier") or {}
    assert kv.get("status") == "indeterminate"
    assert kv.get("indeterminate_policy_applied") is True


@pytest.mark.anyio
async def test_phase6a_verifier_exception_indeterminate_integration(minimal_runtime):
    """Verifier internal fault surfaces INDETERMINATE; answer unchanged (pass_through)."""
    runtime = replace(
        minimal_runtime,
        enable_knowledge_faithfulness_shadow=True,
        enable_knowledge_faithfulness_gate=True,
    )
    route = knowledge_decision(top_k=5)
    ranked = [
        _sample_ranked("inv-1", "c1", "Invoice INV-100 is overdue."),
    ]
    llm_answer = "Invoice INV-100 is overdue [Source 1]."

    from app.api.v2 import retrieve_chat_api as chat_mod
    from app.services import knowledge_faithfulness_verifier as kv_mod

    with _knowledge_chat_patches(chat_mod, runtime, route, ranked, llm_answer):
        with patch.object(
            kv_mod,
            "extract_claims",
            side_effect=RuntimeError("integration fault"),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await _post_chat(client, "Which invoices are overdue?")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("answer") == llm_answer
    kv = (body.get("debug_info") or {}).get("knowledge_verifier") or {}
    assert kv.get("status") == "indeterminate"
    assert "integration fault" in (kv.get("error_detail") or "")


@pytest.mark.anyio
async def test_phase6a_pii_redaction_bundle_alignment(minimal_runtime):
    """Evidence bundle uses post-PII context_str; gated token in sanitized text passes."""
    runtime = replace(
        minimal_runtime,
        enable_knowledge_faithfulness_shadow=True,
        enable_knowledge_faithfulness_gate=True,
        enable_knowledge_faithfulness_debug=True,
    )
    route = knowledge_decision(top_k=5)
    ranked = [
        _sample_ranked(
            "inv-1",
            "c1",
            "Invoice INV-100 is overdue. Contact payer@test.com. Tax rate 25%.",
        ),
    ]
    llm_answer = "Tax rate is 25% [Source 1]."

    from app.api.v2 import retrieve_chat_api as chat_mod

    with _knowledge_chat_patches(chat_mod, runtime, route, ranked, llm_answer):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(client, "Which invoices are overdue?")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("answer") == llm_answer
    kv = (body.get("debug_info") or {}).get("knowledge_verifier") or {}
    assert kv.get("status") == "pass"
    claims = kv.get("claims") or []
    pct_claim = next(
        (c for c in claims if c.get("category") == "numeric" and c.get("status") == "pass"),
        None,
    )
    assert pct_claim is not None
    snippets = [
        link.get("snippet", "")
        for link in (pct_claim.get("evidence_links") or [])
    ]
    assert snippets
    joined = " ".join(snippets).lower()
    assert "payer@test.com" not in joined
    assert "25%" in joined


@pytest.mark.anyio
async def test_phase6a_percentage_gate_integration(minimal_runtime):
    """Unsupported percentage claim triggers gate refusal."""
    from app.api.v2.retrieve_chat_api import _GROUNDING_REFUSAL_PHRASE

    runtime = replace(
        minimal_runtime,
        enable_knowledge_faithfulness_shadow=False,
        enable_knowledge_faithfulness_gate=True,
    )
    route = knowledge_decision(top_k=5)
    ranked = [
        _sample_ranked("inv-1", "c1", "Invoice INV-100 with 10% tax noted."),
    ]
    llm_answer = "Tax rate is 25% [Source 1]."

    from app.api.v2 import retrieve_chat_api as chat_mod

    with _knowledge_chat_patches(chat_mod, runtime, route, ranked, llm_answer):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(client, "What is the tax rate?")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("answer") == _GROUNDING_REFUSAL_PHRASE
    kv = (body.get("debug_info") or {}).get("knowledge_verifier") or {}
    assert kv.get("status") == "fail"
    assert (kv.get("gated_claims_failed") or 0) >= 1


@pytest.mark.anyio
async def test_phase6a_post_verifier_integration_fault_gate_safe(minimal_runtime):
    """Post-verify integration fault with gate=true must stay observable and not fail open."""
    from app.api.v2.retrieve_chat_api import _GROUNDING_REFUSAL_PHRASE
    from app.services.knowledge_faithfulness_verifier import AnswerVerificationResult

    runtime = replace(
        minimal_runtime,
        enable_knowledge_faithfulness_shadow=False,
        enable_knowledge_faithfulness_gate=True,
    )
    route = knowledge_decision(top_k=5)
    ranked = [
        _sample_ranked("inv-1", "c1", "Invoice INV-100 is overdue."),
    ]
    llm_answer = "Total due is $999.00 [Source 1]."

    from app.api.v2 import retrieve_chat_api as chat_mod

    trust_outcomes: list[str] = []

    def _capture_trust_gate(route: str, outcome: str) -> None:
        trust_outcomes.append(outcome)

    with _knowledge_chat_patches(chat_mod, runtime, route, ranked, llm_answer):
        with (
            patch.object(
                AnswerVerificationResult,
                "to_debug_dict",
                side_effect=RuntimeError("debug fault"),
            ),
            patch.object(chat_mod, "record_trust_gate", side_effect=_capture_trust_gate),
            patch.object(chat_mod, "record_knowledge_verifier") as mock_kv_metric,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await _post_chat(client, "Which invoices are overdue?")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("answer") == _GROUNDING_REFUSAL_PHRASE
    kv = (body.get("debug_info") or {}).get("knowledge_verifier") or {}
    assert kv.get("status") in ("pass", "fail", "indeterminate")
    assert kv.get("status") == "fail"
    assert trust_outcomes
    assert trust_outcomes[-1] != "skipped"
    assert trust_outcomes[-1] == "fail"
    assert mock_kv_metric.called
    metric_status = mock_kv_metric.call_args.kwargs.get("status") or (
        mock_kv_metric.call_args.args[1] if len(mock_kv_metric.call_args.args) > 1 else None
    )
    if metric_status is None and mock_kv_metric.call_args.kwargs:
        metric_status = mock_kv_metric.call_args.kwargs.get("status")
    assert metric_status == "fail"

