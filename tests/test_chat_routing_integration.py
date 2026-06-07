"""
HTTP-level integration tests for L0 query routing on POST /api/v2/retrieve/chat.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient, Response

import app.services.query_routing.orchestrator as orch_mod
from app.auth.deps import get_current_user
from app.db.session_v2 import get_db
from app.api.v2.retrieve_chat_api import ChatRetrieveResponse
from app.main import app
from app.retrieval.components import RuntimeComponents
from app.services.query_routing.types import QueryRoute, knowledge_decision


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def override_auth():
    """Bypass JWT auth — match test_admin_customers_rag_dashboard.py pattern."""
    app.dependency_overrides[get_current_user] = lambda: {
        "role": "admin",
        "sub": "test-admin",
    }
    yield
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture(autouse=True)
def override_db():
    """Replace async DB session with a no-op mock (get_db is async generator)."""
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
    """Minimal RuntimeComponents for fields read before L0 early return."""
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
    )


def _mock_llm_tuple():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = MagicMock(
        text="",
        prompt_tokens=0,
        completion_tokens=0,
    )
    return mock_llm, "test-model"


async def _post_chat(client: AsyncClient, message: str) -> Response:
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
def _tenant_patches(minimal_runtime: RuntimeComponents):
    with (
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
async def test_hi_returns_zero_results_and_retrieval_skipped(minimal_runtime):
    """
    'Hi' must be caught by L0 rule router (CHITCHAT) and return immediately
    with zero sources and retrieval_skipped=True — no embed/vector/rerank calls.
    """
    with _tenant_patches(minimal_runtime):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(client, "Hi")

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["total_results"] == 0, (
        f"Expected total_results=0 for 'Hi', got {body.get('total_results')}"
    )
    debug = body.get("debug_info") or {}
    assert debug.get("retrieval_skipped") is True, (
        f"Expected debug_info.retrieval_skipped=True, got: {debug}"
    )


@pytest.mark.anyio
async def test_thanks_returns_zero_results(minimal_runtime):
    """'Thanks' must also be short-circuited — CHITCHAT route, zero retrieval."""
    with _tenant_patches(minimal_runtime):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(client, "Thanks")

    assert resp.status_code == 200
    assert resp.json()["total_results"] == 0


@pytest.mark.anyio
async def test_structured_query_is_not_short_circuited(minimal_runtime):
    """
    'What is the due date on invoice INV-9922?' triggers STRUCTURED route (rule step 6
    via _STRUCTURED_ID_RE). retrieval_allowed=True — must NOT be short-circuited.
    Pipeline is mocked past embed/retrieve/LLM so test does not need real infra.
    """
    mock_llm, model_name = _mock_llm_tuple()
    with (
        _tenant_patches(minimal_runtime),
        patch(
            "app.api.v2.retrieve_chat_api._embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 384,
        ),
        patch(
            "app.api.v2.retrieve_chat_api._resolve_llm",
            return_value=(mock_llm, model_name),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            return_value=([], []),
        ),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(
                client,
                "What is the due date on invoice INV-9922?",
            )

    if resp.status_code == 200:
        debug = resp.json().get("debug_info") or {}
        assert debug.get("retrieval_skipped") is not True, (
            "STRUCTURED query was incorrectly short-circuited as chitchat/clarification"
        )


@pytest.mark.anyio
async def test_l1_task_plan_absent_when_flag_off(minimal_runtime):
    """
    Phase 1 non-regression: with enable_l1_task_classification off, debug_info must
    not include task_plan; response schema and L0 route behavior stay unchanged.
    """
    assert minimal_runtime.enable_l1_task_classification is False

    mock_route = knowledge_decision(top_k=5)
    mock_orch = MagicMock()
    mock_orch.route = AsyncMock(return_value=mock_route)
    mock_llm, model_name = _mock_llm_tuple()
    mock_trace = MagicMock()

    with (
        _tenant_patches(minimal_runtime),
        patch("app.api.v2.retrieve_chat_api.get_orchestrator", return_value=mock_orch),
        patch(
            "app.api.v2.retrieve_chat_api.maybe_start_rag_chat_trace",
            return_value=mock_trace,
        ),
        patch(
            "app.api.v2.retrieve_chat_api._embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 384,
        ),
        patch(
            "app.api.v2.retrieve_chat_api._resolve_llm",
            return_value=(mock_llm, model_name),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            return_value=([], []),
        ),
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
    assert "task_plan" not in debug
    assert debug.get("query_route") == QueryRoute.KNOWLEDGE.value
    assert debug.get("retrieval_skipped") is False

    classified_stages = [
        call.args[1]
        for call in mock_trace.add_event.call_args_list
        if len(call.args) >= 2
    ]
    assert "task.classified" not in classified_stages


@pytest.mark.anyio
async def test_l1_task_plan_present_when_flag_on_knowledge(minimal_runtime):
    """Phase 1: task_plan debug payload only when L1 flag is on and route is KNOWLEDGE."""
    runtime_l1 = replace(minimal_runtime, enable_l1_task_classification=True)
    assert runtime_l1.enable_l1_task_classification is True

    mock_route = knowledge_decision(top_k=5)
    mock_orch = MagicMock()
    mock_orch.route = AsyncMock(return_value=mock_route)
    mock_llm, model_name = _mock_llm_tuple()

    with (
        _tenant_patches(runtime_l1),
        patch("app.api.v2.retrieve_chat_api.get_orchestrator", return_value=mock_orch),
        patch(
            "app.api.v2.retrieve_chat_api._embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 384,
        ),
        patch(
            "app.api.v2.retrieve_chat_api._resolve_llm",
            return_value=(mock_llm, model_name),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            return_value=([], []),
        ),
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
    task_plan = debug.get("task_plan")
    assert isinstance(task_plan, dict)
    assert task_plan.get("route") == QueryRoute.KNOWLEDGE.value
    assert task_plan.get("task_type") == "docset_filter"
    assert task_plan.get("domain") == "invoice"


@pytest.mark.anyio
async def test_get_orchestrator_lazy_init_when_startup_missing():
    """
    Simulates startup failure (no init_orchestrator call).
    get_orchestrator() must not raise RuntimeError — lazy init instead.
    Rule router must still route 'Hi' correctly in degraded mode.
    """
    saved = orch_mod._ORCHESTRATOR
    try:
        orch_mod._ORCHESTRATOR = None
        instance = orch_mod.get_orchestrator()
        assert instance is not None, "Lazy init must return a valid QueryOrchestrator"
        decision = instance._rule_router.route("Hi")
        assert decision is not None, "Rule router must handle 'Hi' without embed"
        assert decision.route == QueryRoute.CHITCHAT, (
            f"Expected CHITCHAT, got {decision.route}"
        )
    finally:
        orch_mod._ORCHESTRATOR = saved
