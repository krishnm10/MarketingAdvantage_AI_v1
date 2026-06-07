"""Phase 5B answer-polish flag scaffold tests (no polish behavior)."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.retrieval.components import RuntimeComponents
from app.services.query_routing.types import QueryRoute, knowledge_decision


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


def _tenant_patches(minimal_runtime: RuntimeComponents, route):
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        from app.api.v2 import retrieve_chat_api as chat_mod

        mock_orch = MagicMock()
        mock_orch.route = AsyncMock(return_value=route)
        with (
            patch.object(chat_mod, "get_orchestrator", return_value=mock_orch),
            patch(
                "app.retrieval.components.resolve_config_or_fail",
                return_value=(MagicMock(client_id="pytest-tenant"), "authoritative_config"),
            ),
            patch(
                "app.retrieval.components.resolve_runtime_components",
                return_value=minimal_runtime,
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
        ):
            yield

    return _ctx()


async def _post_chat(client: AsyncClient, message: str):
    return await client.post(
        "/api/v2/retrieve/chat",
        json={
            "messages": [{"role": "user", "content": message}],
            "client_id": "pytest-tenant",
            "top_k": 5,
            "enable_hyde": False,
        },
    )


def test_answer_mutation_contract_version_defined():
    from app.api.v2.retrieve_chat_api import ANSWER_MUTATION_CONTRACT_VERSION

    assert ANSWER_MUTATION_CONTRACT_VERSION == "v1"


@pytest.mark.anyio
async def test_polish_flag_defaults_off_in_runtime_components():
    rc = RuntimeComponents(
        runtime_mode="authoritative_config",
        config_fingerprint="x",
        client_id="t",
        embedder_type="huggingface",
        embedder_model="m",
        vectordb_type="chroma",
        collection="c",
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
    assert rc.enable_chat_answer_polish is False
    assert rc.enable_chat_answer_polish_debug is False


@pytest.mark.anyio
async def test_polish_flag_off_no_debug_snapshot(minimal_runtime):
    """Flag OFF: no answer_integrity_snapshot in debug_info."""
    route = knowledge_decision(top_k=5)

    async def _fake_retrieve(*args, **kwargs):
        from app.retrieval.types_retrieve import RankedResult

        return (
            [
                RankedResult(
                    chunk_id="c1",
                    text="Invoice INV-1 overdue $10.00.",
                    score=0.9,
                    explanation={},
                    trust_decision=None,
                    file_id="inv-1",
                )
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

    runtime = replace(
        minimal_runtime,
        enable_chat_answer_polish=False,
        enable_chat_answer_polish_debug=False,
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
            return_value=(
                MagicMock(
                    generate=MagicMock(
                        return_value=MagicMock(text="INV-1 overdue [Source 1].")
                    )
                ),
                "model",
            ),
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
            resp = await _post_chat(client, "What is overdue?")

    assert resp.status_code == 200, resp.text
    debug = resp.json().get("debug_info") or {}
    assert "answer_integrity_snapshot" not in debug


@pytest.mark.anyio
async def test_polish_flag_on_debug_emits_observability_only(minimal_runtime):
    """Flag ON + debug ON: snapshot emitted; answer path unchanged."""
    route = knowledge_decision(top_k=5)

    async def _fake_retrieve(*args, **kwargs):
        from app.retrieval.types_retrieve import RankedResult

        return (
            [
                RankedResult(
                    chunk_id="c1",
                    text="Invoice INV-1 overdue $10.00.",
                    score=0.9,
                    explanation={},
                    trust_decision=None,
                    file_id="inv-1",
                )
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

    runtime = replace(
        minimal_runtime,
        enable_chat_answer_polish=True,
        enable_chat_answer_polish_debug=True,
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
            return_value=(
                MagicMock(
                    generate=MagicMock(
                        return_value=MagicMock(text="INV-1 overdue [Source 1].")
                    )
                ),
                "model",
            ),
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
            resp = await _post_chat(client, "What is overdue?")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    debug = body.get("debug_info") or {}
    snap = debug.get("answer_integrity_snapshot")
    assert snap is not None
    assert snap.get("answer_mutation_contract_version") == "v1"
    assert "known_limitations" in snap
    # Phase 5B shadow foundation observability fields.
    assert "integrity_severity" in snap
    assert "integrity_findings" in snap
    assert snap["integrity_severity"] in ("none", "info", "warning")
    assert isinstance(snap["integrity_findings"], list)
    assert body.get("answer") == "INV-1 overdue [Source 1]."


@pytest.mark.anyio
async def test_polish_debug_failure_emits_fallback_snapshot(minimal_runtime):
    """If snapshot debug formatting fails, a minimal observable snapshot is still emitted."""
    route = knowledge_decision(top_k=5)

    async def _fake_retrieve(*args, **kwargs):
        from app.retrieval.types_retrieve import RankedResult

        return (
            [
                RankedResult(
                    chunk_id="c1",
                    text="Invoice INV-1 overdue $10.00.",
                    score=0.9,
                    explanation={},
                    trust_decision=None,
                    file_id="inv-1",
                )
            ],
            [],
        )

    from app.api.v2 import retrieve_chat_api as chat_mod
    from app.retrieval.types_retrieve import DomainType
    from app.retrieval.task_classifier import TaskPlan as _TaskPlan, TaskType
    from app.retrieval.answer_integrity import AnswerIntegritySnapshot

    def _generic_task_plan(*args, **kwargs):
        return _TaskPlan(
            route=QueryRoute.KNOWLEDGE,
            task_type=TaskType.SINGLE_QA,
            domain=DomainType.GENERIC,
            predicates=None,
            confidence=0.9,
        )

    runtime = replace(
        minimal_runtime,
        enable_chat_answer_polish=True,
        enable_chat_answer_polish_debug=True,
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
            return_value=(
                MagicMock(
                    generate=MagicMock(
                        return_value=MagicMock(text="INV-1 overdue [Source 1].")
                    )
                ),
                "model",
            ),
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
        patch.object(
            AnswerIntegritySnapshot,
            "to_debug_dict",
            side_effect=RuntimeError("debug fault"),
        ),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await _post_chat(client, "What is overdue?")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    debug = body.get("debug_info") or {}
    snap = debug.get("answer_integrity_snapshot")
    assert snap is not None
    # Fallback snapshot should at least carry a minimal integration error signal.
    assert "integration_error" in snap
    # Answer text remains unchanged.
    assert body.get("answer") == "INV-1 overdue [Source 1]."
