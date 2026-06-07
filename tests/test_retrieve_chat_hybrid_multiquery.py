"""
Focused tests for hybrid + multi-query chat retrieval helpers and behavior.
"""

from __future__ import annotations

import asyncio
import types
from contextlib import asynccontextmanager, contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v2.retrieve_chat_api import (
    ChatRetrieveRequest,
    ChatRetrieveResponse,
    _chat_runtime_max_results,
    _is_multi_query_enabled,
    _rebuild_ranked_from_fused_chunks,
    _resolve_search_params,
)
from app.auth.deps import get_current_user
from app.core.multi_query_expander import fuse_multi_query_results
from app.db.session_v2 import get_db
from app.main import app
from app.retrieval.components import RuntimeComponents
from app.retrieval.policy import TrustDecision
from app.retrieval.types_retrieve import RankedResult
from app.services.faithfulness_verifier import VerificationResult, VerificationStatus
from app.services.query_routing.types import QueryRoute, knowledge_decision


class DummyRuntimeComponents(types.SimpleNamespace):
    pass


def _make_basic_request(**overrides):
    base = dict(
        session_id="sess-1",
        messages=[{"role": "user", "content": "Hello"}],
        client_id="tenant-1",
    )
    base.update(overrides)
    return ChatRetrieveRequest(**base)


def _sample_ranked(
    chunk_id: str,
    *,
    score: float = 0.9,
    trust_state: str = "validated",
) -> RankedResult:
    return RankedResult(
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        score=score,
        explanation={"interpretation": {"trust_state": trust_state}},
        trust_decision=TrustDecision.TRUSTED,
        file_id="file-abc",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Unit: config defaults, gating, recall window, fusion rebuild
# ─────────────────────────────────────────────────────────────────────────────


class TestTenantConfigDefaults:
    def test_tenant_defaults_used_when_fields_absent(self):
        rc = DummyRuntimeComponents(search_mode="hybrid", top_k_final=13)
        req = _make_basic_request()

        search_mode, top_k = _resolve_search_params(
            req=req, rc=rc, runtime_mode_chat="authoritative_config"
        )

        assert search_mode == "hybrid"
        assert top_k == 13

    def test_explicit_request_overrides_tenant_defaults(self):
        rc = DummyRuntimeComponents(search_mode="semantic", top_k_final=7)
        req = _make_basic_request(search_mode="keyword", top_k=25)

        search_mode, top_k = _resolve_search_params(
            req=req, rc=rc, runtime_mode_chat="authoritative_config"
        )

        assert search_mode == "keyword"
        assert top_k == 25


class DummyRetrievalConfig:
    def __init__(self, enable_multi_query: bool, count: int):
        self.enable_multi_query = enable_multi_query
        self.multi_query_count = count


class TestMultiQueryGating:
    def test_multi_query_enabled_for_knowledge_route_with_config(self):
        cfg = DummyRetrievalConfig(enable_multi_query=True, count=3)
        assert _is_multi_query_enabled(
            route=QueryRoute.KNOWLEDGE,
            retrieval_cfg=cfg,
            has_tenant_pipeline=True,
        )

    def test_multi_query_disabled_for_structured_route_even_when_config_true(self):
        cfg = DummyRetrievalConfig(enable_multi_query=True, count=4)
        assert not _is_multi_query_enabled(
            route=QueryRoute.STRUCTURED,
            retrieval_cfg=cfg,
            has_tenant_pipeline=True,
        )

    def test_multi_query_disabled_when_pipeline_missing(self):
        cfg = DummyRetrievalConfig(enable_multi_query=True, count=3)
        assert not _is_multi_query_enabled(
            route=QueryRoute.KNOWLEDGE,
            retrieval_cfg=cfg,
            has_tenant_pipeline=False,
        )


class TestHybridRecallWindow:
    def test_hybrid_uses_route_recall_limit_not_final_top_k(self):
        recall = knowledge_decision(top_k=5).max_recall_candidates
        assert recall == 200
        assert _chat_runtime_max_results("hybrid", recall, effective_top_k=5) == 200

    def test_semantic_uses_effective_top_k(self):
        assert _chat_runtime_max_results("semantic", 200, effective_top_k=5) == 5


class TestFusionReconstruction:
    def test_rebuild_preserves_chunk_id_and_trust_metadata(self):
        r1 = _sample_ranked("chunk-a", trust_state="validated")
        r2 = _sample_ranked("chunk-b", trust_state="provisional")
        chunk_index = {r1.chunk_id: r1, r2.chunk_id: r2}

        mq = fuse_multi_query_results(
            [
                [{"id": "chunk-b", "text": r2.text, "score": 0.5}],
                [{"id": "chunk-a", "text": r1.text, "score": 0.8}],
            ],
            top_k=2,
        )

        rebuilt = _rebuild_ranked_from_fused_chunks(mq.fused_chunks, chunk_index)

        assert [r.chunk_id for r in rebuilt] == ["chunk-b", "chunk-a"]
        assert rebuilt[0].explanation == r2.explanation
        assert rebuilt[0].trust_decision == TrustDecision.TRUSTED
        assert rebuilt[0].file_id == "file-abc"
        assert rebuilt[1].chunk_id == "chunk-a"


class TestChatResponseSchema:
    def test_chat_response_schema_unchanged(self):
        response = ChatRetrieveResponse(
            session_id="test",
            query="test",
            intent="answer",
            search_mode="semantic",
            total_results=0,
            total_dropped=0,
            latency_ms=100.0,
            results=[],
        )
        for field in (
            "answer",
            "answer_model",
            "answer_latency_ms",
            "answer_error",
            "debug_info",
            "rewritten_query",
            "results",
        ):
            assert hasattr(response, field)


# ─────────────────────────────────────────────────────────────────────────────
# HTTP-level: hybrid recall, expansion fallback, trust gate
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def override_auth():
    app.dependency_overrides[get_current_user] = lambda: {
        "role": "admin",
        "sub": "test-admin",
    }
    yield
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture(autouse=True)
def override_db():
    mock_db = AsyncMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=None)

    async def _override_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture()
def hybrid_runtime() -> RuntimeComponents:
    return RuntimeComponents(
        runtime_mode="authoritative_config",
        config_fingerprint="test",
        client_id="pytest-tenant",
        embedder_type="huggingface",
        embedder_model="test-model",
        vectordb_type="chroma",
        collection="test_collection",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        llm_api_key_env=None,
        llm_base_url=None,
        reranker_name="none",
        reranker_model="",
        search_mode="semantic",
        enable_hyde=False,
        enable_reranking=False,
        hybrid_alpha=0.65,
        similarity_threshold=0.0,
        top_k_retrieval=20,
        top_k_final=5,
        rag_min_score=0.25,
        chunking_strategy="unknown",
    )


def _mock_retrieval_cfg(*, multi_query: bool = False, count: int = 3):
    cfg = MagicMock()
    cfg.client_id = "pytest-tenant"
    cfg.retrieval = MagicMock(
        enable_multi_query=multi_query,
        multi_query_count=count,
        hybrid_alpha=0.65,
    )
    cfg.is_reranking_enabled = MagicMock(return_value=False)
    return cfg


@contextmanager
def _chat_patches(
    hybrid_runtime: RuntimeComponents,
    *,
    multi_query: bool = False,
    retrieval_cfg: MagicMock | None = None,
):
    if retrieval_cfg is None:
        retrieval_cfg = _mock_retrieval_cfg(multi_query=multi_query)

    mock_route = knowledge_decision(top_k=5)
    mock_orch = MagicMock()
    mock_orch.route = AsyncMock(return_value=mock_route)

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
            return_value=(retrieval_cfg, "authoritative_config"),
        ),
        patch(
            "app.retrieval.components.resolve_runtime_components",
            return_value=hybrid_runtime,
        ),
        patch("app.api.v2.retrieve_chat_api.get_orchestrator", return_value=mock_orch),
        patch(
            "app.api.v2.retrieve_chat_api.maybe_start_rag_chat_trace",
            return_value=None,
        ),
    ):
        yield


@pytest.mark.anyio
async def test_hybrid_retrieve_uses_recall_limit_override(hybrid_runtime):
    """Hybrid mode must pass router recall window to RetrievalRuntime, not tiny top_k."""
    captured: dict = {}

    async def _capture_retrieve(*args, **kwargs):
        captured.update(kwargs)
        return ([_sample_ranked("c1")], [])

    mock_llm = MagicMock()
    mock_llm.generate.return_value = MagicMock(text="answer.", prompt_tokens=0, completion_tokens=0)

    with (
        _chat_patches(hybrid_runtime),
        patch(
            "app.api.v2.retrieve_chat_api._embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch(
            "app.api.v2.retrieve_chat_api._resolve_llm",
            return_value=(mock_llm, "test-model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_capture_retrieve,
        ),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/api/v2/retrieve/chat",
                json={
                    "messages": [{"role": "user", "content": "Explain our Q4 strategy"}],
                    "client_id": "pytest-tenant",
                    "search_mode": "hybrid",
                    "top_k": 5,
                    "enable_hyde": False,
                    "generate_answer": False,
                },
            )

    assert resp.status_code == 200
    assert captured.get("max_results_override") == 200
    assert captured.get("recall_limit_override") == 200


@pytest.mark.anyio
async def test_expansion_single_variant_skips_concurrent_fanout(hybrid_runtime):
    """Only original query → no asyncio.gather fan-out for variants."""
    gather_calls: list = []

    async def _tracking_gather(*args, **kwargs):
        gather_calls.append((args, kwargs))
        return await asyncio.gather(*args, **kwargs)

    retrieval_cfg = _mock_retrieval_cfg(multi_query=True, count=3)

    with (
        _chat_patches(hybrid_runtime, retrieval_cfg=retrieval_cfg),
        patch(
            "app.api.v2.retrieve_chat_api._embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch(
            "app.api.v2.retrieve_chat_api._resolve_llm",
            return_value=(MagicMock(), "test-model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.core.multi_query_expander.generate_query_variants",
            return_value=["Explain our Q4 strategy"],
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            return_value=([], []),
        ),
        patch(
            "app.api.v2.retrieve_chat_api.asyncio.gather",
            side_effect=_tracking_gather,
        ),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/api/v2/retrieve/chat",
                json={
                    "messages": [{"role": "user", "content": "Explain our Q4 strategy"}],
                    "client_id": "pytest-tenant",
                    "enable_hyde": False,
                    "generate_answer": False,
                },
            )

    assert resp.status_code == 200
    assert len(gather_calls) == 0


@pytest.mark.anyio
async def test_expansion_exception_falls_back_to_single_query(hybrid_runtime):
    """Expansion failure must not fail the request; no variant gather."""
    gather_calls: list = []

    async def _tracking_gather(*args, **kwargs):
        gather_calls.append(1)
        return await asyncio.gather(*args, **kwargs)

    retrieval_cfg = _mock_retrieval_cfg(multi_query=True, count=3)
    retrieve_calls = 0

    async def _count_retrieve(*args, **kwargs):
        nonlocal retrieve_calls
        retrieve_calls += 1
        return ([_sample_ranked("c1")], [])

    with (
        _chat_patches(hybrid_runtime, retrieval_cfg=retrieval_cfg),
        patch(
            "app.api.v2.retrieve_chat_api._embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch(
            "app.api.v2.retrieve_chat_api._resolve_llm",
            return_value=(MagicMock(), "test-model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.core.multi_query_expander.generate_query_variants",
            side_effect=RuntimeError("expansion timeout"),
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            side_effect=_count_retrieve,
        ),
        patch(
            "app.api.v2.retrieve_chat_api.asyncio.gather",
            side_effect=_tracking_gather,
        ),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/api/v2/retrieve/chat",
                json={
                    "messages": [{"role": "user", "content": "Explain our Q4 strategy"}],
                    "client_id": "pytest-tenant",
                    "enable_hyde": False,
                    "generate_answer": False,
                },
            )

    assert resp.status_code == 200
    assert len(gather_calls) == 0
    assert retrieve_calls == 1


@pytest.mark.anyio
async def test_trust_gate_runs_after_multi_query_retrieval(hybrid_runtime):
    """Faithfulness verifier must still run when multi-query fusion produced results."""
    verify_calls: list = []

    def _fake_verify(*, answer, source_texts, focus_id, route):
        verify_calls.append(
            {"answer": answer, "sources": source_texts, "route": route}
        )
        return answer, VerificationResult(
            status=VerificationStatus.PASS,
            checks=[],
            fail_reason=None,
        )

    mock_llm = MagicMock()
    mock_llm.generate.return_value = MagicMock(
        text="Grounded answer [Source 1].",
        prompt_tokens=1,
        completion_tokens=1,
    )

    ranked = [_sample_ranked("chunk-x", score=0.95)]
    retrieval_cfg = _mock_retrieval_cfg(multi_query=True, count=3)

    @asynccontextmanager
    async def _fake_get_async_session():
        yield AsyncMock()

    with (
        _chat_patches(hybrid_runtime, retrieval_cfg=retrieval_cfg),
        patch(
            "app.api.v2.retrieve_chat_api._embed_in_thread",
            new_callable=AsyncMock,
            return_value=[0.1] * 8,
        ),
        patch(
            "app.api.v2.retrieve_chat_api._resolve_llm",
            return_value=(mock_llm, "test-model"),
        ),
        patch(
            "app.services.ingestion.ingestion_service_v2.get_query_pipeline_for_client",
            return_value=MagicMock(vectordb=MagicMock()),
        ),
        patch(
            "app.core.multi_query_expander.generate_query_variants",
            return_value=["Explain our Q4 strategy", "Q4 business plan overview"],
        ),
        patch(
            "app.retrieval.runtime.RetrievalRuntime.retrieve",
            new_callable=AsyncMock,
            return_value=(ranked, []),
        ),
        patch(
            "app.api.v2.retrieve_chat_api.get_async_session",
            _fake_get_async_session,
        ),
        patch(
            "app.api.v2.retrieve_chat_api.verify_or_refuse",
            side_effect=_fake_verify,
        ),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/api/v2/retrieve/chat",
                json={
                    "messages": [{"role": "user", "content": "Explain our Q4 strategy"}],
                    "client_id": "pytest-tenant",
                    "enable_hyde": False,
                    "generate_answer": True,
                },
            )

    assert resp.status_code == 200
    assert len(verify_calls) == 1
    assert verify_calls[0]["route"] == QueryRoute.KNOWLEDGE.value
    assert verify_calls[0]["sources"]
    assert "chunk-x" in verify_calls[0]["sources"][0]
