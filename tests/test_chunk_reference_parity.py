"""ChunkReference parity across rag_api and retrieve/chat paths."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v2.rag_api import router as rag_router, _build_retrieved_chunks_for_eval
from app.auth.deps import get_current_user
from app.retrieval.types_retrieve import (
    ChunkReference,
    chunk_ref_from_postgres,
    chunk_ref_from_vector_payload,
)


def test_chunk_reference_contract_postgres_vs_vector_payload() -> None:
    postgres_id = "550e8400-e29b-41d4-a716-446655440000"
    pg_ref = chunk_ref_from_postgres(postgres_id)
    assert pg_ref == ChunkReference(chunk_id=postgres_id, source="postgres")

    vector_item = {
        "metadata": {"chunk_id": postgres_id},
        "text": "sample",
        "score": 0.9,
    }
    vec_ref = chunk_ref_from_vector_payload(vector_item)
    assert vec_ref.chunk_id == postgres_id
    assert vec_ref.source == "vector_payload"

    # When metadata carries the canonical postgres UUID, both paths align.
    assert pg_ref.chunk_id == vec_ref.chunk_id


def test_build_retrieved_chunks_for_eval_uses_vector_payload_source() -> None:
    chunk_id = "550e8400-e29b-41d4-a716-446655440001"
    result = MagicMock()
    result.reranked = False
    result.retrieved_chunks = [
        {"metadata": {"chunk_id": chunk_id}, "text": "hello", "score": 0.88}
    ]
    result.reranked_chunks = []

    items = _build_retrieved_chunks_for_eval(result)
    assert len(items) == 1
    assert items[0].chunk_id == chunk_id
    assert items[0].chunk_source == "vector_payload"


@pytest.fixture()
def rag_app() -> FastAPI:
    app = FastAPI()
    app.include_router(rag_router, prefix="/api/v2/rag")

    def _admin():
        return {"role": "admin", "sub": "admin-1"}

    app.dependency_overrides[get_current_user] = _admin
    yield app
    app.dependency_overrides.clear()


def test_rag_query_http_includes_chunk_source(rag_app: FastAPI) -> None:
    chunk_id = "550e8400-e29b-41d4-a716-446655440002"
    mock_result = MagicMock()
    mock_result.reranked = False
    mock_result.retrieved_chunks = [
        {"metadata": {"chunk_id": chunk_id}, "text": "parity", "score": 0.77}
    ]
    mock_result.reranked_chunks = []
    mock_result.context_chunks = mock_result.retrieved_chunks
    mock_result.final_answer = "answer"
    mock_result.trust_score = 0.9
    mock_result.latency = {"total_ms": 1}
    mock_result.metadata = {}

    mock_pipeline = MagicMock()
    mock_pipeline.query.return_value = mock_result
    mock_pipeline.config = MagicMock()

    with patch("app.api.v2.rag_api.pipeline_factory") as mock_factory:
        mock_factory.get_cached.return_value = mock_pipeline
        client = TestClient(rag_app, raise_server_exceptions=True)
        r = client.post(
            "/api/v2/rag/query",
            json={"client_id": "default", "query": "hello"},
        )

    assert r.status_code == 200, r.text
    chunks = r.json().get("retrieved_chunks") or []
    assert chunks
    assert chunks[0]["chunk_source"] == "vector_payload"
    assert chunks[0]["chunk_id"] == chunk_id
