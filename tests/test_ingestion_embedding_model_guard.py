"""Tests for embedding model consistency guard in ingestion pipeline."""

import pytest

from app.services.ingestion.ingestion_service_v2 import IngestionServiceV2


def test_embedding_model_guard_fills_missing_model():
    chunks = [
        {"semantic_hash": "a1", "cleaned_text": "hello"},
        {"semantic_hash": "a2", "embedding_model": "qwen3-embedding:8b"},
    ]

    IngestionServiceV2._assert_chunk_embedding_model(
        chunks=chunks,
        expected_model="qwen3-embedding:8b",
        file_id="file-1",
    )

    assert chunks[0]["embedding_model"] == "qwen3-embedding:8b"


def test_embedding_model_guard_raises_on_mismatch():
    chunks = [
        {"semantic_hash": "a1", "embedding_model": "wrong-model"},
    ]

    with pytest.raises(ValueError, match="Embedding model mismatch"):
        IngestionServiceV2._assert_chunk_embedding_model(
            chunks=chunks,
            expected_model="qwen3-embedding:8b",
            file_id="file-1",
        )
