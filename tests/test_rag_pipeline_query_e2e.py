"""RAGPipeline.query() end-to-end with mocked vectordb/embedder/llm."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.config.client_config_schema import ClientConfig
from app.core.embedders.base import EmbedderInfo
from app.core.llms.base import LLMInfo, LLMResponse
from app.core.rag_pipeline import RAGPipeline
from app.core.vectordb.base import VectorHit


def _minimal_config() -> ClientConfig:
    return ClientConfig.from_dict(
        {
            "client_id": "test-tenant",
            "vectordb": {
                "type": "chroma",
                "collection": "docs",
                "chroma": {"persist_directory": "./test_db"},
            },
            "embedder": {
                "type": "huggingface",
                "huggingface": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            },
            "llm": {
                "single": {
                    "type": "ollama",
                    "model": "llama3",
                    "base_url": "http://localhost:11434",
                }
            },
            "retrieval": {
                "top_k_retrieval": 5,
                "top_k_final": 3,
                "enable_trust_scoring": False,
            },
        }
    )


def test_rag_pipeline_query_returns_answer_with_mocked_stack() -> None:
    config = _minimal_config()

    mock_vdb = MagicMock()
    mock_vdb.kind = "chroma"
    mock_vdb.search.return_value = [
        VectorHit(
            id="chunk-1",
            text="Revenue grew 12% in Q3.",
            score=0.91,
            metadata={"business_id": "test-tenant", "chunk_id": "chunk-1"},
        )
    ]

    mock_embedder = MagicMock()
    mock_embedder.info = EmbedderInfo(
        provider="huggingface",
        model="all-MiniLM-L6-v2",
        dim=384,
    )
    mock_embedder.embed_query.return_value = [0.1] * 384

    mock_llm = MagicMock()
    mock_llm.info = LLMInfo(
        provider="ollama",
        model="llama3",
        supports_streaming=False,
    )
    mock_llm.generate.return_value = LLMResponse(
        text="Revenue grew 12% in Q3 according to the documents.",
        model="llama3",
        prompt_tokens=10,
        completion_tokens=12,
        total_tokens=22,
    )

    pipeline = RAGPipeline(
        vectordb=mock_vdb,
        embedder=mock_embedder,
        llm=mock_llm,
        reranker=None,
        config=config,
        nodes=None,
    )

    result = pipeline.query("What was Q3 revenue?")

    assert result.final_answer
    assert "12%" in result.final_answer or result.final_answer
    assert result.context_chunks
    mock_vdb.search.assert_called_once()
    mock_embedder.embed_query.assert_called()
    mock_llm.generate.assert_called_once()
