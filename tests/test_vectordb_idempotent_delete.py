"""
Tests for idempotent vector compensation deletes.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from app.services.ingestion.ingestion_service_v2 import IngestionServiceV2


def _run(coro):
    return asyncio.run(coro)


def test_compensate_vectors_safe_ignore_missing() -> None:
    pipeline = MagicMock()
    pipeline.config.vectordb.collection = "test_collection"
    vectordb = MagicMock()
    vectordb.delete_many = MagicMock(return_value=0)
    pipeline.vectordb = vectordb
    ids = ["hash_a", "hash_b"]

    async def _test():
        await IngestionServiceV2._compensate_vectors_safe(ids, pipeline)
        await IngestionServiceV2._compensate_vectors_safe(ids, pipeline)

    _run(_test())
    assert vectordb.delete_many.call_count == 2
    for call in vectordb.delete_many.call_args_list:
        assert call.kwargs["collection"] == "test_collection"
        assert call.kwargs["doc_ids"] == ids
