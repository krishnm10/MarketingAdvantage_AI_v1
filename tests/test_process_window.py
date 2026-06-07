"""
Unit tests for streaming ingestion window processing.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.services.ingestion import ingestion_service_v2 as isv2
from app.services.ingestion.streaming_state import StreamingIngestionState


def _run(coro):
    return asyncio.run(coro)


def _file_record(*, file_type: str = "pdf", file_path: str = "/tmp/test.pdf"):
    return SimpleNamespace(
        id=uuid4(),
        business_id=uuid4(),
        file_type=file_type,
        file_path=file_path,
        file_name="test.pdf",
        source_url=None,
    )


def _ready_chunk(i: int) -> dict:
    return {
        "chunk_index": i,
        "text": f"t{i}",
        "cleaned_text": f"t{i}",
        "semantic_hash": f"hash_{i}",
        "embedding_model": "m",
        "page_number": i // 10,
        "char_offset_start": 0,
        "char_offset_end": 10,
        "is_visual": False,
        "status": "ready",
        "metadata": {},
    }


async def _chunk_iter_260():
    for i in range(260):
        yield _ready_chunk(i)


@pytest.fixture(autouse=True)
def _patch_streaming_flag(monkeypatch):
    monkeypatch.setattr(isv2, "STREAMING_INGESTION_ENABLED", True)


def test_end_of_stream_partial_window_is_flushed() -> None:
    record = _file_record()
    mock_db = AsyncMock()

    async def _test():
        with patch.object(
            isv2.IngestionServiceV2,
            "_set_file_processing",
            new_callable=AsyncMock,
        ), patch.object(
            isv2.IngestionServiceV2,
            "_process_window",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_pw, patch.object(
            isv2.StreamingIngestionState,
            "from_checkpoint",
            new_callable=AsyncMock,
        ) as mock_fc, patch.object(
            isv2,
            "_get_ingestion_pipeline",
        ) as mock_pipe, patch.object(
            isv2.IngestionServiceV2,
            "_update_file_status",
            new_callable=AsyncMock,
        ), patch(
            "app.services.ingestion.ingestion_service_v2.iter_pdf_pages",
        ), patch(
            "app.services.ingestion.chunk_stream.iter_chunks",
            return_value=_chunk_iter_260(),
        ):
            pipeline = MagicMock()
            pipeline.embedder.info.model = "test-model"
            pipeline.config.ingestion.batch_size = 256
            pipeline.vectordb.kind = "chroma"
            mock_pipe.return_value = pipeline
            mock_fc.return_value = StreamingIngestionState(
                file_id=record.id,
                business_id=record.business_id,
            )

            session = AsyncMock()
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock(return_value=None)
            session.commit = AsyncMock()

            with patch.object(
                isv2.IngestionServiceV2,
                "_async_session_factory",
                return_value=session,
            ):
                await isv2.IngestionServiceV2._run_pipeline_streaming(
                    mock_db,
                    record,
                    {"file_path": record.file_path},
                    pre_embed_hook=lambda chunks: chunks,
                )

        assert mock_pw.await_count == 2

    _run(_test())


def test_fresh_session_per_window() -> None:
    record = _file_record()
    mock_db = AsyncMock()
    sessions_created: list = []

    def _session_factory():
        session = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.commit = AsyncMock()
        sessions_created.append(session)
        return session

    async def _three_chunks():
        for i in range(3):
            yield _ready_chunk(i)

    async def _test():
        with patch.object(
            isv2.IngestionServiceV2,
            "_set_file_processing",
            new_callable=AsyncMock,
        ), patch.object(
            isv2.IngestionServiceV2,
            "_process_window",
            new_callable=AsyncMock,
            return_value=[],
        ), patch.object(
            isv2.StreamingIngestionState,
            "from_checkpoint",
            new_callable=AsyncMock,
        ) as mock_fc, patch.object(
            isv2,
            "_get_ingestion_pipeline",
        ) as mock_pipe, patch.object(
            isv2.IngestionServiceV2,
            "_update_file_status",
            new_callable=AsyncMock,
        ), patch(
            "app.services.ingestion.ingestion_service_v2.iter_pdf_pages",
        ), patch(
            "app.services.ingestion.chunk_stream.iter_chunks",
            return_value=_three_chunks(),
        ):
            pipeline = MagicMock()
            pipeline.embedder.info.model = "test-model"
            pipeline.config.ingestion.batch_size = 1
            pipeline.vectordb.kind = "chroma"
            mock_pipe.return_value = pipeline
            mock_fc.return_value = StreamingIngestionState(
                file_id=record.id,
                business_id=record.business_id,
            )

            with patch.object(
                isv2.IngestionServiceV2,
                "_async_session_factory",
                side_effect=_session_factory,
            ):
                await isv2.IngestionServiceV2._run_pipeline_streaming(
                    mock_db,
                    record,
                    {"file_path": record.file_path},
                    pre_embed_hook=lambda chunks: chunks,
                )

        assert len(sessions_created) == 3

    _run(_test())


def test_batch_gci_lookup_splits_large_lists() -> None:
    hashes = [f"h{i}" for i in range(1200)]
    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = []
    mock_db.execute = AsyncMock(return_value=mock_result)
    business_id = uuid4()

    async def _test():
        await isv2.IngestionServiceV2._batch_gci_lookup(
            mock_db, hashes, business_id, batch_size=500
        )

    _run(_test())
    assert mock_db.execute.await_count == 3


def test_visual_placeholder_resolved_before_embedding() -> None:
    state = StreamingIngestionState(file_id=uuid4(), business_id=uuid4())
    window_db = AsyncMock()
    record = _file_record()
    pipeline = MagicMock()
    pipeline.config.vectordb.collection = "ingested_content"

    async def _test():
        async def _resolve():
            return [
                {
                    "chunk_index": 99,
                    "text": "v1",
                    "cleaned_text": "v1",
                    "semantic_hash": "vh1",
                    "embedding_model": "m",
                    "page_number": 1,
                    "char_offset_start": 0,
                    "char_offset_end": 5,
                    "is_visual": True,
                    "status": "ready",
                    "metadata": {},
                },
                {
                    "chunk_index": 100,
                    "text": "v2",
                    "cleaned_text": "v2",
                    "semantic_hash": "vh2",
                    "embedding_model": "m",
                    "page_number": 1,
                    "char_offset_start": 5,
                    "char_offset_end": 10,
                    "is_visual": True,
                    "status": "ready",
                    "metadata": {},
                },
            ]

        task = asyncio.create_task(_resolve())
        window = [
            {
                "chunk_index": 0,
                "text": "",
                "cleaned_text": "",
                "semantic_hash": "__visual_pending__",
                "embedding_model": "m",
                "page_number": 1,
                "char_offset_start": 0,
                "char_offset_end": 10,
                "is_visual": True,
                "status": "visual_pending",
                "metadata": {},
                "_visual_task": task,
            }
        ]

        with patch.object(
            isv2.IngestionServiceV2,
            "_dedup_chunks",
            new_callable=AsyncMock,
            return_value=([], {"total": 0, "unique": 0, "duplicates": 0, "dedup_ratio": 0.0}),
        ) as mock_dedup, patch.object(
            isv2.IngestionServiceV2,
            "_insert_chunks",
            new_callable=AsyncMock,
        ), patch.object(
            isv2.IngestionServiceV2,
            "embed_and_store",
            new_callable=AsyncMock,
        ), patch.object(
            isv2,
            "register_unique_chunks_in_gci",
            new_callable=AsyncMock,
        ):
            await isv2.IngestionServiceV2._process_window(
                window,
                state,
                window_db,
                pipeline=pipeline,
                file_record=record,
                file_id=record.id,
                business_id=record.business_id,
                file_type="pdf",
                embedding_model="m",
                pre_embed_hook=lambda c: c,
            )

        mock_dedup.assert_awaited_once()
        dedup_arg_chunks = mock_dedup.await_args[0][1]
        assert len(dedup_arg_chunks) == 2
        assert all(c.get("status") == "ready" for c in dedup_arg_chunks)
        assert not any(c.get("status") == "visual_pending" for c in dedup_arg_chunks)

    _run(_test())


def test_checkpoint_updated_after_successful_window() -> None:
    state = StreamingIngestionState(file_id=uuid4(), business_id=uuid4())
    window_db = AsyncMock()
    record = _file_record()
    pipeline = MagicMock()
    pipeline.config.vectordb.collection = "ingested_content"
    window = [_ready_chunk(5)]

    async def _test():
        with patch.object(
            isv2.IngestionServiceV2,
            "_dedup_chunks",
            new_callable=AsyncMock,
            return_value=(
                [window[0]],
                {"total": 1, "unique": 1, "duplicates": 0, "dedup_ratio": 0.0},
            ),
        ), patch.object(
            isv2.IngestionServiceV2,
            "_insert_chunks",
            new_callable=AsyncMock,
        ), patch.object(
            isv2.IngestionServiceV2,
            "embed_and_store",
            new_callable=AsyncMock,
        ), patch.object(
            isv2,
            "register_unique_chunks_in_gci",
            new_callable=AsyncMock,
        ), patch.object(
            state,
            "persist_checkpoint",
            new_callable=AsyncMock,
        ) as mock_persist:
            await isv2.IngestionServiceV2._process_window(
                window,
                state,
                window_db,
                pipeline=pipeline,
                file_record=record,
                file_id=record.id,
                business_id=record.business_id,
                file_type="pdf",
                embedding_model="m",
                pre_embed_hook=lambda c: c,
            )
            mock_persist.assert_awaited_once_with(window_db)
            assert state.last_processed_chunk_index >= 5
            assert state.last_processed_page >= 0

    _run(_test())


def test_db_commit_failure_triggers_vector_compensation() -> None:
    record = _file_record()
    mock_db = AsyncMock()

    async def _one_chunk():
        yield _ready_chunk(0)

    async def _test():
        with patch.object(
            isv2.IngestionServiceV2,
            "_set_file_processing",
            new_callable=AsyncMock,
        ), patch.object(
            isv2.IngestionServiceV2,
            "_process_window",
            new_callable=AsyncMock,
            return_value=["hash_0"],
        ), patch.object(
            isv2.IngestionServiceV2,
            "_compensate_vectors_safe",
            new_callable=AsyncMock,
        ) as mock_comp, patch.object(
            isv2.StreamingIngestionState,
            "from_checkpoint",
            new_callable=AsyncMock,
        ) as mock_fc, patch.object(
            isv2,
            "_get_ingestion_pipeline",
        ) as mock_pipe, patch(
            "app.services.ingestion.ingestion_service_v2.iter_pdf_pages",
        ), patch(
            "app.services.ingestion.chunk_stream.iter_chunks",
            return_value=_one_chunk(),
        ):
            pipeline = MagicMock()
            pipeline.embedder.info.model = "test-model"
            pipeline.config.ingestion.batch_size = 256
            pipeline.vectordb.kind = "chroma"
            mock_pipe.return_value = pipeline
            mock_fc.return_value = StreamingIngestionState(
                file_id=record.id,
                business_id=record.business_id,
            )

            session = AsyncMock()
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock(return_value=None)
            session.commit = AsyncMock(side_effect=RuntimeError("commit failed"))
            session.rollback = AsyncMock()

            with patch.object(
                isv2.IngestionServiceV2,
                "_async_session_factory",
                return_value=session,
            ):
                with pytest.raises(RuntimeError, match="commit failed"):
                    await isv2.IngestionServiceV2._run_pipeline_streaming(
                        mock_db,
                        record,
                        {"file_path": record.file_path},
                        pre_embed_hook=lambda chunks: chunks,
                    )

            session.rollback.assert_awaited()
            mock_comp.assert_awaited_once()
            assert mock_comp.await_args[0][0] == ["hash_0"]

    _run(_test())
