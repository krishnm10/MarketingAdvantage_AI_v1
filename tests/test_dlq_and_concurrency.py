"""
Tests for durable ingestion DLQ and orchestrator ingest concurrency.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.services.ingestion.ingestion_dlq_service import (
    _sanitize_payload,
    record_ingestion_failure,
)
from app.services.ingestion.ingestion_failure_context import (
    IngestionFailureContext,
    pop_ingestion_failure_context,
    set_ingestion_failure_context,
)
from app.services.ingestion.ingestion_orchestrator import IngestionOrchestrator
from app.services.ingestion.ingestion_worker import (
    IngestionCommand,
    IngestionWorker,
    InMemoryIngestionQueue,
    _MAX_RETRY_ATTEMPTS,
)


def _run(coro):
    return asyncio.run(coro)


def test_sanitize_payload_strips_private_keys_and_tasks() -> None:
    async def _make_payload():
        task = asyncio.create_task(asyncio.sleep(60))
        return {
            "file_id": str(uuid4()),
            "_visual_task": task,
            "nested": {"ok": 1, "_hidden": "x"},
        }

    payload = _run(_make_payload())
    safe = _sanitize_payload(payload)
    assert "_visual_task" not in safe
    assert "nested" in safe
    assert "_hidden" not in safe.get("nested", {})


def test_record_ingestion_failure_persists_row() -> None:
    mock_session = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.close = AsyncMock()
    mock_session.add = MagicMock()

    biz = uuid4()
    fid = uuid4()

    async def _assign_id_on_flush():
        obj = mock_session.add.call_args[0][0]
        if obj.id is None:
            obj.id = uuid4()

    mock_session.flush = AsyncMock(side_effect=_assign_id_on_flush)

    async def _test():
        with patch(
            "app.services.ingestion.ingestion_dlq_service._dlq_session_factory",
            return_value=mock_session,
        ):
            row_id = await record_ingestion_failure(
                business_id=biz,
                file_id=fid,
                stage="worker.ingest_file",
                error=RuntimeError("boom"),
                payload_snapshot={"file_id": str(fid)},
                retry_count=3,
                window_index=None,
            )
        assert row_id is not None
        mock_session.add.assert_called_once()
        added = mock_session.add.call_args[0][0]
        assert added.business_id == biz
        assert added.file_id == fid
        assert added.stage == "worker.ingest_file"
        assert added.retry_count == 3
        mock_session.flush.assert_awaited_once()
        mock_session.commit.assert_awaited_once()

    _run(_test())


def test_worker_records_dlq_on_terminal_failure() -> None:
    queue = InMemoryIngestionQueue()
    worker = IngestionWorker(queue=queue)
    command = IngestionCommand(
        file_id=str(uuid4()),
        business_id=str(uuid4()),
        attempt=_MAX_RETRY_ATTEMPTS,
    )

    async def _test():
        with patch(
            "app.services.ingestion.tenant_guard.resolve_ingestion_tenant",
            return_value=SimpleNamespace(tenant_id=command.business_id),
        ), patch(
            "app.core.config.pipeline_runtime.get_pipeline_identity",
            return_value={"embedder": "x", "vectordb": "y"},
        ), patch.object(
            IngestionOrchestrator,
            "ingest_file",
            new_callable=AsyncMock,
            side_effect=RuntimeError("ingest failed"),
        ), patch(
            "app.services.ingestion.ingestion_dlq_service.record_ingestion_failure",
            new_callable=AsyncMock,
        ) as mock_dlq:
            await worker._process_command(command, worker_id=0)

        mock_dlq.assert_awaited_once()
        kwargs = mock_dlq.await_args.kwargs
        assert kwargs["file_id"] == command.file_id
        assert kwargs["business_id"] == command.business_id
        assert kwargs["stage"] == "worker.ingest_file"
        assert kwargs["retry_count"] == command.attempt

    _run(_test())


def test_worker_records_streaming_window_context_in_dlq() -> None:
    queue = InMemoryIngestionQueue()
    worker = IngestionWorker(queue=queue)
    command = IngestionCommand(
        file_id=str(uuid4()),
        business_id=str(uuid4()),
        attempt=_MAX_RETRY_ATTEMPTS,
    )

    async def _test():
        set_ingestion_failure_context(
            IngestionFailureContext(
                stage="streaming_window_commit",
                window_index=7,
                pipeline_mode="streaming",
            )
        )
        with patch(
            "app.services.ingestion.tenant_guard.resolve_ingestion_tenant",
            return_value=SimpleNamespace(tenant_id=command.business_id),
        ), patch(
            "app.core.config.pipeline_runtime.get_pipeline_identity",
            return_value={},
        ), patch.object(
            IngestionOrchestrator,
            "ingest_file",
            new_callable=AsyncMock,
            side_effect=ValueError("window commit failed"),
        ), patch(
            "app.services.ingestion.ingestion_dlq_service.record_ingestion_failure",
            new_callable=AsyncMock,
        ) as mock_dlq:
            await worker._process_command(command, worker_id=0)

        kwargs = mock_dlq.await_args.kwargs
        assert kwargs["stage"] == "streaming_window_commit"
        assert kwargs["window_index"] == 7

    _run(_test())
    assert pop_ingestion_failure_context() is None


def test_stats_endpoint_reads_dlq_from_database() -> None:
    async def _test():
        with patch(
            "app.services.ingestion.ingestion_dlq_service.count_dlq_entries",
            new_callable=AsyncMock,
            return_value=2,
        ), patch(
            "app.services.ingestion.ingestion_dlq_service.fetch_recent_dlq_entries",
            new_callable=AsyncMock,
            return_value=[{"id": "dlq-1"}],
        ), patch(
            "app.services.ingestion.ingestion_worker.get_ingestion_queue",
        ) as mock_queue_fn, patch(
            "app.services.ingestion.ingestion_worker.get_ingestion_worker",
        ) as mock_worker_fn:
            mock_queue_fn.return_value = SimpleNamespace(
                stats={"enqueued": 1, "processed": 0, "failed": 0, "pending": 0}
            )
            mock_worker_fn.return_value = SimpleNamespace(_running=True)

            from app.main import ingestion_worker_stats

            result = await ingestion_worker_stats()

        assert result["dead_letter"] == [{"id": "dlq-1"}]
        assert result["queue"]["dlq_total"] == 2
        assert "dead_letter" not in mock_queue_fn.return_value.stats

    _run(_test())


def test_ingest_file_semaphore_limits_concurrency() -> None:
    IngestionOrchestrator._tenant_ingest_semaphores.clear()
    concurrent = 0
    max_seen = 0
    gate = asyncio.Event()
    gate.set()

    async def slow_process_file(**kwargs):
        nonlocal concurrent, max_seen
        concurrent += 1
        max_seen = max(max_seen, concurrent)
        await gate.wait()
        concurrent -= 1

    config = SimpleNamespace(
        client_id="tenant-a",
        embedder=SimpleNamespace(type=SimpleNamespace(value="huggingface")),
        security=SimpleNamespace(data_sensitivity="low"),
        features=SimpleNamespace(max_concurrent_ingestions=2),
    )

    async def _test():
        with patch(
            "app.services.ingestion.ingestion_orchestrator._resolve_config",
            return_value=config,
        ), patch(
            "app.services.ingestion.ingestion_orchestrator._log_resolved_config",
        ), patch.object(
            IngestionOrchestrator,
            "_enforce_embedding_policy",
        ), patch.object(
            IngestionOrchestrator,
            "_make_pii_hook",
            return_value=lambda chunks: chunks,
        ), patch(
            "app.services.ingestion.ingestion_service_v2.IngestionServiceV2.process_file",
            side_effect=slow_process_file,
        ):
            orch = IngestionOrchestrator()
            tasks = [
                asyncio.create_task(
                    orch.ingest_file(file_id=f"f{i}", client_id="tenant-a")
                )
                for i in range(4)
            ]
            await asyncio.sleep(0.05)
            assert max_seen <= 2
            gate.clear()
            await asyncio.gather(*tasks)

    _run(_test())


def test_ingest_file_semaphore_released_on_exception() -> None:
    IngestionOrchestrator._tenant_ingest_semaphores.clear()
    calls = {"n": 0}

    config = SimpleNamespace(
        client_id="tenant-b",
        embedder=SimpleNamespace(type=SimpleNamespace(value="huggingface")),
        security=SimpleNamespace(data_sensitivity="low"),
        features=SimpleNamespace(max_concurrent_ingestions=1),
    )

    async def fail_process(**kwargs):
        calls["n"] += 1
        raise RuntimeError("fail once")

    async def ok_process(**kwargs):
        calls["n"] += 1

    async def _test():
        with patch(
            "app.services.ingestion.ingestion_orchestrator._resolve_config",
            return_value=config,
        ), patch(
            "app.services.ingestion.ingestion_orchestrator._log_resolved_config",
        ), patch.object(
            IngestionOrchestrator,
            "_enforce_embedding_policy",
        ), patch.object(
            IngestionOrchestrator,
            "_make_pii_hook",
            return_value=lambda chunks: chunks,
        ), patch(
            "app.services.ingestion.ingestion_service_v2.IngestionServiceV2.process_file",
            side_effect=fail_process,
        ):
            orch = IngestionOrchestrator()
            with pytest.raises(RuntimeError):
                await orch.ingest_file(file_id="f1", client_id="tenant-b")

        with patch.object(
            IngestionOrchestrator,
            "_make_pii_hook",
            return_value=lambda chunks: chunks,
        ), patch(
            "app.services.ingestion.ingestion_service_v2.IngestionServiceV2.process_file",
            side_effect=ok_process,
        ):
            orch = IngestionOrchestrator()
            await orch.ingest_file(file_id="f2", client_id="tenant-b")

        assert calls["n"] == 2

    _run(_test())

