"""
End-to-end integration tests for durable ingestion DLQ and ingest concurrency.

Requires Postgres with the ``ingestion_dlq`` table (``alembic upgrade head``).
Skipped automatically when the database is unavailable.

Run:
    pytest tests/test_dlq_and_concurrency_e2e.py -m integration -v
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.db.models.ingestion_dlq import IngestionDlq
from app.db.session_v2 import DATABASE_URL
from app.services.ingestion import ingestion_service_v2 as isv2
from app.services.ingestion.ingestion_dlq_service import (
    _dlq_session_factory,
    record_ingestion_failure,
)
from app.services.ingestion.ingestion_failure_context import (
    pop_ingestion_failure_context,
)
from app.services.ingestion.ingestion_orchestrator import IngestionOrchestrator
from app.services.ingestion.ingestion_worker import (
    IngestionCommand,
    IngestionWorker,
    InMemoryIngestionQueue,
    _MAX_RETRY_ATTEMPTS,
)
from app.services.ingestion.streaming_state import StreamingIngestionState

# Controlled delay injected into ingest body for timing assertions (seconds).
_E2E_INGEST_HOLD_S = 0.4
_E2E_TIMING_TOLERANCE = 0.65


def _run(coro):
    return asyncio.run(coro)


def _loop_local_dlq_resources() -> tuple[async_sessionmaker[AsyncSession], Any]:
    """
    Engine/session factory bound to the current asyncio.run() loop.

    Avoids asyncpg 'send' errors from reusing the app-wide engine across loops.
    """
    engine = create_async_engine(DATABASE_URL, poolclass=NullPool, echo=False)
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    return factory, engine


async def _postgres_dlq_available() -> bool:
    try:
        from sqlalchemy import text

        from app.db.session_v2 import async_engine

        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            result = await conn.execute(
                text(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name = 'ingestion_dlq'"
                )
            )
            return result.scalar() is not None
    except Exception:
        return False


@pytest.fixture(scope="session")
def postgres_dlq_available() -> None:
    if not _run(_postgres_dlq_available()):
        pytest.skip(
            "Postgres with ingestion_dlq table not available "
            "(run alembic upgrade head and set DATABASE_URL)"
        )


@pytest.fixture
def dlq_row_ids() -> List[UUID]:
    """Collect DLQ row ids for teardown."""
    return []


@pytest.fixture(autouse=True)
def _clear_orchestrator_semaphores() -> None:
    IngestionOrchestrator._tenant_ingest_semaphores.clear()
    yield
    IngestionOrchestrator._tenant_ingest_semaphores.clear()


async def _fetch_dlq_rows(
    *,
    business_id: UUID,
    file_id: UUID,
    session_factory: Optional[async_sessionmaker[AsyncSession]] = None,
) -> List[IngestionDlq]:
    factory = session_factory or _dlq_session_factory
    async with factory() as session:
        result = await session.execute(
            select(IngestionDlq)
            .where(
                IngestionDlq.business_id == business_id,
                IngestionDlq.file_id == file_id,
            )
            .order_by(IngestionDlq.created_at.desc())
        )
        return list(result.scalars().all())


async def _delete_dlq_rows(
    row_ids: List[UUID],
    *,
    session_factory: Optional[async_sessionmaker[AsyncSession]] = None,
) -> None:
    if not row_ids:
        return
    factory = session_factory or _dlq_session_factory
    async with factory() as session:
        await session.execute(delete(IngestionDlq).where(IngestionDlq.id.in_(row_ids)))
        await session.commit()


@pytest.fixture
def dlq_cleanup(dlq_row_ids: List[UUID]) -> List[UUID]:
    yield dlq_row_ids
    factory, engine = _loop_local_dlq_resources()

    async def _cleanup() -> None:
        try:
            await _delete_dlq_rows(dlq_row_ids, session_factory=factory)
        finally:
            await engine.dispose()

    _run(_cleanup())


def _e2e_orchestrator_config(
    *,
    client_id: str,
    max_concurrent_ingestions: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        client_id=client_id,
        embedder=SimpleNamespace(type=SimpleNamespace(value="huggingface")),
        security=SimpleNamespace(data_sensitivity="low"),
        features=SimpleNamespace(
            max_concurrent_ingestions=max_concurrent_ingestions,
        ),
    )


def _track_dlq_ids(dlq_row_ids: List[UUID], rows: List[IngestionDlq]) -> None:
    for row in rows:
        if row.id and row.id not in dlq_row_ids:
            dlq_row_ids.append(row.id)


async def _one_ready_chunk():
    yield {
        "chunk_index": 0,
        "text": "t0",
        "cleaned_text": "t0",
        "semantic_hash": "hash_0",
        "embedding_model": "m",
        "page_number": 0,
        "char_offset_start": 0,
        "char_offset_end": 10,
        "is_visual": False,
        "status": "ready",
        "metadata": {},
    }


async def _run_streaming_window_commit_failure(
    *,
    business_id: UUID,
    file_id: UUID,
    window_index: int = 3,
    order: Optional[List[str]] = None,
) -> None:
    """Drive the real streaming flush failure path (compensation + context)."""
    record = SimpleNamespace(
        id=file_id,
        business_id=business_id,
        file_type="pdf",
        file_path="/tmp/e2e-test.pdf",
        file_name="e2e.pdf",
        source_url=None,
    )
    mock_db = AsyncMock()

    async def _one_chunk():
        async for chunk in _one_ready_chunk():
            yield chunk

    async def _compensate_with_optional_trace(
        vector_ids: List[str], pipeline: Any
    ) -> None:
        if order is not None:
            order.append("compensate")

    with patch.object(
        isv2.IngestionServiceV2,
        "_set_file_processing",
        new_callable=AsyncMock,
    ), patch.object(
        isv2.IngestionServiceV2,
        "_update_file_status",
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
        side_effect=_compensate_with_optional_trace,
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
        side_effect=lambda *a, **k: _one_chunk(),
    ):
        pipeline = MagicMock()
        pipeline.embedder.info.model = "test-model"
        pipeline.config.ingestion.batch_size = 256
        pipeline.vectordb.kind = "chroma"
        mock_pipe.return_value = pipeline
        mock_fc.return_value = StreamingIngestionState(
            file_id=file_id,
            business_id=business_id,
            current_window_index=window_index,
        )

        session = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.commit = AsyncMock(
            side_effect=RuntimeError("e2e streaming window commit failed"),
        )
        session.rollback = AsyncMock()

        with patch.object(
            isv2.IngestionServiceV2,
            "_async_session_factory",
            return_value=session,
        ):
            try:
                await isv2.IngestionServiceV2._run_pipeline_streaming(
                    mock_db,
                    record,
                    {"file_path": record.file_path},
                    pre_embed_hook=lambda chunks: chunks,
                )
            except RuntimeError as err:
                if "commit failed" not in str(err):
                    raise
                session.rollback.assert_awaited()
                mock_comp.assert_awaited_once()
                raise
            else:
                raise AssertionError(
                    "expected streaming window commit failure"
                )


# ─────────────────────────────────────────────────────────────────────────────
# 1. DLQ persistence E2E (worker → Postgres)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_e2e_worker_terminal_failure_persists_dlq_row(
    postgres_dlq_available: None,
    dlq_cleanup: List[UUID],
) -> None:
    business_id = uuid4()
    file_id = uuid4()
    command = IngestionCommand(
        file_id=str(file_id),
        business_id=str(business_id),
        attempt=_MAX_RETRY_ATTEMPTS,
    )
    queue = InMemoryIngestionQueue()
    worker = IngestionWorker(queue=queue)

    async def _test():
        factory, engine = _loop_local_dlq_resources()
        try:
            with patch(
                "app.services.ingestion.ingestion_dlq_service._dlq_session_factory",
                factory,
            ), patch.object(
                IngestionOrchestrator,
                "ingest_file",
                new_callable=AsyncMock,
                side_effect=RuntimeError("e2e terminal ingest failure"),
            ):
                await worker._process_command(command, worker_id=0)

            rows = await _fetch_dlq_rows(
                business_id=business_id,
                file_id=file_id,
                session_factory=factory,
            )
        finally:
            await engine.dispose()
        assert len(rows) >= 1
        row = rows[0]
        _track_dlq_ids(dlq_cleanup, rows)

        assert row.business_id == business_id
        assert row.file_id == file_id
        assert row.stage == "worker.ingest_file"
        assert row.error_type == "RuntimeError"
        assert "e2e terminal ingest failure" in row.error_message
        assert row.retry_count == command.attempt
        assert isinstance(row.payload_snapshot, dict)
        assert row.payload_snapshot.get("file_id") == str(file_id)

    _run(_test())


# ─────────────────────────────────────────────────────────────────────────────
# 2. Stats endpoint E2E (Postgres DLQ, preserved response shape)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_e2e_stats_endpoint_reads_postgres_dlq(
    postgres_dlq_available: None,
    dlq_cleanup: List[UUID],
) -> None:
    business_id = uuid4()
    file_id = uuid4()

    async def _test():
        factory, engine = _loop_local_dlq_resources()
        try:
            with patch(
                "app.services.ingestion.ingestion_dlq_service._dlq_session_factory",
                factory,
            ):
                row_id = await record_ingestion_failure(
                    business_id=business_id,
                    file_id=file_id,
                    stage="worker.ingest_file",
                    error=RuntimeError("e2e stats failure"),
                    payload_snapshot={"file_id": str(file_id), "e2e": True},
                    retry_count=2,
                )
            assert row_id is not None
            dlq_cleanup.append(row_id)

            queue = InMemoryIngestionQueue()
            worker = IngestionWorker(queue=queue)
            worker._running = True

            with patch(
                "app.services.ingestion.ingestion_worker.get_ingestion_queue",
                return_value=queue,
            ), patch(
                "app.services.ingestion.ingestion_worker.get_ingestion_worker",
                return_value=worker,
            ), patch(
                "app.services.ingestion.ingestion_dlq_service._dlq_session_factory",
                factory,
            ):
                from app.main import ingestion_worker_stats

                result = await ingestion_worker_stats()
        finally:
            await engine.dispose()

        assert result["status"] == "running"
        assert "queue" in result
        assert "dlq_total" in result["queue"]
        assert isinstance(result["dead_letter"], list)
        assert result["queue"]["dlq_total"] >= 1
        assert "dead_letter" not in queue.stats

        match = [
            entry
            for entry in result["dead_letter"]
            if entry.get("file_id") == str(file_id)
            and entry.get("business_id") == str(business_id)
        ]
        assert match, "expected DLQ entry from Postgres in dead_letter list"
        assert match[0]["stage"] == "worker.ingest_file"
        assert match[0]["error_type"] == "RuntimeError"

    _run(_test())


# ─────────────────────────────────────────────────────────────────────────────
# 3. Semaphore concurrency E2E (timing + observed parallelism cap)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_e2e_ingest_file_semaphore_enforced_via_timing_and_parallelism(
    postgres_dlq_available: None,
) -> None:
    """
    With max_concurrent_ingestions=1 and hold=_E2E_INGEST_HOLD_S, three overlapping
    ingest_file calls should take at least ~2 * hold seconds wall-clock time.
    """
    tenant_id = f"e2e-tenant-timing-{uuid4().hex[:8]}"
    hold_s = _E2E_INGEST_HOLD_S
    n_tasks = 3
    active = 0
    max_active = 0
    started = 0

    async def held_process_file(**_kwargs: Any) -> None:
        nonlocal active, max_active, started
        started += 1
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(hold_s)
        active -= 1

    config = _e2e_orchestrator_config(
        client_id=tenant_id,
        max_concurrent_ingestions=1,
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
            side_effect=held_process_file,
        ):
            orch = IngestionOrchestrator()
            t0 = time.monotonic()
            await asyncio.gather(
                *[
                    orch.ingest_file(file_id=f"f{i}", client_id=tenant_id)
                    for i in range(n_tasks)
                ]
            )
            elapsed = time.monotonic() - t0
            assert max_active <= 1, "concurrency cap must limit in-flight ingest work"

        min_expected = (n_tasks - 1) * hold_s * _E2E_TIMING_TOLERANCE
        assert elapsed >= min_expected, (
            f"expected serial/capped ingest duration >= {min_expected:.2f}s, got {elapsed:.2f}s"
        )
        assert started == n_tasks

    _run(_test())


@pytest.mark.integration
def test_e2e_ingest_file_semaphore_parallel_cap_two_batches_timing(
    postgres_dlq_available: None,
) -> None:
    """max_concurrent_ingestions=2 with four tasks should need at least two hold windows."""
    tenant_id = f"e2e-tenant-cap2-{uuid4().hex[:8]}"
    hold_s = _E2E_INGEST_HOLD_S
    n_tasks = 4
    max_parallel = 2
    max_active = 0
    active = 0

    async def held_process_file(**_kwargs: Any) -> None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(hold_s)
        active -= 1

    config = _e2e_orchestrator_config(
        client_id=tenant_id,
        max_concurrent_ingestions=max_parallel,
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
            side_effect=held_process_file,
        ):
            orch = IngestionOrchestrator()
            t0 = time.monotonic()
            await asyncio.gather(
                *[
                    orch.ingest_file(file_id=f"g{i}", client_id=tenant_id)
                    for i in range(n_tasks)
                ]
            )
            elapsed = time.monotonic() - t0
            assert max_active <= max_parallel

        batches = (n_tasks + max_parallel - 1) // max_parallel
        min_expected = (batches - 1) * hold_s * _E2E_TIMING_TOLERANCE
        assert elapsed >= min_expected

    _run(_test())


# ─────────────────────────────────────────────────────────────────────────────
# 4. Semaphore release E2E (exception + cancellation)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_e2e_semaphore_released_after_exception(
    postgres_dlq_available: None,
) -> None:
    tenant_id = f"e2e-tenant-exc-{uuid4().hex[:8]}"
    calls: List[str] = []
    config = _e2e_orchestrator_config(
        client_id=tenant_id,
        max_concurrent_ingestions=1,
    )

    async def fail_once(**_kwargs: Any) -> None:
        calls.append("fail")
        raise RuntimeError("e2e first ingest fails")

    async def succeed(**_kwargs: Any) -> None:
        calls.append("ok")

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
            side_effect=fail_once,
        ):
            orch = IngestionOrchestrator()
            with pytest.raises(RuntimeError):
                await orch.ingest_file(file_id="fail-1", client_id=tenant_id)

        with patch(
            "app.services.ingestion.ingestion_service_v2.IngestionServiceV2.process_file",
            side_effect=succeed,
        ):
            orch = IngestionOrchestrator()
            await orch.ingest_file(file_id="ok-2", client_id=tenant_id)

        assert calls == ["fail", "ok"]

    _run(_test())


@pytest.mark.integration
def test_e2e_semaphore_released_after_cancellation(
    postgres_dlq_available: None,
) -> None:
    tenant_id = f"e2e-tenant-cancel-{uuid4().hex[:8]}"
    started = asyncio.Event()
    release = asyncio.Event()
    calls: List[str] = []
    config = _e2e_orchestrator_config(
        client_id=tenant_id,
        max_concurrent_ingestions=1,
    )

    async def long_running(**_kwargs: Any) -> None:
        calls.append("long")
        started.set()
        await release.wait()

    async def quick_ok(**_kwargs: Any) -> None:
        calls.append("after_cancel")

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
            side_effect=long_running,
        ):
            orch = IngestionOrchestrator()
            task = asyncio.create_task(
                orch.ingest_file(file_id="cancel-me", client_id=tenant_id)
            )
            await asyncio.wait_for(started.wait(), timeout=5.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            release.set()

        with patch(
            "app.services.ingestion.ingestion_service_v2.IngestionServiceV2.process_file",
            side_effect=quick_ok,
        ):
            orch = IngestionOrchestrator()
            await orch.ingest_file(file_id="after-cancel", client_id=tenant_id)

        assert calls == ["long", "after_cancel"]

    _run(_test())


# ─────────────────────────────────────────────────────────────────────────────
# 5. Streaming failure context E2E (compensation before DLQ tombstone)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_e2e_streaming_window_failure_dlq_records_stage_and_window(
    postgres_dlq_available: None,
    dlq_cleanup: List[UUID],
) -> None:
    business_id = uuid4()
    file_id = uuid4()
    window_index = 5
    queue = InMemoryIngestionQueue()
    worker = IngestionWorker(queue=queue)

    async def _streaming_ingest(*_args: Any, **_kwargs: Any) -> None:
        await _run_streaming_window_commit_failure(
            business_id=business_id,
            file_id=file_id,
            window_index=window_index,
        )

    async def _test():
        factory, engine = _loop_local_dlq_resources()
        try:
            command = IngestionCommand(
                file_id=str(file_id),
                business_id=str(business_id),
                attempt=_MAX_RETRY_ATTEMPTS,
            )
            with patch(
                "app.services.ingestion.ingestion_dlq_service._dlq_session_factory",
                factory,
            ), patch.object(
                IngestionOrchestrator,
                "ingest_file",
                new_callable=AsyncMock,
                side_effect=_streaming_ingest,
            ):
                await worker._process_command(command, worker_id=0)

            rows = await _fetch_dlq_rows(
                business_id=business_id,
                file_id=file_id,
                session_factory=factory,
            )
            assert len(rows) >= 1
            row = rows[0]
            _track_dlq_ids(dlq_cleanup, rows)

            assert row.stage == "streaming_window_commit"
            assert row.window_index == window_index
            assert row.error_type == "RuntimeError"
            assert "commit failed" in row.error_message
            assert pop_ingestion_failure_context() is None
        finally:
            await engine.dispose()

    _run(_test())


@pytest.mark.integration
def test_e2e_streaming_compensation_before_dlq_persistence_order(
    postgres_dlq_available: None,
    dlq_cleanup: List[UUID],
) -> None:
    """Compensation must complete before the durable DLQ write in the worker path."""
    business_id = uuid4()
    file_id = uuid4()
    order: List[str] = []

    async def _streaming_ingest(*_args: Any, **_kwargs: Any) -> None:
        await _run_streaming_window_commit_failure(
            business_id=business_id,
            file_id=file_id,
            window_index=2,
            order=order,
        )

    queue = InMemoryIngestionQueue()
    worker = IngestionWorker(queue=queue)

    async def _recording_dlq(**kwargs: Any) -> Optional[UUID]:
        order.append("dlq")
        return await record_ingestion_failure(**kwargs)

    async def _test():
        factory, engine = _loop_local_dlq_resources()
        try:
            command = IngestionCommand(
                file_id=str(file_id),
                business_id=str(business_id),
                attempt=_MAX_RETRY_ATTEMPTS,
            )
            with patch(
                "app.services.ingestion.ingestion_dlq_service._dlq_session_factory",
                factory,
            ), patch.object(
                IngestionOrchestrator,
                "ingest_file",
                new_callable=AsyncMock,
                side_effect=_streaming_ingest,
            ), patch(
                "app.services.ingestion.ingestion_dlq_service.record_ingestion_failure",
                side_effect=_recording_dlq,
            ):
                await worker._process_command(command, worker_id=0)

            assert order.index("compensate") < order.index("dlq")
            rows = await _fetch_dlq_rows(
                business_id=business_id,
                file_id=file_id,
                session_factory=factory,
            )
            _track_dlq_ids(dlq_cleanup, rows)
            assert rows and rows[0].stage == "streaming_window_commit"
        finally:
            await engine.dispose()

    _run(_test())


# ─────────────────────────────────────────────────────────────────────────────
# 6. Legacy (non-streaming) failure E2E
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_e2e_legacy_non_streaming_failure_dlq_stage(
    postgres_dlq_available: None,
    dlq_cleanup: List[UUID],
) -> None:
    business_id = uuid4()
    file_id = uuid4()
    command = IngestionCommand(
        file_id=str(file_id),
        business_id=str(business_id),
        attempt=_MAX_RETRY_ATTEMPTS,
    )
    queue = InMemoryIngestionQueue()
    worker = IngestionWorker(queue=queue)

    async def _legacy_fail(*_args: Any, **_kwargs: Any) -> None:
        assert pop_ingestion_failure_context() is None
        raise ValueError("e2e legacy pipeline failure")

    async def _test():
        factory, engine = _loop_local_dlq_resources()
        try:
            with patch(
                "app.services.ingestion.ingestion_dlq_service._dlq_session_factory",
                factory,
            ), patch.object(
                IngestionOrchestrator,
                "ingest_file",
                new_callable=AsyncMock,
                side_effect=_legacy_fail,
            ):
                await worker._process_command(command, worker_id=0)

            rows = await _fetch_dlq_rows(
                business_id=business_id,
                file_id=file_id,
                session_factory=factory,
            )
            assert len(rows) >= 1
            row = rows[0]
            _track_dlq_ids(dlq_cleanup, rows)

            assert row.stage == "worker.ingest_file"
            assert row.window_index is None
            assert row.error_type == "ValueError"
            assert "legacy pipeline failure" in row.error_message
        finally:
            await engine.dispose()

    _run(_test())

