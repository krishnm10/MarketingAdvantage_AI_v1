"""
Unit tests for streaming ingestion state and dedup helpers.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.services.ingestion.streaming_state import (
    StreamingDedupState,
    StreamingIngestionState,
)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# StreamingDedupState
# ---------------------------------------------------------------------------


def test_streaming_dedup_state_add_and_is_seen() -> None:
    dedup = StreamingDedupState()
    dedup.add("h1")
    dedup.add("h2")
    dedup.add("h3")
    assert dedup.is_seen("h1")
    assert dedup.is_seen("h2")
    assert dedup.is_seen("h3")
    assert not dedup.is_seen("unknown")


def test_streaming_dedup_state_at_capacity_logs_warning() -> None:
    dedup = StreamingDedupState()
    with patch.object(StreamingDedupState, "MAX_SEEN_HASHES", 3):
        dedup.add("a")
        dedup.add("b")
        dedup.add("c")
        with patch(
            "app.services.ingestion.streaming_state.logger"
        ) as mock_logger:
            dedup.add("d")
        mock_logger.warning.assert_called_once()
        msg = mock_logger.warning.call_args[0][0]
        assert "capacity" in msg.lower() or "StreamingDedupState" in msg
    assert "d" not in dedup.seen_hashes
    assert len(dedup.seen_hashes) == 3


def test_streaming_dedup_state_is_seen_returns_false_at_capacity() -> None:
    dedup = StreamingDedupState()
    with patch.object(StreamingDedupState, "MAX_SEEN_HASHES", 2):
        dedup.add("x")
        dedup.add("y")
        assert not dedup.is_seen("x")
        assert not dedup.is_seen("y")
        assert not dedup.is_seen("z")


# ---------------------------------------------------------------------------
# StreamingIngestionState.from_checkpoint / persist_checkpoint
# ---------------------------------------------------------------------------


def _fake_file_record(
    *,
    last_processed_page: int = 0,
    last_processed_chunk_index: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        last_processed_page=last_processed_page,
        last_processed_chunk_index=last_processed_chunk_index,
    )


def test_streaming_state_from_checkpoint_loads_values() -> None:
    file_id = uuid4()
    business_id = uuid4()
    fake = _fake_file_record(
        last_processed_page=5,
        last_processed_chunk_index=120,
    )
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = fake
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=mock_result)

    async def _test():
        return await StreamingIngestionState.from_checkpoint(
            mock_db, file_id, business_id
        )

    state = _run(_test())
    assert state.last_processed_page == 5
    assert state.last_processed_chunk_index == 120
    assert state.file_id == file_id
    assert state.business_id == business_id


def test_streaming_state_from_checkpoint_enforces_tenant_guard() -> None:
    file_id = uuid4()
    business_id = uuid4()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=mock_result)

    async def _test():
        await StreamingIngestionState.from_checkpoint(
            mock_db, file_id, business_id
        )

    with pytest.raises(ValueError) as exc_info:
        _run(_test())
    msg = str(exc_info.value)
    assert str(file_id) in msg
    assert str(business_id) in msg


def test_streaming_state_from_checkpoint_wrong_business_id_raises() -> None:
    file_id = uuid4()
    wrong_business = uuid4()
    wrong_hex = wrong_business.hex

    async def _execute(stmt):
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        assert wrong_hex in compiled.replace("-", "")
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        return result

    mock_db = AsyncMock()
    mock_db.execute = _execute

    async def _test():
        await StreamingIngestionState.from_checkpoint(
            mock_db, file_id, wrong_business
        )

    with pytest.raises(ValueError):
        _run(_test())


def test_persist_checkpoint_includes_business_id_in_where() -> None:
    file_id = uuid4()
    business_id = uuid4()
    state = StreamingIngestionState(
        file_id=file_id,
        business_id=business_id,
        last_processed_chunk_index=42,
        last_processed_page=3,
    )
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock()

    async def _test():
        await state.persist_checkpoint(mock_db)

    _run(_test())
    assert mock_db.execute.await_count == 1
    stmt = mock_db.execute.await_args[0][0]
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "business_id" in compiled
    assert str(business_id) in compiled or "business_id" in compiled.lower()


# ---------------------------------------------------------------------------
# STREAMING_INGESTION_ENABLED feature flag
# ---------------------------------------------------------------------------


def test_streaming_ingestion_enabled_flag_default_off(monkeypatch) -> None:
    monkeypatch.delenv("MAI_STREAMING_INGESTION", raising=False)
    module_name = "app.services.ingestion.streaming_state"
    if module_name in sys.modules:
        del sys.modules[module_name]
    mod = importlib.import_module(module_name)
    assert mod.STREAMING_INGESTION_ENABLED is False


def test_streaming_ingestion_enabled_flag_on(monkeypatch) -> None:
    monkeypatch.setenv("MAI_STREAMING_INGESTION", "1")
    module_name = "app.services.ingestion.streaming_state"
    if module_name in sys.modules:
        del sys.modules[module_name]
    mod = importlib.import_module(module_name)
    assert mod.STREAMING_INGESTION_ENABLED is True
