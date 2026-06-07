"""
Unit tests for streaming chunk generator (iter_chunks, ChunkDict).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, AsyncIterator, List
from unittest.mock import MagicMock, patch

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.ingestion.pdf_parser_v2 import ParsedPage
from app.services.ingestion import chunk_stream as cs
from app.services.ingestion.chunk_stream import ChunkDict, iter_chunks


def _run(coro):
    """Run async test coroutine without pytest-asyncio."""
    return asyncio.run(coro)


class _DummySession(AsyncSession):  # lightweight stub; methods never awaited
    pass


async def _single_page_iter(page: ParsedPage) -> AsyncIterator[ParsedPage]:
    yield page


async def _multi_page_iter(pages: List[ParsedPage]) -> AsyncIterator[ParsedPage]:
    for p in pages:
        yield p


def _dummy_pipeline(model: str = "dummy-model") -> Any:
    return SimpleNamespace(
        embedder=SimpleNamespace(info=SimpleNamespace(model=model)),
        config=None,
    )


def _semantic_hash_fn(text: str) -> str:
    return f"H::{text}"


def test_text_page_yields_correct_chunk_count() -> None:
    page = ParsedPage(
        page_number=0,
        text="page text",
        cleaned_text="page text",
        char_offset_start=0,
        char_offset_end=100,
        is_visual=False,
        is_continuation=False,
    )

    async def fake_chunk(text: str, **kwargs: Any) -> List[dict]:
        return [
            {"text": "c1"},
            {"text": "c2"},
            {"text": "c3"},
        ]

    async def _test():
        with patch.object(cs, "_chunk_text_with_strategy", side_effect=fake_chunk):
            chunks = [
                c
                async for c in iter_chunks(
                    _single_page_iter(page),
                    pipeline=_dummy_pipeline(),
                    db=_DummySession(),
                    file_id=MagicMock(),
                    business_id=MagicMock(),
                    file_type="pdf",
                    semantic_hash_fn=_semantic_hash_fn,
                )
            ]
        return chunks

    chunks = _run(_test())
    assert len(chunks) == 3
    assert all(c["status"] == "ready" for c in chunks)


def test_chunk_index_is_globally_monotonic() -> None:
    pages = [
        ParsedPage(0, "p0", "p0", 0, 10, False, False),
        ParsedPage(1, "p1", "p1", 10, 30, False, False),
    ]

    async def fake_chunk(text: str, **kwargs: Any) -> List[dict]:
        # First call → 2 chunks, second call → 3 chunks
        if not hasattr(fake_chunk, "_call"):
            fake_chunk._call = 0  # type: ignore[attr-defined]
        fake_chunk._call += 1  # type: ignore[attr-defined]
        if fake_chunk._call == 1:  # type: ignore[attr-defined]
            return [{"text": "a"}, {"text": "b"}]
        return [{"text": "c"}, {"text": "d"}, {"text": "e"}]

    async def _test():
        with patch.object(cs, "_chunk_text_with_strategy", side_effect=fake_chunk):
            chunks = [
                c
                async for c in iter_chunks(
                    _multi_page_iter(pages),
                    pipeline=_dummy_pipeline(),
                    db=_DummySession(),
                    file_id=MagicMock(),
                    business_id=MagicMock(),
                    file_type="pdf",
                    semantic_hash_fn=_semantic_hash_fn,
                )
            ]
        return chunks

    chunks = _run(_test())
    assert [c["chunk_index"] for c in chunks] == [0, 1, 2, 3, 4]


def test_visual_page_yields_placeholder_immediately() -> None:
    page = ParsedPage(
        page_number=0,
        text="visual page",
        cleaned_text="visual page",
        char_offset_start=0,
        char_offset_end=50,
        is_visual=True,
        is_continuation=False,
    )

    async def _test():
        async def fake_pages() -> AsyncIterator[ParsedPage]:
            yield page

        with patch.object(cs, "_resolve_visual_chunk", return_value=[]):
            chunks = [
                c
                async for c in iter_chunks(
                    fake_pages(),
                    pipeline=_dummy_pipeline(),
                    db=_DummySession(),
                    file_id=MagicMock(),
                    business_id=MagicMock(),
                    file_type="pdf",
                    semantic_hash_fn=_semantic_hash_fn,
                )
            ]
        return chunks

    chunks = _run(_test())
    assert len(chunks) == 1
    c0 = chunks[0]
    assert c0["status"] == "visual_pending"
    assert isinstance(c0["_visual_task"], asyncio.Task)


def test_visual_llm_not_awaited_inline() -> None:
    page = ParsedPage(
        page_number=0,
        text="visual page",
        cleaned_text="visual page",
        char_offset_start=0,
        char_offset_end=50,
        is_visual=True,
        is_continuation=False,
    )

    async def _test():
        async def fake_pages() -> AsyncIterator[ParsedPage]:
            yield page

        create_calls: list[asyncio.Task[Any]] = []

        async def fake_resolve(*args: Any, **kwargs: Any) -> list[ChunkDict]:
            # Should be scheduled via create_task, not awaited directly here.
            await asyncio.sleep(0)
            return []

        async def _runner():
            loop = asyncio.get_running_loop()
            real_create = asyncio.create_task

            def _wrapped_create(coro):
                t = real_create(coro)
                create_calls.append(t)
                return t

            with patch.object(cs, "_resolve_visual_chunk", side_effect=fake_resolve):
                with patch.object(cs.asyncio, "create_task", side_effect=_wrapped_create):
                    agen = iter_chunks(
                        fake_pages(),
                        pipeline=_dummy_pipeline(),
                        db=_DummySession(),
                        file_id=MagicMock(),
                        business_id=MagicMock(),
                        file_type="pdf",
                        semantic_hash_fn=_semantic_hash_fn,
                    )
                    # Pull just the first placeholder and then stop.
                    first = await agen.__anext__()
                    assert first["status"] == "visual_pending"
                    await agen.aclose()

        await _runner()
        return create_calls

    calls = _run(_test())
    # At least one visual task must have been scheduled.
    assert len(calls) == 1


def test_empty_chunk_strategy_yields_empty_status() -> None:
    page = ParsedPage(
        page_number=0,
        text="",
        cleaned_text="",
        char_offset_start=10,
        char_offset_end=20,
        is_visual=False,
        is_continuation=False,
    )

    async def fake_chunk(text: str, **kwargs: Any) -> List[dict]:
        return []

    async def _test():
        with patch.object(cs, "_chunk_text_with_strategy", side_effect=fake_chunk):
            chunks = [
                c
                async for c in iter_chunks(
                    _single_page_iter(page),
                    pipeline=_dummy_pipeline(),
                    db=_DummySession(),
                    file_id=MagicMock(),
                    business_id=MagicMock(),
                    file_type="pdf",
                    semantic_hash_fn=_semantic_hash_fn,
                )
            ]
        return chunks

    chunks = _run(_test())
    assert len(chunks) == 1
    c0 = chunks[0]
    assert c0["status"] == "empty"
    assert c0["text"] == ""


def test_semantic_hash_determinism() -> None:
    page1 = ParsedPage(0, "t", "t", 0, 10, False, False)
    page2 = ParsedPage(1, "t", "t", 10, 20, False, False)

    async def fake_chunk(text: str, **kwargs: Any) -> List[dict]:
        return [{"text": "same"}]

    async def _test():
        with patch.object(cs, "_chunk_text_with_strategy", side_effect=fake_chunk):
            chunks = [
                c
                async for c in iter_chunks(
                    _multi_page_iter([page1, page2]),
                    pipeline=_dummy_pipeline(),
                    db=_DummySession(),
                    file_id=MagicMock(),
                    business_id=MagicMock(),
                    file_type="pdf",
                    semantic_hash_fn=_semantic_hash_fn,
                )
            ]
        return chunks

    chunks = _run(_test())
    assert len(chunks) == 2
    assert chunks[0]["semantic_hash"] == chunks[1]["semantic_hash"]


def test_semantic_hash_uniqueness() -> None:
    page1 = ParsedPage(0, "t1", "t1", 0, 10, False, False)
    page2 = ParsedPage(1, "t2", "t2", 10, 20, False, False)

    async def fake_chunk(text: str, **kwargs: Any) -> List[dict]:
        return [{"text": text}]

    async def _test():
        with patch.object(cs, "_chunk_text_with_strategy", side_effect=fake_chunk):
            chunks = [
                c
                async for c in iter_chunks(
                    _multi_page_iter([page1, page2]),
                    pipeline=_dummy_pipeline(),
                    db=_DummySession(),
                    file_id=MagicMock(),
                    business_id=MagicMock(),
                    file_type="pdf",
                    semantic_hash_fn=_semantic_hash_fn,
                )
            ]
        return chunks

    chunks = _run(_test())
    assert len(chunks) == 2
    assert chunks[0]["semantic_hash"] != chunks[1]["semantic_hash"]


def test_offset_inherited_from_page() -> None:
    page = ParsedPage(
        page_number=3,
        text="page",
        cleaned_text="page",
        char_offset_start=100,
        char_offset_end=500,
        is_visual=False,
        is_continuation=False,
    )

    async def fake_chunk(text: str, **kwargs: Any) -> List[dict]:
        return [{"text": "c1"}, {"text": "c2"}, {"text": "c3"}]

    async def _test():
        with patch.object(cs, "_chunk_text_with_strategy", side_effect=fake_chunk):
            chunks = [
                c
                async for c in iter_chunks(
                    _single_page_iter(page),
                    pipeline=_dummy_pipeline(),
                    db=_DummySession(),
                    file_id=MagicMock(),
                    business_id=MagicMock(),
                    file_type="pdf",
                    semantic_hash_fn=_semantic_hash_fn,
                )
            ]
        return chunks

    chunks = _run(_test())
    assert len(chunks) == 3
    for c in chunks:
        assert c["char_offset_start"] == 100
        assert c["char_offset_end"] == 500


def test_page_iter_cancellation_does_not_hang() -> None:
    pages = [
        ParsedPage(0, "t0", "t0", 0, 10, False, False),
        ParsedPage(1, "t1", "t1", 10, 20, False, False),
    ]

    async def fake_chunk(text: str, **kwargs: Any) -> List[dict]:
        return [{"text": text}]

    async def _test():
        async def page_gen():
            for p in pages:
                yield p

        with patch.object(cs, "_chunk_text_with_strategy", side_effect=fake_chunk):
            agen = iter_chunks(
                page_gen(),
                pipeline=_dummy_pipeline(),
                db=_DummySession(),
                file_id=MagicMock(),
                business_id=MagicMock(),
                file_type="pdf",
                semantic_hash_fn=_semantic_hash_fn,
            )
            _ = await agen.__anext__()  # consume first chunk
            await agen.aclose()         # cancel/abandon mid-stream

    _run(_test())

