"""
Unit tests for streaming PDF page extraction (iter_pdf_pages, ParsedPage).
"""

from __future__ import annotations

import asyncio
from typing import Any, List
from unittest.mock import MagicMock, patch

from app.services.ingestion import pdf_parser_v2 as pdf_mod
from app.services.ingestion.pdf_parser_v2 import (
    ParsedPage,
    _StreamingPdfReader,
    _compute_initial_offset,
    _get_page_count,
    _get_page_text_only_sync,
    _parse_single_page_sync,
    iter_pdf_pages,
    parse_pdf,
)

# Keys that parse_pdf must always return (backward-compat contract).
def _run(coro):
    """Run async test coroutine without pytest-asyncio."""
    return asyncio.run(coro)


PARSE_PDF_REQUIRED_KEYS = frozenset({
    "raw_text",
    "cleaned_text",
    "normalized_text",
    "pages",
    "page_map",
    "visual_explanations",
    "source_type",
    "metadata",
})


class _MockPage:
    def __init__(self, text: str, images: int = 0, *, fail: bool = False):
        self._text = text
        self._images = images
        self._fail = fail

    def get_text(self, mode: str = "text") -> str:
        if self._fail:
            raise RuntimeError("mock page extraction failed")
        return self._text

    def get_images(self) -> List[Any]:
        return [object()] * self._images


class _MockDoc:
    def __init__(self, pages: List[_MockPage]):
        self._pages = pages
        self.page_count = len(pages)

    def __getitem__(self, index: int) -> _MockPage:
        return self._pages[index]

    def close(self) -> None:
        pass


def _mock_reader(pages: List[_MockPage]) -> _StreamingPdfReader:
    return _StreamingPdfReader(doc=_MockDoc(pages), path="/mock/test.pdf")


def test_parse_pdf_returns_identical_shape_as_before() -> None:
    pages = [
        ParsedPage(
            page_number=0,
            text="Page one content here.",
            cleaned_text="Page one content here.",
            char_offset_start=0,
            char_offset_end=22,
            is_visual=False,
            is_continuation=False,
        ),
        ParsedPage(
            page_number=1,
            text="Page two content here.",
            cleaned_text="Page two content here.",
            char_offset_start=22,
            char_offset_end=44,
            is_visual=False,
            is_continuation=False,
        ),
        ParsedPage(
            page_number=2,
            text="Page three content.",
            cleaned_text="Page three content.",
            char_offset_start=44,
            char_offset_end=63,
            is_visual=False,
            is_continuation=False,
        ),
    ]

    async def _fake_iter(_path: str, start_page: int = 0):
        for p in pages:
            yield p

    async def _run_test():
        with patch.object(pdf_mod, "iter_pdf_pages", _fake_iter):
            with patch.object(pdf_mod, "LOCAL_VISUAL_INTERCEPT_TOGGLE", False):
                with patch.object(pdf_mod, "is_llm_enabled", return_value=False):
                    return await parse_pdf("/mock/doc.pdf")

    result = _run(_run_test())

    assert isinstance(result, dict)
    assert PARSE_PDF_REQUIRED_KEYS.issubset(result.keys())
    assert isinstance(result["raw_text"], str)
    assert isinstance(result["cleaned_text"], str)
    assert isinstance(result["normalized_text"], str)
    assert isinstance(result["pages"], int)
    assert isinstance(result["page_map"], list)
    assert isinstance(result["visual_explanations"], list)
    assert result["source_type"] == "pdf"
    assert isinstance(result["metadata"], dict)
    assert result["pages"] == 3
    assert len(result["page_map"]) == 3


def test_iter_pdf_pages_yields_all_pages() -> None:
    reader = _mock_reader([_MockPage("a"), _MockPage("b"), _MockPage("c"), _MockPage("d"), _MockPage("e")])

    async def _collect():
        with patch.object(pdf_mod, "_open_pdf_reader", return_value=reader):
            return [p async for p in iter_pdf_pages("/mock.pdf", start_page=0)]

    collected = _run(_collect())
    assert len(collected) == 5
    assert [p.page_number for p in collected] == [0, 1, 2, 3, 4]


def test_iter_pdf_pages_respects_start_page() -> None:
    reader = _mock_reader([_MockPage("a"), _MockPage("b"), _MockPage("c"), _MockPage("d"), _MockPage("e")])

    async def _collect():
        with patch.object(pdf_mod, "_open_pdf_reader", return_value=reader):
            return [p async for p in iter_pdf_pages("/mock.pdf", start_page=2)]

    collected = _run(_collect())
    assert len(collected) == 3
    assert collected[0].page_number == 2


def test_char_offset_is_cumulative() -> None:
    reader = _mock_reader([_MockPage("abc"), _MockPage("de"), _MockPage("fghi")])

    async def _collect():
        with patch.object(pdf_mod, "_open_pdf_reader", return_value=reader):
            return [p async for p in iter_pdf_pages("/mock.pdf")]

    collected = _run(_collect())
    for n in range(1, len(collected)):
        assert collected[n].char_offset_start == collected[n - 1].char_offset_end


def test_char_offset_correct_after_start_page_resume() -> None:
    page0 = "x" * 100
    page1 = "y" * 200
    reader = _mock_reader(
        [_MockPage(page0), _MockPage(page1), _MockPage("z"), _MockPage("w"), _MockPage("v")]
    )

    async def _collect():
        with patch.object(pdf_mod, "_open_pdf_reader", return_value=reader):
            return [p async for p in iter_pdf_pages("/mock.pdf", start_page=2)]

    collected = _run(_collect())
    assert collected[0].char_offset_start == 300


def test_bad_page_yields_empty_parsedpage_and_continues() -> None:
    pages = [
        _MockPage("ok0"),
        _MockPage("ok1"),
        _MockPage("ok2"),
        _MockPage("", fail=True),
        _MockPage("ok4"),
    ]
    reader = _mock_reader(pages)

    real_parse = pdf_mod._parse_single_page_sync

    def _parse_with_fail(r, page_number, char_offset):
        if page_number == 3:
            raise RuntimeError("simulated page 3 failure")
        return real_parse(r, page_number, char_offset)

    async def _collect():
        with patch.object(pdf_mod, "_open_pdf_reader", return_value=reader):
            with patch.object(pdf_mod, "_parse_single_page_sync", side_effect=_parse_with_fail):
                return [p async for p in iter_pdf_pages("/mock.pdf")]

    collected = _run(_collect())
    assert len(collected) == 5
    assert collected[3].text == ""
    assert collected[3].is_visual is False
    assert collected[4].text == "ok4"


def test_run_in_executor_called_once_per_page() -> None:
    reader = _mock_reader([_MockPage("a"), _MockPage("b"), _MockPage("c"), _MockPage("d")])
    call_count = 0

    async def _collect():
        nonlocal call_count
        loop = asyncio.get_running_loop()
        real_run = loop.run_in_executor

        async def _counting_run(executor, fn, *args):
            nonlocal call_count
            call_count += 1
            return await real_run(executor, fn, *args)

        with patch.object(pdf_mod, "_open_pdf_reader", return_value=reader):
            with patch.object(loop, "run_in_executor", _counting_run):
                return [p async for p in iter_pdf_pages("/mock.pdf")]

    collected = _run(_collect())
    assert len(collected) == 4
    assert call_count == 4


def test_reader_closed_on_generator_abandonment() -> None:
    reader = _mock_reader([_MockPage("a"), _MockPage("b"), _MockPage("c")])
    close_called = False
    original_close = reader.close

    def _tracking_close() -> None:
        nonlocal close_called
        close_called = True
        original_close()

    reader.close = _tracking_close  # type: ignore[method-assign]

    async def _abandon():
        with patch.object(pdf_mod, "_open_pdf_reader", return_value=reader):
            agen = iter_pdf_pages("/mock.pdf")
            first = await agen.__anext__()
            assert first.page_number == 0
            await agen.aclose()

    _run(_abandon())
    assert close_called is True


def test_start_page_beyond_total_raises_or_yields_nothing() -> None:
    reader = _mock_reader([_MockPage("a"), _MockPage("b"), _MockPage("c")])

    async def _collect():
        with patch.object(pdf_mod, "_open_pdf_reader", return_value=reader):
            return [p async for p in iter_pdf_pages("/mock.pdf", start_page=10)]

    collected = _run(_collect())
    assert collected == []


def test_compute_initial_offset_sums_raw_lengths() -> None:
    reader = _mock_reader([_MockPage("x" * 100), _MockPage("y" * 200), _MockPage("z")])
    assert _compute_initial_offset(reader, 2) == 300


def test_get_page_text_only_sync_matches_parse_raw_length() -> None:
    reader = _mock_reader([_MockPage("hello world")])
    raw_only = _get_page_text_only_sync(reader, 0)
    page, new_off = _parse_single_page_sync(reader, 0, 0)
    assert page.text == raw_only
    assert new_off == len(raw_only)


def test_get_page_count() -> None:
    reader = _mock_reader([_MockPage("a"), _MockPage("b")])
    assert _get_page_count(reader) == 2
