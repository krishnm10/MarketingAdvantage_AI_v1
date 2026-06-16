import asyncio
import os

from app.core.chunking_stratagies.chunking_registry import (
    clear_chunker_cache,
    get_chunker,
    list_chunking_strategies,
)


def _run(coro):
    return asyncio.run(coro)


def test_chunking_registry_lists_expected_strategies():
    strategies = set(list_chunking_strategies())
    assert "semantic" in strategies
    assert "overlap" in strategies
    assert "smart_check" in strategies
    assert "recursive_overlap" in strategies
    assert "rust" in strategies


def test_chunking_registry_rejects_unknown_strategy():
    clear_chunker_cache()
    try:
        get_chunker("unknown_strategy_123")
        assert False, "Expected ValueError for unknown strategy"
    except ValueError as e:
        assert "Unknown chunking strategy" in str(e)


def test_overlap_chunker_generates_chunks_for_long_text():
    clear_chunker_cache()
    chunker = get_chunker("overlap")
    text = "This is a long paragraph. " * 200
    chunks = _run(
        chunker.chunk(
            text,
            file_id="test-file",
            source_type="txt",
            embedding_model="test-embed",
        )
    )
    assert len(chunks) >= 2
    assert all("semantic_hash" in c for c in chunks)


def test_env_chunking_strategy_is_respected():
    clear_chunker_cache()
    os.environ["CHUNKING_STRATEGY"] = "smart_check"
    try:
        chunker = get_chunker()
        chunks = _run(
            chunker.chunk(
                "Revenue increased 25 percent in Q4 and margins improved.",
                file_id="test-file",
                source_type="txt",
                embedding_model="test-embed",
            )
        )
        assert len(chunks) >= 1
        assert chunks[0].get("reasoning_ingestion", {}).get("chunking_strategy") == "smart_check_v3"
    finally:
        os.environ.pop("CHUNKING_STRATEGY", None)
        clear_chunker_cache()
