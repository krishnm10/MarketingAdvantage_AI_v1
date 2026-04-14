from app.services.ingestion.token_chunking_service import TokenChunkingService


class FakeTokenizer:
    @property
    def name(self) -> str:
        return "fake"

    def tokenize(self, text: str):
        return text.split()

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class CharTokenizer:
    @property
    def name(self) -> str:
        return "char"

    def tokenize(self, text: str):
        return list(text)

    def count_tokens(self, text: str) -> int:
        return len(text)


def test_token_aware_merge_preserves_chunk_budget():
    service = TokenChunkingService(
        tokenizer=FakeTokenizer(),
        chunk_size=10,
        chunk_overlap=0,
        min_chunk_tokens=6,
    )

    chunks = service._merge_small_chunks(
        [
            "one two three four five",
            "six seven eight nine ten",
            "tail",
        ]
    )

    assert len(chunks) == 2
    assert "tail" in " ".join(chunks)
    assert all(service._count(chunk) <= service.chunk_size for chunk in chunks)


def test_split_oversized_sentence_handles_single_overlong_token():
    service = TokenChunkingService(
        tokenizer=CharTokenizer(),
        chunk_size=5,
        chunk_overlap=0,
        min_chunk_tokens=1,
    )

    chunks = service.chunk_text("abcdefghijk")

    assert chunks == ["abcde", "fghij", "k"]
    assert all(service._count(chunk) <= service.chunk_size for chunk in chunks)
