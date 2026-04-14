# =============================================================================
# app/services/ingestion/token_chunking_service.py
# =============================================================================
#
# Token-Aware Chunking Service — Orchestrates parsing → tokenizing → chunking.
#
# This service provides TRUE token-count-aware chunking using the pluggable
# tokenization factory (app.core.tokenization). Unlike the existing
# whitespace-based count_tokens() in segmenter_v2.py, this service uses
# actual subword tokenizers (HuggingFace, spaCy, NLTK) to measure chunk
# boundaries accurately.
#
# ARCHITECTURE:
#   - Does NOT modify ingestion_service_v2.py or any existing chunker
#   - Registers as a new chunking strategy "token_aware" in the chunking registry
#   - Can be called standalone or integrated via the registry pattern
#   - Reuses existing make_chunk_dict() for output format compatibility
#   - Reuses existing preprocess_document_text() for text cleaning
#
# Configuration (env vars):
#   DEFAULT_TOKENIZER_BACKEND = huggingface | spacy | nltk | whitespace
#   CHUNK_SIZE                = 512   (max tokens per chunk)
#   CHUNK_OVERLAP             = 64    (overlap tokens between chunks)
#   HF_TOKENIZER_MODEL        = bert-base-multilingual-cased
#
# Usage:
#   from app.services.ingestion.token_chunking_service import (
#       token_aware_chunk,
#       TokenChunkingService,
#   )
#
#   # Standalone usage:
#   chunks = await token_aware_chunk(text, db_session=db, file_id="...", ...)
#
#   # Via chunking registry (after import):
#   chunker = get_chunker("token_aware")
#   chunks = await chunker.chunk(text, ...)
# =============================================================================

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from app.core.chunking_stratagies.chunking_registry import Chunker, register_chunker
from app.core.chunking_stratagies.text_preprocessor import preprocess_document_text
from app.core.tokenization.factory import get_tokenizer
from app.core.tokenization.base import BaseTokenizer
from app.core.chunking_stratagies.segmenter_v2 import (
    make_chunk_dict,
)
from app.utils.logger import log_info


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — read from env, with sane defaults
# ─────────────────────────────────────────────────────────────────────────────

def _get_chunk_size() -> int:
    return int(os.getenv("CHUNK_SIZE", "512"))


def _get_chunk_overlap() -> int:
    return int(os.getenv("CHUNK_OVERLAP", "64"))


def _get_min_chunk_tokens() -> int:
    """Minimum token count — chunks below this are merged into neighbours."""
    return int(os.getenv("MIN_CHUNK_TOKENS", "30"))


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANUP — extra whitespace / special character normalization
# ─────────────────────────────────────────────────────────────────────────────

_MULTI_WHITESPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINES = re.compile(r"\n{3,}")


def _clean_whitespace(text: str) -> str:
    """
    Normalize excessive whitespace and special characters.
    Runs AFTER preprocess_document_text() for additional cleanup.
    """
    text = _MULTI_WHITESPACE.sub(" ", text)
    text = _MULTI_NEWLINES.sub("\n\n", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# SENTENCE SPLITTER — reuse the proven regex from segmenter_v2
# ─────────────────────────────────────────────────────────────────────────────

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n")


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences. Falls back to paragraph split."""
    sentences = _SENTENCE_BOUNDARY.split(text)
    if len(sentences) <= 1:
        # No sentence boundaries — try paragraph boundaries
        sentences = _PARAGRAPH_BOUNDARY.split(text)
    return [s.strip() for s in sentences if s.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# CORE: TOKEN-AWARE CHUNKING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class TokenChunkingService:
    """
    Token-aware text chunker that measures boundaries using real tokenizers.

    Algorithm:
      1. Preprocess + clean text
      2. Split into sentences
      3. Greedily pack sentences into chunks where token_count <= chunk_size
      4. Apply overlap by re-including trailing sentences from previous chunk
      5. Merge undersized chunks
      6. Build chunk dicts compatible with existing pipeline

    Why token-aware matters:
      - Character-based chunking can split mid-word or mid-subword
      - Whitespace counting underestimates tokens for Indic scripts (which
        use conjuncts and lack spaces between some word forms)
      - Subword tokenizers (BPE/WordPiece) give the true token count that
        the embedding model will see — chunk boundaries match model limits
    """

    def __init__(
        self,
        tokenizer: Optional[BaseTokenizer] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        min_chunk_tokens: Optional[int] = None,
    ):
        self._tokenizer = tokenizer
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_chunk_tokens = min_chunk_tokens

    @property
    def tokenizer(self) -> BaseTokenizer:
        if self._tokenizer is None:
            self._tokenizer = get_tokenizer()
        return self._tokenizer

    @property
    def chunk_size(self) -> int:
        return self._chunk_size if self._chunk_size is not None else _get_chunk_size()

    @property
    def chunk_overlap(self) -> int:
        return self._chunk_overlap if self._chunk_overlap is not None else _get_chunk_overlap()

    @property
    def min_chunk_tokens(self) -> int:
        return self._min_chunk_tokens if self._min_chunk_tokens is not None else _get_min_chunk_tokens()

    def _count(self, text: str) -> int:
        """Count tokens using the configured backend."""
        return self.tokenizer.count_tokens(text)

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """
        Merge undersized chunks without violating the token budget.

        The legacy merge_small_chunks() helper is character-based, which is
        fine for legacy char chunkers but unsafe for a token-budgeted strategy.
        """
        if not chunks:
            return []

        merged: List[str] = []
        i = 0
        while i < len(chunks):
            current = chunks[i].strip()
            if not current:
                i += 1
                continue

            current_tokens = self._count(current)
            if current_tokens >= self.min_chunk_tokens:
                merged.append(current)
                i += 1
                continue

            if i + 1 < len(chunks):
                candidate = f"{current} {chunks[i + 1].strip()}".strip()
                if self._count(candidate) <= self.chunk_size:
                    chunks[i + 1] = candidate
                    i += 1
                    continue

            if merged:
                candidate = f"{merged[-1]} {current}".strip()
                if self._count(candidate) <= self.chunk_size:
                    merged[-1] = candidate
                    i += 1
                    continue

            merged.append(current)
            i += 1

        return merged

    def _split_overlong_token(self, word: str) -> List[str]:
        """
        Split a single whitespace-free token that exceeds chunk_size.

        This catches pathological OCR/base64/minified-json cases where
        whitespace splitting alone cannot enforce the hard token ceiling.
        """
        if not word:
            return []

        pieces: List[str] = []
        start = 0
        length = len(word)

        while start < length:
            low = start + 1
            high = length
            best_end = None

            while low <= high:
                mid = (low + high) // 2
                candidate = word[start:mid]
                token_count = self._count(candidate)
                if token_count <= self.chunk_size:
                    best_end = mid
                    low = mid + 1
                else:
                    high = mid - 1

            if best_end is None or best_end == start:
                # Last-resort progress guarantee even with unexpected tokenizer behaviour.
                best_end = min(start + 1, length)

            pieces.append(word[start:best_end])
            start = best_end

        return pieces

    def _split_oversized_sentence(self, sentence: str) -> List[str]:
        """
        Handle a single sentence that exceeds chunk_size tokens.

        Strategy: split on whitespace boundaries into sub-segments that each
        fit within chunk_size tokens. This preserves word boundaries even when
        a single sentence is extremely long (e.g. OCR output without punctuation).
        """
        words = sentence.split()
        segments: List[str] = []
        current_words: List[str] = []
        current_tokens = 0

        for word in words:
            word_tokens = self._count(word)
            if word_tokens > self.chunk_size:
                if current_words:
                    segments.append(" ".join(current_words))
                    current_words = []
                    current_tokens = 0
                segments.extend(self._split_overlong_token(word))
                continue

            if current_tokens + word_tokens > self.chunk_size and current_words:
                segments.append(" ".join(current_words))
                # Apply overlap: keep last N tokens worth of words
                overlap_words: List[str] = []
                overlap_tokens = 0
                for w in reversed(current_words):
                    wt = self._count(w)
                    if overlap_tokens + wt > self.chunk_overlap:
                        break
                    overlap_words.insert(0, w)
                    overlap_tokens += wt
                current_words = overlap_words + [word]
                current_tokens = overlap_tokens + word_tokens
            else:
                current_words.append(word)
                current_tokens += word_tokens

        if current_words:
            segments.append(" ".join(current_words))

        return segments

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into token-aware chunks.

        Returns a list of text strings, each guaranteed to be <= chunk_size tokens
        (as measured by the configured tokenizer backend).
        """
        if not text or not text.strip():
            return []

        # Step 1: preprocess + cleanup
        text = preprocess_document_text(text)
        text = _clean_whitespace(text)
        if not text:
            return []

        sentences = _split_sentences(text)

        # Step 2: greedy sentence packing with token counting
        chunks: List[str] = []
        current_sentences: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self._count(sentence)

            # Handle oversized single sentence
            if sent_tokens > self.chunk_size:
                # Flush current buffer first
                if current_sentences:
                    chunks.append(" ".join(current_sentences))
                    current_sentences = []
                    current_tokens = 0
                # Split the oversized sentence
                sub_segments = self._split_oversized_sentence(sentence)
                chunks.extend(sub_segments)
                continue

            # Would adding this sentence exceed chunk_size?
            if current_tokens + sent_tokens > self.chunk_size and current_sentences:
                # Flush current chunk
                chunks.append(" ".join(current_sentences))

                # Apply overlap: carry trailing sentences from previous chunk
                overlap_sents: List[str] = []
                overlap_tokens = 0
                for s in reversed(current_sentences):
                    st = self._count(s)
                    if overlap_tokens + st > self.chunk_overlap:
                        break
                    overlap_sents.insert(0, s)
                    overlap_tokens += st

                current_sentences = overlap_sents
                current_tokens = overlap_tokens

            current_sentences.append(sentence)
            current_tokens += sent_tokens

        # Flush final buffer
        if current_sentences:
            chunks.append(" ".join(current_sentences))

        # Step 3: merge undersized chunks without breaking the token ceiling
        merged = self._merge_small_chunks(chunks)

        log_info(
            f"[TokenChunking] {len(merged)} chunks | "
            f"backend={self.tokenizer.name} | "
            f"chunk_size={self.chunk_size} | overlap={self.chunk_overlap}"
        )

        return merged

    async def chunk(
        self,
        text: str,
        *,
        db_session=None,
        file_id: Optional[str] = None,
        business_id: Optional[Any] = None,
        source_type: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Full chunking pipeline: text → preprocess → tokenize → chunk → dict.

        Returns chunk dicts in the same format as segmenter_v2.make_chunk_dict(),
        ensuring full compatibility with the existing ingestion pipeline
        (dedup, GCI registration, embedding, vector upsert).

        The `tokens` field in each chunk dict is set from the tokenizer backend
        (accurate count) rather than whitespace approximation.
        """
        text_chunks = self.chunk_text(text)
        if not text_chunks:
            return []

        result: List[Dict[str, Any]] = []
        for chunk_text in text_chunks:
            chunk_dict = make_chunk_dict(
                chunk_text,
                db_session=db_session,
                file_id=file_id,
                business_id=business_id,
                source_type=source_type,
                embedding_model=embedding_model,
            )
            if not chunk_dict:
                continue

            # Override the whitespace-based token count with accurate count
            chunk_dict["tokens"] = self._count(
                chunk_dict.get("cleaned_text") or chunk_text
            )
            chunk_dict["tokenizer_backend"] = self.tokenizer.name

            result.append(chunk_dict)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING REGISTRY INTEGRATION
#
# Registers "token_aware" as a new chunking strategy in the existing registry.
# After this module is imported, get_chunker("token_aware") will work.
# ─────────────────────────────────────────────────────────────────────────────

@register_chunker("token_aware")
class TokenAwareChunker(Chunker):
    """
    Chunking registry adapter for TokenChunkingService.

    Registered as "token_aware" strategy. Can be selected via:
      - CHUNKING_STRATEGY=token_aware in .env
      - strategy_override="token_aware" in _chunk_text_with_strategy()
      - get_chunker("token_aware") directly
    """

    def __init__(self):
        self._service = TokenChunkingService()

    async def chunk(
        self,
        text: str,
        *,
        db_session=None,
        file_id=None,
        business_id=None,
        source_type: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return await self._service.chunk(
            text,
            db_session=db_session,
            file_id=file_id,
            business_id=business_id,
            source_type=source_type,
            embedding_model=embedding_model,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTION — drop-in async call
# ─────────────────────────────────────────────────────────────────────────────

async def token_aware_chunk(
    text: str,
    *,
    db_session=None,
    file_id: Optional[str] = None,
    business_id: Optional[Any] = None,
    source_type: Optional[str] = None,
    embedding_model: Optional[str] = None,
    tokenizer_backend: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience async function for token-aware chunking.

    Usage:
        chunks = await token_aware_chunk(
            text,
            db_session=db,
            file_id="abc-123",
            source_type="pdf",
            embedding_model="BAAI/bge-large-en-v1.5",
            tokenizer_backend="huggingface",
            chunk_size=512,
            chunk_overlap=64,
        )
    """
    tokenizer = get_tokenizer(tokenizer_backend) if tokenizer_backend else None
    service = TokenChunkingService(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return await service.chunk(
        text,
        db_session=db_session,
        file_id=file_id,
        business_id=business_id,
        source_type=source_type,
        embedding_model=embedding_model,
    )
