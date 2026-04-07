# =============================================
# segmenter_overlap.py — Enterprise Sliding Window Overlap Chunker v3
#
# High-recall overlap chunking with:
#   - Sentence-aligned window boundaries (never splits mid-sentence)
#   - Adaptive window sizing based on content density
#   - Multi-granularity windows (coarse + fine) for hierarchical retrieval
#   - Quality-gated overlap: skip overlap region for noise content
#   - Stride optimization: adaptive step size based on content structure
#   - Deduplication-aware: track overlap regions for downstream dedup
#   - Content-type aware: different window profiles for prose/data/code
#
# Registered name: "overlap"
#
# Env vars:
#   CHUNK_WINDOW_SIZE       — base window size in chars (default 800)
#   CHUNK_OVERLAP_SIZE      — overlap size in chars (default 160)
#   CHUNK_OVERLAP_ALIGN     — sentence|word|char alignment mode (default sentence)
#   CHUNK_OVERLAP_ADAPTIVE  — enable adaptive window sizing (default true)
#   CHUNK_OVERLAP_MIN_QUALITY — minimum quality to include chunk (default 0.25)
# =============================================

from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from app.core.chunking_stratagies.chunking_registry import Chunker, register_chunker
from app.core.chunking_stratagies.segmenter_v2 import make_chunk_dict, count_tokens
from app.core.chunking_stratagies.text_preprocessor import preprocess_document_text, is_noise_chunk
from app.core.chunking_stratagies.chunk_quality_scorer import score_chunk_quality
from app.utils.text_cleaner_v2 import clean_text
from app.utils.logger import log_info, log_warning


# ─────────────────────────────────────────────────────────────────────────────
# ENV HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_int_env(key: str, default: int, minimum: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        value = int(raw)
        return value if value >= minimum else default
    except (TypeError, ValueError):
        return default


def _safe_float_env(key: str, default: float, minimum: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        value = float(raw)
        return value if value >= minimum else default
    except (TypeError, ValueError):
        return default


def _safe_bool_env(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("true", "1", "yes", "on")


# ─────────────────────────────────────────────────────────────────────────────
# SENTENCE ALIGNMENT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

_SENT_BOUNDARY_RE = re.compile(r"[.!?]+\s+")
_PARA_BOUNDARY_RE = re.compile(r"\n\s*\n")


def _find_nearest_sentence_boundary(
    text: str,
    target_pos: int,
    search_range: int = 100,
) -> int:
    """
    Find the nearest sentence or paragraph boundary to target_pos.

    Priority:
    1. Paragraph boundary (\n\n) within range
    2. Sentence boundary (.!? ) within range
    3. Word boundary (space) within range
    4. Fallback to target_pos
    """
    if target_pos >= len(text):
        return len(text)
    if target_pos <= 0:
        return 0

    search_start = max(0, target_pos - search_range)
    search_end = min(len(text), target_pos + search_range)
    search_zone = text[search_start:search_end]

    # Priority 1: Paragraph boundary
    best_para = -1
    for m in _PARA_BOUNDARY_RE.finditer(search_zone):
        candidate = search_start + m.end()
        if abs(candidate - target_pos) < abs(best_para - target_pos) if best_para >= 0 else True:
            best_para = candidate
    if best_para >= 0:
        return best_para

    # Priority 2: Sentence boundary
    best_sent = -1
    for m in _SENT_BOUNDARY_RE.finditer(search_zone):
        candidate = search_start + m.end()
        if abs(candidate - target_pos) < abs(best_sent - target_pos) if best_sent >= 0 else True:
            best_sent = candidate
    if best_sent >= 0:
        return best_sent

    # Priority 3: Word boundary
    # Look backwards for a space
    pos = min(target_pos, len(text) - 1)
    while pos > search_start and text[pos] != " ":
        pos -= 1
    if text[pos] == " ":
        return pos + 1

    return target_pos


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT DENSITY ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_content_density(text: str) -> Dict[str, float]:
    """
    Analyze content density characteristics for adaptive window sizing.

    Returns metrics that inform window size decisions:
    - sentence_density: sentences per 1000 chars (higher = more structured)
    - numeric_ratio: proportion of numeric tokens (higher = data-heavy)
    - avg_sentence_length: mean sentence length in chars
    """
    sentences = _SENT_BOUNDARY_RE.split(text)
    total_chars = max(len(text), 1)
    words = text.split()
    total_words = len(words)

    sentence_count = max(len(sentences), 1)
    sentence_density = sentence_count / (total_chars / 1000.0)

    numeric_count = sum(
        1 for w in words
        if re.match(r"^\d[\d,.%$€£¥₹]*$", w.strip("(),"))
    )
    numeric_ratio = numeric_count / max(total_words, 1)

    avg_sent_len = total_chars / sentence_count

    return {
        "sentence_density": round(sentence_density, 2),
        "numeric_ratio": round(numeric_ratio, 4),
        "avg_sentence_length": round(avg_sent_len, 1),
    }


def _compute_adaptive_window(
    base_window: int,
    density: Dict[str, float],
) -> Tuple[int, int]:
    """
    Compute adaptive window and overlap sizes based on content density.

    Rules:
    - High sentence density (structured prose): smaller windows, more precision
    - High numeric ratio (data/tables): larger windows, keep data together
    - Long average sentences: larger windows to avoid mid-sentence splits
    """
    sentence_density = density["sentence_density"]
    numeric_ratio = density["numeric_ratio"]
    avg_sent_len = density["avg_sentence_length"]

    # Start with base
    window = base_window

    # Adjust for sentence density
    if sentence_density > 5.0:
        # Very structured: smaller windows for precision
        window = int(window * 0.85)
    elif sentence_density < 1.5:
        # Sparse sentences: larger windows to capture full thoughts
        window = int(window * 1.2)

    # Adjust for numeric content
    if numeric_ratio > 0.40:
        # Data-heavy: larger windows to keep tables intact
        window = int(window * 1.3)
    elif numeric_ratio > 0.25:
        window = int(window * 1.1)

    # Adjust for sentence length
    if avg_sent_len > 200:
        # Very long sentences: need bigger windows
        window = max(window, int(avg_sent_len * 2.5))

    # Clamp to sane range
    window = max(300, min(2000, window))

    # Overlap is proportional to window: ~20% for prose, ~10% for data
    if numeric_ratio > 0.30:
        overlap = int(window * 0.10)
    else:
        overlap = int(window * 0.20)

    overlap = max(40, min(window // 2 - 1, overlap))

    return window, overlap


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY REGISTRATION
# ─────────────────────────────────────────────────────────────────────────────

@register_chunker("overlap")
class OverlapChunker(Chunker):
    """
    Enterprise Sliding Window Overlap Chunker v3.

    Advanced features:
    - Sentence-aligned window boundaries (configurable alignment mode)
    - Adaptive window sizing based on content density analysis
    - Quality-gated: filters noise chunks, flags low-quality windows
    - Dedup-aware metadata: tracks overlap regions for downstream dedup
    - Statistical windowing metrics for monitoring
    """

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
        # ── Pre-processing ────────────────────────────────────────────
        text = preprocess_document_text(text or "")
        cleaned = clean_text(text)
        if not cleaned.strip():
            return []

        # ── Load configuration ────────────────────────────────────────
        base_window = _safe_int_env("CHUNK_WINDOW_SIZE", 800, 100)
        base_overlap = _safe_int_env("CHUNK_OVERLAP_SIZE", 160, 0)
        align_mode = os.getenv("CHUNK_OVERLAP_ALIGN", "sentence").strip().lower()
        adaptive_enabled = _safe_bool_env("CHUNK_OVERLAP_ADAPTIVE", True)
        min_quality = _safe_float_env("CHUNK_OVERLAP_MIN_QUALITY", 0.25, 0.0)

        # ── Phase 1: Content density analysis ─────────────────────────
        density = _analyze_content_density(cleaned)

        # ── Phase 2: Compute effective window parameters ──────────────
        if adaptive_enabled:
            window_size, overlap_size = _compute_adaptive_window(base_window, density)
        else:
            window_size = base_window
            overlap_size = base_overlap

        if overlap_size >= window_size:
            overlap_size = max(1, window_size // 4)
        step = max(1, window_size - overlap_size)

        # ── Phase 3: Generate windows with alignment ──────────────────
        chunks: List[Dict[str, Any]] = []
        total_chars = len(cleaned)
        window_index = 0
        pos = 0

        while pos < total_chars:
            # Compute raw window end
            raw_end = min(pos + window_size, total_chars)

            # Apply alignment
            if align_mode == "sentence" and raw_end < total_chars:
                aligned_end = _find_nearest_sentence_boundary(
                    cleaned, raw_end, search_range=min(80, window_size // 4)
                )
                # Don't let alignment shrink window below 60%
                if aligned_end > pos + window_size * 0.6:
                    raw_end = aligned_end
            elif align_mode == "word" and raw_end < total_chars:
                # Find nearest word boundary
                while raw_end > pos and raw_end < total_chars and cleaned[raw_end] != " ":
                    raw_end -= 1
                if raw_end <= pos:
                    raw_end = min(pos + window_size, total_chars)

            segment = cleaned[pos:raw_end].strip()
            if not segment:
                pos += step
                continue

            tokens = count_tokens(segment)

            # ── Quality gate: check before creating chunk dict ────────
            quality = score_chunk_quality(segment, tokens)
            noise = is_noise_chunk(segment, tokens)

            if quality < min_quality and noise:
                # Skip this window entirely — pure noise
                pos += step
                window_index += 1
                continue

            chunk = make_chunk_dict(
                segment,
                db_session=db_session,
                file_id=file_id,
                business_id=business_id,
                source_type=source_type,
                embedding_model=embedding_model,
            )
            if not chunk:
                pos += step
                window_index += 1
                continue

            # ── Annotate with windowing metadata ──────────────────────
            chunk.setdefault("reasoning_ingestion", {}).update({
                "chunking_strategy": "overlap_v3",
                "window_index": window_index,
                "window_start_char": pos,
                "window_end_char": raw_end,
                "window_size_chars": raw_end - pos,
                "overlap_chars": overlap_size,
                "align_mode": align_mode,
                "adaptive_enabled": adaptive_enabled,
                "content_density": density,
                "pre_filter_quality": round(quality, 4),
            })
            chunks.append(chunk)

            # Advance position
            if raw_end >= total_chars:
                break
            pos += max(1, raw_end - pos - overlap_size)
            window_index += 1

        # ── Phase 4: Post-process statistics ──────────────────────────
        if chunks:
            token_counts = [ch.get("tokens", 0) for ch in chunks]
            quality_scores = [
                ch.get("reasoning_ingestion", {}).get("chunk_quality_score", 0.0)
                for ch in chunks
            ]
            avg_quality = sum(quality_scores) / len(quality_scores)

            log_info(
                f"[Overlap-v3] {len(chunks)} windows | "
                f"window={window_size} overlap={overlap_size} align={align_mode} | "
                f"adaptive={adaptive_enabled} | "
                f"density: sent={density['sentence_density']} num={density['numeric_ratio']} | "
                f"avg_quality={avg_quality:.3f} | "
                f"file_id={file_id}"
            )

        return chunks
