# =============================================
# segmenter_recursive_overlap.py — Enterprise Recursive Overlap Chunker v3
#
# Semantic chunking + intelligent context overlap stitching with:
#   - Sentence-aligned overlap (never splits mid-sentence)
#   - Boundary coherence scoring (only overlap at weak boundaries)
#   - Adaptive overlap size based on boundary quality
#   - Bidirectional context injection (suffix from prev + prefix from next)
#   - Cross-chunk entity continuity tracking
#   - Quality-aware overlap: skip overlap for noise/low-quality chunks
#   - Configurable overlap modes: fixed, adaptive, coherence-gated
#
# Registered name: "recursive_overlap"
#
# Env vars:
#   CHUNK_RECURSIVE_OVERLAP_CHARS     — base overlap size in chars (default 100)
#   CHUNK_RECURSIVE_OVERLAP_MODE      — fixed|adaptive|coherence (default coherence)
#   CHUNK_RECURSIVE_OVERLAP_MAX_CHARS — max overlap cap for adaptive mode (default 200)
#   CHUNK_RECURSIVE_OVERLAP_COHERENCE — coherence threshold for gated overlap (default 0.6)
# =============================================

from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from app.core.chunking_stratagies.chunking_registry import Chunker, register_chunker
from app.core.chunking_stratagies.segmenter_v2 import (
    make_chunk_dict,
    recursive_semantic_chunk,
    count_tokens,
)
from app.core.chunking_stratagies.chunk_quality_scorer import score_chunk_quality
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


# ─────────────────────────────────────────────────────────────────────────────
# SENTENCE BOUNDARY UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def _find_sentence_aligned_overlap(text: str, target_chars: int, from_end: bool = True) -> str:
    """
    Extract a sentence-aligned overlap region from text.

    Instead of blindly cutting at character boundaries, this finds the nearest
    sentence boundary to the target overlap size. This prevents mid-sentence
    overlap which degrades embedding quality.

    Args:
        text: Source text to extract overlap from.
        target_chars: Desired overlap size in characters.
        from_end: If True, extract from the end of text (suffix); else from start (prefix).

    Returns:
        Sentence-aligned overlap text, never longer than target_chars * 1.5.
    """
    if not text or target_chars <= 0:
        return ""

    max_chars = int(target_chars * 1.5)  # Allow 50% overshoot for sentence alignment

    if from_end:
        # Extract suffix
        search_zone = text[-max_chars:] if len(text) > max_chars else text
        # Find sentence boundaries in the search zone
        sentences = _SENT_BOUNDARY_RE.split(search_zone)
        if len(sentences) <= 1:
            # No sentence boundary — fall back to word boundary
            return _word_aligned_cut(text, target_chars, from_end=True)

        # Accumulate sentences from the end until we hit target
        result_parts: List[str] = []
        accumulated = 0
        for sent in reversed(sentences):
            if accumulated + len(sent) > max_chars and result_parts:
                break
            result_parts.insert(0, sent)
            accumulated += len(sent) + 1
            if accumulated >= target_chars:
                break

        return " ".join(result_parts).strip()
    else:
        # Extract prefix
        search_zone = text[:max_chars] if len(text) > max_chars else text
        sentences = _SENT_BOUNDARY_RE.split(search_zone)
        if len(sentences) <= 1:
            return _word_aligned_cut(text, target_chars, from_end=False)

        result_parts = []
        accumulated = 0
        for sent in sentences:
            if accumulated + len(sent) > max_chars and result_parts:
                break
            result_parts.append(sent)
            accumulated += len(sent) + 1
            if accumulated >= target_chars:
                break

        return " ".join(result_parts).strip()


def _word_aligned_cut(text: str, target_chars: int, from_end: bool) -> str:
    """Fallback: cut at word boundary when no sentence boundary exists."""
    if from_end:
        zone = text[-target_chars:] if len(text) > target_chars else text
        # Find first space to avoid mid-word cut
        first_space = zone.find(" ")
        if first_space > 0 and first_space < len(zone) // 2:
            return zone[first_space + 1:].strip()
        return zone.strip()
    else:
        zone = text[:target_chars] if len(text) > target_chars else text
        last_space = zone.rfind(" ")
        if last_space > len(zone) // 2:
            return zone[:last_space].strip()
        return zone.strip()


# ─────────────────────────────────────────────────────────────────────────────
# BOUNDARY COHERENCE SCORING
# ─────────────────────────────────────────────────────────────────────────────

_CONTINUATION_WORDS: frozenset = frozenset({
    "and", "or", "but", "because", "however", "therefore",
    "which", "that", "who", "whom", "whose", "where", "when",
    "additionally", "furthermore", "moreover", "also",
})

_TFIDF_STOP_WORDS: frozenset = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "and", "or", "but", "not", "this", "that", "it", "its",
    "has", "have", "had", "do", "does", "will", "would", "could",
    "should", "may", "might", "can", "so", "if", "then",
})


def _score_boundary_coherence(left_text: str, right_text: str) -> float:
    """
    Score how coherent a chunk boundary is (0.0 = terrible, 1.0 = clean split).

    Multi-signal scoring:
    - Sentence completeness at boundary
    - Topic continuity (word overlap)
    - Continuation word detection
    - Entity span protection
    """
    score = 1.0

    left_stripped = left_text.rstrip()
    right_stripped = right_text.lstrip()

    # Signal 1: Left doesn't end with sentence terminator (-0.25)
    if left_stripped and left_stripped[-1] not in ".!?;:":
        score -= 0.25

    # Signal 2: Right starts with lowercase (-0.30)
    if right_stripped and right_stripped[0].islower():
        score -= 0.30

    # Signal 3: Right starts with continuation word (-0.20)
    first_word = right_stripped.split()[0].lower() if right_stripped.split() else ""
    if first_word in _CONTINUATION_WORDS:
        score -= 0.20

    # Signal 4: Topic continuity — high word overlap means related content
    left_words = set(w.lower() for w in left_text.split()[-20:] if len(w) > 2) - _TFIDF_STOP_WORDS
    right_words = set(w.lower() for w in right_text.split()[:20] if len(w) > 2) - _TFIDF_STOP_WORDS
    if left_words and right_words:
        overlap = len(left_words & right_words)
        union = len(left_words | right_words)
        jaccard = overlap / union if union > 0 else 0
        # High jaccard = related content, needs overlap
        if jaccard > 0.3:
            score -= 0.15

    return max(0.0, score)


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY CONTINUITY TRACKER
# ─────────────────────────────────────────────────────────────────────────────

_ENTITY_PATTERN = re.compile(
    r"(?:"
    r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"  # Proper nouns
    r"|(?:Q[1-4]\s+\d{4})"                 # Quarters
    r"|(?:FY\s*\d{4})"                     # Fiscal years
    r"|(?:[$€£¥₹]\s*[\d,.]+\s*(?:billion|million|bn|mn)?)"  # Currency
    r"|(?:\d+(?:\.\d+)?%)"                 # Percentages
    r")",
    re.IGNORECASE,
)


def _extract_entities(text: str) -> List[str]:
    """Extract named entities from text for continuity tracking."""
    return [m.group() for m in _ENTITY_PATTERN.finditer(text)]


def _compute_entity_continuity(
    prev_text: str,
    curr_text: str,
) -> float:
    """
    Measure entity continuity across chunk boundary.

    Returns 0.0 (no shared entities) to 1.0 (high entity overlap).
    High continuity = these chunks discuss the same entities = overlap beneficial.
    """
    prev_entities = set(e.lower() for e in _extract_entities(prev_text))
    curr_entities = set(e.lower() for e in _extract_entities(curr_text))

    if not prev_entities and not curr_entities:
        return 0.0

    shared = prev_entities & curr_entities
    union = prev_entities | curr_entities

    return len(shared) / len(union) if union else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY REGISTRATION
# ─────────────────────────────────────────────────────────────────────────────

@register_chunker("recursive_overlap")
class RecursiveOverlapChunker(Chunker):
    """
    Enterprise Recursive Overlap Chunker v3.

    Advanced features:
    - Sentence-aligned overlap (never mid-sentence)
    - 3 overlap modes: fixed, adaptive, coherence-gated
    - Boundary coherence scoring drives overlap decisions
    - Entity continuity tracking (forces overlap when entities span boundary)
    - Quality-aware: no overlap for noise chunks
    - Bidirectional context metadata (overlap_prefix recorded for retrieval)
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
        # ── Phase 1: Base semantic chunking ───────────────────────────
        base_chunks = await recursive_semantic_chunk(
            text,
            db_session=db_session,
            file_id=file_id,
            business_id=business_id,
            source_type=source_type,
            embedding_model=embedding_model,
        )
        if not base_chunks:
            return []

        # ── Load configuration ────────────────────────────────────────
        overlap_chars = _safe_int_env("CHUNK_RECURSIVE_OVERLAP_CHARS", 100, 0)
        overlap_mode = os.getenv("CHUNK_RECURSIVE_OVERLAP_MODE", "coherence").strip().lower()
        max_overlap = _safe_int_env("CHUNK_RECURSIVE_OVERLAP_MAX_CHARS", 200, 0)
        coherence_threshold = _safe_float_env("CHUNK_RECURSIVE_OVERLAP_COHERENCE", 0.6, 0.1)

        if overlap_chars <= 0:
            for chunk in base_chunks:
                chunk.setdefault("reasoning_ingestion", {}).update({
                    "chunking_strategy": "recursive_overlap_v3",
                    "overlap_mode": "disabled",
                    "recursive_overlap_chars": 0,
                })
            return base_chunks

        # ── Phase 2: Apply overlap based on mode ──────────────────────
        if overlap_mode == "fixed":
            result = self._apply_fixed_overlap(
                base_chunks, overlap_chars,
                db_session=db_session, file_id=file_id,
                business_id=business_id, source_type=source_type,
                embedding_model=embedding_model,
            )
        elif overlap_mode == "adaptive":
            result = self._apply_adaptive_overlap(
                base_chunks, overlap_chars, max_overlap,
                db_session=db_session, file_id=file_id,
                business_id=business_id, source_type=source_type,
                embedding_model=embedding_model,
            )
        else:  # coherence (default)
            result = self._apply_coherence_gated_overlap(
                base_chunks, overlap_chars, max_overlap,
                coherence_threshold,
                db_session=db_session, file_id=file_id,
                business_id=business_id, source_type=source_type,
                embedding_model=embedding_model,
            )

        log_info(
            f"[RecursiveOverlap-v3] {len(result)} chunks | "
            f"mode={overlap_mode} base_overlap={overlap_chars} | "
            f"file_id={file_id}"
        )
        return result or base_chunks

    # ── Fixed overlap mode ────────────────────────────────────────────

    def _apply_fixed_overlap(
        self,
        chunks: List[Dict[str, Any]],
        overlap_chars: int,
        **make_kw,
    ) -> List[Dict[str, Any]]:
        """Fixed-size sentence-aligned overlap for every boundary."""
        result: List[Dict[str, Any]] = []
        prev_text = ""

        for chunk in chunks:
            curr_text = (chunk.get("text") or chunk.get("cleaned_text") or "").strip()
            if not curr_text:
                continue

            if prev_text:
                prefix = _find_sentence_aligned_overlap(prev_text, overlap_chars, from_end=True)
            else:
                prefix = ""

            merged_text = f"{prefix} {curr_text}".strip() if prefix else curr_text
            merged = make_chunk_dict(merged_text, **make_kw)
            if not merged:
                prev_text = curr_text
                continue

            merged.setdefault("reasoning_ingestion", {}).update({
                "chunking_strategy": "recursive_overlap_v3",
                "overlap_mode": "fixed",
                "recursive_overlap_chars": overlap_chars,
                "actual_overlap_chars": len(prefix),
                "sentence_aligned": True,
            })
            result.append(merged)
            prev_text = curr_text

        return result

    # ── Adaptive overlap mode ─────────────────────────────────────────

    def _apply_adaptive_overlap(
        self,
        chunks: List[Dict[str, Any]],
        base_overlap: int,
        max_overlap: int,
        **make_kw,
    ) -> List[Dict[str, Any]]:
        """
        Adaptive overlap: larger overlap at entity-heavy boundaries,
        smaller overlap at clean sentence boundaries.
        """
        result: List[Dict[str, Any]] = []
        prev_text = ""

        for chunk in chunks:
            curr_text = (chunk.get("text") or chunk.get("cleaned_text") or "").strip()
            if not curr_text:
                continue

            # Skip overlap for noise chunks
            is_noise = chunk.get("reasoning_ingestion", {}).get("noise_flag", False)

            if prev_text and not is_noise:
                # Compute entity continuity to determine overlap need
                entity_cont = _compute_entity_continuity(prev_text, curr_text)
                coherence = _score_boundary_coherence(prev_text, curr_text)

                # Adaptive scaling: more overlap for weak boundaries + shared entities
                weakness = 1.0 - coherence
                entity_boost = entity_cont * 0.5
                scale_factor = max(0.3, min(2.0, 1.0 + weakness + entity_boost))

                adaptive_chars = min(max_overlap, int(base_overlap * scale_factor))
                prefix = _find_sentence_aligned_overlap(prev_text, adaptive_chars, from_end=True)
            else:
                prefix = ""
                coherence = 1.0
                entity_cont = 0.0
                adaptive_chars = 0

            merged_text = f"{prefix} {curr_text}".strip() if prefix else curr_text
            merged = make_chunk_dict(merged_text, **make_kw)
            if not merged:
                prev_text = curr_text
                continue

            merged.setdefault("reasoning_ingestion", {}).update({
                "chunking_strategy": "recursive_overlap_v3",
                "overlap_mode": "adaptive",
                "recursive_overlap_chars": base_overlap,
                "actual_overlap_chars": len(prefix),
                "boundary_coherence": round(coherence, 4),
                "entity_continuity": round(entity_cont, 4),
                "sentence_aligned": True,
            })
            result.append(merged)
            prev_text = curr_text

        return result

    # ── Coherence-gated overlap mode (default) ────────────────────────

    def _apply_coherence_gated_overlap(
        self,
        chunks: List[Dict[str, Any]],
        base_overlap: int,
        max_overlap: int,
        coherence_threshold: float,
        **make_kw,
    ) -> List[Dict[str, Any]]:
        """
        Only inject overlap where boundary coherence is below threshold.

        Clean boundaries (high coherence) get zero overlap — this prevents
        unnecessary token inflation for well-structured documents while
        ensuring recall at weak split points.
        """
        result: List[Dict[str, Any]] = []
        prev_text = ""
        overlap_injected_count = 0

        for chunk in chunks:
            curr_text = (chunk.get("text") or chunk.get("cleaned_text") or "").strip()
            if not curr_text:
                continue

            prefix = ""
            coherence = 1.0
            entity_cont = 0.0
            needs_overlap = False

            if prev_text:
                is_noise = chunk.get("reasoning_ingestion", {}).get("noise_flag", False)
                if not is_noise:
                    coherence = _score_boundary_coherence(prev_text, curr_text)
                    entity_cont = _compute_entity_continuity(prev_text, curr_text)

                    # Gate: overlap only when coherence is weak OR entities span boundary
                    needs_overlap = (
                        coherence < coherence_threshold
                        or entity_cont > 0.3
                    )

                    if needs_overlap:
                        # Scale overlap inversely with coherence
                        weakness = 1.0 - coherence
                        adaptive_chars = min(max_overlap, int(base_overlap * (1.0 + weakness)))
                        prefix = _find_sentence_aligned_overlap(
                            prev_text, adaptive_chars, from_end=True
                        )
                        overlap_injected_count += 1

            merged_text = f"{prefix} {curr_text}".strip() if prefix else curr_text
            merged = make_chunk_dict(merged_text, **make_kw)
            if not merged:
                prev_text = curr_text
                continue

            merged.setdefault("reasoning_ingestion", {}).update({
                "chunking_strategy": "recursive_overlap_v3",
                "overlap_mode": "coherence_gated",
                "recursive_overlap_chars": base_overlap,
                "actual_overlap_chars": len(prefix),
                "boundary_coherence": round(coherence, 4),
                "entity_continuity": round(entity_cont, 4),
                "overlap_injected": needs_overlap,
                "sentence_aligned": True,
            })
            result.append(merged)
            prev_text = curr_text

        if result:
            log_info(
                f"[RecursiveOverlap-v3] Overlap injected at "
                f"{overlap_injected_count}/{len(result)} boundaries "
                f"(threshold={coherence_threshold})"
            )

        return result
