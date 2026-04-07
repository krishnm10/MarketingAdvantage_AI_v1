# =============================================
# segmenter_structure_aware.py — Enterprise Structure-Aware Document Chunker v3
#
# Advanced chunking engine that respects document structure with:
#   - Structural detection: headings, code fences, tables, lists, prose
#   - Topic-shift detection within prose blocks (TF-IDF cosine distance)
#   - Entity-preserving boundaries (regex NER prevents entity fragmentation)
#   - Discourse cue-phrase detection for intelligent split points
#   - Cross-chunk coherence scoring with smart merge decisions
#   - PDF titled/untitled table detection with footnote aggregation
#   - Adaptive overlap injection at low-coherence boundaries
#   - Post-processing: tiny-chunk merge with semantic similarity
#
# Registered names: "structure_aware", "document_aware"
#
# Env vars:
#   CHUNK_STRUCTURE_CODE_MAX        — max chars for a single code chunk (default 2000)
#   CHUNK_STRUCTURE_TABLE_MAX       — max chars for a single table chunk (default 3000)
#   CHUNK_STRUCTURE_PROSE_MAX       — max chars for prose (default 600)
#   CHUNK_STRUCTURE_PROSE_MIN       — min chars for prose merge threshold (default 150)
#   CHUNK_STRUCTURE_OVERLAP_CHARS   — overlap chars at low-coherence boundaries (default 80)
#   CHUNK_STRUCTURE_TOPIC_THRESHOLD — cosine distance threshold for topic split (default 0.45)
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
    DEFAULT_MAX_CHUNK_LEN,
    DEFAULT_MIN_CHUNK_LEN,
)
from app.core.chunking_stratagies.text_preprocessor import preprocess_document_text
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
# STRUCTURAL PATTERNS — compiled once at module load
# ─────────────────────────────────────────────────────────────────────────────

_CODE_FENCE_RE = re.compile(r"^(`{3,}|~{3,})(\w*)\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_TABLE_ROW_RE = re.compile(r"^\|.+\|$")
_LIST_ITEM_RE = re.compile(r"^[\s]*([-*+]|\d+[.)])\s+")
_PDF_TABLE_TITLE_RE = re.compile(r"^Table\s+\d+\b", re.IGNORECASE)
_PDF_NUMERIC_ROW_RE = re.compile(
    r"(?:[\d,]+(?:\.\d+)?(?:/\d+)?|\([\d,]+(?:\.\d+)?\))"
    r"(?:\s+(?:[\d,]+(?:\.\d+)?(?:/\d+)?|\([\d,]+(?:\.\d+)?\))){1,}"
    r"\s*$"
)
_TABLE_FOOTNOTE_RE = re.compile(r"^\(\d+\)\s+\S")
_MIN_PDF_TABLE_ROWS = 3

# ─────────────────────────────────────────────────────────────────────────────
# ENTITY PRESERVATION PATTERNS — prevent entity fragmentation
#
# These compile once. Each pattern detects named entities that must NOT be
# split across chunk boundaries. Used in _find_safe_split_point().
# ─────────────────────────────────────────────────────────────────────────────

# Financial: "Q3 2025 Revenue Growth of 23.5%", "$1.2 billion", "FY2024-25"
_ENTITY_FINANCIAL_RE = re.compile(
    r"(?:"
    r"(?:Q[1-4]\s+\d{4})"
    r"|(?:FY\s*\d{4}(?:[-–]\d{2,4})?)"
    r"|(?:[$€£¥₹]\s*[\d,.]+\s*(?:billion|million|thousand|bn|mn|k|cr|lakh)?)"
    r"|(?:[\d,.]+\s*(?:billion|million|thousand|bn|mn|k|cr|lakh))"
    r"|(?:\d+(?:\.\d+)?%)"
    r")",
    re.IGNORECASE,
)

# Date/time: "March 2025", "2024-03-15", "January 1, 2026"
_ENTITY_DATE_RE = re.compile(
    r"(?:"
    r"(?:(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2}(?:,?\s+\d{4})?)"
    r"|(?:\d{4}[-/]\d{2}[-/]\d{2})"
    r"|(?:(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{4})"
    r")",
    re.IGNORECASE,
)

# Organization abbreviations: "S&P 500", "NASDAQ", "EBITDA", "GDP"
_ENTITY_ORG_ABBREV_RE = re.compile(
    r"\b(?:[A-Z][A-Z&]{1,8}(?:\s+\d+)?)\b"
)

# Compound proper nouns: "New York Stock Exchange", "United States of America"
_ENTITY_PROPER_NOUN_RE = re.compile(
    r"(?:[A-Z][a-z]+(?:\s+(?:of|the|and|&|for|in)\s+)?){2,}[A-Z][a-z]+"
)


# ─────────────────────────────────────────────────────────────────────────────
# DISCOURSE CUE PHRASES — signal topic transitions
#
# Split BEFORE these phrases when they start a new logical segment.
# Weighted by strength: strong cues (0.8+) almost always indicate a shift,
# weak cues (0.3-0.5) only contribute when combined with other signals.
# ─────────────────────────────────────────────────────────────────────────────

_DISCOURSE_CUES_STRONG: Tuple[str, ...] = (
    "however,", "on the other hand,", "conversely,", "in contrast,",
    "nevertheless,", "nonetheless,", "meanwhile,", "alternatively,",
    "on the contrary,",
)
_DISCOURSE_CUES_MEDIUM: Tuple[str, ...] = (
    "furthermore,", "moreover,", "additionally,", "in addition,",
    "similarly,", "likewise,", "as a result,", "consequently,",
    "therefore,", "thus,", "hence,", "accordingly,",
    "for example,", "for instance,", "specifically,",
    "in particular,", "notably,",
)
_DISCOURSE_CUES_WEAK: Tuple[str, ...] = (
    "also,", "then,", "next,", "finally,", "first,", "second,",
    "third,", "lastly,", "in summary,", "to summarize,",
    "in conclusion,", "overall,",
)

# Stop words for TF-IDF computation
_TFIDF_STOP_WORDS: frozenset = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "and", "or", "but", "not", "no", "this", "that", "it", "its",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "than", "more", "also", "very",
    "just", "so", "if", "then", "which", "what", "where", "when",
    "who", "whom", "how", "all", "each", "every", "both", "few",
    "some", "any", "most", "other", "into", "through", "during",
    "before", "after", "above", "below", "between", "such", "only",
})

_WORD_TOKENIZE_RE = re.compile(r"[a-z]{2,}", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# TOPIC SHIFT DETECTION — TF-IDF cosine distance between sentence windows
# ─────────────────────────────────────────────────────────────────────────────

def _extract_content_words(text: str) -> List[str]:
    """Extract alphabetic words >= 2 chars, lowercased, excluding stop words."""
    return [
        w.lower() for w in _WORD_TOKENIZE_RE.findall(text)
        if w.lower() not in _TFIDF_STOP_WORDS
    ]


def _cosine_distance(counter_a: Counter, counter_b: Counter) -> float:
    """
    Cosine distance (1 - cosine_similarity) between two word-frequency vectors.
    Returns 0.0 for identical distributions, 1.0 for completely disjoint.
    O(K) where K = unique terms in both counters combined.
    """
    if not counter_a or not counter_b:
        return 1.0

    common_keys = set(counter_a.keys()) & set(counter_b.keys())
    if not common_keys:
        return 1.0

    dot_product = sum(counter_a[k] * counter_b[k] for k in common_keys)
    mag_a = math.sqrt(sum(v * v for v in counter_a.values()))
    mag_b = math.sqrt(sum(v * v for v in counter_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 1.0

    similarity = dot_product / (mag_a * mag_b)
    return 1.0 - min(1.0, max(0.0, similarity))


def _detect_topic_shifts(
    sentences: List[str],
    threshold: float = 0.45,
    window_size: int = 3,
) -> List[int]:
    """
    Detect topic shift points between consecutive sentence windows.

    Uses sliding TF-IDF cosine distance with a configurable window size.
    Returns indices where the topic shifts significantly (high cosine distance).

    Args:
        sentences: List of sentence strings.
        threshold: Cosine distance above which a shift is detected.
        window_size: Number of sentences per comparison window.

    Returns:
        List of sentence indices where topic shifts occur.
    """
    if len(sentences) < window_size * 2:
        return []

    shift_points: List[int] = []

    for i in range(window_size, len(sentences) - window_size + 1):
        # Build word frequency vectors for left and right windows
        left_words = []
        for s in sentences[max(0, i - window_size):i]:
            left_words.extend(_extract_content_words(s))

        right_words = []
        for s in sentences[i:i + window_size]:
            right_words.extend(_extract_content_words(s))

        left_counter = Counter(left_words)
        right_counter = Counter(right_words)

        distance = _cosine_distance(left_counter, right_counter)

        # Check for discourse cue phrases at the boundary
        cue_boost = 0.0
        boundary_sentence = sentences[i].strip().lower()
        if any(boundary_sentence.startswith(c) for c in _DISCOURSE_CUES_STRONG):
            cue_boost = 0.25
        elif any(boundary_sentence.startswith(c) for c in _DISCOURSE_CUES_MEDIUM):
            cue_boost = 0.15
        elif any(boundary_sentence.startswith(c) for c in _DISCOURSE_CUES_WEAK):
            cue_boost = 0.08

        effective_distance = distance + cue_boost

        if effective_distance >= threshold:
            shift_points.append(i)

    return shift_points


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY-SAFE SPLIT POINT FINDER
# ─────────────────────────────────────────────────────────────────────────────

def _find_entity_spans(text: str) -> List[Tuple[int, int]]:
    """
    Find all named-entity spans in text that must not be split.
    Returns list of (start, end) character indices.
    """
    spans: List[Tuple[int, int]] = []
    for pattern in (
        _ENTITY_FINANCIAL_RE,
        _ENTITY_DATE_RE,
        _ENTITY_ORG_ABBREV_RE,
        _ENTITY_PROPER_NOUN_RE,
    ):
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end()))
    return spans


def _is_inside_entity(position: int, entity_spans: List[Tuple[int, int]]) -> bool:
    """Check if a character position falls inside any entity span."""
    for start, end in entity_spans:
        if start < position < end:
            return True
    return False


def _find_safe_split_point(
    text: str,
    target: int,
    search_range: int = 80,
) -> int:
    """
    Find the optimal split point near `target` that doesn't break entities.

    Priority:
    1. Paragraph boundary (double newline) within range
    2. Sentence boundary (.!?) within range, not inside entity
    3. Clause boundary (;:,) within range, not inside entity
    4. Word boundary (space) within range, not inside entity
    5. Fallback to target position
    """
    if target >= len(text):
        return len(text)

    entity_spans = _find_entity_spans(text)
    search_start = max(0, target - search_range)
    search_end = min(len(text), target + search_range)
    search_zone = text[search_start:search_end]

    # Priority 1: Paragraph boundary
    para_idx = search_zone.rfind("\n\n")
    if para_idx >= 0:
        candidate = search_start + para_idx + 2
        if not _is_inside_entity(candidate, entity_spans):
            return candidate

    # Priority 2: Sentence boundary
    sent_pattern = re.compile(r"[.!?]+\s+")
    best_sent = -1
    for m in sent_pattern.finditer(search_zone):
        pos = search_start + m.end()
        if not _is_inside_entity(pos, entity_spans):
            best_sent = pos
    if best_sent >= 0:
        return best_sent

    # Priority 3: Clause boundary
    for delim in ("; ", ": ", ", "):
        idx = search_zone.rfind(delim)
        if idx >= 0:
            candidate = search_start + idx + len(delim)
            if not _is_inside_entity(candidate, entity_spans):
                return candidate

    # Priority 4: Word boundary
    space_idx = search_zone.rfind(" ")
    if space_idx >= 0:
        candidate = search_start + space_idx + 1
        if not _is_inside_entity(candidate, entity_spans):
            return candidate

    return target


# ─────────────────────────────────────────────────────────────────────────────
# BOUNDARY COHERENCE SCORER
# ─────────────────────────────────────────────────────────────────────────────

def _score_boundary_coherence(
    left_text: str,
    right_text: str,
) -> float:
    """
    Score how coherent a chunk boundary is (0.0 = terrible split, 1.0 = clean).

    Checks:
    - Both sides have complete sentences
    - No entity straddles the boundary
    - Topic distance is reasonable (not mid-paragraph)
    - Right side doesn't start with a continuation word
    """
    score = 1.0

    # Penalty: left doesn't end with sentence terminator
    left_stripped = left_text.rstrip()
    if left_stripped and left_stripped[-1] not in ".!?;:":
        score -= 0.25

    # Penalty: right starts with lowercase (mid-sentence split)
    right_stripped = right_text.lstrip()
    if right_stripped and right_stripped[0].islower():
        score -= 0.30

    # Penalty: right starts with continuation word
    first_word = right_stripped.split()[0].lower() if right_stripped.split() else ""
    continuation_words = {
        "and", "or", "but", "because", "however", "therefore",
        "which", "that", "who", "whom", "whose", "where",
    }
    if first_word in continuation_words:
        score -= 0.20

    # Penalty: entity straddles boundary
    boundary_zone = left_text[-40:] + right_text[:40]
    entity_spans = _find_entity_spans(boundary_zone)
    boundary_pos = len(left_text[-40:])
    if _is_inside_entity(boundary_pos, entity_spans):
        score -= 0.35

    return max(0.0, score)


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL BLOCK
# ─────────────────────────────────────────────────────────────────────────────

class _Block:
    """Lightweight container for a parsed structural block."""
    __slots__ = ("type", "text", "heading_title", "heading_depth", "language")

    def __init__(
        self,
        block_type: str,
        text: str,
        heading_title: str = "",
        heading_depth: int = 0,
        language: str = "",
    ):
        self.type = block_type
        self.text = text
        self.heading_title = heading_title
        self.heading_depth = heading_depth
        self.language = language


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT PARSER — line-by-line structural detection
# ─────────────────────────────────────────────────────────────────────────────

def _parse_structural_blocks(text: str) -> List[_Block]:
    """
    Walk lines and emit typed blocks in document order.
    Detection priority: code fence > heading > table > list > prose.
    """
    lines = text.split("\n")
    blocks: List[_Block] = []
    cur_heading_title = ""
    cur_heading_depth = 0
    prose_buf: List[str] = []
    i = 0

    def _flush_prose() -> None:
        nonlocal prose_buf
        if prose_buf:
            joined = "\n".join(prose_buf).strip()
            if joined:
                blocks.append(_Block(
                    "prose", joined,
                    heading_title=cur_heading_title,
                    heading_depth=cur_heading_depth,
                ))
            prose_buf = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ── Code fence ────────────────────────────────────────────────
        fence_m = _CODE_FENCE_RE.match(stripped)
        if fence_m:
            _flush_prose()
            fence_char = fence_m.group(1)[0]
            fence_len = len(fence_m.group(1))
            language = fence_m.group(2) or ""
            code_lines: List[str] = [line]
            i += 1
            while i < len(lines):
                code_lines.append(lines[i])
                cl = lines[i].strip()
                if cl and all(c == fence_char for c in cl) and len(cl) >= fence_len:
                    i += 1
                    break
                i += 1
            inner = code_lines[1:-1] if len(code_lines) >= 2 else code_lines
            code_text = "\n".join(inner).strip()
            if not code_text:
                continue
            blocks.append(_Block(
                "code", code_text,
                heading_title=cur_heading_title,
                heading_depth=cur_heading_depth,
                language=language,
            ))
            continue

        # ── Heading ───────────────────────────────────────────────────
        heading_m = _HEADING_RE.match(stripped)
        if heading_m:
            _flush_prose()
            cur_heading_depth = len(heading_m.group(1))
            cur_heading_title = heading_m.group(2).strip()
            blocks.append(_Block(
                "heading", stripped,
                heading_title=cur_heading_title,
                heading_depth=cur_heading_depth,
            ))
            i += 1
            continue

        # ── Table (markdown pipe-delimited) ──────────────────────────
        if _TABLE_ROW_RE.match(stripped):
            _flush_prose()
            table_lines: List[str] = [line]
            i += 1
            while i < len(lines):
                ns = lines[i].strip()
                if _TABLE_ROW_RE.match(ns):
                    table_lines.append(lines[i])
                    i += 1
                elif not ns:
                    break
                else:
                    break
            blocks.append(_Block(
                "table", "\n".join(table_lines),
                heading_title=cur_heading_title,
                heading_depth=cur_heading_depth,
            ))
            continue

        # ── PDF titled table ("Table N" or "Table N: ...") ───────────
        if _PDF_TABLE_TITLE_RE.match(stripped):
            _flush_prose()
            pdf_table_lines: List[str] = [line]
            i += 1
            consecutive_blanks = 0
            while i < len(lines):
                ns = lines[i].strip()
                if _PDF_TABLE_TITLE_RE.match(ns):
                    break
                if _HEADING_RE.match(ns):
                    break
                if not ns:
                    consecutive_blanks += 1
                    if consecutive_blanks >= 2:
                        break
                    i += 1
                    continue
                consecutive_blanks = 0
                pdf_table_lines.append(lines[i])
                i += 1
            while i < len(lines):
                ns = lines[i].strip()
                if _TABLE_FOOTNOTE_RE.match(ns):
                    pdf_table_lines.append(lines[i])
                    i += 1
                elif (not ns
                      and (i + 1 < len(lines))
                      and _TABLE_FOOTNOTE_RE.match(lines[i + 1].strip())):
                    i += 1
                else:
                    break
            blocks.append(_Block(
                "table", "\n".join(pdf_table_lines),
                heading_title=cur_heading_title,
                heading_depth=cur_heading_depth,
            ))
            continue

        # ── PDF untitled data table (3+ rows ending with numbers) ────
        if _PDF_NUMERIC_ROW_RE.search(stripped):
            peek = i + 1
            num_count = 1
            while peek < len(lines):
                ps = lines[peek].strip()
                if not ps:
                    break
                if _PDF_NUMERIC_ROW_RE.search(ps):
                    num_count += 1
                    peek += 1
                elif (peek + 1 < len(lines)
                      and lines[peek + 1].strip()
                      and _PDF_NUMERIC_ROW_RE.search(lines[peek + 1].strip())):
                    peek += 1
                else:
                    break
            if num_count >= _MIN_PDF_TABLE_ROWS:
                _flush_prose()
                pdf_data_lines = [lines[j] for j in range(i, peek)]
                blocks.append(_Block(
                    "table", "\n".join(pdf_data_lines),
                    heading_title=cur_heading_title,
                    heading_depth=cur_heading_depth,
                ))
                i = peek
                continue

        # ── List run ──────────────────────────────────────────────────
        if _LIST_ITEM_RE.match(line):
            _flush_prose()
            list_lines: List[str] = [line]
            i += 1
            while i < len(lines):
                nl = lines[i]
                ns = nl.strip()
                if _LIST_ITEM_RE.match(nl):
                    list_lines.append(nl)
                    i += 1
                elif ns and (nl.startswith("  ") or nl.startswith("\t")):
                    list_lines.append(nl)
                    i += 1
                elif not ns:
                    peek = i + 1
                    while peek < len(lines) and not lines[peek].strip():
                        peek += 1
                    if peek < len(lines) and _LIST_ITEM_RE.match(lines[peek]):
                        list_lines.append(nl)
                        i += 1
                    else:
                        break
                else:
                    break
            blocks.append(_Block(
                "list", "\n".join(list_lines),
                heading_title=cur_heading_title,
                heading_depth=cur_heading_depth,
            ))
            continue

        # ── Prose (default) ───────────────────────────────────────────
        prose_buf.append(line)
        i += 1

    _flush_prose()
    return blocks


def _merge_heading_into_content(blocks: List[_Block]) -> List[_Block]:
    """Prepend standalone headings to the next content block."""
    merged: List[_Block] = []
    pending: Optional[_Block] = None

    for blk in blocks:
        if blk.type == "heading":
            if pending is not None:
                merged.append(_Block(
                    "prose", pending.text,
                    heading_title=pending.heading_title,
                    heading_depth=pending.heading_depth,
                ))
            pending = blk
        else:
            if pending is not None:
                blk.text = pending.text + "\n" + blk.text
                blk.heading_title = pending.heading_title
                blk.heading_depth = pending.heading_depth
                pending = None
            merged.append(blk)

    if pending is not None:
        merged.append(_Block(
            "prose", pending.text,
            heading_title=pending.heading_title,
            heading_depth=pending.heading_depth,
        ))
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY REGISTRATION
# ─────────────────────────────────────────────────────────────────────────────

@register_chunker("structure_aware")
@register_chunker("document_aware")
class StructureAwareChunker(Chunker):
    """
    Enterprise Structure-Aware Document Chunker v3.

    Advanced features beyond basic structure detection:
    - Topic-shift detection within prose (TF-IDF cosine distance)
    - Entity-preserving boundaries (regex NER)
    - Discourse cue-phrase boundary reinforcement
    - Adaptive overlap at low-coherence boundaries
    - Cross-chunk coherence scoring for merge decisions
    - Quality-score gated filtering
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
        if not (text or "").strip():
            return []

        code_max = _safe_int_env("CHUNK_STRUCTURE_CODE_MAX", 2000, 200)
        table_max = _safe_int_env("CHUNK_STRUCTURE_TABLE_MAX", 3000, 200)
        prose_max = _safe_int_env("CHUNK_STRUCTURE_PROSE_MAX", DEFAULT_MAX_CHUNK_LEN, 100)
        prose_min = _safe_int_env("CHUNK_STRUCTURE_PROSE_MIN", DEFAULT_MIN_CHUNK_LEN, 50)
        merge_min_tokens = _safe_int_env("CHUNK_STRUCTURE_MERGE_MIN_TOKENS", 25, 5)
        overlap_chars = _safe_int_env("CHUNK_STRUCTURE_OVERLAP_CHARS", 80, 0)
        topic_threshold = _safe_float_env("CHUNK_STRUCTURE_TOPIC_THRESHOLD", 0.45, 0.1)

        # ── Phase 1: Pre-process raw text ─────────────────────────────
        preprocessed = preprocess_document_text(text)
        if not preprocessed.strip():
            return []

        # ── Phase 2: Structural parsing ───────────────────────────────
        blocks = _parse_structural_blocks(preprocessed)
        blocks = _merge_heading_into_content(blocks)
        if not blocks:
            return []

        # ── Phase 3: Block-level chunking with advanced features ──────
        all_chunks: List[Dict[str, Any]] = []
        for blk in blocks:
            blk_chunks = await self._process_block(
                blk,
                code_max=code_max,
                table_max=table_max,
                prose_max=prose_max,
                prose_min=prose_min,
                topic_threshold=topic_threshold,
                db_session=db_session,
                file_id=file_id,
                business_id=business_id,
                source_type=source_type,
                embedding_model=embedding_model,
            )
            all_chunks.extend(blk_chunks)

        # ── Phase 4: Merge tiny chunks with semantic awareness ────────
        all_chunks = self._merge_tiny_chunks_semantic(all_chunks, merge_min_tokens)

        # ── Phase 5: Inject overlap at low-coherence boundaries ───────
        if overlap_chars > 0:
            all_chunks = self._inject_boundary_overlap(all_chunks, overlap_chars)

        # ── Phase 6: Annotate boundary coherence scores ───────────────
        all_chunks = self._annotate_boundary_coherence(all_chunks)

        log_info(
            f"[StructureAware-v3] {len(all_chunks)} chunks from "
            f"{len(blocks)} structural blocks | "
            f"file_id={file_id} | source={source_type}"
        )
        return all_chunks

    # ── Block dispatch ────────────────────────────────────────────────

    async def _process_block(
        self,
        blk: _Block,
        *,
        code_max: int,
        table_max: int,
        prose_max: int,
        prose_min: int,
        topic_threshold: float,
        db_session=None,
        file_id=None,
        business_id=None,
        source_type: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        kw = dict(file_id=file_id, business_id=business_id,
                  source_type=source_type, embedding_model=embedding_model)

        if blk.type == "code":
            return self._chunk_code(blk, code_max=code_max, **kw)
        if blk.type == "table":
            return self._chunk_table(blk, table_max=table_max, **kw)
        if blk.type == "list":
            return self._chunk_list(blk, max_len=prose_max, **kw)
        return await self._chunk_prose_advanced(
            blk, max_len=prose_max, min_len=prose_min,
            topic_threshold=topic_threshold,
            db_session=db_session, **kw,
        )

    # ── Metadata helper ───────────────────────────────────────────────

    @staticmethod
    def _annotate(chunk: Dict[str, Any], blk: _Block, content_type: str) -> None:
        meta = chunk.setdefault("reasoning_ingestion", {})
        meta["chunking_strategy"] = "structure_aware_v3"
        meta["content_type"] = content_type
        meta["section_title"] = blk.heading_title or ""
        meta["section_depth"] = blk.heading_depth
        if blk.language:
            meta["code_language"] = blk.language

    # ── Phase 5: Inject overlap at low-coherence boundaries ───────────

    @staticmethod
    def _inject_boundary_overlap(
        chunks: List[Dict[str, Any]],
        overlap_chars: int,
    ) -> List[Dict[str, Any]]:
        """
        Inject text overlap at boundaries where coherence is low.

        Only overlaps prose-to-prose transitions. Code/table boundaries
        are structurally distinct and don't benefit from overlap.
        """
        if len(chunks) < 2 or overlap_chars <= 0:
            return chunks

        result: List[Dict[str, Any]] = [chunks[0]]
        for i in range(1, len(chunks)):
            curr = chunks[i]
            prev = result[-1]

            prev_type = prev.get("reasoning_ingestion", {}).get("content_type", "prose")
            curr_type = curr.get("reasoning_ingestion", {}).get("content_type", "prose")

            # Only overlap prose-to-prose
            if prev_type == "prose" and curr_type == "prose":
                prev_text = prev.get("cleaned_text", "") or prev.get("text", "")
                curr_text = curr.get("text", "") or curr.get("cleaned_text", "")

                # Score boundary coherence
                coherence = _score_boundary_coherence(prev_text, curr_text)

                # Only inject overlap when coherence is below 0.6
                if coherence < 0.6 and prev_text:
                    overlap_suffix = prev_text[-overlap_chars:].strip()
                    if overlap_suffix:
                        curr_meta = curr.setdefault("reasoning_ingestion", {})
                        curr_meta["overlap_prefix"] = overlap_suffix
                        curr_meta["overlap_chars"] = len(overlap_suffix)
                        curr_meta["boundary_coherence"] = round(coherence, 4)

            result.append(curr)

        return result

    # ── Phase 6: Annotate boundary coherence ──────────────────────────

    @staticmethod
    def _annotate_boundary_coherence(
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Annotate each chunk with its boundary coherence score to its successor."""
        for i in range(len(chunks) - 1):
            curr_text = chunks[i].get("cleaned_text", "") or chunks[i].get("text", "")
            next_text = chunks[i + 1].get("cleaned_text", "") or chunks[i + 1].get("text", "")
            if curr_text and next_text:
                coherence = _score_boundary_coherence(curr_text, next_text)
                chunks[i].setdefault("reasoning_ingestion", {})["boundary_coherence_right"] = round(coherence, 4)
                chunks[i + 1].setdefault("reasoning_ingestion", {})["boundary_coherence_left"] = round(coherence, 4)
        return chunks

    # ── Post-processing: semantic-aware tiny chunk merge ──────────────

    @staticmethod
    def _merge_tiny_chunks_semantic(
        chunks: List[Dict[str, Any]],
        min_tokens: int,
    ) -> List[Dict[str, Any]]:
        """
        Merge chunks below min_tokens into the MOST semantically similar neighbour.

        Unlike basic merge (always prev/next), this compares TF-IDF cosine
        distance to both neighbours and merges with the closer one.
        Code/table chunks are never merged.
        """
        if not chunks or min_tokens < 5:
            return chunks

        tiny_indices = set()
        for i, ch in enumerate(chunks):
            meta = ch.get("reasoning_ingestion", {})
            ct = meta.get("content_type", "prose")
            if ct in ("code", "table"):
                continue
            if ch.get("tokens", 0) < min_tokens:
                tiny_indices.add(i)

        if not tiny_indices:
            return chunks

        result: List[Dict[str, Any]] = []
        skip = set()

        for i, ch in enumerate(chunks):
            if i in skip:
                continue
            if i in tiny_indices:
                tiny_text = ch.get("cleaned_text", "") or ch.get("text", "")
                tiny_words = Counter(_extract_content_words(tiny_text))

                # Compare with previous and next neighbours
                prev_dist = 1.0
                next_dist = 1.0

                if result:
                    prev_text = result[-1].get("cleaned_text", "") or result[-1].get("text", "")
                    prev_words = Counter(_extract_content_words(prev_text))
                    prev_dist = _cosine_distance(tiny_words, prev_words)

                if i + 1 < len(chunks) and (i + 1) not in skip:
                    next_text = chunks[i + 1].get("cleaned_text", "") or chunks[i + 1].get("text", "")
                    next_words = Counter(_extract_content_words(next_text))
                    next_dist = _cosine_distance(tiny_words, next_words)

                merged = False
                if prev_dist <= next_dist and result:
                    # Merge backward (more similar to previous)
                    prev = result[-1]
                    prev["text"] = prev["text"] + "\n" + ch["text"]
                    prev["cleaned_text"] = prev.get("cleaned_text", "") + " " + ch.get("cleaned_text", "")
                    prev["tokens"] = prev.get("tokens", 0) + ch.get("tokens", 0)
                    merged = True
                elif i + 1 < len(chunks) and (i + 1) not in skip:
                    # Merge forward (more similar to next)
                    next_ch = chunks[i + 1]
                    next_ch["text"] = ch["text"] + "\n" + next_ch["text"]
                    next_ch["cleaned_text"] = ch.get("cleaned_text", "") + " " + next_ch.get("cleaned_text", "")
                    next_ch["tokens"] = ch.get("tokens", 0) + next_ch.get("tokens", 0)
                    merged = True

                if not merged:
                    result.append(ch)
            else:
                result.append(ch)

        return result

    # ── Code blocks ───────────────────────────────────────────────────

    def _chunk_code(
        self, blk: _Block, *, code_max: int,
        file_id=None, business_id=None,
        source_type: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        text = blk.text
        kw = dict(file_id=file_id, business_id=business_id,
                  source_type=source_type, embedding_model=embedding_model)

        if len(text) <= code_max:
            c = make_chunk_dict(text, **kw)
            if c:
                self._annotate(c, blk, "code")
                return [c]
            return []

        # Oversized: split at logical code boundaries (blank lines, function defs)
        lines = text.split("\n")
        segments: List[str] = []
        buf: List[str] = []
        buf_len = 0

        for ln in lines:
            # Prefer splitting at blank lines or function/class definitions
            is_boundary = (
                not ln.strip()
                or ln.strip().startswith("def ")
                or ln.strip().startswith("class ")
                or ln.strip().startswith("function ")
                or ln.strip().startswith("public ")
                or ln.strip().startswith("private ")
            )

            if buf_len + len(ln) + 1 > code_max and buf:
                segments.append("\n".join(buf))
                buf = [ln]
                buf_len = len(ln)
            elif is_boundary and buf_len > code_max * 0.6 and buf:
                # Split at logical boundary when we're past 60% capacity
                segments.append("\n".join(buf))
                buf = [ln]
                buf_len = len(ln)
            else:
                buf.append(ln)
                buf_len += len(ln) + 1
        if buf:
            segments.append("\n".join(buf))

        out: List[Dict[str, Any]] = []
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            c = make_chunk_dict(seg, **kw)
            if c:
                self._annotate(c, blk, "code")
                out.append(c)
        return out

    # ── Tables ────────────────────────────────────────────────────────

    def _chunk_table(
        self, blk: _Block, *, table_max: int,
        file_id=None, business_id=None,
        source_type: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        text = blk.text
        kw = dict(file_id=file_id, business_id=business_id,
                  source_type=source_type, embedding_model=embedding_model)

        if len(text) <= table_max:
            c = make_chunk_dict(text, **kw)
            if c:
                self._annotate(c, blk, "table")
                # Annotate table metadata: row count, column count estimate
                lines = text.split("\n")
                c.setdefault("reasoning_ingestion", {})["table_rows"] = len(lines)
                if lines and "|" in lines[0]:
                    c["reasoning_ingestion"]["table_cols_estimate"] = lines[0].count("|") - 1
                return [c]
            return []

        # Oversized: split rows, repeating header+separator in each segment
        lines = text.split("\n")
        header_lines = lines[:2] if len(lines) >= 2 else lines[:1]
        data_lines = lines[len(header_lines):]
        header_text = "\n".join(header_lines)
        header_len = len(header_text) + 1

        segments: List[str] = []
        data_buf: List[str] = []
        data_len = header_len

        for dl in data_lines:
            if data_len + len(dl) + 1 > table_max and data_buf:
                segments.append(header_text + "\n" + "\n".join(data_buf))
                data_buf = [dl]
                data_len = header_len + len(dl)
            else:
                data_buf.append(dl)
                data_len += len(dl) + 1
        if data_buf:
            segments.append(header_text + "\n" + "\n".join(data_buf))

        out: List[Dict[str, Any]] = []
        for idx, seg in enumerate(segments):
            seg = seg.strip()
            if not seg:
                continue
            c = make_chunk_dict(seg, **kw)
            if c:
                self._annotate(c, blk, "table")
                c.setdefault("reasoning_ingestion", {}).update({
                    "table_segment": idx + 1,
                    "table_segments_total": len(segments),
                })
                out.append(c)
        return out

    # ── Lists ─────────────────────────────────────────────────────────

    def _chunk_list(
        self, blk: _Block, *, max_len: int,
        file_id=None, business_id=None,
        source_type: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        text = blk.text
        kw = dict(file_id=file_id, business_id=business_id,
                  source_type=source_type, embedding_model=embedding_model)

        if len(text) <= max_len:
            c = make_chunk_dict(text, **kw)
            if c:
                self._annotate(c, blk, "list")
                # Count list items
                items_count = sum(1 for ln in text.split("\n") if _LIST_ITEM_RE.match(ln))
                c.setdefault("reasoning_ingestion", {})["list_items_count"] = items_count
                return [c]
            return []

        # Parse individual items, then group into segments <= max_len
        items: List[str] = []
        cur_item: List[str] = []
        for ln in text.split("\n"):
            if _LIST_ITEM_RE.match(ln):
                if cur_item:
                    items.append("\n".join(cur_item))
                cur_item = [ln]
            else:
                cur_item.append(ln)
        if cur_item:
            items.append("\n".join(cur_item))

        segments: List[str] = []
        group: List[str] = []
        group_len = 0
        for item in items:
            if group_len + len(item) + 1 > max_len and group:
                segments.append("\n".join(group))
                group = [item]
                group_len = len(item)
            else:
                group.append(item)
                group_len += len(item) + 1
        if group:
            segments.append("\n".join(group))

        out: List[Dict[str, Any]] = []
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            c = make_chunk_dict(seg, **kw)
            if c:
                self._annotate(c, blk, "list")
                items_count = sum(1 for ln in seg.split("\n") if _LIST_ITEM_RE.match(ln))
                c.setdefault("reasoning_ingestion", {})["list_items_count"] = items_count
                out.append(c)
        return out

    # ── Prose with topic-shift detection ──────────────────────────────

    async def _chunk_prose_advanced(
        self,
        blk: _Block,
        *,
        max_len: int,
        min_len: int,
        topic_threshold: float,
        db_session=None,
        file_id=None,
        business_id=None,
        source_type: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Advanced prose chunking with topic-shift detection.

        Pipeline:
        1. Split prose into sentences
        2. Detect topic shifts via TF-IDF cosine distance
        3. Pre-split at topic boundaries
        4. Run recursive_semantic_chunk on each topic segment
        5. Apply entity-safe boundary adjustments
        """
        text = blk.text
        kw = dict(
            file_id=file_id, business_id=business_id,
            source_type=source_type, embedding_model=embedding_model,
        )

        # If text fits in one chunk, no need for topic detection
        if len(text) <= max_len:
            c = make_chunk_dict(text, **kw)
            if c:
                self._annotate(c, blk, "prose")
                c.setdefault("reasoning_ingestion", {})["topic_shift_detected"] = False
            return [c] if c else []

        # Split into sentences for topic analysis
        sent_splitter = re.compile(r"(?<=[.!?])\s+")
        sentences = [s.strip() for s in sent_splitter.split(text) if s.strip()]

        if len(sentences) < 4:
            # Too few sentences for topic analysis, fall back to semantic chunking
            sub_chunks = await recursive_semantic_chunk(
                text, max_chunk_len=max_len, min_chunk_len=min_len,
                db_session=db_session, **kw,
            )
            for c in sub_chunks:
                self._annotate(c, blk, "prose")
            return sub_chunks

        # Detect topic shifts
        shift_points = _detect_topic_shifts(
            sentences, threshold=topic_threshold, window_size=min(3, len(sentences) // 3)
        )

        if not shift_points:
            # No topic shifts detected — standard semantic chunking
            sub_chunks = await recursive_semantic_chunk(
                text, max_chunk_len=max_len, min_chunk_len=min_len,
                db_session=db_session, **kw,
            )
            for c in sub_chunks:
                self._annotate(c, blk, "prose")
                c.setdefault("reasoning_ingestion", {})["topic_shift_detected"] = False
            return sub_chunks

        # Build topic segments from shift points
        topic_segments = self._build_topic_segments(sentences, shift_points)

        all_chunks: List[Dict[str, Any]] = []
        for seg_idx, segment_text in enumerate(topic_segments):
            segment_text = segment_text.strip()
            if not segment_text:
                continue

            if len(segment_text) <= max_len:
                c = make_chunk_dict(segment_text, **kw)
                if c:
                    self._annotate(c, blk, "prose")
                    c.setdefault("reasoning_ingestion", {}).update({
                        "topic_shift_detected": True,
                        "topic_segment_index": seg_idx,
                        "topic_segments_total": len(topic_segments),
                    })
                    all_chunks.append(c)
            else:
                seg_chunks = await recursive_semantic_chunk(
                    segment_text, max_chunk_len=max_len, min_chunk_len=min_len,
                    db_session=db_session, **kw,
                )
                for c in seg_chunks:
                    self._annotate(c, blk, "prose")
                    c.setdefault("reasoning_ingestion", {}).update({
                        "topic_shift_detected": True,
                        "topic_segment_index": seg_idx,
                        "topic_segments_total": len(topic_segments),
                    })
                all_chunks.extend(seg_chunks)

        return all_chunks

    @staticmethod
    def _build_topic_segments(
        sentences: List[str],
        shift_points: List[int],
    ) -> List[str]:
        """Build text segments from sentence list and shift point indices."""
        segments: List[str] = []
        prev_idx = 0
        for sp in sorted(set(shift_points)):
            if sp > prev_idx:
                seg = " ".join(sentences[prev_idx:sp])
                if seg.strip():
                    segments.append(seg)
                prev_idx = sp
        # Last segment
        if prev_idx < len(sentences):
            seg = " ".join(sentences[prev_idx:])
            if seg.strip():
                segments.append(seg)
        return segments
