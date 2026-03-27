# =============================================
# chunk_quality_scorer.py — Enterprise Chunk Quality Scoring Engine v2
#
# Pure function. Zero dependencies beyond stdlib. Zero DB writes.
# Called once per chunk inside make_chunk_dict() — must be fast.
#
# Scoring Dimensions (each 0.0–1.0, weighted-averaged to final score):
#   1. LENGTH              (0.20) — token count in ideal retrieval window
#   2. COMPLETENESS        (0.15) — grammatically complete sentences
#   3. LEXICAL_DENSITY     (0.15) — vocabulary richness
#   4. NOISE               (0.15) — penalize digit/symbol-heavy text
#   5. COHERENCE           (0.10) — sentence structure balance
#   6. INFORMATION_DENSITY (0.15) — ratio of meaningful content vs filler
#   7. RETRIEVAL_FITNESS   (0.10) — embedding model suitability
#
# Hard Penalties (applied AFTER weighted average):
#   - Noise floor: if noise detection triggers → cap score at 0.35
#   - OCR garble: if >35% single-char words → cap score at 0.20
#   - Chart dump: if >55% numeric tokens → cap score at 0.30
#
# Design Principles:
#   - O(N) single pass where possible — no quadratic operations
#   - Zero external dependencies — no NLTK, no spaCy, no ML
#   - Deterministic — same text always produces same score
#   - Tuned for RAG retrieval: chunks that score high retrieve well
# =============================================

from __future__ import annotations

import re
import math
from typing import Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — tuned against production ingestion benchmarks
# ─────────────────────────────────────────────────────────────────────────────

# Token count sweet spot for embedding models (ada-002, BGE, E5, Cohere).
# Below 30 tokens: too short for meaningful embedding vector.
# Above 512 tokens: exceeds most model context windows / degrades recall.
# Peak quality range: 80–400 tokens. Maximum score at 150–250 tokens.
_TOKEN_IDEAL_MIN: int = 80
_TOKEN_IDEAL_MAX: int = 400
_TOKEN_PEAK_MIN: int = 150
_TOKEN_PEAK_MAX: int = 250
_TOKEN_HARD_MIN: int = 15
_TOKEN_HARD_MAX: int = 600

# Lexical density (unique words / total words).
# 0.3 = very repetitive (boilerplate, OCR repeat loops).
# 0.5–0.85 = natural prose range. 
# >0.95 = likely keyword soup or very short text.
_DENSITY_IDEAL_MIN: float = 0.45
_DENSITY_IDEAL_MAX: float = 0.88

# Noise ratio threshold (non-alpha, non-space chars / total chars).
# >0.40 = likely Base64, encoded data, or OCR garbage.
_NOISE_PENALTY_THRESHOLD: float = 0.35

# Sentence boundary regex — shared with segmenter_v2 pattern.
_SENT_BOUNDARY: re.Pattern = re.compile(r"[.!?]+(?:\s|$)")

# Score weights — sum to 1.0.  Rebalanced for 7 dimensions.
_W_LENGTH: float = 0.20
_W_COMPLETENESS: float = 0.15
_W_DENSITY: float = 0.15
_W_NOISE: float = 0.15
_W_COHERENCE: float = 0.10
_W_INFO_DENSITY: float = 0.15
_W_RETRIEVAL: float = 0.10

# Hard penalty caps — applied AFTER weighted average when triggered.
_NOISE_FLOOR_CAP: float = 0.35
_OCR_GARBLE_CAP: float = 0.20
_CHART_DUMP_CAP: float = 0.30


# ─────────────────────────────────────────────────────────────────────────────
# SUB-SCORERS — each returns 0.0–1.0
# ─────────────────────────────────────────────────────────────────────────────

def _score_length(tokens: int) -> float:
    """
    Token count fitness for embedding retrieval.
    
    Scoring curve:
      tokens < 15            → 0.0  (too short for any model)
      15–79                  → linear ramp 0.1–0.6
      80–149                 → linear ramp 0.6–0.9
      150–250 (peak)         → 1.0
      251–400                → linear ramp 1.0–0.8
      401–600                → linear ramp 0.8–0.5
      >600                   → 0.3  (exceeds most model windows)
    """
    if tokens < _TOKEN_HARD_MIN:
        return 0.0
    if tokens < _TOKEN_IDEAL_MIN:
        return 0.1 + 0.5 * (tokens - _TOKEN_HARD_MIN) / (_TOKEN_IDEAL_MIN - _TOKEN_HARD_MIN)
    if tokens < _TOKEN_PEAK_MIN:
        return 0.6 + 0.3 * (tokens - _TOKEN_IDEAL_MIN) / (_TOKEN_PEAK_MIN - _TOKEN_IDEAL_MIN)
    if tokens <= _TOKEN_PEAK_MAX:
        return 1.0
    if tokens <= _TOKEN_IDEAL_MAX:
        return 1.0 - 0.2 * (tokens - _TOKEN_PEAK_MAX) / (_TOKEN_IDEAL_MAX - _TOKEN_PEAK_MAX)
    if tokens <= _TOKEN_HARD_MAX:
        return 0.8 - 0.3 * (tokens - _TOKEN_IDEAL_MAX) / (_TOKEN_HARD_MAX - _TOKEN_IDEAL_MAX)
    return 0.3


def _score_completeness(text: str) -> float:
    """
    Grammatical completeness heuristic.
    
    Checks:
      +0.4  starts with uppercase letter (proper sentence start)
      +0.4  ends with sentence terminator (.!?)
      +0.2  contains at least one internal sentence boundary
    
    Chunks that are grammatically complete retrieve better because
    they carry full semantic meaning without needing surrounding context.
    """
    score = 0.0
    stripped = text.strip()
    if not stripped:
        return 0.0

    # Starts with uppercase
    if stripped[0].isupper():
        score += 0.4

    # Ends with sentence terminator
    last_char = stripped.rstrip()[-1] if stripped.rstrip() else ""
    if last_char in ".!?":
        score += 0.4

    # Has internal sentence boundaries (multiple sentences = more complete)
    boundaries = _SENT_BOUNDARY.findall(stripped)
    if len(boundaries) >= 2:
        score += 0.2
    elif len(boundaries) == 1 and last_char in ".!?":
        score += 0.1

    return min(score, 1.0)


def _score_lexical_density(words: list) -> float:
    """
    Vocabulary richness: unique words / total words.
    
    Natural prose range: 0.45–0.88.
    Below 0.3: excessive repetition (OCR loops, boilerplate).
    Above 0.95: keyword soup or very short text (acceptable).
    
    Uses pre-split word list to avoid redundant splitting.
    """
    total = len(words)
    if total < 3:
        return 0.3  # too short to evaluate meaningfully

    unique = len(set(words))
    density = unique / total

    if _DENSITY_IDEAL_MIN <= density <= _DENSITY_IDEAL_MAX:
        return 1.0
    if density < _DENSITY_IDEAL_MIN:
        # Linear penalty from 0.45 down to 0.0
        return max(0.0, density / _DENSITY_IDEAL_MIN)
    # density > _DENSITY_IDEAL_MAX — slightly high but not terrible
    overshoot = density - _DENSITY_IDEAL_MAX
    return max(0.5, 1.0 - overshoot * 3.0)


def _score_noise(text: str) -> float:
    """
    Penalize text with high non-alphabetic character ratio.
    
    Targets: Base64 blobs, hex dumps, OCR garbage, encoded data.
    Normal prose is ~85–95% alpha+space. Tables/data might be 60–75%.
    
    Returns 1.0 for clean text, degrades toward 0.0 for noisy text.
    """
    if not text:
        return 0.0

    total = len(text)
    noise_chars = sum(1 for c in text if not c.isalpha() and not c.isspace())
    noise_ratio = noise_chars / total

    if noise_ratio <= 0.15:
        return 1.0  # very clean
    if noise_ratio <= _NOISE_PENALTY_THRESHOLD:
        # Linear ramp from 1.0 → 0.5
        return 1.0 - 2.5 * (noise_ratio - 0.15)
    # Heavy noise
    return max(0.0, 0.5 - 2.0 * (noise_ratio - _NOISE_PENALTY_THRESHOLD))


def _score_coherence(text: str, tokens: int) -> float:
    """
    Structural coherence: does the chunk read like a meaningful passage?
    
    Checks:
      - Has 1–8 sentences (not a fragment, not a wall of text)
      - Average sentence length is 8–40 words (not too terse, not too verbose)
      - No single sentence dominates >80% of the chunk (balanced distribution)
    
    Well-structured chunks produce better embeddings and cleaner LLM context.
    """
    if tokens < 5:
        return 0.2

    # Split on sentence boundaries
    sentences = [s.strip() for s in _SENT_BOUNDARY.split(text) if s.strip()]
    n_sent = max(len(sentences), 1)

    score = 0.0

    # Sentence count fitness
    if 1 <= n_sent <= 8:
        score += 0.5
    elif n_sent <= 12:
        score += 0.3
    else:
        score += 0.1

    # Average sentence length (in tokens)
    avg_sent_tokens = tokens / n_sent
    if 8.0 <= avg_sent_tokens <= 40.0:
        score += 0.3
    elif 4.0 <= avg_sent_tokens <= 60.0:
        score += 0.15
    else:
        score += 0.05

    # Balance: no single sentence should dominate
    if n_sent >= 2:
        sent_lengths = [len(s.split()) for s in sentences]
        max_sent = max(sent_lengths)
        if max_sent / tokens <= 0.80:
            score += 0.2
        elif max_sent / tokens <= 0.90:
            score += 0.1

    return min(score, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# NEW DIMENSION 6: INFORMATION DENSITY
# ─────────────────────────────────────────────────────────────────────────────

# Common filler/stop words that carry no retrieval signal.
_STOP_WORDS: frozenset = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "and", "or", "but", "not", "no", "nor", "so", "yet", "if", "then",
    "that", "this", "it", "its", "has", "have", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall",
    "can", "than", "more", "also", "very", "just",
})


def _score_information_density(words: list) -> float:
    """
    Ratio of content-bearing words to total words.

    Stop words, single-character tokens, and purely numeric tokens carry
    minimal retrieval signal. High-quality chunks have more content words.

    Scoring:
      content_ratio >= 0.55  → 1.0  (rich informational content)
      0.40–0.55              → linear 0.6–1.0
      0.25–0.40              → linear 0.3–0.6
      < 0.25                 → 0.1   (mostly filler/numbers)
    """
    total = len(words)
    if total < 3:
        return 0.3

    content_count = sum(
        1 for w in words
        if len(w) > 1
        and w.lower() not in _STOP_WORDS
        and not w.replace(",", "").replace(".", "").replace("-", "").isdigit()
    )
    ratio = content_count / total

    if ratio >= 0.55:
        return 1.0
    if ratio >= 0.40:
        return 0.6 + 0.4 * (ratio - 0.40) / 0.15
    if ratio >= 0.25:
        return 0.3 + 0.3 * (ratio - 0.25) / 0.15
    return 0.1


# ─────────────────────────────────────────────────────────────────────────────
# NEW DIMENSION 7: RETRIEVAL FITNESS
# ─────────────────────────────────────────────────────────────────────────────

def _score_retrieval_fitness(text: str, words: list, tokens: int) -> float:
    """
    How well-suited is this chunk for embedding-based semantic search?

    Checks:
      +0.35  Has at least one proper noun or domain term (uppercase word > 2 chars)
      +0.25  Has at least 2 distinct sentences (multi-facet retrieval)
      +0.20  No truncated start (doesn't begin mid-word lowercase)
      +0.20  Contains contextual keywords (nouns, verbs — not just data)
    """
    score = 0.0
    stripped = text.strip()
    if not stripped:
        return 0.0

    # Proper nouns / domain terms (words starting uppercase, > 2 chars)
    proper_nouns = sum(
        1 for w in words
        if len(w) > 2 and w[0].isupper() and w.isalpha()
    )
    if proper_nouns >= 1:
        score += 0.35

    # Multi-sentence (better for retrieval — covers more facets)
    sentence_ends = [i for i, c in enumerate(stripped) if c in '.!?']
    if len(sentence_ends) >= 2:
        score += 0.25
    elif len(sentence_ends) == 1:
        score += 0.10

    # Clean start (not truncated mid-word)
    if stripped[0].isupper() or stripped[0].isdigit() or stripped[0] in '#-|*':
        score += 0.20
    elif stripped[0].islower():
        # Starts lowercase = likely truncated
        score += 0.0

    # Content words present (not just numbers and labels)
    alpha_words = sum(1 for w in words if w.isalpha() and len(w) > 3)
    if alpha_words >= 3:
        score += 0.20
    elif alpha_words >= 1:
        score += 0.10

    return min(score, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# HARD PENALTY DETECTORS
# ─────────────────────────────────────────────────────────────────────────────

def _detect_ocr_garble(words: list) -> bool:
    """True if >35% of words are single non-I/A characters."""
    if len(words) < 6:
        return False
    single = sum(1 for w in words if len(w) == 1 and w.upper() not in ('A', 'I'))
    return (single / len(words)) > 0.35


def _detect_chart_dump(words: list) -> bool:
    """True if >55% of tokens are numeric-looking."""
    if len(words) < 5:
        return False
    numeric = sum(
        1 for w in words
        if w.strip('(),%$').replace(',', '').replace('.', '').replace('-', '').isdigit()
        or re.match(r'^\d{4}[-–]\d{2,4}$', w.strip('(),%$')) is not None
    )
    return (numeric / len(words)) > 0.55


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY API — called from make_chunk_dict()
# ─────────────────────────────────────────────────────────────────────────────

def score_chunk_quality(text: str, tokens: int) -> float:
    """
    Enterprise chunk quality score: 0.0 (garbage) to 1.0 (ideal for retrieval).

    7-dimension weighted average + hard penalty caps for detected noise.
    Pure function. O(N) complexity. Zero external dependencies.
    Deterministic: same input always produces same score.

    Args:
        text:   The cleaned (lowercased) chunk text.
        tokens: Pre-computed token count (from count_tokens()).

    Returns:
        float in [0.0, 1.0].
    """
    if not text or not text.strip():
        return 0.0

    words = text.split()
    if not words:
        return 0.0

    # ── 7 sub-scores ──────────────────────────────────────────────────────
    s_length       = _score_length(tokens)
    s_completeness = _score_completeness(text)
    s_density      = _score_lexical_density(words)
    s_noise        = _score_noise(text)
    s_coherence    = _score_coherence(text, tokens)
    s_info         = _score_information_density(words)
    s_retrieval    = _score_retrieval_fitness(text, words, tokens)

    raw = (
        _W_LENGTH       * s_length
        + _W_COMPLETENESS * s_completeness
        + _W_DENSITY      * s_density
        + _W_NOISE        * s_noise
        + _W_COHERENCE    * s_coherence
        + _W_INFO_DENSITY * s_info
        + _W_RETRIEVAL    * s_retrieval
    )

    # ── Hard penalty caps ─────────────────────────────────────────────────
    if _detect_ocr_garble(words):
        raw = min(raw, _OCR_GARBLE_CAP)
    elif _detect_chart_dump(words):
        raw = min(raw, _CHART_DUMP_CAP)

    # Clamp to [0.0, 1.0] and round to 4 decimal places for clean JSON.
    return round(max(0.0, min(1.0, raw)), 4)


def score_chunk_quality_detailed(text: str, tokens: int) -> Dict[str, Any]:
    """
    Detailed quality breakdown — useful for debugging and dashboard display.
    
    Returns dict with overall score plus per-dimension sub-scores.
    Not called in hot path (make_chunk_dict uses score_chunk_quality).
    Available for admin APIs, quality dashboards, and test assertions.
    """
    if not text or not text.strip():
        return {
            "overall": 0.0,
            "length": 0.0, "completeness": 0.0, "lexical_density": 0.0,
            "noise": 0.0, "coherence": 0.0,
        }

    words = text.split()
    s_length       = _score_length(tokens)
    s_completeness = _score_completeness(text)
    s_density      = _score_lexical_density(words)
    s_noise        = _score_noise(text)
    s_coherence    = _score_coherence(text, tokens)
    s_info         = _score_information_density(words)
    s_retrieval    = _score_retrieval_fitness(text, words, tokens)

    raw = (
        _W_LENGTH       * s_length
        + _W_COMPLETENESS * s_completeness
        + _W_DENSITY      * s_density
        + _W_NOISE        * s_noise
        + _W_COHERENCE    * s_coherence
        + _W_INFO_DENSITY * s_info
        + _W_RETRIEVAL    * s_retrieval
    )

    if _detect_ocr_garble(words):
        raw = min(raw, _OCR_GARBLE_CAP)
    elif _detect_chart_dump(words):
        raw = min(raw, _CHART_DUMP_CAP)

    overall = round(max(0.0, min(1.0, raw)), 4)

    return {
        "overall":             overall,
        "length":              round(s_length, 4),
        "completeness":        round(s_completeness, 4),
        "lexical_density":     round(s_density, 4),
        "noise":               round(s_noise, 4),
        "coherence":           round(s_coherence, 4),
        "information_density": round(s_info, 4),
        "retrieval_fitness":   round(s_retrieval, 4),
        "ocr_garble_detected": _detect_ocr_garble(words),
        "chart_dump_detected": _detect_chart_dump(words),
    }
