# =============================================
# text_preprocessor.py â€” Enterprise Document Text Pre-Processing Engine
#
# Shared pre-processing pipeline that ALL chunking strategies benefit from.
# Called on raw extracted text BEFORE structural parsing or chunking.
#
# Zero dependencies beyond stdlib. Zero DB writes. Pure functions.
#
# PRE-PROCESSING PIPELINE (in order):
#   0. UNICODE_NORMALIZE   â€” NFKC normalization + control char stripping
#   1. PAGE_BREAK_CLEANUP  â€” strip ---PAGE BREAK--- markers â†’ double newline
#   2. IMAGE_STUB_REMOVAL  â€” remove "The image is a real-world photograph" stubs
#   3. LLM_PREFIX_STRIP    â€” strip LLM prompt prefixes ("The following content
#                            is extracted from a chart...")
#   4. OCR_GARBLE_REMOVAL  â€” detect scrambled diagram OCR ("N N N O O O S A S")
#   5. CHART_DUMP_CLEANUP  â€” detect raw chart axis number sequences
#   6. HEADER_FOOTER_STRIP â€” remove repeated page headers/footers/page numbers
#   7. TOC_BLEED_STRIP     â€” remove Table of Contents dot-leader lines
#   8. WHITESPACE_NORMALIZEâ€” collapse excessive whitespace/newlines
#
# CHUNK-LEVEL NOISE DETECTION:
#   is_noise_chunk()       â€” returns True if chunk text is junk
#   classify_chunk_noise() â€” returns (is_noise, noise_type, confidence)
#
# HOW TO USE:
#   from app.core.chunking_stratagies.text_preprocessor import (
#       preprocess_document_text,
#       is_noise_chunk,
#   )
#   clean = preprocess_document_text(raw_text)   # before chunking
#   ...
#   for chunk in chunks:
#       if is_noise_chunk(chunk["cleaned_text"]):
#           chunk["reasoning_ingestion"]["noise_flag"] = True
# =============================================

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Tuple

logger = logging.getLogger(__name__)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# COMPILED PATTERNS â€” one-time cost at module load
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# 1. Page break markers (various formats seen in production)
_PAGE_BREAK_RE = re.compile(
    r"[-â€”]{2,}\s*PAGE\s*BREAK\s*[-â€”]{2,}",
    re.IGNORECASE,
)

# 2. Image description stubs (from OCR/vision pipelines)
_IMAGE_STUB_RE = re.compile(
    r"(?:^|\n)\s*The image (?:is a real-world photograph|contains visual information|appears to be a photograph)"
    r"[^\n]*(?:\n[^\n]*?(?:photograph|illustration|detected text)[^\n]*)*",
    re.IGNORECASE,
)

# 3. LLM prompt prefixes that bleed into extracted content
_LLM_PREFIX_RE = re.compile(
    r"(?:^|\n)\s*(?:"
    r"The following content is extracted from a (?:chart|graph|table|visual|image)[.,]?\s*"
    r"Explain clearly in plain English[^\n]*\.\s*"
    r"(?:Focus on trends[^\n]*\.\s*)?"
    r"(?:Do NOT repeat[^\n]*\.\s*)?"
    r"Content:\s*"
    r")",
    re.IGNORECASE,
)

# 4. OCR garble: sequences of single chars separated by spaces
#    e.g. "N N N O O O S A S S" â€” from diagram text that overlaps
_OCR_GARBLE_RE = re.compile(
    r"(?:^|\n)(?:[A-Z0-9]\s+){6,}[A-Z0-9]",
)

# 5. Chart number dump: long sequences of bare numbers
#    e.g. "446 459 471 485 427 406 390 188 198 210 222 231 240 248"
_CHART_NUMBERS_RE = re.compile(
    r"\b(?:\d[\d,.]*\s+){5,}\d[\d,.]*\b"
)

# 6. Repeated page headers/footers and page numbers
#    Catches: "Page 14 of 92", "- 14 -", "Page 14 | Q3 Financials", standalone page nums
_PAGE_NUMBER_RE = re.compile(
    r"(?:^|\n)\s*(?:"
    r"[Pp]age\s+\d+\s*(?:of\s+\d+)?\s*(?:[|â€”â€“-]\s*[^\n]*)?"
    r"|[-â€“â€”]\s*\d+\s*[-â€“â€”]"
    r"|\d+\s*[|]\s*[^\n]{0,80}"
    r")\s*(?:\n|$)",
)

# 7. Table of Contents dot-leader lines
#    Catches: "1. Introduction .............. 4", "Chapter 3 ......... 42"
_TOC_LINE_RE = re.compile(
    r"(?:^|\n)[^\n]{0,80}\.{4,}\s*\d+\s*(?:\n|$)",
)

# 8. Control characters that should never appear in clean text
#    Null bytes, soft hyphens, zero-width spaces, BOM, etc.
_CONTROL_CHARS_RE = re.compile(
    r"[\x00\xad\u200b\u200c\u200d\ufeff\u2028\u2029]",
)

# 9. Repeated percentage labels from chart axes / legends.
_PERCENTAGE_SEQ_RE = re.compile(
    r"(?:(?:\d{1,3}(?:\.\d+)?%)\s+){4,}\d{1,3}(?:\.\d+)?%"
)

# 10. Bare page numbers isolated between blank lines (slide footers).
# Conservative pattern to avoid removing numbered lists in prose.
_BARE_PAGE_NUM_RE = re.compile(
    r"(?:^|\n\s*\n)\s*\d{1,3}\s*(?=\n\s*\n|$)",
    re.MULTILINE,
)

# 11. Excessive whitespace
_MULTI_NEWLINE_RE = re.compile(r"\n{4,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{3,}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# DOCUMENT-LEVEL PRE-PROCESSOR â€” call before chunking
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def preprocess_document_text(text: str) -> str:
    """
    Clean raw extracted text before chunking.

    Removes PDF artifacts, OCR noise, LLM prompt prefixes, page break
    markers, repeated headers/footers, and ToC bleed. Applies Unicode
    NFKC normalization to collapse ligatures and variant codepoints.

    Returns cleaned text suitable for any chunking strategy.

    Pure function. O(N) complexity. Deterministic.
    """
    if not text or not text.strip():
        return ""

    original_len = len(text)
    modifications: list[str] = []

    # 0. Unicode NFKC normalization â€” collapse ligatures (ï¬â†’fi),
    #    fullwidth chars (ï¼¡â†’A), and variant codepoints into canonical forms.
    #    Without this, embeddings diverge for visually identical text.
    result = unicodedata.normalize("NFKC", text)
    if result != text:
        modifications.append("UNICODE_NFKC")

    # 0.5. Strip control characters (null bytes, soft hyphens, zero-width)
    cleaned = _CONTROL_CHARS_RE.sub("", result)
    if cleaned != result:
        modifications.append("CONTROL_CHARS")
    result = cleaned

    # 1. Page break markers â†’ double newline (section separator)
    result = _PAGE_BREAK_RE.sub("\n\n", result)

    # 2. Strip image description stubs entirely
    result = _IMAGE_STUB_RE.sub("", result)

    # 3. Strip LLM prompt prefixes (keep the actual Content:)
    result = _LLM_PREFIX_RE.sub("\n", result)

    # 4. Remove OCR garble sequences (single-char runs)
    #    Replace with empty â€” these are unrecoverable without re-OCR
    result = _OCR_GARBLE_RE.sub("", result)

    # 5. Strip repeated page headers/footers and page numbers
    cleaned = _PAGE_NUMBER_RE.sub("\n", result)
    if cleaned != result:
        modifications.append("PAGE_HEADERS")
    result = cleaned

    # 6. Strip Table of Contents dot-leader lines
    cleaned = _TOC_LINE_RE.sub("\n", result)
    if cleaned != result:
        modifications.append("TOC_BLEED")
    result = cleaned

    # 6.5 Strip repeated percentage axis labels from chart OCR.
    cleaned = _PERCENTAGE_SEQ_RE.sub(" ", result)
    if cleaned != result:
        modifications.append("PERCENT_SEQ")
    result = cleaned

    # 6.6 Strip isolated bare page-number footer lines.
    cleaned = _BARE_PAGE_NUM_RE.sub("\n\n", result)
    if cleaned != result:
        modifications.append("BARE_PAGE_NUM")
    result = cleaned

    # 7. Normalize whitespace
    result = _MULTI_NEWLINE_RE.sub("\n\n", result)
    result = _MULTI_SPACE_RE.sub(" ", result)

    result = result.strip()

    # Observability: log what changed so pipeline debugging is possible
    chars_removed = original_len - len(result)
    if modifications:
        logger.debug(
            "text_preprocessor: removed %d chars, applied: %s",
            chars_removed,
            ", ".join(modifications),
        )

    return result


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CHUNK-LEVEL NOISE DETECTION â€” call per chunk after chunking
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def is_noise_chunk(text: str, tokens: int = 0) -> bool:
    """
    Fast check: is this chunk noise that should be flagged/skipped?

    Returns True if the chunk is clearly non-informative:
    - Pure chart number dumps
    - OCR garble
    - Image stubs
    - Too short to carry meaning (<8 tokens)
    """
    if not text or not text.strip():
        return True

    tok = tokens if tokens > 0 else len(text.split())

    # Too short for meaningful embedding
    if tok < 8:
        return True

    stripped = text.strip()

    # Image description stub
    if _is_image_stub(stripped):
        return True

    # Chart number dump: >60% of tokens are numbers
    if _is_chart_number_dump(stripped, tok):
        return True

    # OCR garble: >50% single-character words
    if _is_garbled_ocr(stripped, tok):
        return True

    return False


def classify_chunk_noise(text: str, tokens: int = 0) -> Tuple[bool, str, float]:
    """
    Detailed noise classification for quality dashboards.

    Returns:
        (is_noise: bool, noise_type: str, confidence: float)

    noise_type values:
        "clean"           â€” not noise
        "too_short"       â€” < 8 tokens
        "image_stub"      â€” OCR/vision placeholder text
        "chart_numbers"   â€” raw chart axis/data dump
        "garbled_ocr"     â€” scrambled diagram/overlay text
        "number_soup"     â€” mostly numbers with no prose context
    """
    if not text or not text.strip():
        return True, "empty", 1.0

    tok = tokens if tokens > 0 else len(text.split())

    if tok < 8:
        return True, "too_short", 0.95

    stripped = text.strip()

    if _is_image_stub(stripped):
        return True, "image_stub", 0.95

    if _is_chart_number_dump(stripped, tok):
        return True, "chart_numbers", 0.90

    if _is_garbled_ocr(stripped, tok):
        return True, "garbled_ocr", 0.85

    if _is_number_soup(stripped, tok):
        return True, "number_soup", 0.80

    if _is_toc_bleed(stripped):
        return True, "toc_bleed", 0.85

    if _is_page_header_noise(stripped, tok):
        return True, "page_header", 0.90

    return False, "clean", 0.0


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# INTERNAL DETECTORS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _is_image_stub(text: str) -> bool:
    """Detect image-description placeholder text from OCR/vision pipelines."""
    lower = text.lower()
    stubs = (
        "the image is a real-world photograph",
        "the image contains visual information",
        "this image appears to be a photograph",
        "detected text:",
    )
    # If most of the text is just the stub, it's noise
    for stub in stubs:
        if stub in lower and len(text) < len(stub) + 100:
            return True
    return False


def _is_chart_number_dump(text: str, tokens: int) -> bool:
    """
    Detect raw chart axis/data number sequences.

    Heuristic: if >55% of tokens are numeric-looking and there
    are no complete sentences, it's a chart dump.
    """
    words = text.split()
    if not words:
        return False

    numeric_count = sum(
        1 for w in words
        if _is_numeric_token(w)
    )
    numeric_ratio = numeric_count / len(words)

    # High numeric ratio AND minimal prose structure
    if numeric_ratio > 0.55 and not _has_sentence_structure(text):
        return True

    # Long unbroken number sequences
    if _CHART_NUMBERS_RE.search(text):
        # Only flag if there's no surrounding prose majority
        if numeric_ratio > 0.40:
            return True

    return False


def _is_garbled_ocr(text: str, tokens: int) -> bool:
    """
    Detect scrambled OCR from diagrams with overlapping text layers.

    Heuristic: if >40% of words are single characters (excluding
    common single-char words like "a", "I") it's garbled.
    """
    words = text.split()
    if len(words) < 6:
        return False

    single_chars = sum(
        1 for w in words
        if len(w) == 1 and w.upper() not in ("A", "I")
    )
    ratio = single_chars / len(words)
    return ratio > 0.35


def _is_number_soup(text: str, tokens: int) -> bool:
    """
    Detect text that is mostly numbers with scattered labels.

    Catches chart data like:
    "446 459 471 485 427 406 390 188 198 210 222 231 240 248 2018-19 2019-20..."
    """
    words = text.split()
    if len(words) < 8:
        return False

    numeric_count = sum(1 for w in words if _is_numeric_token(w))
    ratio = numeric_count / len(words)

    # Very high numeric ratio with minimal alpha content
    if ratio > 0.65:
        return True

    return False


def _is_toc_bleed(text: str) -> bool:
    """
    Detect Table of Contents fragments that survived into chunks.

    ToC lines contain dot-leaders: "1. Introduction .............. 4"
    If >40% of non-empty lines match the ToC pattern, it's junk.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False

    toc_lines = sum(1 for ln in lines if _TOC_LINE_RE.match("\n" + ln + "\n"))
    return toc_lines / len(lines) > 0.40


def _is_page_header_noise(text: str, tokens: int) -> bool:
    """
    Detect chunks that are mostly repeated page headers/footers.

    If >50% of non-empty lines are page number patterns and the
    chunk is short, it's header/footer noise.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 1:
        return False

    header_lines = sum(1 for ln in lines if _PAGE_NUMBER_RE.match("\n" + ln + "\n"))
    if not header_lines:
        return False

    # Short chunk dominated by page headers â†’ noise
    if header_lines / len(lines) > 0.50 and tokens < 30:
        return True

    return False


def _is_numeric_token(word: str) -> bool:
    """Check if a word is a number, percentage, currency, or year-range."""
    # Percentage labels from charts: 35%, 3.7%
    if re.match(r"^\d+(?:\.\d+)?%$", word.strip("(),.*")):
        return True

    cleaned = word.strip("(),%$₹€£*")
    if not cleaned:
        return False
    # Pure digits
    if cleaned.replace(",", "").replace(".", "").isdigit():
        return True
    # Year ranges: 2018-19, 2023-24
    if re.match(r"^\d{4}[-\u2013]\d{2,4}$", cleaned):
        return True
    # Decimal with commas: 1,234.56
    if re.match(r"^\d[\d,]*\.?\d*$", cleaned):
        return True
    return False

def _has_sentence_structure(text: str) -> bool:
    """Check if text has at least one proper sentence (subject + verb pattern)."""
    # Simple heuristic: has at least one period followed by space+uppercase
    if re.search(r"\.\s+[A-Z]", text):
        return True
    # Starts with uppercase and ends with period
    stripped = text.strip()
    if stripped and stripped[0].isupper() and stripped[-1] in ".!?":
        return True
    return False
