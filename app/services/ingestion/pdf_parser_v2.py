# pdf_parser_v2.py — Hybrid PDF Extraction Engine (Production-Ready)
# Enhanced for ingestion_v2 pipeline with async-safe extraction, fallback logic,
# and unified LLM normalization toggle.
import pdfplumber
import fitz  # PyMuPDF
import asyncio
import os
from typing import Dict, Any, List, Optional

from app.utils.text_cleaner_v2 import clean_text
from app.utils.logger import log_info, log_warning
from app.services.ingestion.llm_rewriter import rewrite_batch  # ✅ Added LLM integration
from app.config.ingestion_settings import ENABLE_LLM_NORMALIZATION  # ✅ Global flag

import re as _re

# -------------------------------------------------------------------
# CID CHARACTER RESOLUTION
# -------------------------------------------------------------------
# Common CID → Unicode mappings for typical PDF fonts.
_CID_UNICODE_MAP = {
    133: "\u2026",  # ellipsis  (…)
    145: "\u2018",  # left single quote  (')
    146: "\u2019",  # right single quote  (')
    147: "\u201C",  # left double quote  (\u201c)
    148: "\u201D",  # right double quote  (\u201d)
    149: "\u2022",  # bullet  (•)
    150: "\u2013",  # en-dash  (\u2013)
    151: "\u2014",  # em-dash  (\u2014)
    160: "\u00A0",  # non-breaking space
    169: "\u00A9",  # copyright  (©)
    174: "\u00AE",  # registered  (®)
    176: "\u00B0",  # degree  (°)
    188: "\u00BC",  # 1/4
    189: "\u00BD",  # 1/2
    190: "\u00BE",  # 3/4
    210: "\u2013",  # en-dash variant
    211: "\u2014",  # em-dash variant
    212: "\u201C",  # left double quote variant
    213: "\u201D",  # right double quote variant
}

_CID_PATTERN = _re.compile(r"\(cid:(\d+)\)")


def resolve_cid_characters(text: str) -> str:
    """Replace (cid:NNN) placeholders with their Unicode equivalents."""
    def _replace(match):
        cid = int(match.group(1))
        return _CID_UNICODE_MAP.get(cid, "\uFFFD")  # U+FFFD = replacement char
    return _CID_PATTERN.sub(_replace, text)


# -------------------------------------------------------------------
# PAGE HEADER / FOOTER STRIPPING
# -------------------------------------------------------------------
_HEADER_FOOTER_PATTERNS = [
    # "Page X of Y" anywhere on a line
    _re.compile(r"^\s*.*?Page\s+\d+\s+of\s+\d+\s*$", _re.MULTILINE | _re.IGNORECASE),
    # Standalone "— N —" page numbers
    _re.compile(r"^\s*[\-\u2013\u2014]+\s*\d+\s*[\-\u2013\u2014]+\s*$", _re.MULTILINE),
]


def strip_page_headers_footers(text: str) -> str:
    """Remove common page header/footer lines from extracted PDF text."""
    for pattern in _HEADER_FOOTER_PATTERNS:
        text = pattern.sub("", text)
    text = _re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# -------------------------------------------------------------------
# Local parser-level toggle
# -------------------------------------------------------------------
# True  → Force enable LLM normalization for PDF parser
# False → Force disable LLM normalization
# None  → Inherit from global ENABLE_LLM_NORMALIZATION
LOCAL_LLM_TOGGLE = None
PAGE_BREAK_TOKEN = "\n\n---PAGE BREAK---\n\n"
LOCAL_VISUAL_INTERCEPT_TOGGLE = True
VISUAL_INTERCEPT_TIMEOUT_SEC = 60
MAX_VISUAL_EXPLANATIONS = 24


def is_llm_enabled() -> bool:
    """Determine whether LLM normalization is enabled for this parser."""
    return ENABLE_LLM_NORMALIZATION if LOCAL_LLM_TOGGLE is None else LOCAL_LLM_TOGGLE


# -------------------------------------------------------------------
# PDF EXTRACTION (PRIMARY)
# -------------------------------------------------------------------
def extract_with_pdfplumber(file_path: str) -> List[str]:
    """Extracts text page-by-page using pdfplumber (high accuracy for structured PDFs)."""
    pages_text = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    text = resolve_cid_characters(text)
                    pages_text.append(text)
    except Exception as e:
        log_warning(f"[pdf_parser_v2] pdfplumber failed: {e}")
    return pages_text


# -------------------------------------------------------------------
# PDF EXTRACTION (FALLBACK)
# -------------------------------------------------------------------
def extract_with_pymupdf(file_path: str) -> List[str]:
    """Fallback extraction using PyMuPDF for scanned or unusual PDFs."""
    pages_text = []
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text = page.get_text("text") or ""
            if text.strip():
                text = resolve_cid_characters(text)
                pages_text.append(text)
    except Exception as e:
        log_warning(f"[pdf_parser_v2] PyMuPDF fallback failed: {e}")
    return pages_text


# -------------------------------------------------------------------
# PAGE MERGER
# -------------------------------------------------------------------
def merge_page_text(pages: List[str]) -> str:
    """Concatenates pages with structured separation markers."""
    return PAGE_BREAK_TOKEN.join(pages)


def _is_presentation_pdf(pages: List[str]) -> bool:
    """
    Conservative heuristic for slide-style PDFs.
    """
    non_empty = [p for p in (pages or []) if isinstance(p, str) and p.strip()]
    if len(non_empty) < 5:
        return False
    lengths = [len(p) for p in non_empty]
    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)
    return avg_len < 420 and max_len < 1300


def build_page_map(pages: List[str]) -> List[Dict[str, Any]]:
    """
    Build deterministic page offset metadata against the merged raw_text buffer.
    Offsets are 0-based [start_char, end_char) within the merged text.
    """
    page_map: List[Dict[str, Any]] = []
    cursor = 0

    for i, page_text in enumerate(pages, start=1):
        text = page_text or ""
        start = cursor
        end = start + len(text)
        page_map.append(
            {
                "page_number": i,
                "text": text,
                "cleaned_text": clean_text(text),
                "start_char": start,
                "end_char": end,
            }
        )
        cursor = end + len(PAGE_BREAK_TOKEN)

    return page_map


# -------------------------------------------------------------------
# PARALLEL PAGE EXTRACTION (ASYNC)
# -------------------------------------------------------------------
async def parallel_extract_pdf(file_path: str) -> List[str]:
    """Runs extraction concurrently using both pdfplumber and pymupdf for redundancy."""
    # use get_running_loop() — safer inside an active event loop
    loop = asyncio.get_running_loop()
    plumber_task = loop.run_in_executor(None, lambda: extract_with_pdfplumber(file_path))
    pymupdf_task = loop.run_in_executor(None, lambda: extract_with_pymupdf(file_path))

    results = await asyncio.gather(plumber_task, pymupdf_task, return_exceptions=True)
    plumber_pages, pymupdf_pages = results

    # Prefer pdfplumber if sufficient text found, otherwise use PyMuPDF fallback
    if plumber_pages and sum(len(p) for p in plumber_pages) > 200:
        return plumber_pages
    if pymupdf_pages:
        return pymupdf_pages
    raise ValueError(f"[pdf_parser_v2] No extractable text found in: {file_path}")


# -------------------------------------------------------------------
# MAIN PARSER PIPELINE
# -------------------------------------------------------------------
async def parse_pdf(file_path: str) -> Dict[str, Any]:
    """
    Hybrid PDF parser with async concurrency and optional LLM normalization.
    Steps:
      1. Attempt parallel extraction (pdfplumber + PyMuPDF)
      2. Merge and clean results
      3. (Optional) Normalize with LLM
      4. Return standardized ingestion output
    """

    log_info(f"[pdf_parser_v2] Reading PDF: {file_path}")

    pages_text = await parallel_extract_pdf(file_path)

    if not pages_text:
        raise ValueError(f"[pdf_parser_v2] Empty extraction result: {file_path}")

    page_map = build_page_map(pages_text)

    # ----------------------------------------------------------------
    # Optional visual interception (explanations-only, fail-open)
    # ----------------------------------------------------------------
    visual_texts: List[str] = []
    if LOCAL_VISUAL_INTERCEPT_TOGGLE:
        try:
            from app.services.ingestion.media.document_visual_interceptor_v1 import (
                DocumentVisualInterceptorV1,
            )

            interceptor = DocumentVisualInterceptorV1()
            visual_texts = await asyncio.wait_for(
                interceptor.intercept_explanations_only(
                    file_path=file_path,
                    parsed_output={},
                    file_type="pdf",
                    max_visuals=MAX_VISUAL_EXPLANATIONS,
                ),
                timeout=VISUAL_INTERCEPT_TIMEOUT_SEC,
            )
            if visual_texts:
                log_info(
                    f"[pdf_parser_v2] Extracted {len(visual_texts)} visual explanations"
                )
        except asyncio.TimeoutError:
            log_warning(
                f"[pdf_parser_v2] Visual interception timed out after "
                f"{VISUAL_INTERCEPT_TIMEOUT_SEC}s (non-fatal)"
            )
        except Exception as e:
            log_warning(
                f"[pdf_parser_v2] Visual interception failed (non-fatal): {e}"
            )

    # ----------------------------------------------------------------
    # Strip page headers / footers before merge
    # ----------------------------------------------------------------
    pages_text = [strip_page_headers_footers(p) for p in pages_text]
    pages_text = [p for p in pages_text if p.strip()]

    is_slide_mode = _is_presentation_pdf(pages_text)
    if is_slide_mode:
        # Keep each slide as an independent semantic unit to avoid hybrid merges.
        entries: List[Dict[str, str]] = [
            {"text": p.strip()} for p in pages_text if isinstance(p, str) and p.strip()
        ]
        if visual_texts:
            entries.extend({"text": t} for t in visual_texts if isinstance(t, str) and t.strip())

        combined = merge_page_text([e["text"] for e in entries if e.get("text")])
        cleaned = clean_text(combined)
        normalized_text = cleaned
    else:
        combined = merge_page_text(pages_text)
        if visual_texts:
            combined = combined + "\n\n" + "\n\n".join(visual_texts)
        cleaned = clean_text(combined)
        normalized_text = cleaned

    # ----------------------------------------------------------------
    # ✅ Optional LLM normalization
    # ----------------------------------------------------------------
    if is_llm_enabled():
        try:
            log_info(f"[pdf_parser_v2] Sending {len(pages_text)} pages for LLM normalization...")
            normalized_results = await rewrite_batch([cleaned])
            if normalized_results:
                normalized_text = normalized_results[0]
                log_info("[pdf_parser_v2] ✅ LLM normalization complete.")
        except Exception as e:
            log_warning(f"[pdf_parser_v2] ⚠️ LLM normalization failed: {e}")
    else:
        log_info("[pdf_parser_v2] LLM normalization skipped (disabled).")

    # ----------------------------------------------------------------
    # ✅ Return standardized output
    # ----------------------------------------------------------------
    result = {
        "raw_text": combined,
        "cleaned_text": cleaned,
        "normalized_text": normalized_text,
        "pages": len(pages_text),
        "page_map": page_map,
        "visual_explanations": visual_texts,
        "source_type": "pdf",
        "metadata": {
            "file_name": os.path.basename(file_path),
            "pages": len(pages_text),
            "parser": "pdfplumber + pymupdf hybrid + LLM optional",
            "llm_normalization": is_llm_enabled(),
            "mode": "slide" if is_slide_mode else "prose",
            "visual_explanations_count": len(visual_texts),
        },
    }
    if is_slide_mode:
        result["entries"] = entries
    return result
