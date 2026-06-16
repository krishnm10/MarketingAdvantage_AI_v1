# pdf_parser_v2.py — Hybrid PDF Extraction Engine (Production-Ready)
# Enhanced for ingestion_v2 pipeline with async-safe extraction, fallback logic,
# and unified LLM normalization toggle.
import pdfplumber
import fitz  # PyMuPDF
import asyncio
import logging
import os
import re as _re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from app.utils.text_cleaner_v2 import clean_text
from app.utils.logger import log_info, log_warning
from app.services.ingestion.llm_rewriter import rewrite_batch  # ✅ Added LLM integration
from app.config.ingestion_settings import ENABLE_LLM_NORMALIZATION  # ✅ Global flag

logger = logging.getLogger(__name__)

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
# Streaming page model (todo: pdf-parser-streaming)
# char_offset advancement uses raw `text` length (same as _get_page_text_only_sync).
# -------------------------------------------------------------------
@dataclass
class ParsedPage:
    """One PDF page for streaming ingestion."""

    page_number: int           # 0-based
    text: str                  # raw extracted text (pre-cleaning)
    cleaned_text: str          # after CID resolution and whitespace normalization
    char_offset_start: int     # cumulative char offset at page start (raw-text space)
    char_offset_end: int       # cumulative char offset at page end (raw-text space)
    is_visual: bool            # True if image-heavy with low text density
    is_continuation: bool      # True if page appears to start mid-sentence


@dataclass
class _StreamingPdfReader:
    """Thin wrapper around an open PyMuPDF document."""

    doc: Any
    path: str

    def close(self) -> None:
        if self.doc is not None:
            self.doc.close()
            self.doc = None


def _open_pdf_reader(file_path: Union[Path, str]) -> _StreamingPdfReader:
    """Open PDF once via PyMuPDF (used by iter_pdf_pages)."""
    path = str(file_path)
    doc = fitz.open(path)
    return _StreamingPdfReader(doc=doc, path=path)


def _get_page_count(reader: _StreamingPdfReader) -> int:
    return int(reader.doc.page_count)


def _get_page_text_only_sync(reader: _StreamingPdfReader, page_index: int) -> str:
    """
    Raw text only — no cleaning, CID resolution, or heuristics.
    Used for checkpoint offset pre-pass; must match raw `text` in ParsedPage.
    """
    page = reader.doc[page_index]
    return page.get_text("text") or ""


def _compute_initial_offset(reader: _StreamingPdfReader, start_page: int) -> int:
    """Sum raw text lengths for pages [0, start_page) for checkpoint resume."""
    initial_offset = 0
    for p in range(0, start_page):
        initial_offset += len(_get_page_text_only_sync(reader, p))
    return initial_offset


def _page_image_count(reader: _StreamingPdfReader, page_number: int) -> int:
    try:
        page = reader.doc[page_number]
        return len(page.get_images())
    except Exception as exc:
        logger.debug(
            "pdf_parser: cannot determine image count for page %d: %s",
            page_number,
            exc,
        )
        return 0


def _is_continuation_page(cleaned_text: str) -> bool:
    stripped = cleaned_text.strip()
    if not stripped:
        return False
    first = stripped[0]
    return first.islower()


def _is_visual_page(cleaned_text: str, image_count: int) -> bool:
    if len(cleaned_text.strip()) < 100 and image_count >= 1:
        return True
    return False


def _parse_single_page_sync(
    reader: _StreamingPdfReader,
    page_number: int,
    char_offset: int,
) -> tuple[ParsedPage, int]:
    """
    CPU-bound per-page parse. Reader must already be open; never open/close here.
    Offset advancement: len(raw text) — identical to _get_page_text_only_sync.
    """
    raw = _get_page_text_only_sync(reader, page_number)
    cid_resolved = resolve_cid_characters(raw)
    cleaned = clean_text(cid_resolved)
    image_count = _page_image_count(reader, page_number)
    is_visual = _is_visual_page(cleaned, image_count)
    is_continuation = _is_continuation_page(cleaned)
    end_offset = char_offset + len(raw)
    page = ParsedPage(
        page_number=page_number,
        text=raw,
        cleaned_text=cleaned,
        char_offset_start=char_offset,
        char_offset_end=end_offset,
        is_visual=is_visual,
        is_continuation=is_continuation,
    )
    return page, end_offset


async def iter_pdf_pages(
    file_path: Union[Path, str],
    start_page: int = 0,
) -> AsyncIterator[ParsedPage]:
    """
    Lazy async generator over PDF pages. One run_in_executor call per page.
    """
    reader = _open_pdf_reader(file_path)
    try:
        page_count = _get_page_count(reader)
        if start_page >= page_count:
            return

        current_offset = 0
        if start_page > 0:
            current_offset = _compute_initial_offset(reader, start_page)

        loop = asyncio.get_running_loop()
        for page_num in range(start_page, page_count):
            try:
                page, current_offset = await loop.run_in_executor(
                    None,
                    _parse_single_page_sync,
                    reader,
                    page_num,
                    current_offset,
                )
                yield page
            except Exception as page_err:
                logger.error(
                    "pdf_parser: failed to parse page %d in %s: %s",
                    page_num,
                    file_path,
                    page_err,
                )
                yield ParsedPage(
                    page_number=page_num,
                    text="",
                    cleaned_text="",
                    char_offset_start=current_offset,
                    char_offset_end=current_offset,
                    is_visual=False,
                    is_continuation=False,
                )
    finally:
        reader.close()


# -------------------------------------------------------------------
# PDF EXTRACTION (PRIMARY) — legacy bulk helpers (still used by parallel_extract_pdf)
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

    DEPRECATED for new streaming ingestion paths — use ParsedPage.char_offset_* instead.
    Kept for parse_pdf backward compatibility. [todo: pdf-parser-streaming]
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


async def _collect_pages_via_stream(file_path: str) -> List[str]:
    """Aggregate non-empty header-stripped pages from iter_pdf_pages."""
    pages_text: List[str] = []
    async for page in iter_pdf_pages(file_path):
        stripped = strip_page_headers_footers(page.text)
        if stripped.strip():
            pages_text.append(stripped)
    return pages_text


async def _collect_pages_via_stream_with_stats(file_path: str) -> tuple[List[str], int, int]:
    """Like _collect_pages_via_stream but also returns (page_count, raw_char_total)."""
    pages_text: List[str] = []
    page_count = 0
    raw_chars = 0
    async for page in iter_pdf_pages(file_path):
        page_count += 1
        raw_chars += len(page.text or "")
        stripped = strip_page_headers_footers(page.text)
        if stripped.strip():
            pages_text.append(stripped)
    return pages_text, page_count, raw_chars


async def _fetch_visual_explanations(file_path: str) -> List[str]:
    """Optional visual/OCR explanations from embedded PDF images (fail-open)."""
    if not LOCAL_VISUAL_INTERCEPT_TOGGLE:
        return []
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
        return visual_texts or []
    except asyncio.TimeoutError:
        log_warning(
            f"[pdf_parser_v2] Visual interception timed out after "
            f"{VISUAL_INTERCEPT_TIMEOUT_SEC}s (non-fatal)"
        )
    except Exception as e:
        log_warning(
            f"[pdf_parser_v2] Visual interception failed (non-fatal): {e}"
        )
    return []


def _non_empty_stripped_pages(pages: List[str]) -> List[str]:
    return [
        strip_page_headers_footers(p)
        for p in pages
        if isinstance(p, str) and strip_page_headers_footers(p).strip()
    ]


async def _ocr_scanned_pdf_pages(file_path: str, max_pages: int = 24) -> List[str]:
    """
    Last-resort OCR for image-only PDFs: rasterize each page and run the captioner.
    Used only when text-layer and embedded-image extraction both return nothing.
    """
    import fitz
    import tempfile

    from app.services.ingestion.media.image_ingestor_v1 import ImageIngestorV1

    ingestor = ImageIngestorV1()
    loop = asyncio.get_running_loop()
    pages_out: List[str] = []

    def _render_page_to_temp(page_index: int) -> tuple[int, str]:
        doc = fitz.open(file_path)
        try:
            if page_index >= len(doc):
                return page_index, ""
            page = doc[page_index]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            fd, temp_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            pix.save(temp_path)
            return page_index, temp_path
        finally:
            doc.close()

    doc = fitz.open(file_path)
    n_pages = min(len(doc), max_pages)
    doc.close()

    for page_index in range(n_pages):
        _, temp_path = await loop.run_in_executor(None, _render_page_to_temp, page_index)
        if not temp_path:
            continue
        try:
            result = await ingestor.captioner.caption(temp_path)
            if not isinstance(result, dict):
                continue
            ocr_text = (result.get("ocr_text") or "").strip()
            caption = (result.get("caption") or "").strip()
            page_text = ocr_text or caption
            if page_text:
                pages_out.append(page_text)
        except Exception as e:
            log_warning(
                f"[pdf_parser_v2] Page render OCR failed for page {page_index + 1}: {e}"
            )
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    if pages_out:
        log_info(
            f"[pdf_parser_v2] Recovered {len(pages_out)} page(s) via render+OCR fallback"
        )
    return pages_out


# -------------------------------------------------------------------
# MAIN PARSER PIPELINE
# -------------------------------------------------------------------
async def parse_pdf(file_path: str) -> Dict[str, Any]:
    """
    Hybrid PDF parser with async concurrency and optional LLM normalization.
    Steps:
      1. Stream pages via iter_pdf_pages (PyMuPDF per-page, run_in_executor)
      2. Merge and clean results
      3. (Optional) Normalize with LLM
      4. Return standardized ingestion output
    """
    logger.warning(
        "parse_pdf() is deprecated. Use iter_pdf_pages() for streaming ingestion. "
        "Callers: migrate to iter_pdf_pages before the next major release. "
        "[todo: pdf-parser-streaming]"
    )

    log_info(f"[pdf_parser_v2] Reading PDF: {file_path}")

    pages_text, page_count, raw_chars = await _collect_pages_via_stream_with_stats(file_path)
    extraction_source = "stream"

    if not pages_text:
        try:
            fallback_pages = await parallel_extract_pdf(file_path)
            pages_text = _non_empty_stripped_pages(fallback_pages)
            if pages_text:
                extraction_source = "parallel_extract"
                log_info(
                    f"[pdf_parser_v2] Recovered {len(pages_text)} pages via parallel_extract_pdf"
                )
        except ValueError:
            pass

    visual_texts: List[str] = await _fetch_visual_explanations(file_path)

    if not pages_text and visual_texts:
        pages_text = [
            t.strip() for t in visual_texts if isinstance(t, str) and t.strip()
        ]
        extraction_source = "visual_ocr"
        visual_texts = []
        log_info(
            f"[pdf_parser_v2] Using {len(pages_text)} visual/OCR page(s) as primary text"
        )

    if not pages_text and page_count > 0:
        scanned_pages = await _ocr_scanned_pdf_pages(
            file_path, max_pages=MAX_VISUAL_EXPLANATIONS
        )
        if scanned_pages:
            pages_text = scanned_pages
            extraction_source = "page_render_ocr"

    if not pages_text:
        raise ValueError(
            f"[pdf_parser_v2] Empty extraction result: {file_path} "
            f"(pages={page_count}, raw_chars={raw_chars}). "
            "The PDF may be encrypted, corrupt, or a scanned document with no OCR text layer."
        )

    page_map = build_page_map(pages_text)

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
