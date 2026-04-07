"""
Test Chunking With Your Own Documents
======================================
Load ANY file (PDF, DOCX, CSV, XLSX, TXT, MD, JSON, XML, images, audio, video)
and see how each chunking strategy breaks it into chunks.

Usage:
    python test_chunking_my_document.py <file_path>                          # test with default strategy (semantic)
    python test_chunking_my_document.py <file_path> --strategy overlap       # test with one strategy
    python test_chunking_my_document.py <file_path> --strategy all           # test ALL strategies
    python test_chunking_my_document.py <file_path> --strategy semantic overlap elite_v2   # test specific ones
    python test_chunking_my_document.py <file_path> --show-text              # show full chunk text
    python test_chunking_my_document.py <file_path> --show-text --max-chars 500  # limit displayed text length

Examples:
    python test_chunking_my_document.py "C:\\docs\\report.pdf" --strategy all --show-text
    python test_chunking_my_document.py data.csv --strategy smart_check elite_v2
    python test_chunking_my_document.py presentation.docx --show-text --max-chars 300
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ── ensure project root on path ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ── disable LLM-dependent features for offline testing ───────────────────────
os.environ.setdefault("CHUNK_ELITE_PROPOSITIONS", "false")
os.environ.setdefault("CHUNK_ELITE_AGENTIC_SPLIT", "false")

from app.core.chunking_stratagies.chunking_registry import (
    clear_chunker_cache,
    get_chunker,
    list_chunking_strategies,
)

# ── parsers ──────────────────────────────────────────────────────────────────
from app.services.ingestion.pdf_parser_v2 import parse_pdf
from app.services.ingestion.docx_parser_v2 import parse_docx
from app.services.ingestion.excel_parser_v2 import parse_excel
from app.services.ingestion.csv_parser_v2 import parse_csv
from app.services.ingestion.text_parser_v2 import parse_text
from app.services.ingestion.json_parser_v2 import parse_json
from app.services.ingestion.xml_parser_v2 import parse_xml

# ═══════════════════════════════════════════════════════════════════════════════
#  File Extension → Parser Mapping
# ═══════════════════════════════════════════════════════════════════════════════

PARSER_MAP = {
    # Document formats
    ".pdf":      parse_pdf,
    ".docx":     parse_docx,
    ".xlsx":     parse_excel,
    ".xls":      parse_excel,
    ".csv":      parse_csv,
    ".txt":      parse_text,
    ".md":       parse_text,
    ".markdown": parse_text,
    ".json":     parse_json,
    ".xml":      parse_xml,
}

# Media formats — need special handling (transcription / OCR)
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp"}

# Canonical strategies (no aliases) in recommended test order
CANONICAL_STRATEGIES = [
    "semantic",
    "overlap",
    "rust",
    "smart_check",
    "structure_aware",
    "recursive_overlap",
    "elite",
    "elite_v2",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _print_header(title: str) -> None:
    w = 76
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)


def _print_subheader(title: str) -> None:
    print(f"\n  ── {title} {'─' * max(1, 55 - len(title))}")


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [{len(text) - max_chars} more chars]"


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 1: Extract Text from File
# ═══════════════════════════════════════════════════════════════════════════════

async def extract_text_from_file(file_path: str) -> Dict[str, Any]:
    """
    Parse any supported file and return extracted text + metadata.
    Returns dict with keys: text, source_type, metadata, file_path
    """
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)

    print(f"\n  File      : {file_name}")
    print(f"  Path      : {file_path}")
    print(f"  Size      : {file_size:,} bytes ({file_size / 1024:.1f} KB)")
    print(f"  Extension : {ext}")

    # ── Document formats ─────────────────────────────────────────────────
    if ext in PARSER_MAP:
        parser = PARSER_MAP[ext]
        print(f"  Parser    : {parser.__name__}")
        print(f"\n  Parsing file...")

        parsed = await parser(file_path)

        # Different parsers return text in different keys
        # Priority: normalized_text > cleaned_text > raw_text
        text = (
            parsed.get("normalized_text")
            or parsed.get("cleaned_text")
            or parsed.get("raw_text")
            or ""
        )

        # For Excel/CSV that return pre-chunked data
        if not text and "chunks" in parsed:
            # Merge chunk texts for re-chunking test
            chunk_texts = []
            for c in parsed["chunks"]:
                ct = c.get("text") or c.get("cleaned_text") or ""
                if ct.strip():
                    chunk_texts.append(ct.strip())
            text = "\n\n".join(chunk_texts)

        source_type = parsed.get("source_type", ext.lstrip("."))
        metadata = parsed.get("metadata", {})
        pages = parsed.get("pages", metadata.get("pages"))

        print(f"  Source    : {source_type}")
        if pages:
            print(f"  Pages     : {pages}")
        print(f"  Extracted : {len(text):,} characters")

        return {
            "text": text,
            "source_type": source_type,
            "metadata": metadata,
            "file_path": file_path,
        }

    # ── Audio formats ────────────────────────────────────────────────────
    elif ext in AUDIO_EXTENSIONS:
        try:
            from app.services.ingestion.media.audio_ingestor_v1 import AudioIngestorV1
            print(f"  Parser    : AudioIngestorV1 (Whisper transcription)")
            print(f"\n  Transcribing audio... (this may take a while)")

            ingestor = AudioIngestorV1()
            # Use the transcribe method directly without DB
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(file_path)
            text = result.get("text", "")

            print(f"  Extracted : {len(text):,} characters from audio")
            return {
                "text": text,
                "source_type": "audio",
                "metadata": {"format": ext, "transcription_model": "whisper-base"},
                "file_path": file_path,
            }
        except ImportError as e:
            print(f"\n  WARNING: Audio processing requires 'openai-whisper' package.")
            print(f"  Install with: pip install openai-whisper")
            raise RuntimeError(f"Audio dependencies not available: {e}")

    # ── Video formats ────────────────────────────────────────────────────
    elif ext in VIDEO_EXTENSIONS:
        try:
            print(f"  Parser    : Video extraction (audio track → Whisper)")
            print(f"\n  Extracting audio from video and transcribing...")

            from moviepy.editor import VideoFileClip
            import whisper
            import tempfile

            # Extract audio to temp wav
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_audio = tmp.name
            try:
                clip = VideoFileClip(file_path)
                clip.audio.write_audiofile(tmp_audio, verbose=False, logger=None)
                clip.close()

                model = whisper.load_model("base")
                result = model.transcribe(tmp_audio)
                text = result.get("text", "")
            finally:
                if os.path.exists(tmp_audio):
                    os.unlink(tmp_audio)

            print(f"  Extracted : {len(text):,} characters from video audio")
            return {
                "text": text,
                "source_type": "video",
                "metadata": {"format": ext, "transcription_model": "whisper-base"},
                "file_path": file_path,
            }
        except ImportError as e:
            print(f"\n  WARNING: Video processing requires 'moviepy' and 'openai-whisper'.")
            raise RuntimeError(f"Video dependencies not available: {e}")

    # ── Image formats ────────────────────────────────────────────────────
    elif ext in IMAGE_EXTENSIONS:
        try:
            print(f"  Parser    : Image OCR (pytesseract + Pillow)")
            print(f"\n  Extracting text from image via OCR...")

            from PIL import Image
            import pytesseract

            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)

            print(f"  Extracted : {len(text):,} characters from image OCR")
            if len(text.strip()) < 10:
                print(f"  NOTE: Very little text found. Image may be a photo/chart, not a text document.")

            return {
                "text": text,
                "source_type": "image",
                "metadata": {"format": ext, "size": img.size, "mode": img.mode},
                "file_path": file_path,
            }
        except ImportError as e:
            print(f"\n  WARNING: Image OCR requires 'Pillow' and 'pytesseract'.")
            raise RuntimeError(f"Image dependencies not available: {e}")

    else:
        supported = sorted(
            list(PARSER_MAP.keys())
            | AUDIO_EXTENSIONS
            | VIDEO_EXTENSIONS
            | IMAGE_EXTENSIONS
        )
        raise ValueError(
            f"Unsupported file extension '{ext}'.\n"
            f"Supported: {', '.join(supported)}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 2: Run Chunking Strategy on Extracted Text
# ═══════════════════════════════════════════════════════════════════════════════

async def run_chunking(
    text: str,
    strategy_name: str,
    source_type: str = "txt",
    file_id: str = "test-doc",
    max_chars: int = 300,
) -> Dict[str, Any]:
    """Run a single chunking strategy on the given text."""
    result = {
        "strategy": strategy_name,
        "chunk_count": 0,
        "total_tokens": 0,
        "avg_tokens": 0.0,
        "avg_quality": 0.0,
        "min_quality": None,
        "max_quality": None,
        "elapsed_ms": 0.0,
        "error": None,
        "chunks": [],
    }

    clear_chunker_cache()
    t0 = time.perf_counter()

    try:
        chunker = get_chunker(strategy_name)
        chunks = await chunker.chunk(
            text,
            file_id=f"{file_id}-{strategy_name}",
            source_type=source_type,
            embedding_model="test-embed-model",
        )

        elapsed = (time.perf_counter() - t0) * 1000
        result["elapsed_ms"] = round(elapsed, 1)
        result["chunk_count"] = len(chunks)
        result["chunks"] = chunks

        token_counts = []
        quality_scores = []
        for c in chunks:
            tok = c.get("tokens", 0)
            if isinstance(tok, (int, float)) and tok > 0:
                token_counts.append(int(tok))
            ri = c.get("reasoning_ingestion", {})
            qs = ri.get("chunk_quality_score")
            if isinstance(qs, (int, float)):
                quality_scores.append(float(qs))

        result["total_tokens"] = sum(token_counts)
        result["avg_tokens"] = round(_avg(token_counts), 1)
        result["avg_quality"] = round(_avg(quality_scores), 3)
        result["min_quality"] = round(min(quality_scores), 3) if quality_scores else None
        result["max_quality"] = round(max(quality_scores), 3) if quality_scores else None

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        result["elapsed_ms"] = round(elapsed, 1)
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Display Results
# ═══════════════════════════════════════════════════════════════════════════════

def display_strategy_result(
    result: Dict[str, Any],
    max_chars: int = 300,
) -> None:
    status = "ERROR" if result["error"] else "OK"
    _print_header(f"Strategy: {result['strategy']}  [{status}]")

    if result["error"]:
        print(f"\n  ERROR: {result['error']}")
        return

    print(f"  Chunks    : {result['chunk_count']}")
    print(f"  Tokens    : total={result['total_tokens']}  avg={result['avg_tokens']}")
    print(
        f"  Quality   : avg={result['avg_quality']}  "
        f"min={result['min_quality']}  max={result['max_quality']}"
    )
    print(f"  Time      : {result['elapsed_ms']} ms")

    chunks = result["chunks"]
    for i, chunk in enumerate(chunks):
        ri = chunk.get("reasoning_ingestion", {})
        tok = chunk.get("tokens", "?")
        qs = ri.get("chunk_quality_score", "?")
        sig = ri.get("signal_type", "?")
        biz = ri.get("business_function", "?")
        noise = ri.get("noise_flag", False)
        low_q = ri.get("low_quality_flag", False)
        strat = ri.get("chunking_strategy", "?")
        shash = chunk.get("semantic_hash", "?")

        _print_subheader(f"Chunk {i + 1}/{len(chunks)}")
        print(f"    Tokens          : {tok}")
        print(f"    Quality Score   : {qs}")
        print(f"    Signal Type     : {sig}")
        print(f"    Business Func   : {biz}")
        print(f"    Strategy Tag    : {strat}")
        print(f"    Semantic Hash   : {shash[:16]}..." if isinstance(shash, str) and len(shash) > 16 else f"    Semantic Hash   : {shash}")
        if noise:
            print(f"    ⚠ NOISE FLAG    : True")
        if low_q:
            print(f"    ⚠ LOW QUALITY   : True")

        # Always show chunk text
        text = chunk.get("text", "")
        display_text = _truncate(text, max_chars)
        print(f"    ┌─ Text ────────────────────────────────────────")
        for line in display_text.split("\n"):
            print(f"    │ {line}")
        print(f"    └───────────────────────────────────────────────")


def display_summary(results: List[Dict[str, Any]], file_name: str) -> None:
    _print_header(f"SUMMARY — {file_name}")

    hdr = f"  {'Strategy':<22} {'Status':<8} {'Chunks':<8} {'AvgTok':<8} {'AvgQ':<8} {'Time(ms)':<10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for r in results:
        status = "OK" if not r["error"] else "ERROR"
        print(
            f"  {r['strategy']:<22} {status:<8} {r['chunk_count']:<8} "
            f"{r['avg_tokens']:<8.1f} {r['avg_quality']:<8.3f} {r['elapsed_ms']:<10.1f}"
        )

    ok_count = sum(1 for r in results if not r["error"])
    print(f"\n  Result: {ok_count}/{len(results)} strategies succeeded")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Write semantic_hash.log
# ═══════════════════════════════════════════════════════════════════════════════

def write_semantic_hash_log(
    results: List[Dict[str, Any]],
    file_name: str,
    file_path: str,
    extracted_text: str,
    log_path: str = "semantic_hash.log",
) -> str:
    """Write full chunk data to semantic_hash.log for inspection."""
    lines: List[str] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines.append("=" * 80)
    lines.append(f"  SEMANTIC HASH LOG — {file_name}")
    lines.append(f"  Generated: {ts}")
    lines.append(f"  Source file: {file_path}")
    lines.append(f"  Extracted text length: {len(extracted_text):,} chars")
    lines.append("=" * 80)
    lines.append("")

    for r in results:
        strategy = r["strategy"]
        lines.append("-" * 80)
        lines.append(f"STRATEGY: {strategy}")
        lines.append(f"  Chunks: {r['chunk_count']}  |  Tokens: {r['total_tokens']}  |  "
                     f"Avg Quality: {r['avg_quality']}  |  Time: {r['elapsed_ms']} ms")
        lines.append("-" * 80)

        if r["error"]:
            lines.append(f"  ERROR: {r['error']}")
            lines.append("")
            continue

        for i, chunk in enumerate(r["chunks"]):
            ri = chunk.get("reasoning_ingestion", {})
            lines.append("")
            lines.append(f"  ┌── Chunk {i + 1}/{len(r['chunks'])} ──────────────────────────────────────")
            lines.append(f"  │ Semantic Hash     : {chunk.get('semantic_hash', '?')}")
            lines.append(f"  │ Normalized Hash   : {chunk.get('normalized_hash', '?')}")
            lines.append(f"  │ Tokens            : {chunk.get('tokens', '?')}")
            lines.append(f"  │ Confidence        : {chunk.get('confidence', '?')}")
            lines.append(f"  │ Source Type       : {chunk.get('source_type', '?')}")
            lines.append(f"  │ Quality Score     : {ri.get('chunk_quality_score', '?')}")
            lines.append(f"  │ Signal Type       : {ri.get('signal_type', '?')}")
            lines.append(f"  │ Business Function : {ri.get('business_function', '?')}")
            lines.append(f"  │ Time Horizon      : {ri.get('time_horizon', '?')}")
            lines.append(f"  │ Granularity       : {ri.get('granularity', '?')}")
            lines.append(f"  │ Sentiment         : {ri.get('sentiment_bucket', '?')} "
                         f"(confidence: {ri.get('sentiment_confidence', '?')})")
            lines.append(f"  │ Chunking Strategy : {ri.get('chunking_strategy', '?')}")
            lines.append(f"  │ Noise Flag        : {ri.get('noise_flag', False)}")
            lines.append(f"  │ Low Quality Flag  : {ri.get('low_quality_flag', False)}")
            lines.append(f"  │ Regulated Content : {ri.get('potentially_regulated', False)}")
            lines.append(f"  │ Extraction Time   : {ri.get('extraction_timestamp', '?')}")

            # Strategy-specific metadata
            extra_keys = sorted(
                k for k in ri
                if k not in {
                    "chunk_quality_score", "signal_type", "business_function",
                    "time_horizon", "granularity", "sentiment_bucket",
                    "sentiment_confidence", "chunking_strategy", "noise_flag",
                    "low_quality_flag", "potentially_regulated",
                    "extraction_timestamp", "origin_authority",
                    "extraction_confidence", "data_lineage_id",
                }
            )
            if extra_keys:
                lines.append(f"  │")
                lines.append(f"  │ ── Strategy-Specific Metadata ──")
                for k in extra_keys:
                    v = ri[k]
                    lines.append(f"  │ {k:<22}: {v}")

            # Full chunk text
            text = chunk.get("text", "")
            lines.append(f"  │")
            lines.append(f"  │ ── Chunk Text ({len(text)} chars) ──")
            for tline in text.split("\n"):
                lines.append(f"  │ {tline}")
            lines.append(f"  └────────────────────────────────────────────────────────")

        lines.append("")

    # Write file
    log_content = "\n".join(lines)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_content)

    return log_path


# ═══════════════════════════════════════════════════════════════════════════════
#  Extracted Text Preview
# ═══════════════════════════════════════════════════════════════════════════════

def display_extracted_text(text: str, max_preview: int = 1000) -> None:
    _print_header("EXTRACTED TEXT PREVIEW")
    preview = _truncate(text, max_preview)
    for line in preview.split("\n"):
        print(f"  {line}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test chunking strategies on your own documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available strategies:
  semantic           Default recursive semantic splitting
  overlap            Enterprise sliding window with intelligent overlap
  rust               High-performance with graceful fallback cascade
  smart_check        Adaptive token sizing with domain awareness
  structure_aware    Preserves document structure (headings, tables, code)
  recursive_overlap  Semantic + coherence-gated intelligent overlap
  elite              Hierarchical multi-resolution (parent+child chunks)
  elite_v2           Advanced with optional LLM enrichment

Supported file formats:
  Documents : .pdf .docx .xlsx .xls .csv .txt .md .json .xml
  Audio     : .wav .mp3 .m4a .flac .aac .ogg .opus .wma
  Video     : .mp4 .avi .mov .mkv .webm .flv .wmv .m4v
  Images    : .png .jpg .jpeg .gif .bmp .tiff .tif .webp
        """,
    )
    parser.add_argument("file", help="Path to the document file to test")
    parser.add_argument(
        "--strategy", "-s",
        nargs="*",
        default=["semantic"],
        help="Chunking strategy name(s), or 'all' to test every strategy (default: semantic)",
    )
    parser.add_argument(
        "--max-chars", "-m",
        type=int,
        default=300,
        help="Max characters to display per chunk in console (default: 300)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="semantic_hash.log",
        help="Path for the detailed chunk log file (default: semantic_hash.log)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=1000,
        help="Max characters for extracted text preview (default: 1000, 0 to skip)",
    )

    args = parser.parse_args()

    # ── Resolve strategies ───────────────────────────────────────────────
    all_strategies = set(list_chunking_strategies())
    if args.strategy and args.strategy[0].lower() == "all":
        selected = CANONICAL_STRATEGIES
    else:
        selected = []
        for name in args.strategy:
            key = name.lower().strip()
            if key not in all_strategies:
                print(f"WARNING: '{key}' is not a registered strategy. Skipping.")
                print(f"  Available: {', '.join(sorted(all_strategies))}")
            else:
                selected.append(key)

    if not selected:
        print("No valid strategies selected.")
        return 1

    # ── Step 1: Extract text ─────────────────────────────────────────────
    _print_header("STEP 1: FILE EXTRACTION")
    try:
        extraction = await extract_text_from_file(args.file)
    except Exception as exc:
        print(f"\n  EXTRACTION FAILED: {exc}")
        traceback.print_exc()
        return 1

    text = extraction["text"]
    source_type = extraction["source_type"]
    file_name = os.path.basename(args.file)

    if not text or len(text.strip()) < 10:
        print(f"\n  WARNING: Extracted text is very short ({len(text)} chars).")
        print(f"  The file may be empty, image-only, or require additional dependencies.")
        if not text.strip():
            return 1

    # ── Show extracted text preview ──────────────────────────────────────
    if args.preview > 0:
        display_extracted_text(text, args.preview)

    # ── Step 2: Run chunking strategies ──────────────────────────────────
    _print_header(f"STEP 2: CHUNKING — {len(selected)} strategies on '{file_name}'")
    print(f"  Strategies : {', '.join(selected)}")
    print(f"  Source type: {source_type}")
    print(f"  Input size : {len(text):,} characters")

    results: List[Dict[str, Any]] = []

    for strategy in selected:
        r = await run_chunking(
            text=text,
            strategy_name=strategy,
            source_type=source_type,
            file_id=f"test-{os.path.splitext(file_name)[0]}",
            max_chars=args.max_chars,
        )
        display_strategy_result(r, max_chars=args.max_chars)
        results.append(r)

    # ── Write semantic_hash.log ──────────────────────────────────────────
    log_path = write_semantic_hash_log(
        results=results,
        file_name=file_name,
        file_path=os.path.abspath(args.file),
        extracted_text=text,
        log_path=args.log_file,
    )
    _print_header(f"LOG FILE WRITTEN")
    print(f"  Full chunk data with text → {os.path.abspath(log_path)}")
    print(f"  Open with: notepad {log_path}")
    print()

    # ── Summary ──────────────────────────────────────────────────────────
    if len(results) > 1:
        display_summary(results, file_name)

    return 0 if all(not r["error"] for r in results) else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
