"""
test_structure_aware_real_file.py
===================================
Test the structure_aware chunking strategy with a REAL PDF, DOCX, or TXT file.

Usage:
    python test_structure_aware_real_file.py path/to/document.pdf
    python test_structure_aware_real_file.py path/to/document.docx
    python test_structure_aware_real_file.py path/to/document.txt

The script:
  1. Extracts text from the file using the production parsers (pdf_parser_v2 / docx_parser_v2)
  2. Runs it through the structure_aware chunker
  3. Prints a full report: type, section, token count, quality score, text preview

CHUNKING_STRATEGY in .env is already set to structure_aware, but this script
explicitly uses it so the test is self-contained.
"""

import sys
import os
import asyncio
from pathlib import Path

# Make sure the project is on the path when running from the workspace root
sys.path.insert(0, str(Path(__file__).parent))

os.environ.setdefault("CHUNKING_STRATEGY", "structure_aware")

from app.core.chunking_stratagies.chunking_registry import get_chunker  # noqa: E402


# ─── TEXT EXTRACTORS ─────────────────────────────────────────────────────────

def extract_text_from_pdf(file_path: str) -> str:
    """Use production pdf_parser_v2 extractors (pdfplumber → PyMuPDF fallback)."""
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t.strip():
                    pages.append(t)
        if pages:
            print(f"  [pdfplumber] extracted {len(pages)} page(s)")
            return "\n\n".join(pages)
    except Exception as e:
        print(f"  [pdfplumber] failed: {e} — trying PyMuPDF")

    try:
        import fitz
        pages = []
        doc = fitz.open(file_path)
        for page in doc:
            t = page.get_text("text") or ""
            if t.strip():
                pages.append(t)
        if pages:
            print(f"  [PyMuPDF] extracted {len(pages)} page(s)")
            return "\n\n".join(pages)
    except Exception as e:
        print(f"  [PyMuPDF] failed: {e}")

    raise RuntimeError("Could not extract text from PDF — check that pdfplumber or PyMuPDF is installed.")


def extract_text_from_docx(file_path: str) -> str:
    """Use python-docx to extract paragraphs + tables."""
    from docx import Document
    doc = Document(file_path)
    parts = []

    for para in doc.paragraphs:
        t = para.text.strip()
        if not t:
            continue
        # Detect heading style from DOCX structure and format as markdown heading
        style_name = (para.style.name or "").lower()
        if "heading 1" in style_name:
            parts.append(f"# {t}")
        elif "heading 2" in style_name:
            parts.append(f"## {t}")
        elif "heading 3" in style_name:
            parts.append(f"### {t}")
        elif "heading" in style_name:
            parts.append(f"#### {t}")
        else:
            parts.append(t)

    for table in doc.tables:
        rows = []
        for i, row in enumerate(table.rows):
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            rows.append("| " + " | ".join(cells) + " |")
            if i == 0:
                rows.append("| " + " | ".join("---" for _ in cells) + " |")
        if rows:
            parts.append("\n".join(rows))

    print(f"  [python-docx] extracted {len(doc.paragraphs)} paragraphs, {len(doc.tables)} tables")
    return "\n\n".join(parts)


def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    print(f"  [txt] read {len(content):,} chars")
    return content


def extract_text(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    elif suffix in (".docx", ".doc"):
        return extract_text_from_docx(file_path)
    elif suffix == ".txt":
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf .docx .txt")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

async def run(file_path: str) -> None:
    print(f"\nFile : {file_path}")
    print(f"Size : {Path(file_path).stat().st_size:,} bytes")

    print("\nExtracting text...")
    raw_text = extract_text(file_path)
    print(f"Extracted: {len(raw_text):,} chars  ({len(raw_text.split()):,} words)\n")

    source_type = Path(file_path).suffix.lstrip(".").lower()
    chunker = get_chunker("structure_aware")
    print(f"Chunker: {type(chunker).__name__}\n")

    chunks = await chunker.chunk(
        raw_text,
        source_type=source_type,
        embedding_model="text-embedding-3-small",
    )

    print(f"Total chunks: {len(chunks)}")
    print("=" * 100)

    quality_scores = []
    for i, ch in enumerate(chunks):
        meta = ch.get("reasoning_ingestion", {})
        ctype   = meta.get("content_type", "?")
        section = meta.get("section_title", "")
        depth   = meta.get("section_depth", 0)
        quality = meta.get("chunk_quality_score", -1)
        lang    = meta.get("code_language", "")
        tok     = ch.get("tokens", 0)
        preview = ch.get("text", "")[:120].replace("\n", " ")

        lang_str = f"  lang={lang}" if lang else ""
        heading  = ("  " + "#" * depth + " " + section) if section else ""
        quality_scores.append(quality)

        print(
            f"[{i:03d}] {ctype:6s} | tok={tok:4d} | quality={quality:.3f}"
            f"{lang_str}{heading}"
        )
        print(f"       {preview}...")
        print()

    if quality_scores:
        avg_q = sum(quality_scores) / len(quality_scores)
        min_q = min(quality_scores)
        max_q = max(quality_scores)
        print("─" * 100)
        print(f"Quality summary  →  avg={avg_q:.3f}  min={min_q:.3f}  max={max_q:.3f}")
        print()

    # Content type breakdown
    from collections import Counter
    breakdown = Counter(ch["reasoning_ingestion"].get("content_type", "?") for ch in chunks)
    print("Content type breakdown:")
    for ctype, count in breakdown.most_common():
        print(f"  {ctype:8s}: {count}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        prog = Path(sys.argv[0]).name
        print(f"Usage: python {prog} <path-to-file.pdf|.docx|.txt>")
        print()
        print("Example:")
        print(f"  python {prog} uploaded_files/report.pdf")
        sys.exit(1)

    target = sys.argv[1]
    if not Path(target).exists():
        print(f"Error: file not found: {target}")
        sys.exit(1)

    asyncio.run(run(target))
