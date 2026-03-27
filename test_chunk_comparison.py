"""Analyze real PDF chunks and compare old vs new."""
import asyncio
from app.core.chunking_stratagies.chunking_registry import get_chunker

PDF = "static/uploads/manual/AnimalHusbandary_2025_8964de6740314c5d82d5809f5664e08c.pdf"

# Old chunks from database (from CSV export)
OLD_CHUNKS = [
    {"tokens": 40,  "text": "uman Population according to Population Census-2011 Species-wise Milk Contribution 2024-25 Non-Descript Buffalo, Goat, 3.32%... ---PAGE BREAK---"},
    {"tokens": 32,  "text": "The image contains visual information. Detected text: @ovept_of AHO @ rept of and] deptorano"},
    {"tokens": 74,  "text": "The following content is extracted from a chart... Indian livestock sector demonstrated remarkable growth GVA surging 195%"},
    {"tokens": 15,  "text": "The image is a real-world photograph. This image appears to be a photograph or illustration."},
    {"tokens": 70,  "text": "This rapid growth has enabled livestock to become a dominant pillar of agricultural income, contributing 31%..."},
    {"tokens": 85,  "text": "Furthermore, the sector has witnessed a steady increase in exports, total value ₹66,249 crore in 2024-25..."},
    {"tokens": 65,  "text": "The Animal Husbandry Statistics Division plays a crucial role in providing authentic data..."},
    {"tokens": 54,  "text": "It fulfils the need of timely availability of reliable and current data relating to various livestock indicators..."},
    {"tokens": 82,  "text": "Together, these surveys form a robust statistical framework... ---PAGE BREAK--- Methodology The Survey is conducted in three seasons"},
    {"tokens": 110, "text": "N N N O O O S A S S 1st November to E 1st March to A 1st July to 31st A 28th or 29th E E S..."},
    {"tokens": 53,  "text": "The per-capita availably of milk is 485 grams per day... 446 459 471 485 427 406 390 188 198 210 222 231 240 248"},
    {"tokens": 73,  "text": "The survey is conducted in three stages: villages/urban wards, Households/Enterprises/Butcher Shops..."},
    {"tokens": 28,  "text": "Generally, 5% sample villages/wards selected at random without replacement for every District for SELECTIONSOF every season"},
    {"tokens": 72,  "text": "SEASON 5% Grouping 5% sample in two sub samples... ---PAGE BREAK--- eLISS (End-to-End Livestock Integrated Sample Survey)"},
    {"tokens": 70,  "text": "The responsibility to develop this system was entrusted to ICAR-IASRI, New Delhi..."},
    {"tokens": 84,  "text": "Glimpse of Major Livestock Products ITEMS 2023-24 2024-25 Y-o-Y* Milk Production 239.30 247.87 3.58... ---PAGE BREAK---"},
    {"tokens": 28,  "text": "India ranks 1st in the world in terms of total milk production. (Source: FAO). Increased by 3.58%."},
    {"tokens": 75,  "text": "Uttar Pradesh (15.66%) 2. Rajasthan (14.82%)... ---PAGE BREAK--- Highlights Egg production 2024-25"},
]

def classify_old_chunk(chunk):
    text = chunk["text"]
    issues = []
    if "---PAGE BREAK---" in text:
        issues.append("PAGE_BREAK_POLLUTION")
    if chunk["tokens"] < 20:
        issues.append("TOO_SHORT")
    if "image contains visual" in text.lower() or "real-world photograph" in text.lower():
        issues.append("NOISE_IMAGE_DESCRIPTION")
    if "N N N O O O" in text or "E E S" in text:
        issues.append("SCRAMBLED_DIAGRAM_OCR")
    garbled = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text),1)
    if garbled > 0.15:
        issues.append("HIGH_NOISE_CHARS")
    if "The following content is extracted from a chart" in text:
        issues.append("LLM_PROMPT_PREFIX_EMBEDDED")
    numeric = sum(1 for t in text.split() if any(c.isdigit() for c in t)) / max(len(text.split()),1)
    if numeric > 0.4:
        issues.append("RAW_CHART_NUMBERS")
    quality = "GOOD" if not issues else ("POOR" if len(issues) >= 2 else "MEDIOCRE")
    return quality, issues

async def main():
    print("=" * 90)
    print("OLD CHUNKS ANALYSIS (from database CSV export)")
    print("=" * 90)
    good = mediocre = poor = 0
    for i, ch in enumerate(OLD_CHUNKS):
        quality, issues = classify_old_chunk(ch)
        if quality == "GOOD": good += 1
        elif quality == "MEDIOCRE": mediocre += 1
        else: poor += 1
        issue_str = ", ".join(issues) if issues else "clean"
        print(f"  Old [{i:02d}] tok={ch['tokens']:4d} | {quality:8s} | {issue_str}")
    total = len(OLD_CHUNKS)
    print(f"\n  Old chunks: {total} total → GOOD={good} | MEDIOCRE={mediocre} | POOR={poor}")
    print(f"  Usability: {good}/{total} = {100*good//total}% chunks are clean\n")

    print("=" * 90)
    print("NEW CHUNKS (structure_aware strategy on same PDF)")
    print("=" * 90)
    import pdfplumber
    pages = []
    with pdfplumber.open(PDF) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                pages.append(t)
    raw_text = "\n\n".join(pages)

    chunker = get_chunker("structure_aware")
    chunks = await chunker.chunk(raw_text, source_type="pdf", embedding_model="text-embedding-3-small")

    scores = [ch["reasoning_ingestion"]["chunk_quality_score"] for ch in chunks]
    avg = sum(scores) / len(scores)
    high = sum(1 for s in scores if s >= 0.75)
    mid  = sum(1 for s in scores if 0.50 <= s < 0.75)
    low  = sum(1 for s in scores if s < 0.50)

    for i, ch in enumerate(chunks):
        meta = ch["reasoning_ingestion"]
        ct  = meta.get("content_type", "?")
        qs  = meta.get("chunk_quality_score", 0)
        tok = ch["tokens"]
        preview = ch["text"][:100].replace("\n", " ")
        flag = "✓" if qs >= 0.75 else ("~" if qs >= 0.50 else "✗")
        print(f"  New [{i:02d}] {flag} {ct:6s} tok={tok:4d} quality={qs:.3f} | {preview[:80]}...")

    print(f"\n  New chunks: {len(chunks)} total | avg quality={avg:.3f}")
    print(f"  HIGH(>=0.75)={high}  MID(0.50-0.75)={mid}  LOW(<0.50)={low}")
    print()
    print("=" * 90)
    print("COMPARISON")
    print("=" * 90)
    print(f"  Old strategy : {total} chunks, {good}/{total} ({100*good//total}%) clean, avg quality ~0.55 (estimated)")
    print(f"  New strategy : {len(chunks)} chunks, {high}/{len(chunks)} ({100*high//len(chunks)}%) high quality, avg={avg:.3f}")
    print()
    print("KEY PROBLEMS IN OLD CHUNKS:")
    print("  1. PAGE BREAK markers bleeding into chunk text (5 chunks polluted)")
    print("  2. OCR noise chunks: image description stubs (2 chunks, 15-32 tokens)")
    print("  3. Scrambled diagram OCR: 'N N N O O O S A S' (1 chunk, unreadable)")
    print("  4. Raw chart number dumps without context (2 chunks)")
    print("  5. LLM prompt prefix embedded in content (1 chunk)")
    print("  6. Truncated text starting mid-word: 'uman Population...' (missing H)")
    print("  7. Chunks spanning page boundaries with unrelated content merged")


if __name__ == "__main__":
    asyncio.run(main())
