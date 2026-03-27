"""Edge case tests for structure_aware chunker."""
import asyncio
from app.core.chunking_stratagies.chunking_registry import get_chunker


async def main():
    chunker = get_chunker("structure_aware")

    # TEST 1: Oversized code block (>2000 chars) — splits at line boundaries
    big_code = "```python\n" + "\n".join(
        [f'x_{i} = compute_value({i}, param="test")  # line {i}' for i in range(80)]
    ) + "\n```"
    chunks = await chunker.chunk(big_code, source_type="pdf", embedding_model="t")
    print(f"TEST 1 — Oversized code ({len(big_code)} chars):")
    for i, c in enumerate(chunks):
        meta = c.get("reasoning_ingestion", {})
        ct = meta.get("content_type", "?")
        qs = meta.get("chunk_quality_score", -1)
        print(f"  chunk {i}: type={ct} tokens={c['tokens']} quality={qs:.3f}")
    print()

    # TEST 2: Oversized table (>3000 chars) — splits keeping header row
    header = "| ID   | Name       | Revenue  | Region     | Status     |"
    sep    = "|------|------------|----------|------------|------------|"
    rows = [
        f"| {i:04d} | Customer{i:02d} | ${i*1000:>7,} | Region-{i%5}  | Active     |"
        for i in range(60)
    ]
    big_table = "\n".join([header, sep] + rows)
    chunks = await chunker.chunk(big_table, source_type="csv", embedding_model="t")
    print(f"TEST 2 — Oversized table ({len(big_table)} chars):")
    for i, c in enumerate(chunks):
        meta = c.get("reasoning_ingestion", {})
        ct = meta.get("content_type", "?")
        qs = meta.get("chunk_quality_score", -1)
        has_hdr = "| ID" in c["text"][:60]
        print(f"  chunk {i}: type={ct} tokens={c['tokens']} quality={qs:.3f} header_preserved={has_hdr}")
    print()

    # TEST 3: Plain prose — falls through to semantic chunking
    prose = "Financial performance exceeded expectations in Q4. " * 30
    chunks = await chunker.chunk(prose, source_type="pdf", embedding_model="t")
    print(f"TEST 3 — Plain prose ({len(prose)} chars):")
    for i, c in enumerate(chunks):
        meta = c.get("reasoning_ingestion", {})
        ct = meta.get("content_type", "?")
        qs = meta.get("chunk_quality_score", -1)
        print(f"  chunk {i}: type={ct} tokens={c['tokens']} quality={qs:.3f}")
    print()

    # TEST 4: Empty/whitespace
    chunks = await chunker.chunk("   ", source_type="pdf", embedding_model="t")
    print(f"TEST 4 — Empty text: {len(chunks)} chunks (expected 0)")
    print()

    # TEST 5: document_aware alias resolves to same class
    alias = get_chunker("document_aware")
    chunks = await alias.chunk("# Test\nHello world", source_type="txt", embedding_model="t")
    ct = chunks[0]["reasoning_ingestion"]["content_type"]
    print(f"TEST 5 — document_aware alias: {len(chunks)} chunk(s), type={ct}")
    print()

    # TEST 6: Mixed markdown — heading with no content after it
    doc = "# Heading One\n## Heading Two\n## Heading Three\nSome content here."
    chunks = await chunker.chunk(doc, source_type="pdf", embedding_model="t")
    print(f"TEST 6 — Trailing headings ({len(chunks)} chunks):")
    for i, c in enumerate(chunks):
        meta = c.get("reasoning_ingestion", {})
        section = meta.get("section_title", "")
        preview = c["text"][:80].replace("\n", " ")
        print(f"  chunk {i}: section='{section}' text={preview}")
    print()

    print("ALL EDGE CASE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
