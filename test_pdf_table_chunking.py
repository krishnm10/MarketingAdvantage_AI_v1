"""
Test: PDF-extracted table detection in structure_aware chunker.

Simulates pdfplumber-extracted text from "Design and build accessible PDF tables"
to verify tables are detected and kept as coherent chunks (not split as prose).
"""
import asyncio
from app.core.chunking_stratagies.chunking_registry import get_chunker


# Simulated pdfplumber output — space-separated columns, no pipes
PDF_TABLE_TEXT = """Design and build accessible PDF tables
Sample tables

Table 1
Column header (TH) Column header (TH) Column header (TH)
Row header (TH) Data cell (TD) Data cell (TD)
Row header(TH) Data cell (TD) Data cell (TD)

Table 2: example of footnotes referenced from within a table
Expenditure by function £ million 2009/10 2010/11 1
Policy functions Financial 22.5 30.57
Information 2 10.2 14.8
Contingency 2.6 1.2
Remunerated functions Agency services 3 44.7 35.91
Payments 22.41 19.88
Banking 22.90 44.23
Other 12.69 10.32
(1) Provisional total as of publication date.
(2) Costs associated with on-going information programmes.
(3) From the management accounts, net of recoveries, including interest charges.

Table 3: film credits style layout
Main character Daniel Radcliffe
Sidekick 1 Rupert Grint
Sidekick 2 Emma Watson
Lovable ogre Robbie Coltrane
Professor Maggie Smith
Headmaster Richard Harris

Table 5: year-end financial statement (£, thousands)
2010 2009 2008
Non-current assets
Property 345 445 222
Investment 567 654 423
Intangibles 423 123 453
Current assets
Trade and other receivables 435 634 231
Cash and cash equivalents 524 123 482
Other 223 211 254

Table 10: self-contained year-end statement (£, thousands) (multiple
layout problems)
2011 2010 restated
General income 250,000 200,000
Increase in value, WIP 15,000 30,000
265,000 230,000
Administrative costs
Staff costs (200,000) (150,000)
Early departures (10,000) (20,000)
Other (25,000) (10,000)
Depreciation (10,000) (10,000)
Programme costs
Impairment loss (10,000) (5,000)
Other (5,000) (5,000)
(260,000) (200,000)
Surplus 5,000 30,000
"""


async def main():
    chunker = get_chunker("structure_aware")
    chunks = await chunker.chunk(
        PDF_TABLE_TEXT,
        source_type="pdf",
        embedding_model="test-model",
    )

    print(f"Total chunks: {len(chunks)}")
    print("=" * 80)

    table_chunks = 0
    for i, ch in enumerate(chunks):
        ri = ch.get("reasoning_ingestion", {})
        content_type = ri.get("content_type", "?")
        section = ri.get("section_title", "")
        tokens = ch.get("tokens", 0)
        text_preview = ch.get("text", "")[:150].replace("\n", "\\n")

        print(
            f"Chunk {i}: type={content_type:6s} | "
            f"section={section:30s} | tokens={tokens:4d}"
        )
        print(f"  text: {text_preview}")
        print()

        if content_type == "table":
            table_chunks += 1

    print("=" * 80)
    print(f"TABLE chunks: {table_chunks}")
    print(f"TOTAL chunks: {len(chunks)}")

    # Validation checks
    assert table_chunks >= 4, (
        f"Expected at least 4 table chunks for 5 tables, got {table_chunks}"
    )

    # Verify Table 2 data includes footnotes and full data rows
    table2_chunks = [
        ch for ch in chunks
        if "Expenditure by function" in ch.get("text", "")
        or "22.5" in ch.get("text", "")
    ]
    assert table2_chunks, "Table 2 data should exist in at least one chunk"
    t2_text = table2_chunks[0]["text"]
    assert "22.5" in t2_text and "30.57" in t2_text, (
        "Table 2 data row (Financial 22.5 30.57) should be intact"
    )
    assert "44.23" in t2_text, (
        "Table 2 row (Banking 22.90 44.23) should be in same chunk"
    )

    # Verify Table 10 parenthesized negatives are intact
    table10_chunks = [
        ch for ch in chunks
        if "(200,000)" in ch.get("text", "")
    ]
    assert table10_chunks, "Table 10 with parenthesized values should exist"

    print("\n✅ ALL CHECKS PASSED — PDF tables are chunked cleanly!")


if __name__ == "__main__":
    asyncio.run(main())
