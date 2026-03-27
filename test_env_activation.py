"""
Quick live test for structure_aware strategy activated via CHUNKING_STRATEGY env var.
Run with:  python test_env_activation.py
"""
import os
import asyncio

os.environ.setdefault("CHUNKING_STRATEGY", "structure_aware")

from app.core.chunking_stratagies.chunking_registry import get_chunker  # noqa: E402

INVOICE_DOC = """
# Invoice Processing Guide

## Overview

This guide covers the automated invoice processing pipeline.
The system validates, deduplicates, and routes invoices for approval
based on configurable business rules and spending thresholds.

## Steps

1. Receive invoice via API or email attachment
2. Validate line items against purchase orders
3. Flag mismatches for manual review
4. Auto-approve invoices under the threshold amount
5. Route escalated invoices to the appropriate budget owner

## Thresholds

| Category   | Auto-Approve | Manual Review | Escalate   |
|------------|-------------|---------------|------------|
| Supplies   | < $500      | $500–$5,000   | > $5,000   |
| Services   | < $2,000    | $2k–$15k      | > $15,000  |
| Capital    | Never       | Always        | > $100,000 |

## Compliance Notes

All invoices must retain a 7-year audit trail per SOX requirements.
GDPR applies to any invoice containing personal data of EU residents.
Duplicate invoices are automatically suppressed by the deduplication engine.
"""


async def main():
    chunker = get_chunker()          # reads CHUNKING_STRATEGY from env
    print(f"Active chunker: {type(chunker).__name__}")
    print()

    chunks = await chunker.chunk(
        INVOICE_DOC,
        source_type="pdf",
        embedding_model="text-embedding-3-small",
    )

    print(f"Total chunks: {len(chunks)}")
    print("-" * 90)
    for i, ch in enumerate(chunks):
        meta = ch["reasoning_ingestion"]
        ctype   = meta.get("content_type", "?")
        section = meta.get("section_title", "")
        depth   = meta.get("section_depth", 0)
        quality = meta.get("chunk_quality_score", -1)
        tok     = ch["tokens"]
        preview = ch["text"][:100].replace("\n", " ")
        print(f"Chunk {i}: [{ctype:6s}] {'#'*depth} {section:28s} | tok={tok:4d} quality={quality:.3f}")
        print(f"         {preview}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())
