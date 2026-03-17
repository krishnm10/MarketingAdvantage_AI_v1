# Deduplication Governance Guide

This file explains the full L1, L2, and L3 deduplication model in one place so business users, admins, developers, and reviewers can understand how duplicate control works during ingestion.

## Purpose

The deduplication process protects the ingestion pipeline from storing repeated or low-value content multiple times.

It serves four governance goals:

1. Prevent exact repeated content from being stored again in the same file.
2. Prevent already-known content from being re-added when it already exists in shared content history.
3. Flag possible semantic duplicates that need review when the system cannot prove the match strongly enough from current payload evidence.
4. Make duplicate decisions visible in the admin UI so users can understand why content was kept, rejected, or marked for review.

## Deduplication Layers

### L1: Exact Duplicate Inside The Same File

L1 checks for exact repeated semantic hashes inside the current ingestion file.

- Meaning:
  The same normalized content appears again in the same uploaded file.
- Match basis:
  Same semantic hash after cleaning and normalization.
- Governance action:
  Keep one usable version and mark the repeated copies as duplicate.
- Best for:
  Repeated rows, copied paragraphs, duplicate records, repeated cells, or duplicated sections inside one file.

Example:

- File name: `customer_data.xlsx`
- Chunk 3 and Chunk 8 have the same normalized text
- Result:
  Chunk 3 stays as the anchor
  Chunk 8 is marked as L1 duplicate

### L2: Cross-File Duplicate Through Global Content Index

L2 checks whether a duplicate row is already linked to previously committed content through Global Content Index evidence.

- Meaning:
  The current row matches content already recognized in the broader system.
- Match basis:
  Global Content Index reference and occurrence evidence returned by the backend.
- Governance action:
  Mark as duplicate because the content already exists in system history.
- Best for:
  Re-ingested documents, repeated content across uploads, known rows that were already committed earlier.

Example:

- File name: `news_feed.json`
- Chunk 11 matches existing GCI content
- Result:
  Chunk 11 is marked as L2 duplicate
  UI shows GCI reference evidence

### L3: Semantic Review Candidate

L3 is the review lane for duplicates that are still considered duplicate by the backend, but are not fully explained by L1 exact repeat evidence or visible L2 GCI evidence in the current payload.

- Meaning:
  The row appears duplicate, but the current payload does not expose enough exact target evidence for direct proof in the UI.
- Match basis:
  Current duplicate flag plus remaining semantic review classification.
- Governance action:
  Hold this as analyst-reviewable duplicate evidence.
- Best for:
  Near-duplicate meaning, paraphrased content, similar sentences, incomplete evidence cases.

Example:

- File name: `web_scrape.txt`
- Chunk 21 is flagged duplicate
- No exact repeat group is visible
- No GCI link is exposed in current payload
- Result:
  Chunk 21 appears in L3 review

## How Governance Works During Ingestion

When a file is ingested, deduplication should be understood as a staged control process:

1. Content is parsed into chunks or rows.
2. Cleaned and normalized text is prepared.
3. Duplicate checks are applied.
4. One of the following outcomes is assigned:
   - Unique
   - L1 duplicate
   - L2 duplicate
   - L3 review duplicate
5. Storage and downstream retrieval use only the dedup-approved content according to backend rules.

This means governance is not only about detection. It is also about controlled acceptance, rejection, and review of incoming content.

## Governance Decision Table

| Layer | What It Means | Evidence Source | Governance Decision | UI Expectation |
|---|---|---|---|---|
| Unique | Content passed dedup | Current file payload | Keep for normal use | Green unique row |
| L1 | Exact repeat in same file | Repeated semantic hash | Keep one, mark rest duplicate | Amber highlight |
| L2 | Already known in prior content | Global Content Index evidence | Mark duplicate | Blue highlight |
| L3 | Duplicate needs analyst interpretation | Duplicate flag without visible L1/L2 proof | Review / governed duplicate handling | Rose highlight |

## What The UI Should Show

For better governance and layman understanding, the ingestion dedup detail page should show:

1. Ingestion file name
2. Duplicate lane: L1, L2, or L3
3. Chunk number or row number
4. Matched against
5. Matching semantics
6. Duplicate value or chunk text
7. Unique values separately for comparison

This is why the updated UI now includes:

- a dedup match review table
- lane summary cards for L1, L2, and L3
- duplicate highlighting by color
- unique rows shown separately

## Meaning Of “Matched Against”

The phrase “matched against” depends on the layer:

- L1:
  Matched against content in the same ingestion file
- L2:
  Matched against Global Content Index evidence from previously known content
- L3:
  Matched against semantic duplicate review logic from the current backend response

Important limitation:

The current backend payload does not always expose the external matched file name for L2 and L3. Because of that, the UI can only show the best available source label unless the API is expanded later.

## Meaning Of “Matching Semantics”

“Matching semantics” is the plain-language explanation of why the row is considered duplicate.

Examples:

- L1:
  Exact semantic-hash repeat
- L2:
  Cross-file match through Global Content Index
- L3:
  Semantic review candidate based on remaining duplicate evidence

This wording is useful for governance because it explains the reason for the duplicate decision, not just the status.

## Recommended Operational Governance

For practical governance during ingestion:

1. Review L1 first because it is usually the clearest and lowest-risk duplicate category.
2. Review L2 next because it reflects prior known content and helps prevent re-ingestion of committed content.
3. Review L3 carefully because it is the most interpretation-heavy lane.
4. Track duplicate ratios by file to spot suspicious uploads.
5. Treat high L3 volumes as a signal for business review, data quality review, or future backend enrichment.

## Recommended Business Interpretation

- High L1:
  The file itself is repetitive or badly prepared.
- High L2:
  The organization is re-ingesting content already known to the platform.
- High L3:
  The file may contain semantically overlapping content that needs closer human review.
- High Unique with low duplicates:
  The file is mostly clean for ingestion.

## Current UI Scope

The current UI work is intentionally frontend-only.

That means:

- Backend logic is not changed
- Duplicate classification still comes from existing backend behavior
- UI only improves visibility, readability, and governance understanding

## Future Enhancement Options

If the team wants deeper governance later, backend enhancements could expose:

1. Matched external file name for L2
2. Matched target chunk for L3
3. Similarity score for semantic duplicate review
4. Anchor chunk ID for L1 groups
5. Reviewer decision status for L3

These are future API improvements, not part of the current UI-only scope.

## Summary

L1, L2, and L3 should be understood as three governance lanes for duplicate handling during ingestion:

- L1 governs exact repeats inside the current file
- L2 governs duplicates already known to the wider platform
- L3 governs duplicates that still need semantic review context

The best operational model is:

- keep unique content
- suppress exact repeats
- suppress already-known cross-file duplicates
- surface semantic-review duplicates clearly for controlled decision-making

