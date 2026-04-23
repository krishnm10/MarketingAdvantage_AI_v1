---
name: ingestion-optimizer
description: Analyze and optimize RAG and document ingestion pipelines end-to-end, from document loading and chunking through embeddings and storage, with a focus on semantic coherence, duplication, metadata quality, embedding compatibility, and scalability. Use when designing, auditing, or debugging ingestion and indexing flows for vector databases or RAG systems.
---

# Ingestion Optimizer

This skill guides the agent to analyze and optimize the **ingestion pipeline** for RAG and vector-search systems:

`Document loading → chunking → embedding → storage/indexing`

It focuses exclusively on **ingestion-time concerns**, not query-time retrieval or LLM prompting.

The agent should always prioritize:
- **Correctness and semantic coverage**
- **Cost and token efficiency**
- **Scalability and resilience**

---

## When to use this skill

Use this skill when:
- The user asks to **design, review, or optimize an ingestion / indexing pipeline** for a RAG or vector-search system.
- The user mentions **chunking, embeddings, ingestion cost, or duplicate chunks**.
- You see code or configs for **loaders, text splitters, embedding jobs, or ingestion workers**.
- The user reports **high costs, slow ingestion, or inconsistent / missing results from specific documents**.

If the user is asking about **end-to-end RAG behavior including retrieval and LLM prompting**, combine this skill with `rag-pipeline-auditor`. This skill should focus only on **document loading → chunking → embedding → storage**.

---

## Scope

This skill covers:
- **Document loading**: How raw data is read from sources (files, APIs, DBs, message queues).
- **Chunking**: How content is split into chunks for embedding (strategy, size, overlap, structure-awareness).
- **Embedding**: How vectors are created (model, dimensions, batching, caching).
- **Storage & indexing**: How chunks and metadata are persisted in the vector DB and any side stores.

It does **not** cover:
- Query-time retrieval / reranking / LLM prompting (see `rag-pipeline-auditor`).
- UI-level behavior or downstream application logic.

---

## Required inputs

Before running an ingestion audit, try to gather or infer:

- **Data & documents**
  - Document types: PDFs, HTML, markdown, code, tickets, DB rows, logs, etc.
  - Typical document length distributions (short notes vs long manuals).
  - Example documents that must be well-covered (SLOs, runbooks, policies, product docs).
- **Ingestion implementation**
  - Loaders and parsers used (libraries, custom code, ETL jobs).
  - Chunking code or configuration (splitter class, chunk size, overlap, rules).
  - Embedding model and configuration (provider, name, dimension, normalization).
  - Vector DB / index configuration: collection name, vector field, metric, index type, metadata fields.
  - Any supplementary stores (object storage for raw docs, relational DB for metadata).
- **Operational characteristics**
  - Ingestion patterns: one-off batch, periodic batches, or streaming.
  - Throughput goals (docs/hour, tokens/sec) and current bottlenecks.
  - Cost statistics if available (embedding calls, vector DB storage cost).
- **Observability**
  - Logs/metrics on ingestion failures, retries, or dropped documents.
  - Sample of stored chunks and their metadata for inspection.

When repository code is available, **inspect it directly** rather than relying only on user prose.

---

## End-to-end analysis workflow

When applying this skill, follow this workflow:

1. **Map the ingestion pipeline**
   - Draw a concise map: `Source(s) → Loader/Parser → Chunker → Embedder → Vector DB / Storage`.
   - Note the tools, key parameters, and any missing or unclear pieces for each stage.
2. **Analyze chunking and overlap (Sections 1–2)**
   - Check semantic coherence, structure-awareness, chunk size, and overlap strategy.
3. **Analyze duplication and redundancy (Section 3)**
   - Detect duplicate or near-duplicate chunks and redundant embeddings.
4. **Review metadata quality (Section 4)**
   - Validate that metadata supports filtering, debugging, and tenant isolation.
5. **Validate embedding compatibility (Section 5)**
   - Ensure a single, consistent embedding space and correct pairing with the vector DB.
6. **Classify ingestion failures (Section 6)**
   - Map issues to chunking, coverage, redundancy, or metadata failures.
7. **Evaluate cost and efficiency (Section 7)**
   - Analyze chunk counts, embedding call volume, and storage usage.
8. **Evaluate infrastructure and scaling (Section 8)**
   - Assess batching, parallelism, worker design, and separation from the online API.
9. **Check hard failure conditions**
   - If any hard failures apply, mark the system as **"Ingestion Pipeline Unsafe"** and make that explicit.
10. **Produce the final structured report**
   - Use the **Output format** section at the end of this file.

Always make **reasonable assumptions** when details are missing, and clearly label them as assumptions. Prefer **concrete config/code-level suggestions** over vague advice.

---

## Section 1 – Chunking strategy

Goal: Ensure chunking is **semantically coherent**, **task-aligned**, and **not naively fixed-size** for complex data.

Checklist:
- **Chunking approach**
  - Identify whether chunking is: naive fixed-size, recursive/paragraph-based, semantic, or structure-aware (e.g., heading/section-based, code-block-aware).
  - For complex documents (long-form docs, PDFs, legal, policies, multi-page runbooks, code), treat **naive fixed-size chunking without structure** as a **weak design**, and call it out explicitly.
- **Chunk size**
  - Determine the target chunk size in tokens or characters.
  - Check that the chunk size is aligned with:
    - Downstream model context window.
    - Typical document structure (sections, paragraphs, code blocks).
    - Target tasks (Q&A, search, summarization).
- **Boundary handling**
  - Check whether chunking respects **paragraphs, headings, bullet lists, tables, and code blocks**.
  - Detect sentence/section boundary breaks where meaning is split across chunks.
- **Per-document vs global strategy**
  - Verify whether different document types can use different chunking strategies (e.g., FAQ entries vs 100-page manuals).
  - Flag a single global strategy used for all data types as a potential issue when the corpus is heterogeneous.

What to flag:
- **Naive fixed-size chunking** on complex or highly structured documents.
- Chunk sizes that are **too small** (excessive fragmentation, loss of context) or **too large** (risk of truncation and under-utilization of recall).
- Chunking that ignores obvious structure (headings, code blocks, bullet lists, table boundaries).

Recommend:
- For complex documents, prefer **recursive, structure-aware, or semantic chunking** with a target range (e.g., 700–1 500 tokens) rather than a single hard window.
- Tune chunking per major document class when possible.

---

## Section 2 – Overlap & context preservation

Goal: Ensure important context is preserved across chunks without unnecessary duplication.

Checklist:
- **Overlap configuration**
  - Identify the overlap amount (in tokens/characters or sentences).
  - Assess whether overlap is **too low** (context loss at boundaries) or **too high** (explosive duplication).
- **Boundary sensitivity**
  - Verify that overlap is applied around **section transitions, headings, and paragraph boundaries**, not just blind token offsets.
  - For tables or code blocks, confirm that logically related content is kept together in at least one chunk.
- **Task alignment**
  - Check that overlap is sufficient for typical question scopes (e.g., cross-paragraph reasoning, section intros + bullets).

Heuristics:
- For long-form text, overlaps of roughly **10–30% of the target chunk size** are often appropriate.
- Short FAQ-like documents may require **no or minimal overlap**.

What to flag:
- **No overlap** or extremely small overlap on long-form documents where users ask questions that span multiple paragraphs or sections.
- Overlap so high that it creates many nearly-identical chunks and excessive embedding/storage cost.

Recommend:
- Adopt a **document-type-specific** overlap strategy, with defaults tuned per corpus.
- Use **content-aware overlap** (e.g., overlap by heading/paragraph counts) where feasible, not only raw token counts.

---

## Section 3 – Duplicate & redundant embeddings

Goal: Detect and reduce **duplicate chunks** and **near-duplicate embeddings** that waste cost and storage.

Checklist:
- **Duplicate text detection**
  - Sample stored chunks and check for exact duplicate `text` fields.
  - Verify whether boilerplate sections (headers, footers, navigation, legal disclaimers) are repeatedly embedded.
- **Near-duplicate detection**
  - Look for chunks with very minor differences (e.g., page numbers, timestamps) that are otherwise identical.
  - If you can compute similarities, check for embeddings above a high similarity threshold within the same document or corpus.
- **Deduplication strategy**
  - Determine whether ingestion applies **content hashing, fingerprinting, or dedupe rules** before embedding.
  - Check if identical documents or versions are re-embedded instead of reusing existing vectors.

What to flag:
- Large fractions of chunks that are **identical or almost identical**.
- Boilerplate content embedded for every page or document without deduplication.
- No content hashing or deduplication when documents are frequently updated or re-ingested.

Recommend:
- Introduce a **content hash (e.g., normalized text hash)** to avoid re-embedding identical content.
- Deduplicate chunks within and across documents before embedding.
- Cache embeddings for repeated strings or templates when they cannot be avoided.

---

## Section 4 – Metadata quality

Goal: Ensure metadata is rich and consistent enough to support **filtering, hybrid retrieval, debugging, and tenant isolation**.

Checklist:
- **Core identifiers**
  - Each chunk should have a stable **document ID**, **chunk ID**, and **source identifier** (e.g., file path, URL, record key).
  - Verify **document ID consistency** across versions and re-ingestions (or explicit version fields when docs change).
- **Structural metadata**
  - Capture section or heading titles, page numbers, paragraph indices, or code module/function names where relevant.
  - For logs or time-series, include timestamps and event types.
- **Operational metadata**
  - Include ingestion timestamps, pipeline version, and embedding model version.
  - Ensure tenant/environment fields (e.g., `tenant_id`, `environment`) exist where multi-tenant or multi-environment.
- **Search & filtering support**
  - Verify that metadata fields align with likely filter dimensions (tenant, product, language, region, time range, document type).
  - Confirm metadata types are appropriate (e.g., enums/ints instead of arbitrary free-form tags where possible).

What to flag:
- Chunks lacking **document IDs** or clear source references.
- Missing metadata needed for **filtering**, **hybrid search features**, or **tenant isolation**.
- Inconsistent or free-form metadata values that make filtering brittle.

Recommend:
- Standardize a **metadata schema** for all chunks (doc id, source, type, section, timestamps, tenant, language, embedding model, etc.).
- Store enough metadata to reconstruct **which chunks came from which documents and which ingestion run**.

---

## Section 5 – Embedding compatibility

Goal: Ensure that embeddings are **consistent, compatible with the vector DB**, and do not mix multiple embedding spaces.

Checklist:
- **Single embedding model usage**
  - Verify that a **single embedding model** (or clearly separated collections per model) is used for a given index.
  - Confirm that there is **no mixing of embeddings from different models** in the same collection or index.
- **Vector dimension consistency**
  - Confirm that all stored vectors share the same dimensionality and match the embedding model’s output dimension.
  - Ensure the vector DB schema (e.g., column types, dimension) matches the embedding model.
- **Distance metric compatibility**
  - Check that the chosen similarity metric (cosine, dot product, L2) is **appropriate for the embedding model** and is consistently used in indexing and query-time retrieval.
  - Verify any required **normalization** (e.g., unit vectors for cosine) is applied at ingestion when the DB expects it.
- **Embedding lifecycle**
  - Verify there is a strategy to **re-embed and re-index** all affected data when the embedding model changes.
  - Check that embedding model and version are captured in metadata.

Critical condition:
- **Mixed embeddings in the same index (different models, dimensions, or preprocessing)** must be treated as a **critical ingestion failure**.

Recommend:
- Enforce **one embedding model per collection/index**, with clear metadata markers.
- When changing models, plan a **full re-embedding and re-index** rather than incremental mixing.

---

## Section 6 – Ingestion failure detection & classification

Goal: Classify ingestion issues into clear failure types and identify root causes.

Failure types:
- **Chunking failure (semantic loss)**:
  - Important logical units (sections, procedures, API definitions) are split or scattered such that no chunk contains enough context.
- **Coverage failure (important data missing)**:
  - Important documents, sections, or versions are not ingested at all or only partially ingested.
- **Redundancy failure (duplicate chunks)**:
  - Excessive duplication of content leading to high cost and noisy retrieval.
- **Metadata failure (poor filtering capability)**:
  - Inadequate metadata prevents targeting the right chunks or understanding where results came from.

Checklist:
- Trace a few **critical documents** end-to-end:
  - Confirm they are loaded, chunked, embedded, and stored as expected.
  - Inspect whether all key sections are present in at least one chunk with sufficient context.
- Compare **source-of-truth counts** (documents, records, pages) to stored chunks:
  - Look for large discrepancies indicating missing or over-generated chunks.
- Review logs/metrics:
  - Look for ingestion errors, retries, or dropped records.

For each detected issue, label it with the appropriate **failure type(s)** above. This classification must appear in the final report.

---

## Section 7 – Cost & efficiency

Goal: Optimize ingestion for **cost, throughput, and storage efficiency** without sacrificing accuracy.

Checklist:
- **Chunk count and density**
  - Compute or estimate average chunks per document for each major document type.
  - Detect **over-chunking** (far more chunks than necessary) or **under-chunking** (huge chunks that limit recall).
- **Embedding call volume**
  - Estimate embedding calls per document and per ingestion run.
  - Identify opportunities for **batch embedding** and **request batching** to the embedding provider.
- **Redundant embeddings**
  - Combine this with Section 3 to quantify how many chunks are duplicates or near-duplicates.
- **Storage usage**
  - Estimate per-document and total storage usage in the vector DB and any backup stores.

What to flag:
- Overly small chunk sizes creating many low-value chunks.
- Lack of batching (one network call per chunk).
- Re-embedding the same documents or content on each ingestion run without change detection.

Recommend:
- Use **batch embedding APIs** and tune batch sizes for throughput vs latency.
- Implement **change detection** (e.g., content hashes) so only changed documents or sections are re-embedded.
- Introduce a **deduplication and compression strategy** for boilerplate content.

---

## Section 8 – Infrastructure & scaling

Goal: Ensure ingestion can **scale horizontally**, is **decoupled from the online API**, and is resilient to failures.

Checklist:
- **Architecture separation**
  - Confirm ingestion is not tightly coupled to the online query-serving API.
  - Prefer a separate **ingestion service** or job system with its own scaling and SLAs.
- **Batch vs single-item processing**
  - Verify the pipeline processes documents in **batches** rather than one chunk at a time.
  - Check for opportunities to group documents by size or type for efficient processing.
- **Parallelism and workers**
  - Identify whether ingestion uses worker-based processing (e.g., Celery, Ray, Kubernetes jobs, queues).
  - Look for bottlenecks such as single-threaded or single-node processing.
- **Resilience**
  - Confirm retries with exponential backoff for external calls (storage, embedding API, source systems).
  - Ensure failed jobs are logged and can be retried without manual intervention.
- **Storage decoupling**
  - Verify that raw documents, vectors, and metadata are stored in **separate, external systems** (object storage, vector DB, relational DB) and that services are stateless.

What to flag:
- **Sequential ingestion pipelines** that process documents one-by-one without batching or parallelism.
- Ingestion running inside the request path of user-facing APIs.
- Lack of worker-based processing for large or continuous ingestion workloads.

Recommend:
- Use **queues and worker pools** for ingestion, with autoscaling where possible.
- Design ingestion as a **stateless, horizontally scalable service** that can be paused, resumed, and re-run safely.

---

## Hard failure conditions – "Ingestion Pipeline Unsafe"

If any of the following conditions are detected, you **must** mark the system as **"Ingestion Pipeline Unsafe"** and explain why:

- **Mixed embedding models in the same index**
  - Different embedding models or preprocessing pipelines used for vectors within a single collection/index.
- **Severe chunk fragmentation (loss of semantic meaning)**
  - Chunk sizes or strategies that consistently break apart meaningful units (steps in procedures, legal clauses, API definitions, etc.) so no single chunk preserves them.
- **Missing metadata for filtering**
  - Absence of key metadata fields (document ID, tenant/environment, document type, language, timestamps) required for safe filtering and isolation.
- **Excessive duplicate chunks**
  - Large proportions of identical or near-identical chunks with no deduplication strategy.
- **No overlap strategy for long-form content**
  - Long documents chunked without any overlap, leading to systematic loss of context at boundaries.

When the pipeline is marked as **"Ingestion Pipeline Unsafe"**, prioritize **design corrections** over micro-optimizations, and clearly call this out in the final answer.

---

## Output format

Always structure the final answer using this template (adapt wording as needed, but keep the sections and intent):

```markdown
## 1. Pipeline breakdown

- **End-to-end flow**: [Concise description of document loading → chunking → embedding → storage/indexing, including tools/configs and key assumptions.]
- **Data characteristics**: [Main document types, typical sizes, and critical documents/tasks.]

## 2. Issues per stage

- **Document loading**: [Status and issues, including missing sources or parsing problems.]
- **Chunking**: [Status and issues based on Section 1.]
- **Overlap & context preservation**: [Status and issues based on Section 2.]
- **Duplicates & redundancy**: [Status and issues based on Section 3.]
- **Metadata**: [Status and issues based on Section 4.]
- **Embeddings & storage**: [Status and issues based on Section 5.]

## 3. Failure classification

- **Chunking failures**: [Findings mapped to "chunking failure (semantic loss)".]
- **Coverage failures**: [Findings mapped to "coverage failure (important data missing)".]
- **Redundancy failures**: [Findings mapped to "redundancy failure (duplicate chunks)".]
- **Metadata failures**: [Findings mapped to "metadata failure (poor filtering capability)".]

## 4. Optimal chunking strategy

- **Recommended approach**: [Semantic/recursive/structure-aware chunking description.]
- **Chunk size & overlap**: [Target size range and overlap % per main document type.]
- **Special handling**: [Tables, code, FAQs, logs, etc.]

## 5. Cost reduction plan

- **Embedding cost optimizations**: [Batching, deduplication, change detection, model choices.]
- **Storage optimizations**: [Deduplication, compression, pruning obsolete or low-value chunks.]

## 6. Accuracy improvement plan

- **Coverage fixes**: [How to ensure critical docs/sections are fully and correctly ingested.]
- **Quality fixes**: [Chunking and metadata improvements that directly improve retrieval quality.]

## 7. Scaling recommendations

- **Ingestion architecture**: [Proposed worker/queue design, separation from online API, externalized storage.]
- **Throughput & resilience**: [Parallelism, batching, retries, observability for ingestion.]

## 8. Ingestion safety status

- **Status**: [Safe / Needs work / Ingestion Pipeline Unsafe]
- **Reasons**: [Concrete justification, citing any hard failure conditions.]
```

Always make the **"Ingestion safety status"** and any **hard failure conditions** highly visible in the final answer so they can be acted on before production deployment.

