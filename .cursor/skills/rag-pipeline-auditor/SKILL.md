---
name: rag-pipeline-auditor
description: Analyze retrieval-augmented generation (RAG) pipelines end-to-end, auditing chunking, embeddings, vector database schema/indexing, retrieval (including hybrid search), reranking, and LLM grounding. Use when designing, reviewing, or debugging RAG systems, or when the user asks to evaluate or improve RAG retrieval quality, accuracy, or architecture.
---

# RAG Pipeline Auditor

This skill guides the agent to audit a RAG (retrieval-augmented generation) pipeline end-to-end and propose concrete, stage-specific fixes.

The agent should use this skill when:
- The user asks to **analyze, audit, or debug a RAG pipeline**
- The user reports **retrieval quality issues, hallucinations, or low answer accuracy**
- The user wants to **improve chunking, embeddings, vector DB, retrieval, reranking, or LLM prompting** in a RAG system

Always aim for **production-grade**, **cost-aware**, and **evaluated** designs.

---

## Required Inputs

Before auditing, try to gather or infer:

- **High-level pipeline description**
  - Data sources and document types (PDF, HTML, code, tickets, logs, DB rows, etc.)
  - Main user tasks (Q&A, chat, code assist, summarization, agent tools, etc.)
- **Implementation artifacts (as available)**
  - Chunking code/config (ingestion scripts, loaders, text splitters)
  - Embedding model + parameters (provider, model name, dim, pooling, normalization)
  - Vector DB schema and index config (collection name, fields, metric_type, index params)
  - Retrieval code/config (top-k, score thresholds, filters, hybrid search settings)
  - Reranker model/config (cross-encoder or LLM-based reranker, k in/k out)
  - LLM prompt templates and system messages
- **Signals / metrics** (if available)
  - Example bad outputs (hallucinations, irrelevant answers)
  - Any offline eval results, feedback logs, or telemetry

If code or config is in a repo, **inspect it directly** (using search/semantic tools) rather than relying only on user prose.

---

## Overall Analysis Workflow

When applying this skill, follow these steps:

1. **Map the pipeline**
   - Draw a concise mental map: `Input → Chunking → Embedding → Vector DB → Retrieval → Reranking → LLM`
   - Note for each stage: tech stack, parameters, and any missing pieces.
2. **Run the six core checks (Stages 1–6 below)**
3. **Classify failures by type**
   - Retrieval failure, ranking failure, context failure, generation failure (see below).
4. **Propose exact, stage-specific fixes**
   - For each stage, produce concrete config/code-level changes, not vague advice.
5. **Highlight evaluation and observability gaps**
   - Recommend minimal evaluation and tracing setup if missing.

Always produce the **final answer** using the output format defined in the "Output Format" section.

---

## Failure Classification (Use Throughout)

Consistently map issues to these categories:

- **Retrieval failure**: Relevant documents are not retrieved at all (recall problem).
- **Ranking failure**: Relevant docs are present but ranked too low or overshadowed by noisy results.
- **Context failure**: Relevant content exists but is truncated, drowned in noise, or placed where the model under-attends (lost-in-the-middle).
- **Generation failure**: Context was sufficient, but the LLM still hallucinates or misinterprets (prompting, constraints, or model choice issue).

Always identify **which stage is the root cause** before proposing fixes.

---

## Stage 1 – Check Chunking Strategy

Goals:
- Ensure chunks are **semantically coherent**, **task-aligned**, and **not naively fixed-size** for complex data.

Checklist:
- **Chunking approach**
  - Identify whether chunking is: naive fixed-size, recursive/paragraph-based, semantic, or structure-aware (e.g., section/heading-based).
  - For complex documents (long-form docs, PDFs, code, legal, policies), **flag naive fixed-size chunking as a flawed design**.
- **Chunk size and overlap**
  - Verify target token window and overlap are appropriate for the task and model context window.
  - Ensure tables, code blocks, and sections are **not split across chunks** without overlap.
- **Structure awareness**
  - Prefer chunking that respects **headings, sections, bullets, and code blocks**.
  - For multi-doc corpora (e.g., FAQs, tickets), consider preserving per-document boundaries.
- **Metadata**
  - Ensure each chunk carries enough metadata (source, section, doc id, timestamps, tenant, language, etc.) to support filtering and debugging.

Common issues to flag:
- Single global chunk size (e.g., 512 tokens) used for all data types.
- No or minimal overlap causing answers to miss transitions across sections.
- Chunking run only once and never rethought after changing LLM, tasks, or embeddings.

---

## Stage 2 – Validate Embedding Compatibility

Goals:
- Ensure embeddings are **compatible with data, tasks, and vector DB**, and that no mixed embedding spaces exist.

Checklist:
- **Model and domain fit**
  - Check that the embedding model supports the languages present.
  - Prefer domain-tuned or instruction-style embeddings for complex tasks if available.
- **Dimensionality and metric**
  - Confirm dimensionality matches the vector DB schema and index.
  - Verify similarity metric (cosine, dot, L2) is appropriate and **consistent** across ingestion and retrieval.
- **Pooling and normalization**
  - Check whether embeddings are properly pooled (e.g., CLS vs mean) per model recommendations.
  - If cosine similarity is used, confirm vectors are normalized when required by the DB/index.
- **Embedding lifecycle**
  - Ensure there is **no mixing of embeddings from different models** in the same collection; if detected, treat as a **critical error**.
  - Verify there is a strategy for re-embedding and re-indexing when the embedding model changes.

Common issues to flag:
- Embedding model changed without full re-index.
- Metric type in the index does not match embedding assumptions.
- Mixed languages but mono-lingual embeddings.

---

## Stage 3 – Analyze Vector DB Schema & Indexing

Goals:
- Ensure the vector store is configured for **correctness, performance, and filtering**.

Checklist:
- **Collection / table design**
  - Validate schema: vector field, metadata fields, primary key, timestamps, tenant / namespace fields.
  - Check that metadata types are appropriate (e.g., enums/ints instead of free-form strings when possible).
- **Index configuration**
  - Verify index type, metric, and parameters (e.g., HNSW, IVF, PQ) are appropriate for dataset size and latency targets.
  - Ensure recall/latency trade-offs are reasonable and configurable.
- **Sharding and isolation**
  - Check for tenant or project isolation (per-tenant collections / partitions / filters).
  - Ensure no cross-tenant leakage in queries.
- **Data quality**
  - Confirm that all chunks are embedded and indexed, with no silent failures.
  - Validate there is a mechanism for deletes/updates (soft deletes, upserts, tombstones).

Common issues to flag:
- Missing or incorrect metric type.
- No metadata filters even though multi-tenant, multi-project, or multi-language.
- Large collection without an approximate index, leading to high latency.

---

## Stage 4 – Evaluate Retrieval (top-k, hybrid)

Goals:
- Ensure retrieval delivers **high-recall, relevant candidates** and leverages **hybrid search by default**.

Checklist:
- **Top-k and thresholds**
  - Inspect top-k values for initial retrieval and any post-filtering.
  - Tune k based on task complexity (often in the range 5–20); too low harms recall, too high inflates cost and noise.
- **Hybrid search**
  - Enforce hybrid search (dense + sparse/BM25 or keyword) by default for text corpora.
  - Check that sparse and dense scores are combined or reranked coherently.
- **Filtering and routing**
  - Validate metadata filters (tenant, product, time range, language, environment).
  - Ensure filters are **actually applied** in queries (no placeholders or TODOs).
- **Query transformation**
  - Inspect query rewriting (e.g., query expansion, synonym handling, multi-step queries).
  - Ensure transformations do not over-broaden queries into noise.

Common issues to flag:
- Pure dense retrieval without BM25/keyword where exact-term matching is critical.
- Fixed top-k (e.g., 3) for all tasks, leading to missed relevant docs.
- Overly broad filters or missing tenant filters.

---

## Stage 5 – Check Reranking Effectiveness

Goals:
- Ensure a **reranking stage** exists and is tuned to maximize relevance.

Checklist:
- **Presence and placement**
  - Verify there is a reranking step after initial retrieval; if not, **flag this as a design gap** for most non-trivial systems.
- **Model and configuration**
  - Identify reranker type: cross-encoder, bi-encoder with second pass, or LLM-based scoring.
  - Inspect how many candidates are reranked (e.g., top-50 → top-10) and whether this is adequate.
- **Scoring and cutoffs**
  - Confirm scoring direction and thresholds are correct (higher is better vs lower is better).
  - Check that low-scoring candidates can be dropped to reduce context noise.

Common issues to flag:
- No reranker at all.
- Reranking too few or too many candidates.
- Misinterpreted scores or thresholds.

---

## Stage 6 – Analyze LLM Grounding & Prompting

Goals:
- Ensure the LLM is **properly grounded in retrieved context**, with prompts designed to minimize hallucinations and handle context limits.

Checklist:
- **Prompt structure**
  - Inspect system and user prompts: are instructions explicit about using only provided context, citing sources, and saying "I don't know" when necessary?
  - Verify prompt includes clear formatting for citations (e.g., reference IDs/URLs).
- **Context construction**
  - Check how chunks are serialized into the final prompt: order, formatting, separators, and maximum tokens.
  - Detect potential **lost-in-the-middle** issues when many chunks are concatenated.
- **Model choice and parameters**
  - Verify model is suitable for context length and reasoning complexity.
  - Inspect temperature, top_p, and other decoding parameters for faithfulness vs creativity trade-offs.
- **Guardrails and constraints**
  - Look for mechanisms like tools, validators, or answer policies that reduce hallucinations (e.g., retrieval-aware prompting, self-checks).

Common issues to flag:
- LLM prompt does not instruct the model to rely on context or admit uncertainty.
- Context truncation without awareness (longest docs or most relevant chunks cut off).
- No explicit citation or grounding requirement.

---

## Evaluation, Observability, and Cost

In addition to the six stages, always comment on:

- **Evaluation**
  - Whether there is a **golden dataset** or synthetic eval set.
  - Metrics in use: faithfulness, answer relevance, precision/recall, MRR, etc.
  - Presence of regression tests for pipeline changes.
- **Observability**
  - Availability of structured logs, traces, and per-request IDs.
  - Ability to inspect: query → retrieved docs → final answer for any request.
- **Cost & performance**
  - Opportunities to reduce token usage (smaller models, better top-k, deduplicated context).
  - Caching for embeddings, retrieval results, and prompts where appropriate.

If these are missing, treat them as **important, but fix after correctness**.

---


## Minimum Viable Production RAG Standard

A production-ready system must include:

- Semantic or structured chunking (no naive fixed-size chunking for complex data)
- Compatible embedding model and vector DB metric
- Hybrid retrieval (dense + sparse)
- Reranking stage
- Grounded LLM prompting with context control
- Basic evaluation framework (metrics + dataset)

If any of the above is missing:
- Flag as incomplete system design




## Output Format

Always structure the final answer using this template (adapt section names as needed, but keep the hierarchy):

```markdown
## Stage-by-stage breakdown

- **Stage 1 – Chunking**: [Status: OK / Needs attention / Broken] – Short summary of findings.
- **Stage 2 – Embeddings**: [Status: ...] – Short summary.
- **Stage 3 – Vector DB & Indexing**: [Status: ...] – Short summary.
- **Stage 4 – Retrieval (top-k, hybrid)**: [Status: ...] – Short summary.
- **Stage 5 – Reranking**: [Status: ...] – Short summary.
- **Stage 6 – LLM Grounding & Prompting**: [Status: ...] – Short summary.

## Critical issues

- **[Stage] – [Short title]**: One–two sentence description, including why it is critical.
- ...

## Accuracy risks

- **[Failure type: Retrieval / Ranking / Context / Generation]**: Description of how and where this occurs in the current pipeline.
- ...

## Recommended fixes by stage

- **Stage 1 – Chunking**:
  - [Exact fix 1: e.g., "Replace fixed-size splitter with recursive text splitter using headings, target 800–1200 tokens with 15–20% overlap."]
  - [Exact fix 2...]
- **Stage 2 – Embeddings**:
  - [Exact fix...]
- **Stage 3 – Vector DB & Indexing**:
  - [Exact fix...]
- **Stage 4 – Retrieval (top-k, hybrid)**:
  - [Exact fix...]
- **Stage 5 – Reranking**:
  - [Exact fix...]
- **Stage 6 – LLM Grounding & Prompting**:
  - [Exact fix...]

## Optional: Evaluation & observability plan

- **Evaluation**: [Concrete proposal for a minimal but meaningful eval setup.]
- **Observability**: [Concrete proposal for logging/tracing to validate future changes.]
```

## Infrastructure & Scaling Considerations

Consider:

- Separation of ingestion and retrieval services
- Horizontal scalability of retrieval APIs
- Distributed ingestion (batch embedding, workers)
- Vector DB scaling limits and indexing strategy

Flag:
- Single-node bottlenecks
- Tight coupling between pipeline stages

When details are unknown (e.g., embedding model not specified), **call this out explicitly**, make reasonable assumptions, and provide **clear next questions or checks** the user should perform.

