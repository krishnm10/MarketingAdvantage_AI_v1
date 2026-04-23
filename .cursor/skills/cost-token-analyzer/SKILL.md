---
name: cost-token-analyzer
description: Analyze LLM and RAG system cost and token usage across embeddings, retrieval, context construction, and generation, identifying cost drivers, redundant processing, and optimization opportunities. Use when the user asks about cost, pricing, token efficiency, or optimization of AI/RAG pipelines.
---

# Cost & Token Analyzer

This skill guides the agent to analyze **cost drivers and token usage** in LLM and RAG systems.

It focuses on:
- **Embeddings**: model choice, call patterns, batching, duplication.
- **Retrieval & vector DB**: top-k, hybrid search overhead, storage and query costs.
- **LLM calls**: prompt design, context construction, max_tokens, model selection.
- **Redundant processing & caching**: repeated work, missing caches, poor cache keys.

The analysis must always consider the full pipeline:

`Input → Tokenization → Chunking → Embedding → Vector DB → Retrieval/Reranking → LLM Reasoning`

and classify issues by **where cost is spent and how to reduce it with minimal quality loss**.

---

## When to use this skill

Use this skill when:
- The user asks to **analyze or reduce cost** of an AI or RAG system.
- The user mentions **token usage, pricing, latency, or model right-sizing**.
- You see **embedding jobs, retrieval settings, or LLM configs** and want to understand their cost impact.
- The user suspects **wasteful processing, duplicate embeddings, or oversized prompts/context**.

Combine this skill with:
- `ingestion-optimizer` when the primary concern is **ingestion-time cost** (chunk counts, embeddings, storage).
- `rag-pipeline-auditor` when the user wants **end-to-end RAG quality + cost**.
- `retrieval-debugger` when concrete **retrieval examples** are available and you need to see which chunks are being pulled into context.

This skill always keeps a **cost and token-efficiency lens** while respecting **accuracy, robustness, and security**.

---

## Scope

This skill covers:
- **Workload characterization**: volumes, traffic patterns, and usage tiers.
- **Embedding usage**: model choice, frequency, duplication, batching, caching.
- **Vector DB & retrieval costs**: write/read patterns, top-k, hybrid search overhead, storage.
- **LLM calls**: prompt templates, context assembly, model selection, decoding params, streaming.
- **Redundant processing & token waste**: repeated embeddings, repeated retrieval, unbounded history, oversized system prompts.
- **Caching strategy**: what to cache, where, keys, TTLs, invalidation, safety.
- **Unit economics & observability**: per-request and per-feature cost, tracing, and monitoring.

It does **not** focus on:
- Non-LLM infrastructure costs unrelated to the AI/RAG logic (e.g., generic web hosting) except where they are tightly coupled to LLM usage patterns.
- Detailed retrieval quality or chunking semantics beyond what is needed to understand cost — for those, lean on `rag-pipeline-auditor` and `ingestion-optimizer`.

---

## Required inputs

Before running a cost & token audit, try to gather or infer:

- **Traffic & workload**
  - Request types (chat, Q&A, agents/tools, batch jobs).
  - Current and projected volumes (requests/day, concurrency).
  - Latency SLOs and interactivity requirements.
- **LLM configuration**
  - Models in use for each task (generation, tools/agents, reranking).
  - Prompt templates (system + user + tool call formats).
  - Max input/output tokens, temperature, and decoding params.
- **Embedding & vector DB configuration**
  - Embedding model(s), dimensions, and pricing tier.
  - Chunking strategy (size, overlap) and typical chunks per document.
  - Vector DB type, index configuration, metric, and storage pricing.
  - Retrieval settings: top-k, filters, hybrid search flags, reranker settings.
- **Caching & infrastructure**
  - Any existing caches (embeddings, retrieval results, responses, summaries).
  - Deployment pattern (synchronous vs worker-based, batch vs single).
  - Observability: logs/metrics/traces for tokens, costs, and latencies.

When exact numbers are not available, make **reasonable, clearly-labeled assumptions** and base cost estimates on those.

---

## End-to-end cost analysis workflow

When applying this skill, follow this workflow:

1. **Map the pipeline and traffic**
   - Sketch the end-to-end path: `Input → Tokenization → Chunking → Embedding → Vector DB → Retrieval/Reranking → LLM Reasoning → Output`.
   - Identify all places where **external metered APIs** are called (embeddings, LLMs, vector DB).
   - Capture workload assumptions: request mix, QPS, typical vs worst-case paths.
2. **Quantify token usage per representative request**
   - Estimate or compute:
     - Input tokens: system prompt + instructions + conversation history + retrieved context.
     - Output tokens: typical vs max.
   - Distinguish **steady-state chat turns** vs **first-request / cold-start** behavior.
3. **Quantify embedding and vector DB usage**
   - Estimate:
     - Chunks per document and documents per ingestion run.
     - Embedding calls (per doc, per run, per day/month).
     - Vector DB writes (upserts) and reads (per request, per batch).
4. **Identify redundancy and waste**
   - Look for:
     - Duplicate or near-duplicate chunks being embedded repeatedly.
     - Re-embedding unchanged content on every run.
     - Repeating the same retrieval + LLM call patterns for identical or highly similar queries.
     - Unbounded accumulation of conversation history or context.
5. **Propose model right-sizing**
   - For each task (classification, routing, extraction, generation, planning), decide whether a **smaller/cheaper model** or **two-stage cascade** can satisfy requirements.
6. **Design or refine caching layers**
   - Decide what to cache (embeddings, retrieval results, intermediate summaries, full answers).
   - Define cache keys, TTLs, invalidation rules, and **tenant isolation**.
7. **Produce a structured cost breakdown and optimization plan**
   - Break cost down by stage and by unit (per-request, per-1k tokens, per-day/month).
   - Propose concrete changes with estimated savings and potential quality trade-offs.

Always keep **accuracy and safety** as constraints: do **not** recommend cost cuts that obviously break grounding, tenant isolation, or evaluation discipline.

---

## Section 1 – Workload characterization

Goal: Understand **where** and **how often** the system spends tokens and external API calls.

Checklist:
- Identify major **request types** and flows (e.g., single-turn Q&A, multi-turn chat, tools/agents, batch analytics).
- Distinguish **online / interactive** vs **offline / batch** workloads.
- Capture order-of-magnitude estimates:
  - Requests per day/month per request type.
  - Average and P95 call-path length (number of LLM/tool invocations per user action).
- Flag **long-running agents or loops** that can lead to unbounded LLM/tool calls.

What to flag:
- Systems where the **default path always uses the largest model**, even for trivial operations.
- Unbounded agent loops without guardrails or step limits.
- Lack of clear **SLOs or budgets** per feature, making runaway cost more likely.

---

## Section 2 – Embedding usage and cost

Goal: Optimize **embedding-related cost** without sacrificing retrieval quality.

Checklist:
- **Model choice and usage**
  - Identify embedding models and pricing (or relative cost tiers).
  - Confirm a single model per index or clean separation of collections.
- **Chunking and counts**
  - Estimate average chunks per document and deduce total vectors.
  - Assess whether chunk sizes are causing **over-chunking** (too many small chunks).
- **Call patterns**
  - Determine batching strategy (per chunk vs batched).
  - Identify whether identical content is embedded more than once.
- **Caching & dedupe**
  - Check for content hashing or fingerprinting before embedding.
  - Look for any **embedding cache** keyed by normalized text + model id.

What to flag:
- One embedding API call per chunk with no batching.
- Re-embedding the same documents or unchanged content on each ingest.
- Mixed embedding models in the same collection (also a correctness failure).

Recommend:
- Introduce **content hashes** and **embedding caches** to avoid re-embedding identical text.
- Use **batch embedding APIs** with tuned batch sizes.
- Consider **cheaper embedding models** if retrieval quality allows, but require evaluation before switching.

---

## Section 3 – Vector DB, retrieval, and storage costs

Goal: Control **storage and query cost** from the vector DB and retrieval stack.

Checklist:
- **Index design**
  - Estimate number of vectors and storage per vector (dimension, precision, metadata).
  - Understand index type and its memory/CPU trade-offs.
- **Retrieval settings**
  - Inspect `top-k`, filtering, and hybrid search configuration.
  - Determine typical number of **candidate chunks** loaded into context per request.
- **Read/write patterns**
  - Quantify:
    - Writes per ingestion cycle (upserts/deletes).
    - Reads per user request (vector queries, BM25 queries, reranker inputs).

What to flag:
- Very high `top-k` values that pull many low-value chunks into prompts.
- No approximate index for large collections (unnecessary compute cost and latency).
- Lack of **TTL or archival** strategy for outdated or low-value vectors.

Recommend:
- Tune `top-k` to target **5–20 high-quality candidates**, combined with a reranker.
- Consider separating **hot** vs **cold** data into different collections or tiers.
- Periodically prune or archive vectors that are obsolete or never retrieved.

---

## Section 4 – LLM prompt design and token usage

Goal: Minimize **per-request token usage** while preserving model performance.

Checklist:
- **Prompt structure**
  - Inspect system prompts, instructions, and any constant boilerplate.
  - Identify redundant or verbose instructions that appear on every call.
- **Context construction**
  - Analyze how retrieved chunks are serialized (ordering, separators, metadata).
  - Measure or estimate total context tokens and check for truncation.
- **Conversation history**
  - Determine how many previous messages are kept by default.
  - Check for history windowing, summarization, or memory strategies.
- **Model selection**
  - Map tasks to models: small for routing/classification, medium for typical answers, large for rare or high-value paths.

What to flag:
- Very large system prompts or templates repeated for every call.
- Unbounded conversation histories concatenated into every prompt.
- Including **too many chunks** or irrelevant context (lost-in-the-middle and token waste).
- Using a **single very large model** for all steps without cascading.

Recommend:
- Introduce **prompt refactors**: more concise instructions, reusable IDs for rules, and reference to shared docs instead of repeating them in full.
- Implement **conversation windowing and summarization** to cap history tokens.
- Use **two-stage or multi-stage pipelines**:
  - Cheap model for routing/triage/query rewriting.
  - More capable model only when needed, on reduced context.

---

## Section 5 – Redundant processing and token waste

Goal: Detect **unnecessary repeated work** across the pipeline.

Checklist:
- **Repeated queries**
  - Check whether identical or near-identical queries trigger full retrieval + LLM calls every time.
  - Look for absence of result caching on read-heavy but slowly changing corpora.
- **Agents and tools**
  - Inspect agents that repeatedly call tools with similar prompts or context.
  - Identify cycles where the same data is retrieved and re-summarized multiple times.
- **Multi-step flows**
  - Look for flows where the same long context is passed to many intermediate steps instead of using compact intermediate representations.

What to flag:
- No caching in front of expensive, deterministic flows.
- Agent loops that repeatedly recompute the same results.
- Re-summarizing the same documents or search results on every request.

Recommend:
- Introduce **response and intermediate-result caches** (e.g., summaries, retrieval results).
- Use **IDs and references** instead of re-sending full documents between steps.
- Design flows so that expensive summarization is reused across users when safe.

---

## Section 6 – Caching strategy

Goal: Design **safe, effective caches** to reduce repeated cost.

Checklist:
- **Caching layers**
  - Embedding cache keyed by `(normalized_text, embedding_model_id)`.
  - Retrieval result cache keyed by `(canonicalized_query, filters, corpus_version)`.
  - Answer or summary cache for popular questions or heavy documents.
- **Safety & isolation**
  - Include **tenant_id, environment, language, and versioning** in cache keys where relevant.
  - Avoid cross-tenant leakage and respect access controls.
- **TTL & invalidation**
  - Define reasonable TTLs and invalidation triggers (document updates, schema changes).
  - Ensure cache invalidation is not reliant on manual steps.

What to flag:
- Global caches without tenant or environment boundaries.
- Caches with no invalidation strategy when content changes.
- Storing sensitive or PII-rich data without appropriate safeguards or masking.

Recommend:
- Start with **read-mostly caches** (embeddings, retrieval results, summaries) that are safe to recompute.
- Add **per-feature budgets and hit-rate monitoring** to ensure caches are effective.
- Combine caching with **evaluation** to ensure that aggressive caching does not degrade answer freshness or correctness.

---

## Section 7 – Model right-sizing and downgrades

Goal: Propose **model downgrades or cascades** that reduce cost while maintaining acceptable quality.

Checklist:
- **Task taxonomy**
  - Classify tasks: routing, classification, extraction, scoring, summarization, reasoning, generation, tool orchestration.
- **Current model usage**
  - For each task, record the model currently used and its relative cost.
- **Quality requirements**
  - Identify where errors are cheap vs expensive (e.g., internal tooling vs user-facing compliance answers).

What to flag:
- Use of the **same largest model** for low-risk, low-complexity tasks.
- No clear criteria for when to use cheaper vs more capable models.

Recommend:
- Use **smaller models** for:
  - Routing, classification, query rewriting, basic summarization, initial candidate scoring.
- Reserve **larger models** for:
  - High-stakes answers, complex reasoning, multi-document synthesis.
- Consider **cascaded designs**:
  - Start with a smaller model; escalate to a larger one only when confidence is low or stakes are high.

Always pair model changes with **evaluation** (accuracy, faithfulness, user acceptance) before rolling out widely.

---

## Section 8 – Observability and unit economics

Goal: Make cost **measurable, attributable, and tunable**.

Checklist:
- **Per-request metrics**
  - Track input and output tokens, embeddings, and vector queries per request.
  - Record model names, versions, and latency per call.
- **Aggregation**
  - Compute per-feature and per-tenant costs over time.
  - Identify top cost-drivers by endpoint, feature, and customer segment.
- **Tracing**
  - Use request IDs to tie together retrieval, LLM calls, and outputs.

What to flag:
- No visibility into tokens or cost by feature.
- Inability to answer "which endpoint or customer is most expensive?"

Recommend:
- Instrument cost metrics and dashboards (e.g., by endpoint, tenant, and model).
- Use these metrics to validate that **optimization changes** are actually reducing cost without harming quality.

---

## Section 9 – Hard cost anti-patterns

If any of the following are present, explicitly call out that the system has **serious cost design flaws**, even if it functions correctly:

- Unbounded chat history appended to every prompt with no windowing or summarization.
- No batching for embeddings, leading to one API call per chunk.
- No caching of embeddings, retrieval results, or heavy summaries in read-heavy workloads.
- Always using the largest model for every call, regardless of task complexity.
- Agent loops without step limits or safeguards that can spin indefinitely.

When these appear, prioritize **architectural corrections** over micro-optimizations.

---

## Output format

Always structure the final answer from this skill using the following sections (keep the headings, adapt the wording and content):

```markdown
## 1. Problem Breakdown

- **Use case and pipeline**: [Concise description of the system, its main tasks, and the pipeline stages in play (embedding, retrieval, LLM, agents, etc.).]
- **Traffic & workload**: [Key assumptions about request types, volumes, and concurrency.]

## 2. Issues & Risk Analysis

- **Embeddings**: [Key cost drivers, duplication, batching issues, or model problems.]
- **LLM calls**: [Prompt size, model choice, history handling, agent loops.]
- **Retrieval & storage**: [Vector DB, top-k, hybrid search costs, storage growth.]
- **Redundant processing & token waste**: [Where repeated work or unbounded context appears.]

## 3. Structured Solution

- **Stage-by-stage plan**: [Concrete changes for embeddings, retrieval, LLM prompts, and infrastructure.]
- **Ordering & dependencies**: [What to fix first vs later, with rationale.]

## 4. Optimizations

- **Quick wins (low risk)**: [Immediate, low-risk changes with high ROI.]
- **Medium-term improvements**: [Larger refactors or pipeline changes.]
- **Long-term options**: [Deeper architectural shifts or model strategy changes.]

## 5. Risks & Edge Cases

- **Quality risks**: [Where cost reductions might harm accuracy or robustness.]
- **Freshness & consistency**: [Caching, invalidation, and staleness trade-offs.]
- **Security & isolation**: [Tenant isolation in caches and vector DB, PII handling.]

## 6. Cost Breakdown

- **Per-stage cost**: [Estimated cost contribution from embeddings, retrieval, LLM calls, and other metered components, with clear assumptions.]
- **Per-request and monthly view**: [Unit economics and projected monthly spend.]

## 7. Suggested Model Downgrades

- **Candidate downgrades**: [Where smaller/cheaper models can replace current ones, with justification.]
- **Cascades**: [Opportunities for multi-stage (small-then-large) model flows.]

## 8. Caching Strategy

- **What to cache**: [Embeddings, retrieval results, summaries, answers, etc.]
- **Cache design**: [Keys (including tenant and versioning), TTLs, invalidation triggers.]
- **Expected impact**: [Rough estimates of cost savings from caching layers.]
```

Ensure each answer remains **cost-aware, pipeline-aware, and evaluation-conscious**, and clearly separates:
- **Where cost is currently spent**
- **Which optimizations are safe**
- **Which optimizations require careful evaluation before adoption**

