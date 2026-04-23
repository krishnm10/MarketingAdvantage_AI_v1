---
name: retrieval-debugger
description: Analyze query-time retrieval outputs (queries plus retrieved chunks) to diagnose retrieval, ranking, and context failures, and propose concrete fixes to chunking, filters, hybrid search settings, top-k, and rerankers. Use when debugging retrieval quality for a specific RAG query where you have the retrieved chunks available.
---

# Retrieval Debugger

This skill guides the agent to debug **per-query retrieval quality** in a RAG system using a concrete query and its retrieved chunks.

Use this skill when:
- The user provides a **query** and a set of **retrieved chunks/documents** (with any available scores, ranks, or metadata).
- The user reports that **a specific question was answered poorly**, and you want to understand whether the root cause is **retrieval, ranking, or context construction**, not just LLM generation.
- You want concrete, **stage-specific fixes** around retrieval (top-k, filters, hybrid balance, reranker) rather than full pipeline redesign.

If the issues appear to be **systemic across many queries** (not just a single example), consider also applying the `rag-pipeline-auditor` skill for an end-to-end pipeline review.

---

## Required Inputs

Before applying this skill, gather (or infer) the following:

- **Query and task**
  - The exact user query.
  - If available, expected answer or ground truth, and the main task type (e.g., fact Q&A, how-to, policy lookup, code help).
- **Retrieved chunks**
  - The list of chunks/documents retrieved for this query, ideally with:
    - Rank/order
    - Scores (dense/sparse if available)
    - Metadata (source document id, section/heading, tenant, timestamps, etc.).
- **Retrieval configuration (if available)**
  - Top-k used at each stage (initial retrieval, post-filtering, reranking).
  - Whether **hybrid retrieval** (dense + sparse/BM25) is enabled and how scores are combined.
  - Whether a **reranker** is present, and if so:
    - Model type (cross-encoder, LLM reranker, etc.).
    - How many candidates are reranked (e.g., top-50 → top-10).

If some of this is missing, **call out the gaps explicitly** and proceed with best-effort qualitative reasoning based only on the visible chunks.

---

## Analysis Workflow (Per Query)

Always follow this workflow in order.

### 1. Understand the query and information needs

- Summarize the query in your own words.
- Identify the **key information needs** and sub-questions (aspects) required to answer it correctly.
- If a gold answer or expected behavior is provided, extract:
  - Critical facts that must appear.
  - Constraints (time range, product/tenant, jurisdiction, environment, etc.).

### 2. Evaluate per-chunk relevance

For each retrieved chunk:

- Provide a **very short summary** (1 sentence) of the chunk.
- Label its **relevance** to the query using:
  - `critical` – directly contains key facts or instructions needed for a correct answer.
  - `helpful` – partially relevant, adds context or secondary details.
  - `irrelevant/noise` – unrelated or only tangentially related.
  - `harmful/misleading` – likely to push the model toward an incorrect answer.
- If the query has multiple aspects, note **which aspect(s)** each relevant chunk supports.

This labeling is the backbone for later **precision vs recall** and failure classification.

### 3. Identify missing relevant information

Using the query, its aspects, and any expected answer:

- List important **facts, sections, or document types** that are **not covered by any retrieved chunk**.
- Note if you see:
  - Only partial coverage of a document/section (e.g., the middle of a policy without its definitions).
  - Evidence that the correct document family is present but the **exact relevant section** is missing.

If answering the query correctly would clearly require content that is not present in any of the retrieved chunks, treat this as a likely **retrieval failure or chunking/context failure**, depending on whether:
- The entire doc appears to be missing (retrieval issue), or
- The doc is present but chunked/selected poorly (chunking/context issue).

### 4. Precision vs Recall assessment

Given `k = number of retrieved chunks`:

- Approximate **precision**:
  - Let `relevant = critical + helpful chunks`.
  - Compute a qualitative precision: `high / medium / low` based on `relevant / k`:
    - `high`: majority of chunks are relevant, very little noise.
    - `medium`: mixed relevant and irrelevant chunks.
    - `low`: most chunks are irrelevant or misleading.
- Approximate **recall** qualitatively:
  - `high`: All major aspects of the query are well covered by retrieved chunks.
  - `medium`: Some aspects are covered, but 1–2 important ones are missing or under-represented.
  - `low`: Key aspects have little or no representation in the retrieved chunks.

Always explain **why** you rate precision and recall at those levels.

### 5. Analyze top-k effectiveness

Using the relevance labels:

- Examine whether the chosen `top-k` seems:
  - **Too low**: relevant chunks are sparse and coverage of aspects is weak → **recall likely constrained by k**.
  - **Too high**: many clearly irrelevant chunks and near-duplicates → noisy context and unnecessary cost.
- Check for:
  - **Duplicates or near-duplicates** consuming slots that could hold more diverse evidence.
  - Large blocks of clearly irrelevant chunks at the tail (e.g., ranks 8–20).

From this, recommend whether to:
- Increase k (for coverage) with stronger reranking/filters, or
- Decrease k and rely on better scoring/reranking to keep only high-quality chunks.

### 6. Analyze hybrid search balance (dense vs sparse)

When information is available:

- Inspect **dense vs sparse scores** or indicators to see which side dominates.
- Consider typical patterns:
  - If many chunks share key query terms but are semantically off-topic, suspect **overweight sparse/BM25**.
  - If chunks are semantically related but miss critical exact terms (e.g., product names, IDs, codes), suspect **underweight sparse/BM25** or **pure dense retrieval**.

When score details are not provided:

- Infer from content:
  - If top chunks contain exact query terms but miss the needed concept, suspect lexical bias.
  - If top chunks rarely contain key query terms but are loosely related thematically, suspect over-reliance on dense similarity.
- If there is **no evidence of hybrid retrieval**, explicitly flag the absence of hybrid search as a likely weakness for many real-world text corpora.

Recommend concrete adjustments such as:
- Enabling hybrid retrieval if missing.
- Rebalancing score combination or thresholds between dense and sparse components.

### 7. Evaluate reranker presence and impact

Determine whether a **reranker** is used:

- If **no reranker is present**, and the task is non-trivial, flag this as a design gap and recommend adding a reranker.
- If a reranker exists:
  - Check whether **highly relevant chunks** are ranked low or dropped.
  - Check whether noisy or misleading chunks appear at the top despite relevant candidates existing lower in the list.

Use this to assess:
- Whether **initial retrieval** is failing (relevant chunks missing entirely).
- Or whether **ranking** is failing (relevant chunks present but not surfaced).

Recommend:
- Increasing the number of candidates passed into the reranker (e.g., top-50 → rerank → top-10).
- Adjusting thresholds or score interpretation.
- Switching to or tuning a stronger reranker model for the domain.

### 8. Classify failures

Based on the above analysis, classify issues explicitly into:

- **Retrieval failure**:
  - Relevant documents or sections are **not present at all** in the retrieved set.
  - Often indicated by low recall despite a reasonable k.
- **Ranking failure**:
  - Relevant chunks exist but are **ranked too low** or overshadowed by noisy chunks.
  - Precision among top results is poor even though some deeper results are good.
- **Context failure**:
  - Relevant content exists but is:
    - Fragmented across chunks due to poor chunking.
    - Drowned in irrelevant context.
    - Likely to be **lost-in-the-middle** when serialized into the model’s context window.

If the retrieval, ranking, and context all appear solid yet the answer is wrong, call out a **generation failure** and note that this is outside the main scope of this skill but should be handled via prompting/model changes.

---

## Hard Failure Conditions (Must Not Pass)

If any of the following are detected, explicitly mark the system as "Production Unsafe":

- No reranker in a non-trivial retrieval setup
- No hybrid retrieval (dense + sparse) for text-heavy corpora
- Critical query aspects missing entirely from retrieved chunks
- Retrieval returning mostly irrelevant or misleading chunks
- No evaluation or validation mechanism for retrieval quality

When triggered:
- Explicitly state: "This retrieval system is production unsafe"
- Prioritize fixes before optimization


## Minimum Viable Retrieval Standard

A valid retrieval system must include:

- Adequate top-k for query coverage
- Hybrid retrieval (dense + sparse)
- Reranking stage
- Metadata filtering

If any of the above is missing:
- Flag as incomplete or weak retrieval design


## Cost, Context, and Efficiency

Always evaluate:

- Token cost from retrieved chunks
- Impact of top-k on prompt size
- Reranker computational cost
- Redundant or duplicate chunks

Check:
- Whether retrieved chunks fit within LLM context limits
- Whether excessive chunks cause "lost-in-the-middle" issues
- Detect duplicate or near-duplicate chunks consuming top-k slots
- Ensure top-k selection aligns with LLM context window limits

Recommend:
- Reducing k with better reranking
- Deduplicating chunks
- Optimizing chunk size vs context window

Always balance:
Accuracy vs Cost vs Context Size



## Output Format

Always structure the final answer using this template (adapt wording slightly as needed, but keep the sections and intent):

```markdown
## Query & task summary

- **Query**: [original query]
- **Task type**: [e.g., fact lookup / how-to / policy compliance / code help]
- **Key information needs**: [1–3 bullets capturing aspects that must be satisfied]

## Chunk-by-chunk relevance

For each chunk in order:
- **Chunk [i]** – [short 1-sentence summary]
  - **Relevance**: [critical / helpful / irrelevant/noise / harmful]
  - **Supports aspects**: [which query aspects it helps with, if any]

## Precision vs recall analysis

- **k (number of chunks)**: [k]
- **Precision (qualitative)**: [high / medium / low] – [short justification]
- **Recall (qualitative)**: [high / medium / low] – [short justification]
- **Notable missing information**: [bulleted list of missing facts/sections, if any]

## Failure classification

- **Retrieval failure**: [yes/no] – [why]
- **Ranking failure**: [yes/no] – [why]
- **Context failure**: [yes/no] – [why]
- **(Optional) Generation failure**: [yes/no] – [why, if clearly outside retrieval]

## Root cause analysis

- **Primary root cause**: [one-line summary: e.g., "Low recall due to missing hybrid search and too-small top-k"]
- **Secondary contributing factors**: [1–3 bullets, if applicable]

## Recommended fixes (concrete)

- **Chunking / context construction**:
  - [Fixes if context failure or chunking issues are present; otherwise "No change needed."]
- **Retrieval (top-k, filters, hybrid)**:
  - [Exact changes to top-k values, filters, or hybrid scoring strategy.]
- **Reranker configuration**:
  - [Add a reranker if missing, or adjust reranker parameters / candidate counts.]
- **Other notes**:
  - [Cost/latency considerations, evaluation suggestions, or follow-up diagnostics.]
```

Keep recommendations **concrete and actionable** (e.g., "Increase initial dense top-k from 10 to 40 and enable BM25 hybrid, then rerank top-40 down to 8–12 chunks") rather than vague guidance.

---

## Additional Guidance

- Be explicit when information is missing (e.g., lack of hybrid search details, unknown reranker).
- When possible, suggest **small, testable changes** that can be evaluated against a **golden query set** instead of one-off tuning.
- If per-query analysis reveals patterns likely to affect many queries, recommend:
  - Running this analysis on a **representative set of queries**.
  - Performing a broader **pipeline-level audit** using the `rag-pipeline-auditor` skill.

---

## Performance & Scaling Considerations

Consider:

- Query latency impact of high top-k values
- Reranker scalability under load
- Vector DB query performance

Flag:
- Excessive retrieval sizes causing latency spikes
- Reranker bottlenecks
- Inefficient query patterns under scale