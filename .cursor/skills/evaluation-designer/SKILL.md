---
name: evaluation-designer
description: Design end-to-end evaluation frameworks for RAG and LLM systems, including metric selection (precision, recall, MRR, faithfulness), golden dataset creation, and test query design. Use when defining evaluation plans, benchmarks, or test suites for retrieval and generation quality across the full pipeline.
---

# Evaluation Designer

## Instructions

Use this skill to design **production-grade evaluation frameworks** for RAG and LLM systems. Always evaluate the full pipeline:

Input → Tokenization → Chunking → Embedding → Vector DB → Retrieval/Reranking → Context Construction → LLM Reasoning

The goal is to:
- Define **clear metrics** (Precision, Recall, MRR, Faithfulness)
- Build a **high-quality golden dataset**
- Design **test queries** that stress the system realistically
- Produce a concrete **evaluation plan** and **benchmark methodology**

### When to Use This Skill

Use this skill when:
- Launching a new RAG/LLM feature and you need an evaluation strategy
- Comparing models, retrieval settings, chunking strategies, or prompts
- Diagnosing quality regressions or low retrieval/generation accuracy
- Designing golden datasets and test suites for continuous evaluation

### Required Inputs to Gather

Before designing the evaluation, gather:
- **Use case**: task type (QA, summarization, agentic tools, classification, etc.)
- **Domains**: e.g., marketing docs, legal, support, code, analytics
- **System architecture**: high-level RAG or agent pipeline (what components exist)
- **Current retrieval setup**: embeddings model, vector DB, hybrid search, top-k, reranker
- **Constraints**: latency, cost, safety requirements, regulatory constraints
- **Existing data**: logs, previous queries, labeled examples, known failure cases

---

## Outputs You Must Produce

When applying this skill, always produce:

- **Evaluation plan** that includes:
  - Scope (what is and is not evaluated)
  - Metrics and how they are computed
  - Dataset description and size
  - Evaluation procedures (offline and, if relevant, online)
  - Acceptance thresholds or targets

- **Benchmark methodology** that includes:
  - How to construct golden datasets
  - How to design and maintain test queries
  - How to run experiments and compare variants
  - How to interpret metric changes and diagnose failures

Use the templates in the **Templates** section when responding to the user.

---

## Step-by-Step Workflow

### 1. Clarify Evaluation Goals and Scope

1. Identify primary goals:
   - Retrieval quality? Hallucination reduction? Overall task success? Safety?
2. Decide evaluation levels:
   - **Retrieval-only** (vector DB + hybrid search + reranker)
   - **End-to-end** (retrieval + reasoning + tools/agents)
3. Define success criteria:
   - Example: "Precision@5 ≥ 0.6, Recall@20 ≥ 0.8, MRR@10 ≥ 0.5, Faithfulness ≥ 0.9"
4. List non-goals (what is explicitly out of scope for this iteration).

### 2. Map Evaluation to Pipeline Stages

Explicitly link metrics and checks to stages:

- **Chunking & Embedding**
  - Check that golden labels are compatible with chunking strategy.
  - Ensure no mixed embedding spaces (one embedding model per index).
- **Vector DB & Retrieval**
  - Evaluate **Precision@k**, **Recall@k**, **MRR@k** at retrieval level.
  - Validate hybrid search parameters (dense + sparse) and top-k.
- **Context Construction**
  - Monitor context length, risk of overflow, and lost-in-the-middle.
- **LLM Reasoning**
  - Evaluate **Faithfulness** and answer quality given retrieved context.

Classify failures by stage:
- **Retrieval failure** (relevant docs not retrieved)
- **Ranking failure** (relevant docs retrieved but ranked too low)
- **Context failure** (truncation, lost-in-the-middle, irrelevant chunks dominating)
- **Generation failure** (hallucination or incorrect reasoning despite good context)

---

## Metrics


## Metric Weighting and Decision Logic

Define how metrics combine into decisions:

Example:
- Retrieval score = weighted combination of Precision, Recall, MRR
- Final system score = Retrieval score + Faithfulness

Ensure:
- Low faithfulness overrides high retrieval metrics
- Critical metrics have higher weight

Use weighted scoring for:
- Model comparison
- Deployment decisions


## Cost and Latency Metrics

In addition to quality metrics, always track:

- Tokens per query (prompt + completion)
- Cost per query
- Retrieval latency
- End-to-end latency

Evaluate trade-offs:

- Accuracy vs Cost
- Accuracy vs Latency

Flag:
- Improvements that significantly increase cost or latency


### 1. Retrieval Metrics

Assume each query has a set of **relevant documents/chunks** in the golden dataset.

- **Precision@k**
  - Definition: fraction of retrieved items in the top \(k\) that are relevant.
  - Formula: \( \text{Precision@k} = \frac{\#\{\text{relevant in top }k\}}{k} \)
  - Use to control noise: high precision means retrieved items are mostly relevant.

- **Recall@k**
  - Definition: fraction of all relevant items that appear in the top \(k\).
  - Formula: \( \text{Recall@k} = \frac{\#\{\text{relevant in top }k\}}{\#\{\text{all relevant}\}} \)
  - Use to detect retrieval misses: low recall means relevant content is not retrieved.

- **MRR@k (Mean Reciprocal Rank)**
  - For each query:
    - Find rank of the first relevant item \(r\) within the top \(k\) (if none, reciprocal rank is 0).
    - Reciprocal rank: \( \text{RR} = \frac{1}{r} \)
  - \( \text{MRR@k} \) is the average RR across all queries.
  - Use to measure how **quickly** relevant results appear.

Recommend typical settings:
- Retrieval evaluation: **k ∈ {5, 10, 20}** depending on context size and use case.
- Prefer reporting **Precision@5, Recall@20, MRR@10** as a default trio.

### 2. Faithfulness and Answer Quality

Faithfulness is about whether the model’s answer is **grounded in retrieved context** and does not hallucinate.

- **Faithfulness**
  - For each query, judge if the answer:
    - Is fully supported by the retrieved context
    - Does not introduce unsupported facts
  - Labels (typical):
    - Faithful
    - Partially faithful
    - Unfaithful / hallucinated
  - Compute:
    - Faithfulness rate = fraction of answers labeled "faithful"
    - Optionally, weighted scores (Faithful = 1.0, Partial = 0.5, Unfaithful = 0.0)

- **Answer Correctness / Task Success**
  - For QA: exact match or semantic match with expected answer.
  - For summarization: coverage and correctness vs reference.
  - For agents: task completion rate (e.g., correct action/tool result).

Implementation options:
- Human raters with clear guidelines.
- LLM-as-judge prompts with:
  - Access to query, retrieved context, system answer, and reference answer (if any).
  - Explicit instructions to penalize unsupported claims.

---

## Golden Dataset Creation

## Failure-Driven Dataset Expansion

Continuously improve dataset by:

- Adding queries from:
  - Failure simulator outputs
  - Real user failure cases
- Labeling new edge cases and adversarial queries

Ensure:
- Dataset evolves with system weaknesses
- Known failure modes are re-tested in future evaluations


### 1. Define Annotation Granularity

Decide what you will label as "relevant":
- Document-level: entire documents or pages.
- Chunk-level: individual chunks as produced by the chunker.
- Answer-level: expected answer strings or structured outputs.

Align labeling granularity with:
- The **chunking strategy** (semantic/recursive, overlapping, etc.).
- The **retrieval unit** (chunk vs document).

### 2. Source Candidate Queries

Use multiple sources to reduce bias:
- Real user logs (anonymized and de-identified).
- Synthetic queries generated by LLMs, then **reviewed by humans**.
- Domain experts’ questions covering critical workflows.

Ensure coverage of:
- **Head queries** (frequent, common tasks).
- **Tail queries** (rare but important).
- **Edge cases**: ambiguous, adversarial, or multi-hop queries.

### 3. Label Relevant Documents/Chunks

For each query:
1. Present annotators with:
   - The query
   - Candidate documents/chunks
   - Optionally, full source corpus for search
2. Ask them to mark:
   - Which documents/chunks contain information necessary to answer the query.
   - Whether each is:
     - Fully relevant
     - Partially relevant
     - Irrelevant
3. Store labels with:
   - Query ID
   - Document/chunk IDs
   - Relevance grade (binary or graded)

### 4. Label Expected Answers (If Applicable)

For QA or generation tasks:
- Provide canonical answers (for deterministic tasks).
- Or provide multiple acceptable answers / key points to check.
- For summarization, store:
  - Key facts that must appear
  - Forbidden errors/misinterpretations

### 5. Dataset Size and Splits

As a starting point:
- **Minimal**: 50–100 queries for smoke tests and initial tuning.
- **Robust**: 200–500+ queries for stable metrics and A/B tests.
- **Large-scale**: 1,000+ for production-grade benchmarking.

Always create splits:
- **Train/tune**: for prompt tweaking, retrieval settings, and model selection.
- **Dev**: for hyperparameter tuning and intermediate checks.
- **Test/holdout**: for final unbiased evaluation only.

---

## Test Query Design

### 1. Coverage Dimensions

Design the query set to cover:
- Domains (e.g., campaigns, SEO, performance analytics, etc.).
- Query intents (lookup, comparison, explanation, strategy, debugging).
- Difficulty levels (easy, medium, hard; multi-hop vs single-hop).
- Temporal aspects (current vs historical data, time ranges).
- Safety-sensitive or ambiguous cases.

### 2. Query Types

Include a mix of:
- **Retrieval-sensitive queries**
  - Require finding a specific fact or passage.
  - Use to stress-test Precision/Recall/MRR.
- **Reasoning-sensitive queries**
  - Require combining multiple pieces of evidence.
  - Use to test LLM reasoning and faithfulness.
- **Edge-case queries**
  - Ambiguous, under-specified, or adversarial.
  - Use to test robustness, refusal behavior, and safety.

### 3. Annotation for Each Query

For each query, store:
- Query text
- Metadata: domain, intent, difficulty, scenario tags
- Relevant document/chunk IDs (with graded relevance if used)
- Expected answer or evaluation rubric
- Notes on known pitfalls or common errors

---

## Benchmark Methodology

### 1. Retrieval-Only Benchmark

1. For each query in the golden dataset:
   - Run the **retrieval pipeline only** (no LLM reasoning):
     - Hybrid search (dense + sparse/BM25)
     - Vector DB query with top-k
     - Reranker (if present)
2. Compare retrieved IDs against golden relevance labels.
3. Compute:
   - Precision@k, Recall@k, MRR@k (and optionally nDCG@k).
4. Segment metrics by:
   - Domain, intent, difficulty, and other metadata.
5. Diagnose:
   - **Retrieval failures**: relevant docs never retrieved.
   - **Ranking failures**: relevant docs appear but too low.

Use this benchmark to:
- Tune top-k, hybrid weights, and reranker.
- Compare embedding models or vector index configurations.

### 2. End-to-End Benchmark

1. For each query in the golden dataset:
   - Run the **full pipeline** (retrieval + context construction + LLM reasoning).
   - Capture:
     - Retrieved context
     - Final answer
     - Latency and token usage (prompt + completion)
2. Evaluate:
   - Faithfulness (supported vs unsupported claims).
   - Answer correctness / task success.
   - Citation correctness (if the system outputs citations).
3. Use human raters or LLM-as-judge with consistent prompts.
4. Classify error types:
   - Retrieval, ranking, context, or generation failure.

Use this benchmark to:
- Compare prompts, LLM models, and system instructions.
- Validate that retrieval improvements translate into better answers.

### 3. Experiment Design and Comparison

When comparing variants (A/B):
1. Keep **golden dataset and evaluation rubric fixed**.
2. Run both systems A and B on the same queries.
3. Collect metrics per query, then:
   - Compute mean differences and confidence intervals.
   - Optionally, apply statistical tests (e.g., bootstrap).
4. Report:
   - Overall metric deltas (e.g., +0.08 MRR@10).
   - Per-segment deltas (e.g., hard queries improved more).
5. Investigate examples where:
   - A wins strongly over B.
   - B wins strongly over A.

---

## Regression Detection Framework

For every system change:

- Compare against previous baseline
- Track metric deltas:
  - Precision@k
  - Recall@k
  - MRR@k
  - Faithfulness

Flag:
- Any statistically significant degradation
- Silent regressions in specific segments (e.g., hard queries)

Require:
- Baseline locking before experiments
- Versioned evaluation reports

## Templates

### Evaluation Plan Template

Use this template when producing an evaluation plan for the user:

```markdown
## Evaluation Plan

### 1. Problem Overview
- Use case:
- Domains:
- Primary user journeys:

### 2. Evaluation Goals and Scope
- Primary goals:
- In-scope components:
- Out-of-scope components:
- Constraints (latency, cost, safety):

### 3. System Overview
- High-level architecture:
- Retrieval stack (embedding model, vector DB, hybrid search, reranker):
- LLM(s) and prompting setup:

### 4. Datasets and Golden Set
- Source(s) of queries:
- Golden dataset size and splits (train/dev/test):
- Annotation granularity (document/chunk/answer level):
- Labeling process (who, how, guidelines):

### 5. Metrics
- Retrieval: Precision@k, Recall@k, MRR@k (k = ...).
- Generation: Faithfulness, correctness, task success.
- Operational: latency, cost per query, error rates.

### 6. Evaluation Procedures
- Retrieval-only evaluation:
  - Steps:
  - Frequency:
- End-to-end evaluation:
  - Steps:
  - Frequency:

### 7. Acceptance Criteria
- Target thresholds for key metrics:
- Rollout/blocker conditions:

### 8. Analysis and Reporting
- How results will be segmented:
- How frequently reports are generated:
- Who reviews results and acts on them:

### 9. Risks and Limitations
- Known weaknesses of the evaluation:
- Data gaps or bias risks:
- Planned improvements:
```

## Hard Evaluation Failure Conditions

If any of the following occur, mark system as **"Not Production Ready"**:

- Faithfulness below acceptable threshold
- Recall below required minimum
- Significant regression from baseline metrics
- High hallucination rate in end-to-end evaluation
- Critical failure cases (from failure simulator) unresolved

When triggered:
- Block deployment
- Require fixes before iteration continues


### Benchmark Methodology Template

Use this template when describing how benchmarking will be executed:

```markdown
## Benchmark Methodology

### 1. Benchmark Scope
- Components covered (retrieval-only, end-to-end, or both):
- Systems/variants compared:

### 2. Golden Dataset Definition
- Query selection process:
- Annotation protocol:
- Dataset statistics (counts per domain/intent/difficulty):

### 3. Evaluation Pipeline
- Retrieval pipeline configuration (top-k, hybrid weights, reranker):
- LLM and prompt configuration:
- Logging and tracing setup:

### 4. Metric Computation
- Retrieval metrics: how Precision@k, Recall@k, MRR@k are computed.
- Faithfulness and correctness scoring process.
- Aggregation and segmentation strategy.

### 5. Experiment Procedure
- Steps to run a single benchmark:
- Steps to run A/B comparisons:
- Randomization and ordering controls (if any):

### 6. Interpretation and Decision Rules
- How to interpret metric deltas:
- Minimal detectable improvement or effect size:
- Decision rules for rollout vs rollback:

### 7. Maintenance
- How often benchmarks are re-run:
- How new queries or labels are added:
- Versioning of datasets, prompts, and configurations:
```

---

## Relationship to Other Skills

When designing evaluation frameworks, you may also:
- Use `rag-pipeline-auditor` to understand and document the RAG architecture and pipeline risks.
- Use `retrieval-debugger` to inspect specific queries and diagnose retrieval/ranking/context failures.
- Use `ingestion-optimizer` to ensure the golden dataset and evaluation reflect realistic ingestion and chunking.
- Use `cost-token-analyzer` to include cost and token metrics in the evaluation.
- Use `prompt-optimizer` to iterate on prompts while tracking evaluation metrics.
- Use `failure-simulator` to generate adversarial scenarios and incorporate them into the golden dataset and test queries.


## Continuous Evaluation Loop

Define evaluation cadence:

- Offline evaluation:
  - Run on every major change (model, chunking, retrieval, prompt)
- Scheduled evaluation:
  - Daily or weekly batch runs
- Production monitoring:
  - Track real-time metrics (latency, cost, error rates)

Trigger re-evaluation when:
- Model changes
- Data ingestion updates
- Retrieval configuration changes

Ensure:
- Automated reporting
- Alerting on metric degradation