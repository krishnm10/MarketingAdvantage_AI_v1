---
name: llm-output-quality-analyzer
description: Analyze LLM final responses for grounding against retrieved or provided context, hallucination risk, and reasoning quality, assigning quality scores and proposing concrete improvement strategies. Use when the user asks to evaluate, critique, or improve an LLM answer, detect hallucinations, or review RAG-generated responses.
---

# LLM Output Quality Analyzer

## Purpose

This skill guides the agent to critically evaluate final LLM responses, especially in RAG systems, focusing on:

- **Grounding**: Is the answer supported by retrieved or provided context?
- **Hallucination risk**: Are there unsupported claims or speculative leaps?
- **Reasoning quality**: Is the reasoning coherent, correct, and task-aligned?

The agent must always produce a **quality score**, **hallucination assessment**, and **concrete improvement strategy** for the answer under review.

---

## When to Use This Skill

Apply this skill when:

- The user asks to **rate, critique, audit, or review** an LLM answer.
- The user suspects **hallucinations, weak reasoning, or poor grounding** in a response.
- You are comparing **multiple answers** and need structured quality judgments.
- You are validating answers produced by a **RAG pipeline** and want to localize failures.

This skill is **answer-focused**: it analyzes the **final response**, but must stay aware of the full pipeline:

Input → Tokenization → Chunking → Embedding → Vector DB → Retrieval/Reranking → Context Construction → LLM Reasoning → **Final Answer**

---

## Required Inputs

Before analyzing, gather (or infer) as much of the following as is available:

- **User query / task** being answered.
- **Final LLM response** (the answer being evaluated).
- **Retrieved or provided context**, if any:
  - Retrieved chunks, citations, or reference passages.
  - System or tool output that the answer should be grounded in.
- **System/instruction prompt** (if accessible), to understand constraints and style requirements.
- **Pipeline behavior (optional but ideal)**:
  - Retrieval settings (top-k, hybrid search use, reranker presence).
  - Any truncation or context-window limits known.

If context or pipeline details are missing, call this out explicitly and treat quality judgments (especially hallucination risk) as **higher-uncertainty**.

---

## Failure Types (Answer-Level)

Always classify issues using these categories, tying them back to pipeline stages where possible:

- **Retrieval failure**: Relevant information appears to be absent from the context, so the answer cannot be fully correct even if reasoning is solid.
- **Ranking failure**: Relevant information exists but is likely ranked too low or overshadowed by noise, leading the answer to use suboptimal evidence.
- **Context failure**: Relevant content exists but is truncated, diluted by irrelevant chunks, or placed where the model under-attends (lost-in-the-middle).
- **Generation failure**: Context was sufficient, but the LLM hallucinated, misinterpreted, or ignored instructions.

You **must** identify the most likely **root-cause category** before proposing fixes.

---

## Scoring Framework

Compute an **overall quality score (0–100)** derived from four dimensions:

- **Faithfulness / Grounding (F)** – 0–10
  - Are claims supported by the provided context or clearly framed as uncertainty?
  - Penalize unsupported factual statements, mis-citations, or ignoring context.
- **Reasoning Quality (R)** – 0–10
  - Logical consistency, correct use of evidence, and absence of contradictions.
  - Correct handling of edge cases and constraints in the prompt.
- **Completeness & Relevance (C)** – 0–10
  - Coverage of the user’s question and avoidance of irrelevant digressions.
- **Clarity & Instruction Adherence (I)** – 0–10
  - Clear structure, correct format, tone, and adherence to explicit instructions.

Use a **weighted combination** (you may adjust weights per use case, but default to faithfulness-heavy):

- Overall score S (0–100) ≈ 0.45·F + 0.25·R + 0.20·C + 0.10·I, then scaled to 0–100.

Hard constraints:

- If **major hallucinations** or critical unfaithfulness are present, cap S at **≤ 50**, regardless of other dimensions.
- If the answer is **unsafe, clearly non-compliant, or ignores critical constraints**, treat S as **≤ 40** and call this out explicitly.

Always report both the **overall score** and **per-dimension scores**.

---

## Analysis Workflow

Follow this workflow when applying the skill.

### 1. Restate the Evaluation Task

- Summarize:
  - The **user query / task**.
  - The **intended outcome** (e.g., factual answer, strategy, code, reasoning).
  - Any **critical constraints** (safety, compliance, output format, verbosity limits).

This forms the **Problem Breakdown** section in your final answer.

### 2. Inspect Grounding and Faithfulness

When context is available:

- For each major claim or section in the answer:
  - Check whether it is:
    - **Fully supported** by specific context passages.
    - **Partially supported** (some details inferred).
    - **Unsupported / contradicted** by context.
- Verify:
  - Citations (if present) actually support the claim.
  - The answer does **not import external facts** that are absent from context without clearly labeling them as general knowledge or assumptions.
  - The answer admits uncertainty (e.g., “the context does not specify…”) instead of fabricating.

When context is not available:

- Rely on:
  - Internal consistency.
  - Known domain facts (to the extent possible).
  - The specificity and confidence of claims.
- Mark hallucination risk as **uncertain but non-zero**, especially for:
  - Highly specific numeric values, URLs, API names, or citations.
  - Assertions that go beyond what an average domain expert could safely infer.

Explicitly flag:

- **Hallucinated details**: unsupported facts, fabricated identifiers, or contradictions.
- **Overconfident language** when evidence is weak or context is missing.

### 3. Evaluate Reasoning Quality

Assess whether:

- The reasoning steps (even if not fully exposed to the user) are:
  - Coherent and free of logical fallacies.
  - Correctly using the provided evidence.
  - Handling edge cases, ambiguity, and constraints.
- Multi-step conclusions:
  - Do not skip critical dependencies.
  - Are traceable back to context or well-justified domain knowledge.

Flag:

- Internal contradictions within the answer.
- Incorrect mathematical or logical steps.
- Misinterpretations of the question or context.

### 4. Check Completeness, Relevance, and Context Handling

Assess:

- **Completeness**:
  - Does the answer cover all parts of the user query (sub-questions, constraints)?
  - Are important caveats, limitations, or trade-offs mentioned?
- **Relevance**:
  - Is the majority of the content directly useful to the user’s goal?
  - Is there unnecessary repetition or filler that just increases token usage?
- **Context handling**:
  - Signs of **context overflow** or **lost-in-the-middle**:
    - References to early context but ignoring later, more relevant chunks.
    - Inconsistent usage of retrieved facts suggesting truncation or overload.

If you suspect context issues, map them to **retrieval, ranking, or context failure** categories.

### 5. Assess Clarity, Format, and Instruction Adherence

Check:

- Structure:
  - Clear headings, bullets, and separation of concerns where appropriate.
- Style:
  - Tone appropriate to the task.
  - Compliance with user-specified formatting rules (e.g., no emojis, headings, templates).
- Safety and policy:
  - No exposure of sensitive data or internal system prompts.
  - No evidence of **prompt injection success** (e.g., answer echoing “ignore previous instructions” or revealing hidden instructions).

Note where the answer could be **shorter and more efficient** without sacrificing clarity or accuracy.

### 6. Classify Failure Type(s) and Root Cause

Using the Failure Types section:

- Decide whether the dominant issue is:
  - **Retrieval failure**
  - **Ranking failure**
  - **Context failure**
  - **Generation failure**
- Justify your classification by tying:
  - Observed answer flaws → likely pipeline stage issues.

If information is insufficient to be certain, make a **best-effort hypothesis** and clearly mark uncertainty.

### 7. Synthesize Score, Hallucination Assessment, and Improvement Strategy

- Assign:
  - **Per-dimension scores (F, R, C, I)**.
  - **Overall quality score (0–100)** using the specified weighting.
  - **Hallucination risk level**: Low / Medium / High with justification.
- Design a **three-layer improvement plan**:
  - **Prompt & Answer-level fixes**: how to rephrase or restructure the answer now.
  - **Retrieval / Context fixes**: how to adjust chunking, top-k, hybrid search, or reranking if retrieval is part of the problem.
  - **Evaluation & monitoring**: how to catch this failure mode systematically (e.g., golden queries, LLM-as-judge metrics).

---

## Output Format (For This Skill)

Always structure your final answer using this template (adapt wording, but keep all sections):

```markdown
## Problem Breakdown
- **User query / task**: [Short restatement]
- **Intended outcome**: [What a “good” answer should achieve]
- **Available context**: [What context was or was not available]

## Issues & Risk Analysis
- **Overall quality score (0–100)**: [Score] (F=[0–10], R=[0–10], C=[0–10], I=[0–10])
- **Hallucination risk**: [Low / Medium / High] – [Why]
- **Failure classification**: [Retrieval / Ranking / Context / Generation] – [Justification]
- **Key quality issues**:
  - [Issue 1 – brief description]
  - [Issue 2 – brief description]

## Structured Solution (Improved Answer Strategy)
- **Grounding improvements**:
  - [How to better tie claims to context or admit uncertainty]
- **Reasoning improvements**:
  - [How to fix logical gaps or misinterpretations]
- **Completeness & relevance improvements**:
  - [What to add/remove to better answer the question]
- **Format & instruction adherence**:
  - [Changes needed to align with required style/constraints]

## Optimizations (Cost, Latency, Robustness)
- **Token and cost optimizations**:
  - [Ways to reduce verbosity or redundant explanation without losing clarity]
- **Retrieval/context optimizations (if applicable)**:
  - [Adjustments to top-k, hybrid search, reranking, or chunking assumptions]
- **Evaluation and monitoring**:
  - [How to track this failure mode over time (e.g., golden queries, LLM-as-judge)]

## Risks & Edge Cases
- **Residual risks**:
  - [Cases where the improved answer may still struggle]
- **Edge cases to test**:
  - [Ambiguous, adversarial, or data-sparse scenarios to evaluate explicitly]
```

When proposing an improved answer, you may:

- Provide a **revised answer** inline, clearly marked.
- Or provide a **strategy plus outline** when full rewriting is outside scope.

---

## Relationship to Other Skills

When applying this skill, you may also:

- Use `rag-pipeline-auditor` to diagnose structural pipeline issues behind repeated answer failures.
- Use `retrieval-debugger` to analyze specific queries with suspect retrieval or ranking.
- Use `evaluation-designer` to build a systematic evaluation and regression framework for answer quality and hallucination rates.

This skill is meant for **per-answer quality analysis**; for system-wide evaluation, pair it with those complementary skills.

