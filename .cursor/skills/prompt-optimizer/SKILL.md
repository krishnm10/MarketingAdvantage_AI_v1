---
name: prompt-optimizer
description: Optimize LLM prompts for lower token usage, better reasoning, and reduced hallucination. Use when the user asks to improve, refactor, or audit a prompt, reduce token cost or latency, or increase answer reliability and faithfulness.
---

# Prompt Optimizer

This skill guides the agent to take an existing prompt and transform it into a cheaper, more reliable, and more precise version, while preserving the original task intent.

The agent should use this skill when:
- The user provides a prompt and asks to optimize, tighten, refactor, make cheaper, reduce tokens, or reduce hallucinations.
- The user reports hallucinations or unreliable behavior that appears prompt-related (not purely a retrieval or tool failure).
- The user is designing system prompts or templates for RAG, tools, or agent workflows and wants production-grade behavior.

---

## Required Inputs

Before optimizing a prompt, gather or infer:

- **Original prompt text** (system, developer, and user messages if available).
- **Target model family and context window** (e.g., "gpt-4.1, 8K" or "gpt-4.1, 128K") when known.
- **Primary task type** (classification, extraction, summarization, coding, RAG Q&A, multi-step agent, etc.).
- **Known failure modes**, if any (hallucinations, missing constraints, verbosity, refusals, off-topic answers).
- **For RAG prompts**: how retrieved context is inserted (format, delimiters, max chunks) and any citation/grounding requirements.

If some of this information is missing, make reasonable assumptions, but **call out each assumption explicitly** in the analysis.

---

## Optimization Goals (Priority Order)

When applying this skill, optimize in the following order:

1. **Correctness and faithfulness**: Preserve or improve factual accuracy and adherence to provided context or tools.
2. **Hallucination reduction**: Make grounding, allowed knowledge sources, and "I don't know" behavior explicit.
3. **Reasoning quality**: Improve clarity of reasoning and decomposition where appropriate for the task.
4. **Token and latency efficiency**: Reduce unnecessary tokens (boilerplate, repetition, overspecified examples).
5. **Maintainability**: Keep the prompt easy for humans to understand, audit, and modify.

If a trade-off arises (e.g., slightly more tokens for significantly higher faithfulness), prefer **accuracy and reliability** over minimal token count and explain the trade-off.

---

## Analysis Workflow

When you use this skill on a prompt, follow these steps:

1. **Map the current prompt**
   - Identify roles: system, developer, user messages, and any tool or RAG-specific instructions.
   - Segment the prompt into: goals, inputs, instructions, output format, constraints, examples.
2. **Identify issues and risks**
   - Redundant or repeated instructions that waste tokens.
   - Conflicting, vague, or underspecified requirements.
   - Missing grounding instructions for RAG (e.g., not telling the model to rely only on provided context).
   - Overly long or unnecessary few-shot examples, or examples that do not cover real edge cases.
   - Unbounded outputs (no structure, length, or format constraints).
3. **Design the improved prompt**
   - Enforce a clear section structure, for example:
     - Role & high-level goal
     - Inputs and assumptions
     - Instructions and constraints
     - Output format
   - Use concise, direct language and bullet lists where possible instead of long prose.
   - Make explicit:
     - What information sources are allowed or forbidden.
     - How to handle missing or insufficient information (e.g., say "I don't know" or ask for clarification).
     - For RAG: that the model must base answers only on provided context and cite sources or chunk IDs when feasible.
4. **Estimate token impact**
   - Compare original vs improved prompt length in words or approximate tokens.
   - Note structural changes that affect per-call token consumption (e.g., shorter reusable system prompt, fewer or shorter examples).
   - If the optimized prompt is slightly longer but significantly improves faithfulness, explain why this is an acceptable trade.
5. **Document expected behavioral changes**
   - Explain how the new prompt should change:
     - Faithfulness and hallucination rate.
     - Reasoning clarity and robustness to edge cases.
     - Cost and latency (per request and at scale).

---

## Output Format (Mandatory)

Always structure the final answer using the following sections and content:

```markdown
## Problem Breakdown
Briefly restate what the original prompt is trying to achieve, the target model (if known), and any key constraints (context window, RAG vs non-RAG, tools, etc.).

## Issues & Risk Analysis
- **Prompt design issues**: List concrete shortcomings (redundancy, ambiguity, missing constraints, unclear output format, etc.).
- **Hallucination / faithfulness risks**: Identify where the current prompt may allow or encourage unsupported claims.
- **Cost / latency risks**: Call out unnecessary verbosity, repeated instructions, or overly long examples.

## Structured Solution
- **Improved prompt**: Provide the final optimized prompt in a fenced `markdown` code block.
- **Design notes**: Briefly explain the main structural changes (sections, ordering, key constraints).

## Optimizations
- **Token savings estimate**: Roughly estimate token reduction (e.g., "≈30–40% fewer prompt tokens") and describe the main contributors.
- **Expected accuracy / reliability impact**: Qualitative estimate (e.g., "Higher faithfulness due to explicit grounding and refusal behavior").
- **Other improvements**: Note gains in readability, maintainability, and adaptability (e.g., easier to plug into different models).

## Risks & Edge Cases
- **Behavioral risks**: Situations where stricter constraints might cause refusals or overly conservative answers.
- **Context & RAG risks (if applicable)**: Potential context overflow, lost-in-the-middle issues, or retrieval failures that prompt changes alone cannot fix.
- **Follow-up recommendations**: Concrete suggestions for evaluation (e.g., A/B test on a golden set, log-based analysis) and future refinements.
```

These sections are required for every use of this skill. Keep each section concise and information-dense.

---

## Additional Guidelines

- **Be model-agnostic by default**, but call out assumptions when guidance depends on chat-style models with system messages or long context windows.
- **Do not hide retrieval or data quality problems behind prompt changes**:
  - If failures are likely due to retrieval, ranking, or context construction issues, state this explicitly and separate those from prompt-level fixes.
- For RAG systems:
  - Make grounding and citation behavior explicit.
  - Warn about **context overflow** and **lost-in-the-middle** problems when many chunks are concatenated.
  - Emphasize that prompt optimization cannot compensate for fundamentally broken chunking, embeddings, or retrieval.
- Prefer **minimal, reusable prompt templates** over one-off, highly specific prompts.
- When trimming examples:
  - Keep a **small number of high-signal examples** that cover edge cases.
  - Remove generic or redundant examples that only inflate tokens without improving behavior.
- When you tighten prompts, always ensure:
  - The optimized prompt still fully covers the user's business requirements.
  - Any removed section either did not carry unique constraints or has been merged into a more concise instruction.

