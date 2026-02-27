"""
================================================================================
Marketing Advantage AI — Chain-of-LLM Pipeline
File: app/core/llms/chain.py

PURPOSE:
  Compose multiple LLMs in a sequential processing pipeline.
  Each step receives the previous step's output as its input.
  RAG context is injected at the first step only.

REAL-WORLD USE CASES:
  ┌──────────────────────────────────────────────────────────┐
  │ PIPELINE 1 — Fast + Quality (3 steps)                    │
  │  Step 1: Groq/llama3 → Extract key entities from query   │
  │  Step 2: Ollama/mistral → Add Indian market context      │
  │  Step 3: OpenAI/gpt-4o → Polish final answer             │
  └──────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────┐
  │ PIPELINE 2 — Refinement Loop (2 steps)                   │
  │  Step 1: Ollama → Draft answer from context              │
  │  Step 2: Claude → Review + improve draft answer          │
  └──────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────┐
  │ PIPELINE 3 — Translation (2 steps)                       │
  │  Step 1: Gemini → Answer in English from context         │
  │  Step 2: Groq   → Translate to Hindi/Telugu/Tamil        │
  └──────────────────────────────────────────────────────────┘
================================================================================
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.core.llms.base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ChainStep
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChainStep:
    """
    Configuration for a single step in the LLMChain.

    Args:
        llm:           Any BaseLLM connector.
        system_prompt: Step-specific system instruction.
        temperature:   Override temperature for this step (optional).
        max_tokens:    Override max_tokens for this step (optional).
        label:         Optional human label for observability/logging.
    """
    llm:           BaseLLM
    system_prompt: Optional[str] = None
    temperature:   Optional[float] = None
    max_tokens:    Optional[int]   = None
    label:         str             = ""


# ─────────────────────────────────────────────────────────────────────────────
# ChainStepResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChainStepResult:
    """Captures output + metadata for one executed chain step."""
    step_index:       int
    label:            str
    model:            str
    provider:         str
    input_preview:    str        # first 200 chars of input (for debugging)
    output:           str
    prompt_tokens:    int
    completion_tokens:int
    latency_ms:       float


# ─────────────────────────────────────────────────────────────────────────────
# ChainResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChainResult:
    """
    Full result from an LLMChain execution.

    Attributes:
        final_answer:   The output of the last step — this is the response.
        steps:          Detailed metadata for every executed step.
        total_tokens:   Sum of all tokens used across all steps.
        total_latency_ms: Wall-clock time for the full chain.
    """
    final_answer:      str
    steps:             List[ChainStepResult]
    total_tokens:      int
    total_latency_ms:  float

    def __str__(self) -> str:
        return self.final_answer

    def summary(self) -> str:
        """Human-readable chain execution summary."""
        lines = [
            f"Chain: {len(self.steps)} steps | "
            f"{self.total_tokens} tokens | "
            f"{self.total_latency_ms:.0f}ms"
        ]
        for s in self.steps:
            lines.append(
                f"  Step {s.step_index}: [{s.label or s.model}] "
                f"→ {s.completion_tokens} tokens | {s.latency_ms:.0f}ms"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# LLMChain
# ─────────────────────────────────────────────────────────────────────────────

class LLMChain:
    """
    Sequential chain of LLMs.

    Each step:
      1. Receives the output of the previous step as its input prompt.
      2. Can have its own system_prompt, temperature, max_tokens.
      3. RAG context is injected only at step 0 (the first step).

    The chain is fully synchronous. Use asyncio.run() or FastAPI
    BackgroundTasks if you need async execution.

    Args:
        steps:            List of ChainStep (at least 1 required).
        default_temperature: Fallback temperature if step has no override.
        default_max_tokens:  Fallback max_tokens if step has no override.
    """

    def __init__(
        self,
        steps: List[ChainStep],
        *,
        default_temperature: float = 0.3,
        default_max_tokens: int = 1024,
    ):
        if not steps:
            raise ValueError("[LLMChain] At least one ChainStep is required.")

        self.steps = steps
        self._default_temperature = float(default_temperature)
        self._default_max_tokens  = int(default_max_tokens)

        models = [s.llm.info.model for s in steps]
        logger.info(
            "[LLMChain] Initialized | %d steps: %s",
            len(steps), " → ".join(models)
        )

    def run(
        self,
        user_query: str,
        *,
        context: Optional[str] = None,
        metadata_filters: Optional[Dict] = None,
    ) -> ChainResult:
        """
        Execute the full chain.

        Args:
            user_query:       The user's original query (input to step 0).
            context:          RAG-retrieved context string (injected at step 0 only).
            metadata_filters: Optional metadata (passed through for observability).

        Returns:
            ChainResult with final_answer, step-by-step details, and totals.
        """
        chain_start = time.perf_counter()

        current_input = user_query
        step_results:  List[ChainStepResult] = []
        total_tokens = 0

        for idx, step in enumerate(self.steps):
            step_start = time.perf_counter()

            # ── Build prompt ────────────────────────────────────────────
            # RAG context injected ONLY at step 0
            if idx == 0 and context:
                prompt = self._build_rag_prompt(
                    query=current_input,
                    context=context,
                )
            else:
                # Steps 1+ receive previous step's output as-is
                prompt = current_input

            # ── Resolve per-step overrides ────────────────────────────
            temperature = (
                step.temperature
                if step.temperature is not None
                else self._default_temperature
            )
            max_tokens = (
                step.max_tokens
                if step.max_tokens is not None
                else self._default_max_tokens
            )
            label = step.label or f"step_{idx + 1}"

            logger.info(
                "[LLMChain] Step %d/%d | %s | model=%s | temp=%.2f | max_tokens=%d",
                idx + 1, len(self.steps), label,
                step.llm.info.model, temperature, max_tokens,
            )

            # ── LLM call ──────────────────────────────────────────────
            response: LLMResponse = step.llm.generate(
                prompt,
                system_prompt=step.system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            step_ms = (time.perf_counter() - step_start) * 1000
            total_tokens += response.total_tokens

            # ── Capture step result ───────────────────────────────────
            step_results.append(
                ChainStepResult(
                    step_index=idx + 1,
                    label=label,
                    model=step.llm.info.model,
                    provider=step.llm.info.provider,
                    input_preview=prompt[:200],
                    output=response.text,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    latency_ms=round(step_ms, 2),
                )
            )

            # ── Feed output to next step ─────────────────────────────
            current_input = response.text

            logger.debug(
                "[LLMChain] Step %d complete | latency=%.0fms | tokens=%d",
                idx + 1, step_ms, response.total_tokens,
            )

        chain_ms = (time.perf_counter() - chain_start) * 1000
        result = ChainResult(
            final_answer=current_input,
            steps=step_results,
            total_tokens=total_tokens,
            total_latency_ms=round(chain_ms, 2),
        )

        logger.info(
            "[LLMChain] Complete | %s",
            result.summary().replace("\n", " | "),
        )

        return result

    @staticmethod
    def _build_rag_prompt(query: str, context: str) -> str:
        """Build the initial RAG-augmented prompt for step 0."""
        return (
            "Use the following retrieved context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

    # ── Convenience property ─────────────────────────────────────────────

    @property
    def model_names(self) -> List[str]:
        return [s.llm.info.model for s in self.steps]
