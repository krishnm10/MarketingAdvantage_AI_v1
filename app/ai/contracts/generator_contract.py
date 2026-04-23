# =============================================================================
# app/ai/contracts/generator_contract.py
#
# GeneratorContract — Phase 2, Module 3
#
# Defines the canonical interface for LLM text generation in the RAG pipeline.
# This sits above ``app/core/llms/base.py`` (the plugin-level contract) and
# adds RAG-specific semantics: structured context injection, prompt templates,
# citation tracking, and adaptive token budgeting.
#
# Architecture:
#   RerankerContract (retrieve/score)
#        ↓  ScoredCandidate list
#   GeneratorContract (generate)
#        ↓  GenerationResult
#
# Design rules:
#   - The generator owns context assembly and prompt construction;
#     callers supply raw ScoredCandidates, not pre-formatted strings.
#   - Token budget is adaptive: compute actual context tokens, then set
#     max_new_tokens to stay within the model's context window.
#   - Temperature and max_tokens are per-request overrides — never global
#     constants baked into the generator implementation.
#   - System prompts may be supplied via a PromptTemplate or inline string;
#     they are never hard-coded inside the generator.
# =============================================================================

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from app.ai.contracts.reranker_contract import ScoredCandidate


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class GeneratorProvider(str, Enum):
    """Canonical LLM provider identifiers for generation."""
    OPENAI    = "openai"
    ANTHROPIC = "anthropic"
    GROQ      = "groq"
    GEMINI    = "gemini"
    OLLAMA    = "ollama"
    CUSTOM    = "custom"


class CitationStyle(str, Enum):
    """
    How the generator should reference source chunks in its answer.

    INLINE_NUMERIC  — ``[1]``, ``[2]`` inline superscripts (default).
    INLINE_SOURCE   — ``(Source: <filename>)`` after each claim.
    NONE            — No citation markers in the answer.
    """
    INLINE_NUMERIC = "inline_numeric"
    INLINE_SOURCE  = "inline_source"
    NONE           = "none"


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

@dataclass
class PromptTemplate:
    """
    A reusable, parameterised prompt template for RAG generation.

    Template variables (replaced at render time):
      {{SYSTEM_INSTRUCTIONS}} — injected from system_instructions field
      {{CONTEXT}}             — formatted retrieved chunks
      {{QUESTION}}            — user query
      {{CITATION_STYLE}}      — citation format instruction

    Attributes
    ----------
    template_id
        Unique identifier for this template (used in API and UI).
    name
        Human-readable display name.
    system_instructions
        Instructions defining the assistant persona, grounding behaviour,
        and citation policy.  Should NOT contain the actual question or context
        (those are injected at runtime).
    context_format
        Format string for each context chunk.
        Supports ``{index}``, ``{text}``, ``{source}``, ``{page}``.
    question_prefix
        Label placed before the user question (e.g. ``"Question:"``, ``"User:"``)
    answer_prefix
        Label placed after the question to prime the answer
        (e.g. ``"Answer:"``, ``"Assistant:"``)
    citation_style
        Citation annotation strategy.
    max_context_chars
        Hard cap on total context characters before context truncation.
        Prevents prompt overflow for long-context models.  ``None`` = no cap
        (token budgeting is then enforced at the token level).
    version
        Semantic version string for change tracking.
    tags
        Searchable tags for the prompt library UI.
    """
    template_id:       str
    name:              str
    system_instructions: str
    context_format:    str             = "[{index}] {text}\n  [Source: {source}]"
    question_prefix:   str             = "Question:"
    answer_prefix:     str             = "Answer:"
    citation_style:    CitationStyle   = CitationStyle.INLINE_NUMERIC
    max_context_chars: Optional[int]   = None
    version:           str             = "1.0.0"
    tags:              List[str]       = field(default_factory=list)
    metadata:          Dict[str, Any]  = field(default_factory=dict)

    def render_system_prompt(self) -> str:
        """Return the system instructions string, ready to pass to the LLM."""
        return self.system_instructions.strip()

    def render_context(self, chunks: List[ScoredCandidate]) -> str:
        """
        Assemble context from scored candidates using the declared format.
        Applies max_context_chars truncation if set.
        """
        parts: List[str] = []
        total_chars = 0

        for i, chunk in enumerate(chunks, 1):
            source = (
                chunk.metadata.get("source")
                or chunk.metadata.get("file_name")
                or chunk.metadata.get("url")
                or "unknown"
            )
            page = chunk.metadata.get("page_number", "")
            page_str = f", p.{page}" if page else ""
            source_label = f"{source}{page_str}"

            formatted = self.context_format.format(
                index=i,
                text=chunk.text.strip(),
                source=source_label,
                page=str(page),
            )

            if self.max_context_chars is not None:
                if total_chars + len(formatted) > self.max_context_chars:
                    break
                total_chars += len(formatted)

            parts.append(formatted)

        return "\n\n".join(parts) if parts else "No relevant context found."

    def render_full_prompt(
        self,
        query: str,
        chunks: List[ScoredCandidate],
    ) -> str:
        """
        Assemble the complete user-turn prompt (context + question).
        System instructions are passed separately to LLMs that support
        a system role; for single-turn APIs this is concatenated.
        """
        context = self.render_context(chunks)
        return (
            f"Context:\n{context}\n\n"
            f"{self.question_prefix} {query}\n\n"
            f"{self.answer_prefix}"
        )


# ---------------------------------------------------------------------------
# Generation request / result
# ---------------------------------------------------------------------------

@dataclass
class GenerationRequest:
    """
    Fully-specified generation request for a RAG turn.

    Attributes
    ----------
    query
        Raw user query (sanitised by security middleware before this point).
    context_chunks
        Reranked and threshold-gated candidates to use as context.
    prompt_template
        Template governing system instructions and context formatting.
        ``None`` uses the generator's built-in default.
    temperature
        Sampling temperature (0.0 = deterministic).  Generator should
        apply this per-request; never use a fixed global default.
    max_new_tokens
        Maximum tokens to generate.  Should be set adaptively based on
        remaining budget after context token count.  ``None`` defers to
        the generator's configured default.
    stop_sequences
        Optional list of stop tokens.
    correlation_id
        Trace/request ID for log correlation across pipeline stages.
    """
    query:           str
    context_chunks:  List[ScoredCandidate]
    prompt_template: Optional[PromptTemplate] = None
    temperature:     float                    = 0.3
    max_new_tokens:  Optional[int]            = None
    stop_sequences:  List[str]                = field(default_factory=list)
    correlation_id:  Optional[str]            = None


@dataclass
class GenerationResult:
    """
    Structured response from a single generation call.

    Attributes
    ----------
    answer
        Final generated answer text.
    prompt_tokens
        Tokens consumed by the prompt (context + system + question).
    completion_tokens
        Tokens in the generated answer.
    finish_reason
        Why generation stopped: ``"stop"``, ``"length"``, ``"content_filter"``.
    model
        Exact model identifier used for this call.
    latency_ms
        Wall-clock generation time in milliseconds.
    metadata
        Raw provider response fields for debugging.
    """
    answer:            str
    prompt_tokens:     int
    completion_tokens: int
    finish_reason:     str
    model:             str
    latency_ms:        float                  = 0.0
    metadata:          Dict[str, Any]         = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ---------------------------------------------------------------------------
# GeneratorContract ABC
# ---------------------------------------------------------------------------

class GeneratorContract(abc.ABC):
    """
    Canonical interface every Phase 1 LLM generator must satisfy.

    Contract invariants
    -------------------
    I-G1  ``generate()`` never raises on provider errors; it returns a
          ``GenerationResult`` with ``finish_reason="error"`` and the error
          message in ``answer`` when retries are exhausted.
    I-G2  ``generate()`` always respects ``request.max_new_tokens`` when set;
          it must never exceed it.
    I-G3  API keys are read from environment variables at construction time;
          they must never appear in logs or the ``GenerationResult.metadata``.
    I-G4  Temperature 0.0 results in deterministic outputs (or the closest
          approximation the provider supports).
    """

    @property
    @abc.abstractmethod
    def provider(self) -> GeneratorProvider:
        """Provider identifier."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def model_id(self) -> str:
        """Canonical model name (e.g. ``"gpt-4o"``)."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def max_context_tokens(self) -> int:
        """
        Total context window of this model in tokens.
        Used by adaptive token budgeting to compute safe max_new_tokens.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate a grounded answer for the given RAG request.

        The implementation is responsible for:
        1. Rendering the prompt from ``request.prompt_template``
           (or a built-in default).
        2. Calling the underlying LLM with ``temperature`` and token limits.
        3. Returning a fully-populated ``GenerationResult``.
        """
        raise NotImplementedError

    def health_check(self) -> bool:
        """Verify the underlying LLM API is reachable."""
        return True
