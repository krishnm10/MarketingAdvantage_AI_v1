"""
SharedGenerationExecutor — Unified generation orchestration layer.

Extracted from RAGPipeline to serve as the SINGLE generation execution path
for both RAGPipeline and (eventually) RetrievalRuntime. Eliminates duplicated
generation, security, formatting, and trust logic across the two stacks.

Gated behind: ENABLE_SHARED_GENERATION

Responsibilities:
  1. Context window management (CWM budget enforcement)
  2. Prompt assembly (context building, prompt node rendering)
  3. Pre-LLM security (PII scan on assembled context)
  4. LLM invocation (single LLM or LLMChain)
  5. Post-LLM security (PII scan on generated answer)
  6. Trust scoring (confidence gate)
  7. Output formatting & trust enforcement (format, block low-trust, F-10)

Design:
  - Stateless per-request — all state via parameters
  - Component-agnostic — base contracts only
  - Config-driven — all parameters from ClientConfig
  - Tenant-aware — tenant_id for telemetry and audit
  - Error-normalized — raises GenerationError on LLM failures
  - Preserves EXACT prompt, output, and safety behavior from RAGPipeline
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from app.core.llms.base import BaseLLM, LLMResponse
from app.core.llms.chain import LLMChain, ChainResult
from app.core.config.client_config_schema import ClientConfig
from app.core.runtime.runtime_context import RAGRuntimeContext
from app.core.runtime.runtime_telemetry import emit_runtime_event
from app.core.runtime.errors import GenerationError

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """Output of the full generation execution (Steps 3.6–7 of RAGPipeline)."""

    final_answer: Optional[str]
    trust_score: Optional[float]
    latency: Dict[str, float]

    pii_redacted: bool = False
    pii_entities_found: List[str] = field(default_factory=list)

    formatter_applied: bool = False
    response_blocked: bool = False
    block_reason: Optional[str] = None

    context_window_applied: bool = False
    context_window_chunks_dropped: int = 0

    context_chunks: List[Dict[str, Any]] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Executor
# ─────────────────────────────────────────────────────────────────────────────

class SharedGenerationExecutor:
    """
    Unified generation executor extracted from RAGPipeline.

    Encapsulates Steps 3.6–7 (CWM → context → pre-PII → prompt → LLM →
    post-PII → trust → formatter). RAGPipeline delegates to this class;
    later, RetrievalRuntime will too.
    """

    def __init__(
        self,
        *,
        llm: Optional[Union[BaseLLM, LLMChain]],
        config: ClientConfig,
        nodes: Optional[Any] = None,
    ) -> None:
        self._llm = llm
        self._config = config
        self._nodes = nodes

        self._llm_name = (
            f"chain({','.join(llm.model_names)})"
            if isinstance(llm, LLMChain)
            else llm.info.model
            if llm else "none"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────

    def execute(
        self,
        *,
        user_query: str,
        context_chunks: List[Dict[str, Any]],
        tenant_id: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        pii_redacted: bool = False,
        pii_entities_found: Optional[List[str]] = None,
        runtime_context: Optional[RAGRuntimeContext] = None,
    ) -> GenerationResult:
        """
        Execute the full generation pipeline (Steps 3.6–7).

        Preserves EXACT behavior from RAGPipeline.query() inline code.
        """
        latency: Dict[str, float] = {}
        _pii_redacted = pii_redacted
        _pii_entities: List[str] = list(pii_entities_found or [])
        working_chunks = list(context_chunks)

        # ── Step 3.6: Context Window Manager ─────────────────────────
        _cw_chunks_dropped = 0
        latency["context_window_ms"] = 0.0

        cwm = (
            self._nodes.context_window
            if self._nodes and self._nodes.has_context_window()
            else None
        )
        if cwm is not None and self._config.context_window.enabled:
            t0_cw = time.perf_counter()
            original_count = len(working_chunks)
            original_token_est = sum(
                len(c.get("text", "")) // 4 for c in working_chunks
            )

            _sys_prompt = (
                system_prompt
                or (self._config.llm.single.system_prompt
                    if self._config.llm and self._config.llm.single else None)
            )
            _sys_prompt_tokens = len(_sys_prompt) // 4 if _sys_prompt else 0

            _max_ctx = (
                self._config.llm.single.max_tokens * 4
                if self._config.llm and self._config.llm.single
                else 4096
            )

            try:
                cw_result = cwm.apply_budget(
                    working_chunks,
                    max_context_tokens=_max_ctx,
                    system_prompt_tokens=_sys_prompt_tokens,
                )
                trimmed = cw_result["trimmed_chunks"]
                trimmed_token_est = cw_result["total_tokens"]
                chunks_dropped = cw_result["chunks_dropped"]

                if chunks_dropped > 0:
                    working_chunks = trimmed
                    _cw_chunks_dropped = chunks_dropped
                    logger.info(
                        '{"event":"CONTEXT_WINDOW_APPLIED",'
                        '"client_id":"%s",'
                        '"original_chunks":%d,"final_chunks":%d,'
                        '"chunks_dropped":%d,'
                        '"original_tokens_est":%d,"final_tokens":%d,'
                        '"budget_tokens":%d,'
                        '"strategy":"%s",'
                        '"latency_ms":%.2f}',
                        self._config.client_id,
                        original_count, len(trimmed),
                        chunks_dropped,
                        original_token_est, trimmed_token_est,
                        cw_result["budget_tokens"],
                        cw_result["strategy_used"],
                        cw_result["latency_ms"],
                    )
                else:
                    logger.debug(
                        "[RAGPipeline] Context window: all %d chunks within "
                        "budget (%d/%d tokens) — no trimming needed.",
                        original_count, original_token_est,
                        cw_result["budget_tokens"],
                    )
            except Exception as _cw_exc:
                logger.warning(
                    '{"event":"CONTEXT_WINDOW_FAILED",'
                    '"client_id":"%s",'
                    '"error":"%s",'
                    '"fallback":"original_chunks"}',
                    self._config.client_id,
                    str(_cw_exc).replace('"', "'"),
                )

            latency["context_window_ms"] = round(
                (time.perf_counter() - t0_cw) * 1000, 2
            )

        # ── Step 4: Build context string ─────────────────────────────
        context_str = self._build_context(working_chunks)

        # ── Pre-LLM PII scan on context ──────────────────────────────
        if self._nodes and self._nodes.has_pii_middleware():
            pii_mw = self._nodes.pii_middleware
            if "pre_llm" in getattr(pii_mw, '_positions', []):
                t0_pii = time.perf_counter()
                try:
                    ctx_scan = pii_mw.scan_text(context_str, position="pre_llm")
                    if getattr(ctx_scan, "entities_found", []):
                        _pii_redacted = True
                        _pii_entities.extend(ctx_scan.entities_found)
                        context_str = getattr(ctx_scan, "redacted_text", context_str)
                except Exception as e:
                    logger.warning("[RAGPipeline] Pre-LLM PII scan failed: %s", e)
                latency["pii_ms"] = latency.get("pii_ms", 0.0) + round(
                    (time.perf_counter() - t0_pii) * 1000, 2
                )

        # ── Prompt Node rendering ────────────────────────────────────
        if self._nodes and getattr(self._nodes, 'prompt_node', None):
            t0_prompt = time.perf_counter()
            try:
                prompt_result = self._nodes.prompt_node.render(
                    query=user_query,
                    context=context_str,
                    system_prompt=(
                        system_prompt or (
                            self._config.llm.single.system_prompt
                            if self._config.llm and self._config.llm.single
                            else None
                        )
                    ),
                )
                if prompt_result.get("rendered_prompt"):
                    context_str = prompt_result["rendered_prompt"]
                    latency["prompt_ms"] = round(
                        (time.perf_counter() - t0_prompt) * 1000, 2
                    )
            except Exception as e:
                logger.warning(
                    "[RAGPipeline] Prompt node failed, using default: %s", e
                )
                latency["prompt_ms"] = round(
                    (time.perf_counter() - t0_prompt) * 1000, 2
                )

        # ── Step 5: LLM generation ──────────────────────────────────
        final_answer: Optional[str] = None
        latency["llm_ms"] = 0.0

        if self._llm is not None:
            t0 = time.perf_counter()

            llm_temperature = temperature or (
                self._config.llm.single.temperature
                if self._config.llm and self._config.llm.single
                else 0.3
            )
            llm_max_tokens = max_tokens or (
                self._config.llm.single.max_tokens
                if self._config.llm and self._config.llm.single
                else 1024
            )
            llm_system_prompt = system_prompt or (
                self._config.llm.single.system_prompt
                if self._config.llm and self._config.llm.single
                else None
            )

            if isinstance(self._llm, LLMChain):
                chain_result: ChainResult = self._llm.run(
                    user_query=user_query,
                    context=context_str,
                )
                final_answer = chain_result.final_answer
                latency["llm_chain_steps"] = {
                    f"step_{s.step_index}_{s.model}": s.latency_ms
                    for s in chain_result.steps
                }
                logger.info(
                    "[RAGPipeline] LLMChain complete | %s",
                    chain_result.summary(),
                )
            else:
                prompt = self._build_rag_prompt(
                    query=user_query,
                    context=context_str,
                    system_prompt=llm_system_prompt,
                )
                try:
                    llm_response = self._llm.generate(
                        prompt,
                        system_prompt=llm_system_prompt,
                        temperature=llm_temperature,
                        max_tokens=llm_max_tokens,
                    )
                except GenerationError:
                    raise
                except Exception as exc:
                    raise GenerationError(
                        f"LLM generation failed: {exc}",
                        tenant_id=tenant_id,
                        pipeline_id=self._config.client_id,
                        details={"llm": self._llm_name, "operation": "generate"},
                    ) from exc
                final_answer = llm_response.text
                emit_runtime_event(
                    "LLM_GENERATION_COMPLETE",
                    tenant_id=tenant_id,
                    client_id=self._config.client_id,
                    llm_model=llm_response.model,
                    token_usage={"total_tokens": llm_response.total_tokens},
                )

            latency["llm_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ── Step 5.5: Post-LLM PII enforcement ──────────────────────
        latency["pii_post_llm_ms"] = 0.0

        if (
            final_answer is not None
            and self._nodes
            and self._nodes.has_pii_middleware()
        ):
            pii_mw = self._nodes.pii_middleware
            if "post_llm" in getattr(pii_mw, "_positions", []):
                t0_pii_post = time.perf_counter()
                try:
                    post_scan = pii_mw.scan_text(
                        final_answer, position="post_llm",
                    )
                    _post_entities = getattr(post_scan, "entities_found", [])

                    if getattr(post_scan, "blocked", False):
                        _pii_redacted = True
                        _pii_entities.extend(_post_entities)
                        final_answer = (
                            "[BLOCKED] Response blocked by post-LLM PII policy."
                        )
                        logger.warning(
                            '{"event":"POST_LLM_PII_BLOCKED",'
                            '"client_id":"%s",'
                            '"entities":%s,'
                            '"severity_max":"%s"}',
                            self._config.client_id,
                            [e for e in _post_entities],
                            getattr(post_scan, "severity_max", "unknown"),
                        )
                    elif _post_entities:
                        _pii_redacted = True
                        _pii_entities.extend(_post_entities)
                        final_answer = getattr(
                            post_scan, "redacted_text", final_answer,
                        )
                        logger.info(
                            '{"event":"POST_LLM_PII_REDACTED",'
                            '"client_id":"%s",'
                            '"entities":%s,'
                            '"action":"%s",'
                            '"severity_max":"%s"}',
                            self._config.client_id,
                            [e for e in _post_entities],
                            getattr(post_scan, "action_taken", "REDACT"),
                            getattr(post_scan, "severity_max", "unknown"),
                        )
                    else:
                        logger.debug(
                            "[RAGPipeline] Post-LLM PII scan clean | "
                            "client=%s | latency=%.2fms",
                            self._config.client_id,
                            getattr(post_scan, "latency_ms", 0.0),
                        )
                except Exception as _pii_post_exc:
                    logger.warning(
                        '{"event":"POST_LLM_PII_FAILED",'
                        '"client_id":"%s",'
                        '"error":"%s",'
                        '"fallback":"raw_answer"}',
                        self._config.client_id,
                        str(_pii_post_exc).replace('"', "'"),
                    )

                latency["pii_post_llm_ms"] = round(
                    (time.perf_counter() - t0_pii_post) * 1000, 2
                )
                latency["pii_ms"] = (
                    latency.get("pii_ms", 0.0) + latency["pii_post_llm_ms"]
                )

        # ── Step 6: Trust scoring ────────────────────────────────────
        trust_score: Optional[float] = None
        latency["trust_ms"] = 0.0

        if self._config.retrieval.enable_trust_scoring and working_chunks:
            t0 = time.perf_counter()
            from app.core.trust_adapter import TrustAdapter
            trust_score = TrustAdapter().calculate(
                working_chunks, pii_redacted=_pii_redacted,
            )
            latency["trust_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ── Step 7: Output formatting & trust gate ───────────────────
        _fmt_applied = False
        _response_blocked = False
        _block_reason: Optional[str] = None
        latency["formatter_ms"] = 0.0

        ofmt = (
            self._nodes.output_formatter
            if self._nodes and self._nodes.has_output_formatter()
            else None
        )
        if ofmt is not None and self._config.formatter.enabled and final_answer is not None:
            t0_fmt = time.perf_counter()
            try:
                fmt_result = ofmt.format_output(
                    final_answer,
                    pii_redacted=_pii_redacted,
                    pii_entities_found=_pii_entities or None,
                    trust_score=trust_score,
                )
                _fmt_applied = True
                _response_blocked = fmt_result.get("blocked", False)
                _block_reason = fmt_result.get("block_reason")
                final_answer = fmt_result["formatted_output"]

                if _response_blocked:
                    logger.warning(
                        '{"event":"OUTPUT_BLOCKED",'
                        '"client_id":"%s",'
                        '"block_reason":"%s",'
                        '"trust_score":%s,'
                        '"pii_redacted":%s}',
                        self._config.client_id,
                        (_block_reason or "").replace('"', "'"),
                        f"{trust_score:.4f}" if trust_score is not None else "null",
                        str(_pii_redacted).lower(),
                    )
                else:
                    logger.info(
                        '{"event":"OUTPUT_FORMATTED",'
                        '"client_id":"%s",'
                        '"format":"%s",'
                        '"trust_gate_passed":true,'
                        '"trust_score":%s,'
                        '"latency_ms":%.2f}',
                        self._config.client_id,
                        fmt_result.get("format_type", "unknown"),
                        f"{trust_score:.4f}" if trust_score is not None else "null",
                        fmt_result.get("latency_ms", 0.0),
                    )
            except Exception as _fmt_exc:
                # F-10: Even when the formatter crashes, enforce the trust
                # gate manually so low-trust responses can't escape.
                _fmt_cfg = self._config.formatter
                if (
                    _fmt_cfg.block_on_low_trust
                    and trust_score is not None
                    and trust_score < _fmt_cfg.min_trust_score
                ):
                    _response_blocked = True
                    _block_reason = (
                        f"trust_score {trust_score:.4f} < "
                        f"min_trust_score {_fmt_cfg.min_trust_score:.4f} "
                        f"(enforced in formatter fallback)"
                    )
                    final_answer = (
                        "[BLOCKED] Response blocked: trust score below threshold."
                    )
                    logger.warning(
                        '{"event":"OUTPUT_FORMATTER_FAILED_TRUST_BLOCKED",'
                        '"client_id":"%s",'
                        '"error":"%s",'
                        '"trust_score":%.4f,'
                        '"min_trust_score":%.4f,'
                        '"action":"blocked"}',
                        self._config.client_id,
                        str(_fmt_exc).replace('"', "'"),
                        trust_score,
                        _fmt_cfg.min_trust_score,
                    )
                else:
                    logger.warning(
                        '{"event":"OUTPUT_FORMATTER_FAILED",'
                        '"client_id":"%s",'
                        '"error":"%s",'
                        '"fallback":"raw_answer",'
                        '"trust_gate_enforced":false}',
                        self._config.client_id,
                        str(_fmt_exc).replace('"', "'"),
                    )

            latency["formatter_ms"] = round(
                (time.perf_counter() - t0_fmt) * 1000, 2
            )

        return GenerationResult(
            final_answer=final_answer,
            trust_score=trust_score,
            latency=latency,
            pii_redacted=_pii_redacted,
            pii_entities_found=list(set(_pii_entities)),
            formatter_applied=_fmt_applied,
            response_blocked=_response_blocked,
            block_reason=_block_reason,
            context_window_applied=_cw_chunks_dropped > 0,
            context_window_chunks_dropped=_cw_chunks_dropped,
            context_chunks=working_chunks,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Helpers — exact copies from RAGPipeline
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_context(chunks: List[Dict[str, Any]]) -> str:
        """
        Assemble retrieved chunks into a clean numbered context string
        ready to be injected into the LLM prompt.

        EXACT copy of RAGPipeline._build_context.
        """
        if not chunks:
            return "No relevant context found."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "").strip()
            meta = chunk.get("metadata", {})

            citation_parts = []
            source = meta.get("source") or meta.get("file_name") or meta.get("url")
            if source:
                citation_parts.append(f"Source: {source}")
            page_number = meta.get("page_number")
            if isinstance(page_number, int) and page_number > 0:
                citation_parts.append(f"Page: {page_number}")
            section_title = meta.get("section_title")
            if section_title:
                citation_parts.append(f"Section: {section_title}")
            source_line = f"  [{' | '.join(citation_parts)}]" if citation_parts else ""

            parts.append(f"[{i}] {text}{source_line}")

        return "\n\n".join(parts)

    @staticmethod
    def _build_rag_prompt(
        query: str, context: str, system_prompt: Optional[str] = None
    ) -> str:
        """
        Build the final RAG prompt sent to the LLM.

        EXACT copy of RAGPipeline._build_rag_prompt.
        """
        default_system = (
            "You are a Marketing Intelligence Assistant for enterprise businesses.\n"
            "Answer the question using ONLY the context provided below.\n"
            "If the context does not contain enough information, say so clearly.\n"
            "Cite the source number [1], [2], etc. when referencing specific facts."
        )
        prompt_prefix = system_prompt or default_system
        return (
            f"{prompt_prefix}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
