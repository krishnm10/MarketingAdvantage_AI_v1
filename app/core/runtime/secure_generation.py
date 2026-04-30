"""
Secure Generation Handler for RetrievalRuntime — Phase 2A.5

Closes the critical security gap (CB-06) identified in the runtime convergence audit:
When generate_answer=true in RetrievalRuntime, the generation flow must have:
- Pre-LLM PII scanning on context
- Post-LLM PII scanning on answer
- Trust scoring
- Trust gating
- Output formatting protections

This module provides enterprise-grade security alignment with RAGPipeline
WITHOUT modifying API contracts or retrieval ranking behavior.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SecureGenerationResult:
    """Result of secure LLM generation with full security metadata."""
    
    answer: Optional[str] = None
    answer_model: Optional[str] = None
    answer_latency_ms: Optional[float] = None
    answer_error: Optional[str] = None
    
    # Security metadata
    security_aligned: bool = False
    pre_llm_pii_scan_applied: bool = False
    pre_llm_pii_entities: List[str] = field(default_factory=list)
    pre_llm_context_blocked: bool = False
    post_llm_pii_scan_applied: bool = False
    post_llm_pii_entities: List[str] = field(default_factory=list)
    post_llm_blocked: bool = False
    trust_score: Optional[float] = None
    trust_gate_applied: bool = False
    trust_gate_passed: bool = True
    output_formatter_applied: bool = False
    output_blocked: bool = False
    block_reason: Optional[str] = None
    
    # Latency breakdown
    latency_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class SecureGenerationConfig:
    """Configuration for secure generation."""
    
    # PII settings
    enable_pre_llm_pii_scan: bool = True
    enable_post_llm_pii_scan: bool = True
    block_on_critical_pii: bool = True
    pii_action: str = "REDACT"
    
    # Trust settings
    enable_trust_scoring: bool = True
    enable_trust_gate: bool = True
    min_trust_score: float = 0.3
    pii_trust_penalty: float = 0.15
    
    # Output formatting
    enable_output_formatter: bool = True
    response_format: str = "plain_text"
    strip_boilerplate: bool = False
    toxicity_filter: str = "disabled"
    
    @classmethod
    def from_runtime_components(cls, rc: Any) -> "SecureGenerationConfig":
        """Build config from RuntimeComponents."""
        return cls(
            min_trust_score=getattr(rc, "rag_min_score", 0.3),
        )


class SecureGenerationHandler:
    """
    Handles secure LLM generation for RetrievalRuntime.
    
    Provides security parity with RAGPipeline's generation path:
    - Pre-LLM PII scanning and redaction
    - Post-LLM PII scanning
    - Trust scoring using TrustAdapter
    - Trust gating via OutputFormatter
    - Structured telemetry for all security events
    """
    
    def __init__(
        self,
        config: Optional[SecureGenerationConfig] = None,
        tenant_id: str = "default",
        request_id: Optional[str] = None,
    ):
        self._config = config or SecureGenerationConfig()
        self._tenant_id = tenant_id
        self._request_id = request_id
        
        self._pii_middleware = None
        self._trust_adapter = None
        self._output_formatter = None
    
    def _get_pii_middleware(self):
        """Lazy-load PII middleware."""
        if self._pii_middleware is None:
            from app.core.pipeline_nodes.pii_middleware import RegexPIIMiddleware
            self._pii_middleware = RegexPIIMiddleware({
                "position": ["pre_llm", "post_llm"],
                "action": self._config.pii_action,
                "block_on_severity": "CRITICAL" if self._config.block_on_critical_pii else None,
            })
        return self._pii_middleware
    
    def _get_trust_adapter(self):
        """Lazy-load trust adapter."""
        if self._trust_adapter is None:
            from app.core.trust_adapter import TrustAdapter
            self._trust_adapter = TrustAdapter()
        return self._trust_adapter
    
    def _get_output_formatter(self):
        """Lazy-load output formatter."""
        if self._output_formatter is None:
            from app.core.pipeline_nodes.output_formatter import OutputFormatter
            self._output_formatter = OutputFormatter(
                response_format=self._config.response_format,
                min_trust_score=self._config.min_trust_score,
                block_on_low_trust=self._config.enable_trust_gate,
                strip_boilerplate=self._config.strip_boilerplate,
                toxicity_filter=self._config.toxicity_filter,
            )
        return self._output_formatter
    
    def generate_secure(
        self,
        llm: Any,
        llm_model: str,
        query: str,
        context_chunks: List[Dict[str, Any]],
        *,
        max_tokens: int = 700,
        temperature: float = 0.0,
    ) -> SecureGenerationResult:
        """
        Execute secure LLM generation with full security pipeline.
        
        Args:
            llm: Instantiated LLM instance
            llm_model: Model name for telemetry
            query: User query
            context_chunks: Retrieved chunks (list of dicts with 'text', 'score', etc.)
            max_tokens: Max tokens for LLM response
            temperature: LLM temperature
        
        Returns:
            SecureGenerationResult with answer and security metadata
        """
        result = SecureGenerationResult(
            answer_model=llm_model,
            security_aligned=True,
        )
        latency = {}
        total_start = time.perf_counter()
        
        try:
            # ── Step 1: Build context string ──────────────────────────────
            context_parts = [
                f"[Source {i}] (relevance={chunk.get('score', 0):.3f})\n{chunk.get('text', '').strip()}"
                for i, chunk in enumerate(context_chunks, start=1)
            ]
            context_str = "\n\n---\n\n".join(context_parts)
            
            # ── Step 2: Pre-LLM PII Scan ──────────────────────────────────
            if self._config.enable_pre_llm_pii_scan:
                t0 = time.perf_counter()
                try:
                    pii_mw = self._get_pii_middleware()
                    pre_scan = pii_mw.scan_text(context_str, position="pre_llm")
                    
                    result.pre_llm_pii_scan_applied = True
                    result.pre_llm_pii_entities = pre_scan.entities_found.copy()
                    
                    if pre_scan.blocked:
                        result.pre_llm_context_blocked = True
                        result.answer_error = (
                            "Context blocked: Critical PII detected in retrieved documents."
                        )
                        result.block_reason = "pre_llm_pii_blocked"
                        latency["pre_llm_pii_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                        self._log_security_event(
                            "PRE_LLM_PII_BLOCKED",
                            entities=pre_scan.entities_found,
                            severity=str(pre_scan.severity_max) if pre_scan.severity_max else None,
                        )
                        result.latency_breakdown = latency
                        return result
                    
                    context_str = pre_scan.redacted_text
                    
                    if pre_scan.entities_found:
                        self._log_security_event(
                            "PRE_LLM_PII_REDACTED",
                            entities=pre_scan.entities_found,
                            severity=str(pre_scan.severity_max) if pre_scan.severity_max else None,
                        )
                    
                    latency["pre_llm_pii_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                    
                except Exception as e:
                    logger.warning(
                        "[SecureGen] Pre-LLM PII scan failed, continuing without: %s", e
                    )
                    latency["pre_llm_pii_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            
            # ── Step 3: Build RAG Prompt ──────────────────────────────────
            rag_prompt = (
                "You are a precise, grounded enterprise assistant.\n"
                "Your task: answer the user's question using ONLY the retrieved passages below.\n\n"
                "Rules:\n"
                "  1. Cite every source you use with its number: [Source 1], [Source 2], etc.\n"
                "  2. If the passages lack sufficient information to answer confidently, "
                "respond EXACTLY with: "
                "'I could not find a reliable answer in the available documents.'\n"
                "  3. Never invent facts. Do not add information not present in the sources.\n"
                "  4. Keep the answer factual, concise, and professional.\n\n"
                f"QUESTION: {query}\n\n"
                f"RETRIEVED PASSAGES:\n{context_str}\n\n"
                "ANSWER (grounded, with citations):"
            )
            
            # ── Step 4: LLM Generation ────────────────────────────────────
            t0 = time.perf_counter()
            try:
                resp = llm.generate(rag_prompt, temperature=temperature, max_tokens=max_tokens)
                raw_answer = (resp.text or "").strip()
                latency["llm_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                
                if not raw_answer:
                    result.answer_error = "LLM returned an empty response."
                    result.latency_breakdown = latency
                    return result
                    
            except Exception as e:
                latency["llm_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                result.answer_error = f"LLM generation error: {str(e)[:300]}"
                self._log_security_event("LLM_GENERATION_FAILED", error=str(e)[:200])
                result.latency_breakdown = latency
                return result
            
            # ── Step 5: Post-LLM PII Scan ─────────────────────────────────
            pii_redacted = False
            if self._config.enable_post_llm_pii_scan:
                t0 = time.perf_counter()
                try:
                    pii_mw = self._get_pii_middleware()
                    post_scan = pii_mw.scan_text(raw_answer, position="post_llm")
                    
                    result.post_llm_pii_scan_applied = True
                    result.post_llm_pii_entities = post_scan.entities_found.copy()
                    
                    if post_scan.blocked:
                        result.post_llm_blocked = True
                        result.answer_error = (
                            "Response blocked: Critical PII detected in generated answer."
                        )
                        result.block_reason = "post_llm_pii_blocked"
                        latency["post_llm_pii_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                        self._log_security_event(
                            "POST_LLM_PII_BLOCKED",
                            entities=post_scan.entities_found,
                            severity=str(post_scan.severity_max) if post_scan.severity_max else None,
                        )
                        result.latency_breakdown = latency
                        return result
                    
                    raw_answer = post_scan.redacted_text
                    pii_redacted = bool(post_scan.entities_found)
                    
                    if post_scan.entities_found:
                        self._log_security_event(
                            "POST_LLM_PII_REDACTED",
                            entities=post_scan.entities_found,
                            severity=str(post_scan.severity_max) if post_scan.severity_max else None,
                        )
                    
                    latency["post_llm_pii_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                    
                except Exception as e:
                    logger.warning(
                        "[SecureGen] Post-LLM PII scan failed, continuing: %s", e
                    )
                    latency["post_llm_pii_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            
            # ── Step 6: Trust Scoring ─────────────────────────────────────
            trust_score: Optional[float] = None
            if self._config.enable_trust_scoring:
                t0 = time.perf_counter()
                try:
                    trust_adapter = self._get_trust_adapter()
                    trust_score = trust_adapter.calculate(
                        context_chunks,
                        pii_redacted=pii_redacted,
                        pii_penalty=self._config.pii_trust_penalty,
                    )
                    result.trust_score = trust_score
                    latency["trust_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                    
                    self._log_security_event(
                        "TRUST_SCORE_COMPUTED",
                        trust_score=trust_score,
                        pii_penalty_applied=pii_redacted,
                    )
                    
                except Exception as e:
                    logger.warning(
                        "[SecureGen] Trust scoring failed, using None: %s", e
                    )
                    latency["trust_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            
            # ── Step 7: Output Formatting + Trust Gate ────────────────────
            if self._config.enable_output_formatter:
                t0 = time.perf_counter()
                try:
                    formatter = self._get_output_formatter()
                    fmt_result = formatter.format_output(
                        raw_answer,
                        pii_redacted=pii_redacted,
                        pii_entities_found=result.post_llm_pii_entities or None,
                        trust_score=trust_score,
                    )
                    
                    result.output_formatter_applied = True
                    result.trust_gate_applied = self._config.enable_trust_gate
                    result.trust_gate_passed = fmt_result.get("trust_gate_passed", True)
                    result.output_blocked = fmt_result.get("blocked", False)
                    
                    if result.output_blocked:
                        result.block_reason = fmt_result.get("block_reason")
                        result.answer = fmt_result.get("formatted_output")
                        self._log_security_event(
                            "OUTPUT_BLOCKED",
                            reason=result.block_reason,
                            trust_score=trust_score,
                        )
                    else:
                        result.answer = fmt_result.get("formatted_output", raw_answer)
                        self._log_security_event(
                            "OUTPUT_FORMATTED",
                            format_type=fmt_result.get("format_type"),
                            trust_gate_passed=True,
                        )
                    
                    latency["formatter_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                    
                except Exception as e:
                    logger.warning(
                        "[SecureGen] Output formatter failed, returning raw answer: %s", e
                    )
                    result.answer = raw_answer
                    latency["formatter_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            else:
                result.answer = raw_answer
            
            # ── Step 8: Final latency ─────────────────────────────────────
            result.answer_latency_ms = round((time.perf_counter() - total_start) * 1000, 2)
            result.latency_breakdown = latency
            
            self._log_security_event(
                "SECURE_GENERATION_COMPLETE",
                success=result.answer is not None and not result.output_blocked,
                total_ms=result.answer_latency_ms,
            )
            
            return result
            
        except Exception as e:
            logger.error("[SecureGen] Unexpected error: %s", e, exc_info=True)
            result.answer_error = f"Secure generation failed: {str(e)[:300]}"
            result.latency_breakdown = latency
            return result
    
    def _log_security_event(self, event: str, **kwargs) -> None:
        """Emit structured security telemetry."""
        import json
        
        log_data = {
            "event": event,
            "tenant_id": self._tenant_id,
            "request_id": self._request_id,
            "stack": "retrieval_runtime",
            "security_aligned": True,
            **{k: v for k, v in kwargs.items() if v is not None},
        }
        
        logger.info(
            '{"security_event":"%s",%s}',
            event,
            ",".join(f'"{k}":{json.dumps(v)}' for k, v in log_data.items() if k != "event"),
        )


def generate_answer_secure(
    llm: Any,
    llm_model: str,
    query: str,
    context_chunks: List[Dict[str, Any]],
    *,
    tenant_id: str = "default",
    request_id: Optional[str] = None,
    config: Optional[SecureGenerationConfig] = None,
    max_tokens: int = 700,
    temperature: float = 0.0,
) -> SecureGenerationResult:
    """
    Convenience function for secure answer generation.
    
    Use this from retrieve_api.py and retrieve_chat_api.py when
    ENABLE_RETRIEVAL_SECURITY_ALIGNMENT is True.
    
    Args:
        llm: Instantiated LLM instance
        llm_model: Model name
        query: User query
        context_chunks: Retrieved chunks
        tenant_id: Tenant ID for audit logging
        request_id: Request ID for tracing
        config: Optional config override
        max_tokens: LLM max tokens
        temperature: LLM temperature
    
    Returns:
        SecureGenerationResult with full security metadata
    """
    handler = SecureGenerationHandler(
        config=config,
        tenant_id=tenant_id,
        request_id=request_id,
    )
    return handler.generate_secure(
        llm=llm,
        llm_model=llm_model,
        query=query,
        context_chunks=context_chunks,
        max_tokens=max_tokens,
        temperature=temperature,
    )
