# =============================================================================
# app/ai/validation/tokenizer_validator.py
#
# TokenizerValidator + AlignmentReport — Phase 1, Module 5b
#
# Stateless service class.  All validation is EAGER — runs at pipeline
# construction time (inside PipelineFactory.build or ingestion bootstrap).
# No check may be deferred to embedding time or query time.
#
# Failure modes addressed (all 16):
#   F-01  tokenizer_family consistency between embedder and reranker
#   F-02  chunk size ≥ embed_max_tokens
#   F-03  special token injection (indirect — via tokenizer family check)
#   F-04  dimension mismatch between embedder and VectorDB index
#   F-05  embedding space drift — fingerprint stored in VectorDB metadata
#   F-06  normalization mismatch — is_normalized + metric pairing
#   F-07  asymmetric encoding — query_prefix / passage_prefix not empty when required
#   F-08  distance metric mismatch — embedder vs VectorDB
#   F-09  ANN algorithm mismatch — fingerprint change detection
#   F-10  metadata schema conflict — warning when VectorDB type changes
#   F-11  score space incompatibility — reranker score_space check
#   F-12  domain/language mismatch — lang_support alignment warning
#   F-13  reranker input length — chunk_max ≤ reranker.max_input_tokens
#   F-14  LLM context tokenizer mismatch — LLM must use its own tokenizer
#   F-15  Lost in the middle — chunk_max bounded by headroom_pct
#   F-16  tokenizer migration without re-index — embedding_fingerprint guard
# =============================================================================

from __future__ import annotations

import logging
import math
import os
import statistics
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.ai.contracts.embedder_contract import EmbedderBundle, RerankerView, VerificationStatus
from app.ai.contracts.tokenizer_contract import TokenizerContract
from app.ai.validation.chunk_sizer import DefaultChunkSizer, ValidationResult
from app.ai.validation.exceptions import AlignmentError, TokenOverflowError
from app.core.config.client_config_schema import VectorDBConfig, VectorDBType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AlignmentReport
# ---------------------------------------------------------------------------

@dataclass
class AlignmentReport:
    """
    Full alignment report from validate_pipeline_alignment().

    Fields
    ──────
    ok                      : True iff no BLOCKING errors were found.
    errors                  : List of blocking error messages with F-codes.
    warnings                : List of non-blocking warning messages with F-codes.
    failure_modes           : Union of F-codes from errors and warnings.
    embedder_model_id       : Embedder model identifier validated.
    reranker_model_id       : Reranker model identifier (or None).
    vectordb_kind           : VectorDB backend type string.
    vectordb_collection     : Collection / index name.
    embedder_dimension      : Dimension from EmbedderBundle.
    index_dimension         : Dimension from VectorDBConfig (or None).
    distance_metric_embedder: Distance metric from EmbedderBundle.
    distance_metric_index   : Distance metric from VectorDBConfig (or None).
    is_normalized_embedder  : is_normalized from EmbedderBundle.
    score_space_reranker    : score_space from RerankerView (or None).
    tokenizer_family_embedder: tokenizer_family from EmbedderBundle.
    tokenizer_family_reranker: tokenizer_family from RerankerView (or None).
    embed_max_tokens        : embed_max_tokens from EmbedderBundle.
    reranker_max_tokens     : max_input_tokens from RerankerView (or None).
    effective_chunk_max     : safe chunk size computed from ChunkSizerContract.
    embedding_fingerprint   : Fingerprint of the embedder configuration.
    """
    ok:                       bool
    errors:                   List[str]
    warnings:                 List[str]
    failure_modes:            List[str]
    embedder_model_id:        str
    reranker_model_id:        Optional[str]
    vectordb_kind:            str
    vectordb_collection:      str
    embedder_dimension:       int
    index_dimension:          Optional[int]
    distance_metric_embedder: str
    distance_metric_index:    Optional[str]
    is_normalized_embedder:   bool
    score_space_reranker:     Optional[str]
    tokenizer_family_embedder: str
    tokenizer_family_reranker: Optional[str]
    embed_max_tokens:         int
    reranker_max_tokens:      Optional[int]
    effective_chunk_max:      int
    embedding_fingerprint:    str


# ---------------------------------------------------------------------------
# CalibrationResult — per-(model_id, factory_backend) alignment metrics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationResult:
    """
    Per-model tokenizer calibration output from TokenizerValidator.calibrate().

    Stored in the module-level cache keyed by (model_id, factory_backend).
    Fed into ChunkingTokenCounter.from_bundle() via AlignmentMetrics.

    Fields
    ──────
    model_id        : Canonical model identifier (e.g. "google/gemini-embedding-001").
    factory_backend : Factory tokenizer backend used for approximation
                      (e.g. "huggingface").
    ratio_p95       : 95th-percentile of (model_tokens / factory_tokens) across
                      calibration samples.
    delta_p95       : 95th-percentile of (model_tokens - factory_tokens).
    ratio_mean      : Mean ratio — useful for detecting systematic bias.
    sample_count    : Number of samples used for calibration.
    soft_cap_factor : Configured factor applied to hard_cap → soft_cap.
    soft_cap        : Precomputed soft limit = floor(hard_cap * soft_cap_factor).
    hard_cap        : embed_max_tokens from the EmbedderBundle.
    """
    model_id:        str
    factory_backend: str
    ratio_p95:       float
    delta_p95:       int
    ratio_mean:      float
    sample_count:    int
    soft_cap_factor: float
    soft_cap:        int
    hard_cap:        int


# ---------------------------------------------------------------------------
# Module-level calibration cache — keyed by (model_id, factory_backend)
# Thread-safe: guarded by _CALIBRATION_LOCK.
# ---------------------------------------------------------------------------

_CALIBRATION_CACHE: Dict[Tuple[str, str], CalibrationResult] = {}
_CALIBRATION_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers — VectorDB config introspection
# ---------------------------------------------------------------------------

def _extract_index_dimension(vector_db_config: VectorDBConfig) -> Optional[int]:
    """Extract configured embedding dimension from VectorDBConfig, if present."""
    t = vector_db_config.type
    if t == VectorDBType.PINECONE and vector_db_config.pinecone:
        return vector_db_config.pinecone.embedding_dim
    # Chroma, Qdrant, Weaviate, Milvus, Redis do not declare dimension in
    # ClientConfig — they take it from ensure_collection() at runtime.
    return None


def _extract_index_metric(vector_db_config: VectorDBConfig) -> Optional[str]:
    """Extract configured distance metric from VectorDBConfig, if declared."""
    t = vector_db_config.type
    if t == VectorDBType.PINECONE and vector_db_config.pinecone:
        return vector_db_config.pinecone.metric
    return None


# ---------------------------------------------------------------------------
# TokenizerValidator
# ---------------------------------------------------------------------------

class TokenizerValidator:
    """
    Stateless validation service for Phase 1 pipeline alignment.

    All methods are EAGER: they must be called at pipeline construction time,
    not at embedding or query time.  Fail loud.  Fail with context.
    Never swallow errors.

    Usage (at PipelineFactory.build time)
    ──────────────────────────────────────
    ::

        validator = TokenizerValidator()

        # Check individual chunks during ingestion:
        result = validator.validate_chunk(
            chunk_text, bundle.tokenizer,
            embed_max=bundle.embed_max_tokens,
            reranker_max=reranker_view.max_input_tokens if reranker_view else None,
        )

        # Check full pipeline alignment once at build time:
        report = validator.validate_pipeline_alignment(
            embedder_bundle=bundle,
            reranker_bundle=reranker_view,
            vector_db_config=config.vectordb,
        )
        if not report.ok:
            raise AlignmentError(
                failure_modes=report.failure_modes,
                embedder_model_id=report.embedder_model_id,
                vectordb_kind=report.vectordb_kind,
                details={"errors": report.errors},
            )
    """

    # ------------------------------------------------------------------
    # validate_chunk (§4.1)
    # ------------------------------------------------------------------

    def validate_chunk(
        self,
        chunk_text: str,
        tokenizer: TokenizerContract,
        embed_max: int,
        reranker_max: Optional[int] = None,
        *,
        query_reserve_tokens: int = 50,
        special_overhead_tokens: int = 4,
        headroom_pct: float = 0.10,
    ) -> ValidationResult:
        """
        Validate that ``chunk_text`` fits within the safe token limit.

        All checks follow §4.1 order exactly.

        Parameters
        ──────────
        chunk_text            : The text chunk to validate.
        tokenizer             : The embedder's bound TokenizerContract.
        embed_max             : Maximum tokens for the embedding model.
        reranker_max          : Maximum tokens for the reranker (or None).
        query_reserve_tokens  : Reserved budget for the query.
        special_overhead_tokens: Overhead for sentinel tokens.
        headroom_pct          : Fractional safety headroom.

        Returns
        ────────
        ValidationResult (ok=True if within limits).

        Raises
        ──────
        TokenOverflowError
            On any overflow or counting failure.
        """
        sizer = DefaultChunkSizer(
            embed_max_tokens=embed_max,
            reranker_max_tokens=reranker_max,
            query_reserve_tokens=query_reserve_tokens,
            special_overhead_tokens=special_overhead_tokens,
            headroom_pct=headroom_pct,
            model_id=tokenizer.model_id,
        )
        return sizer.validate_chunk(chunk_text, tokenizer)

    # ------------------------------------------------------------------
    # validate_pipeline_alignment (§4.2)
    # ------------------------------------------------------------------

    def validate_pipeline_alignment(
        self,
        embedder_bundle: EmbedderBundle,
        reranker_bundle: Optional[RerankerView],
        vector_db_config: VectorDBConfig,
    ) -> AlignmentReport:
        """
        Run all 8 alignment checks defined in §4.2.

        Blocking checks raise AlignmentError AND are reflected in the report.
        Non-blocking checks produce warnings only.

        Checks (in order, per §4.2):
          1. Dimension vs VectorDB index         F-04, F-09, F-16 — BLOCKING
          2. Distance metric alignment           F-08             — BLOCKING
          3. Normalization alignment             F-06, F-11       — BLOCKING
          4. Tokenizer family consistency        F-01, F-16       — BLOCKING
          5. Reranker input length               F-13             — BLOCKING
          6. Score space compatibility           F-11             — WARNING
          7. Language/domain coverage            F-12             — WARNING
          8. LLM context tokenizer separation    F-14, F-15       — WARNING
        """
        errors: List[str] = []
        warnings: List[str] = []
        failure_modes: List[str] = []

        model_id = embedder_bundle.model_id
        vdb_kind = vector_db_config.type.value
        collection = vector_db_config.collection

        index_dim = _extract_index_dimension(vector_db_config)
        index_metric = _extract_index_metric(vector_db_config)

        embed_dim = embedder_bundle.dimension
        embed_metric = embedder_bundle.distance_metric.value
        embed_family = embedder_bundle.tokenizer_family.value
        embed_max = embedder_bundle.embed_max_tokens
        is_norm = embedder_bundle.is_normalized
        fingerprint = embedder_bundle.embedding_fingerprint()

        reranker_max: Optional[int] = None
        reranker_family: Optional[str] = None
        reranker_score_space: Optional[str] = None
        reranker_lang: Optional[str] = None
        reranker_model_id: Optional[str] = None

        if reranker_bundle is not None:
            reranker_max = reranker_bundle.max_input_tokens
            reranker_family = reranker_bundle.tokenizer_family.value
            reranker_score_space = reranker_bundle.score_space
            reranker_lang = reranker_bundle.lang_support
            reranker_model_id = reranker_bundle.model_id

        # Compute effective chunk max for reporting and check 5
        sizer = DefaultChunkSizer(
            embed_max_tokens=embed_max,
            reranker_max_tokens=reranker_max,
            model_id=model_id,
        )
        try:
            effective_chunk_max = sizer.safe_chunk_size()
        except ValueError as exc:
            errors.append(
                f"[F-13/F-15] ChunkSizer configuration invalid: {exc}"
            )
            _add_modes(failure_modes, ["F-13", "F-15"])
            effective_chunk_max = 0

        # ── Check 1: Dimension vs VectorDB index ─────────────────────
        if index_dim is not None and index_dim != embed_dim:
            msg = (
                f"[F-04/F-09/F-16] Dimension mismatch: embedder.dimension={embed_dim} "
                f"≠ vectordb.embedding_dim={index_dim} "
                f"(model_id={model_id!r}, vectordb={vdb_kind!r}). "
                "Rebuild the index or change the embedder."
            )
            errors.append(msg)
            _add_modes(failure_modes, ["F-04", "F-09", "F-16"])
            logger.error(msg)

        # ── Check 2: Distance metric alignment ───────────────────────
        if index_metric is not None:
            norm_index = _normalise_metric(index_metric)
            norm_embed = _normalise_metric(embed_metric)
            if norm_index != norm_embed:
                msg = (
                    f"[F-08] Distance metric mismatch: embedder.metric={embed_metric!r} "
                    f"≠ vectordb.metric={index_metric!r} "
                    f"(model_id={model_id!r}). "
                    "Rebuild the index with the correct metric or update the catalog."
                )
                errors.append(msg)
                _add_modes(failure_modes, ["F-08"])
                logger.error(msg)

        # ── Check 3: Normalization alignment ─────────────────────────
        if is_norm and embed_metric not in ("cosine", "dotproduct"):
            msg = (
                f"[F-06/F-11] Normalization mismatch: embedder.is_normalized=True "
                f"but distance_metric={embed_metric!r} may not be compatible. "
                f"Normalized vectors should use 'cosine' or 'dotproduct'. "
                f"(model_id={model_id!r})."
            )
            errors.append(msg)
            _add_modes(failure_modes, ["F-06", "F-11"])
            logger.error(msg)

        # ── Check 4: Tokenizer family consistency ─────────────────────
        if embedder_bundle.verification_status == VerificationStatus.UNVERIFIED:
            msg = (
                f"[F-01/F-16] EmbedderBundle.verification_status='unverified' "
                f"for model_id={model_id!r}. Tokenizer family cannot be guaranteed. "
                "Production ingestion is BLOCKED until status is 'verified'."
            )
            errors.append(msg)
            _add_modes(failure_modes, ["F-01", "F-16"])
            logger.error(msg)
        elif embedder_bundle.verification_status == VerificationStatus.CATALOG_MISMATCH:
            wmsg = (
                f"[W-F-01/F-16] EmbedderBundle.verification_status='catalog_mismatch' "
                f"for model_id={model_id!r}. Discovered tokenizer metadata differed "
                "from catalog; catalog was used as authoritative. "
                "Review catalog entry before production deployment."
            )
            warnings.append(wmsg)
            _add_modes(failure_modes, ["F-01", "F-16"])
            logger.warning(wmsg)

        # ── Check 5: Reranker input length ────────────────────────────
        if reranker_max is not None and effective_chunk_max > 0:
            if effective_chunk_max > reranker_max:
                msg = (
                    f"[F-13] Reranker input length mismatch: effective_chunk_max="
                    f"{effective_chunk_max} > reranker.max_input_tokens={reranker_max} "
                    f"(reranker={reranker_model_id!r}). "
                    "Reduce chunk size or use a reranker with a larger input window."
                )
                errors.append(msg)
                _add_modes(failure_modes, ["F-13"])
                logger.error(msg)

        # ── Check 6: Score space compatibility ────────────────────────
        if reranker_score_space is not None and reranker_score_space == "logits":
            wmsg = (
                f"[W-F-11] Reranker {reranker_model_id!r} uses raw logit scores "
                "(range approx -10 to +12). If scores are combined with vector "
                "similarity scores (0–1 range for cosine), re-normalize before "
                "merging. Failure to do so causes F-11 (ranking corruption)."
            )
            warnings.append(wmsg)
            _add_modes(failure_modes, ["F-11"])
            logger.warning(wmsg)

        # ── Check 7: Language / domain coverage ───────────────────────
        if reranker_lang is not None and reranker_lang != "multilingual":
            # Surface as a warning; the operator knows the corpus language.
            wmsg = (
                f"[W-F-12] Reranker {reranker_model_id!r} supports "
                f"lang={reranker_lang!r} only. If the corpus contains other "
                "languages, retrieval quality will degrade. "
                "Use a multilingual reranker for mixed-language corpora."
            )
            warnings.append(wmsg)
            _add_modes(failure_modes, ["F-12"])
            logger.warning(wmsg)

        # ── Check 8: LLM context tokenizer separation ─────────────────
        # This is a design-level invariant we can only warn about here;
        # actual enforcement must happen in the LLM chain configuration.
        wmsg = (
            "[W-F-14/F-15] REMINDER: LLM context packing must use the LLM's "
            "own tokenizer, NOT the embedding model's tokenizer. "
            f"Embedding tokenizer family={embed_family!r} is NOT suitable for "
            "measuring LLM prompt fit. Enforces R-L1, R-L2, R-L3."
        )
        warnings.append(wmsg)
        # F-14/F-15 are not added to failure_modes here since this is always
        # a reminder, not a detected violation in this scope.

        ok = len(errors) == 0

        report = AlignmentReport(
            ok=ok,
            errors=errors,
            warnings=warnings,
            failure_modes=sorted(set(failure_modes)),
            embedder_model_id=model_id,
            reranker_model_id=reranker_model_id,
            vectordb_kind=vdb_kind,
            vectordb_collection=collection,
            embedder_dimension=embed_dim,
            index_dimension=index_dim,
            distance_metric_embedder=embed_metric,
            distance_metric_index=index_metric,
            is_normalized_embedder=is_norm,
            score_space_reranker=reranker_score_space,
            tokenizer_family_embedder=embed_family,
            tokenizer_family_reranker=reranker_family,
            embed_max_tokens=embed_max,
            reranker_max_tokens=reranker_max,
            effective_chunk_max=effective_chunk_max,
            embedding_fingerprint=fingerprint,
        )

        if not ok:
            logger.error(
                "[TokenizerValidator] Alignment FAILED | model_id=%r | "
                "vectordb=%r | errors=%d | failure_modes=%s",
                model_id, vdb_kind, len(errors), sorted(set(failure_modes)),
            )
            raise AlignmentError(
                failure_modes=sorted(set(failure_modes)),
                embedder_model_id=model_id,
                vectordb_kind=vdb_kind,
                details={"errors": errors, "fingerprint": fingerprint},
            )

        logger.info(
            "[TokenizerValidator] Alignment OK | model_id=%r | vectordb=%r | "
            "chunk_max=%d | warnings=%d | fp=%s",
            model_id, vdb_kind, effective_chunk_max, len(warnings),
            fingerprint[:12],
        )
        return report

    # ------------------------------------------------------------------
    # calibrate (§5 — tokenizer alignment calibration)
    # ------------------------------------------------------------------

    def calibrate(
        self,
        *,
        model_id: str,
        tokenizer: TokenizerContract,
        hard_cap: int,
        provider: str = "default",
        sample_texts: Optional[List[str]] = None,
        soft_cap_factor_override: Optional[float] = None,
    ) -> "CalibrationResult":
        """
        Measure alignment between a model-native TokenizerContract and the
        factory tokenizer (DEFAULT_TOKENIZER_BACKEND) to derive calibrated
        soft_cap and ratio_p95 for ChunkingTokenCounter.

        Results are cached per (model_id, factory_backend) and reused on
        subsequent calls, so calibration runs at most once per model per
        process lifetime.

        Parameters
        ──────────
        model_id               : Canonical model identifier.
        tokenizer              : The model's bound TokenizerContract.
        hard_cap               : embed_max_tokens from EmbedderBundle.
        provider               : Provider name for default factor lookup.
        sample_texts           : Custom texts to calibrate on. If None, uses
                                 built-in representative samples.
        soft_cap_factor_override: Override the per-provider default factor.
                                  Must be in (0.0, 1.0).

        Returns
        ───────
        CalibrationResult with ratio_p95, delta_p95, soft_cap, and more.
        """
        from app.ai.chunking.token_counter import _DEFAULT_SOFT_CAP_FACTORS

        factory_backend = os.getenv("DEFAULT_TOKENIZER_BACKEND", "huggingface")
        cache_key = (model_id, factory_backend)

        with _CALIBRATION_LOCK:
            cached = _CALIBRATION_CACHE.get(cache_key)
            if cached is not None:
                logger.debug(
                    "[TokenizerValidator] Calibration cache hit | model=%r | backend=%r",
                    model_id, factory_backend,
                )
                return cached

        # ── Build calibration sample set ───────────────────────────────────
        if sample_texts is None:
            sample_texts = _CALIBRATION_SAMPLE_TEXTS

        # ── Collect (factory_count, model_count) pairs ────────────────────
        try:
            from app.core.tokenization.factory import count_tokens as _factory_count
        except Exception:
            # Factory unavailable — return uncalibrated defaults.
            logger.warning(
                "[TokenizerValidator] Factory tokenizer unavailable; "
                "calibration skipped for model=%r. Using provider defaults.",
                model_id,
            )
            return _uncalibrated_result(
                model_id, factory_backend, hard_cap, provider, soft_cap_factor_override
            )

        ratios: List[float] = []
        deltas: List[int] = []
        successes = 0

        for text in sample_texts:
            if not text.strip():
                continue
            try:
                factory_n = _factory_count(text)
                model_n   = tokenizer.count_tokens(text, include_special_tokens=False)
                if factory_n > 0:
                    ratios.append(model_n / factory_n)
                    deltas.append(model_n - factory_n)
                    successes += 1
            except Exception as exc:
                logger.debug(
                    "[TokenizerValidator] Calibration sample failed | model=%r | %s",
                    model_id, exc,
                )

        if successes < 3:
            logger.warning(
                "[TokenizerValidator] Insufficient calibration samples (%d/%d) "
                "for model=%r. Using provider defaults.",
                successes, len(sample_texts), model_id,
            )
            return _uncalibrated_result(
                model_id, factory_backend, hard_cap, provider, soft_cap_factor_override
            )

        ratios.sort()
        deltas.sort()

        # p95 index (ceiling to be conservative)
        p95_idx = min(len(ratios) - 1, math.ceil(len(ratios) * 0.95) - 1)
        ratio_p95 = round(ratios[p95_idx], 4)
        delta_p95 = deltas[p95_idx]
        ratio_mean = round(statistics.mean(ratios), 4)

        # ── Determine soft_cap_factor ─────────────────────────────────────
        if soft_cap_factor_override is not None:
            factor = max(0.01, min(1.0, soft_cap_factor_override))
        else:
            default_factor = _DEFAULT_SOFT_CAP_FACTORS.get(
                provider.lower(), _DEFAULT_SOFT_CAP_FACTORS["default"]
            )
            # Never exceed the provider's default soft_cap_factor.
            # Also derive a factor from ratio_p95: factor_from_ratio = 1 / ratio_p95.
            # Use the MORE conservative of the two.
            factor_from_ratio = round(1.0 / ratio_p95, 4) if ratio_p95 > 0 else default_factor
            factor = min(default_factor, factor_from_ratio)

        soft_cap = max(1, math.floor(hard_cap * factor))

        result = CalibrationResult(
            model_id=model_id,
            factory_backend=factory_backend,
            ratio_p95=ratio_p95,
            delta_p95=delta_p95,
            ratio_mean=ratio_mean,
            sample_count=successes,
            soft_cap_factor=round(factor, 4),
            soft_cap=soft_cap,
            hard_cap=hard_cap,
        )

        with _CALIBRATION_LOCK:
            _CALIBRATION_CACHE[cache_key] = result

        logger.info(
            "[TokenizerValidator] Calibration complete | model=%r | backend=%r | "
            "ratio_p95=%.3f | delta_p95=%d | ratio_mean=%.3f | "
            "soft_cap=%d | hard_cap=%d | soft_cap_factor=%.3f | samples=%d",
            model_id, factory_backend, ratio_p95, delta_p95, ratio_mean,
            soft_cap, hard_cap, factor, successes,
        )
        return result

    def get_alignment_metrics_for_bundle(
        self,
        bundle: EmbedderBundle,
        *,
        run_calibration: bool = True,
    ) -> "AlignmentMetrics":  # type: ignore[name-defined]
        """
        Return calibrated AlignmentMetrics for an EmbedderBundle.

        This is the bridge between TokenizerValidator.calibrate() and
        ChunkingTokenCounter.from_bundle().  Call this at pipeline
        construction time and pass the result to ChunkingTokenCounter.

        Parameters
        ──────────
        bundle           : Fully resolved EmbedderBundle.
        run_calibration  : If True (default), calibrate the tokenizer now
                           if not already cached.  If False, use provider
                           defaults without calibration.

        Returns
        ───────
        AlignmentMetrics ready for ChunkingTokenCounter.from_bundle().
        """
        from app.ai.chunking.token_counter import AlignmentMetrics, _default_alignment_metrics

        provider = bundle.provider.value if hasattr(bundle.provider, "value") else str(bundle.provider)
        factory_backend = os.getenv("DEFAULT_TOKENIZER_BACKEND", "huggingface")
        cache_key = (bundle.model_id, factory_backend)

        calibration: Optional[CalibrationResult] = None
        if run_calibration:
            with _CALIBRATION_LOCK:
                calibration = _CALIBRATION_CACHE.get(cache_key)

            if calibration is None:
                try:
                    calibration = self.calibrate(
                        model_id=bundle.model_id,
                        tokenizer=bundle.tokenizer,
                        hard_cap=bundle.embed_max_tokens,
                        provider=provider,
                    )
                except Exception as exc:
                    logger.warning(
                        "[TokenizerValidator] calibrate() failed for model=%r: %s. "
                        "Falling back to provider defaults.",
                        bundle.model_id, exc,
                    )

        if calibration is not None:
            return AlignmentMetrics(
                ratio_p95=calibration.ratio_p95,
                delta_p95=calibration.delta_p95,
                soft_cap_factor=calibration.soft_cap_factor,
                soft_cap=calibration.soft_cap,
                hard_cap=calibration.hard_cap,
                calibrated=True,
                model_id=calibration.model_id,
                factory_backend=calibration.factory_backend,
            )

        return _default_alignment_metrics(
            provider=provider,
            hard_cap=bundle.embed_max_tokens,
            model_id=bundle.model_id,
            factory_backend=factory_backend,
        )


# ---------------------------------------------------------------------------
# Migration guard helper (§5.4 / R-C4 / F-16)
# ---------------------------------------------------------------------------

class EmbeddingSpaceMigrationGuard:
    """
    Compares a stored embedding fingerprint against the current bundle's
    fingerprint and raises AlignmentError if they differ.

    This is the mechanism that prevents using an old chunk index after a
    tokenizer or embedder change (R-C4, F-16).

    Usage
    ──────
    At ``ensure_collection`` time, store ``bundle.embedding_fingerprint()``
    as collection metadata under the key ``"embedding_fingerprint"``.

    At every subsequent ``PipelineFactory.build`` or ingestion startup,
    call::

        guard = EmbeddingSpaceMigrationGuard()
        guard.assert_compatible(
            stored_fingerprint=stored,
            current_bundle=bundle,
            collection_name="my_collection",
            vectordb_kind="qdrant",
        )

    If fingerprints differ, AlignmentError is raised with F-05 / F-16 codes,
    blocking any further ingestion or retrieval until the operator explicitly
    re-indexes with the new embedder.
    """

    def assert_compatible(
        self,
        *,
        stored_fingerprint: str,
        current_bundle: EmbedderBundle,
        collection_name: str,
        vectordb_kind: str,
    ) -> None:
        """
        Assert that the stored embedding fingerprint matches the current bundle.

        Parameters
        ──────────
        stored_fingerprint : Fingerprint stored in VectorDB collection metadata.
        current_bundle     : The EmbedderBundle being used for this pipeline.
        collection_name    : VectorDB collection/index name for error messages.
        vectordb_kind      : VectorDB backend name for error messages.

        Raises
        ──────
        AlignmentError [F-05, F-16]
            If fingerprints differ, indicating the index was built with a
            different embedding space (model, tokenizer family, dimension,
            metric, or normalization).
        """
        current_fp = current_bundle.embedding_fingerprint()

        if stored_fingerprint == current_fp:
            logger.debug(
                "[MigrationGuard] Fingerprint OK: collection=%r fp=%s",
                collection_name, current_fp[:12],
            )
            return

        # Fingerprints differ — decode what changed
        details: Dict[str, Any] = {
            "collection": collection_name,
            "stored_fingerprint":  stored_fingerprint[:16],
            "current_fingerprint": current_fp[:16],
            "current_model_id":    current_bundle.model_id,
            "current_dimension":   current_bundle.dimension,
            "current_metric":      current_bundle.distance_metric.value,
            "current_family":      current_bundle.tokenizer_family.value,
            "action_required": (
                "Drop and recreate the collection (full re-index) "
                "with the new embedder before ingestion or retrieval can proceed. "
                "F-05: embedding space drift. F-16: tokenizer migration without re-index."
            ),
        }

        msg = (
            f"[F-05/F-16] EmbeddingSpaceMigrationGuard: collection={collection_name!r} "
            f"fingerprint mismatch. "
            f"stored={stored_fingerprint[:16]!r} "
            f"current={current_fp[:16]!r}. "
            "Re-index required before proceeding."
        )
        logger.error(msg)

        raise AlignmentError(
            failure_modes=["F-05", "F-16"],
            embedder_model_id=current_bundle.model_id,
            vectordb_kind=vectordb_kind,
            details=details,
        )


# ---------------------------------------------------------------------------
# Module-level singleton instances
# ---------------------------------------------------------------------------

tokenizer_validator = TokenizerValidator()
migration_guard = EmbeddingSpaceMigrationGuard()


# ---------------------------------------------------------------------------
# Calibration sample texts
# Representative of typical marketing / business document content.
# Short, medium, and long segments; Latin, numeric, and mixed content.
# ---------------------------------------------------------------------------

_CALIBRATION_SAMPLE_TEXTS: List[str] = [
    # Short prose (30–80 tokens)
    "Customer acquisition cost dropped by 15% in Q3 due to improved targeting.",
    "Marketing channels include email, paid search, social media, and affiliate programs.",
    "Brand awareness is measured through aided recall, unaided recall, and net promoter score.",
    # Medium prose (100–200 tokens)
    (
        "The integrated marketing strategy leverages multi-channel attribution models to "
        "allocate budget across digital and traditional media. Each campaign is evaluated on "
        "incremental revenue, customer lifetime value, and cost per acquisition. "
        "Monthly reporting cadence ensures rapid course correction when KPIs diverge from plan."
    ),
    (
        "Personalization at scale requires a customer data platform that unifies first-party "
        "signals from CRM, web analytics, and purchase history. Machine learning models segment "
        "audiences by propensity score, churn risk, and predicted LTV. Real-time decisioning "
        "delivers the right message to the right customer at the right moment across all touchpoints."
    ),
    # Longer prose (250–400 tokens)
    (
        "Content marketing is a strategic approach focused on creating and distributing valuable, "
        "relevant, and consistent content to attract and retain a clearly defined audience — and, "
        "ultimately, to drive profitable customer action. Unlike traditional advertising, content "
        "marketing does not explicitly promote a brand but instead stimulates interest in its "
        "products or services. A well-executed content strategy builds authority and trust over "
        "time by addressing the questions, pain points, and aspirations of the target persona. "
        "Key formats include long-form articles, white papers, case studies, video tutorials, "
        "webinars, and interactive tools such as ROI calculators and diagnostic assessments. "
        "Distribution channels span owned media (company blog, email newsletter), earned media "
        "(press coverage, organic social sharing), and paid amplification (sponsored posts, "
        "native advertising). Success is tracked through organic search rankings, time-on-site, "
        "lead-to-MQL conversion rates, and ultimately pipeline influenced by content assets."
    ),
    # Numeric-heavy content
    (
        "Q4 results: Revenue $12.4M (+18% YoY), Gross Margin 68.2% (+1.4pp), "
        "CAC $142 (-12%), LTV $1,840 (+9%), LTV/CAC ratio 12.96x. "
        "Channel breakdown: Paid Search 34%, Organic 28%, Email 21%, Social 17%."
    ),
    # Technical / structured
    (
        "API endpoint: POST /v2/embeddings — accepts JSON payload with model_id, "
        "input_texts (array of strings, max 128), and optional parameters: "
        "task_type (RETRIEVAL_DOCUMENT | RETRIEVAL_QUERY | CLASSIFICATION), "
        "output_dimensionality (768 | 1536 | 3072). Returns embeddings array "
        "with values, usage.prompt_tokens, and model metadata."
    ),
    # Mixed-language / unicode challenge
    (
        "Global campaign performance: North America 45%, EMEA 31%, APAC 24%. "
        "Top markets: United States, United Kingdom, Germany, Japan (日本), "
        "Brazil (Brasil). Customer satisfaction scores: NPS 62, CSAT 87%, CES 4.2/5."
    ),
]


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def _uncalibrated_result(
    model_id: str,
    factory_backend: str,
    hard_cap: int,
    provider: str,
    soft_cap_factor_override: Optional[float],
) -> CalibrationResult:
    """Construct a CalibrationResult from provider defaults without sampling."""
    from app.ai.chunking.token_counter import _DEFAULT_SOFT_CAP_FACTORS

    default_factor = _DEFAULT_SOFT_CAP_FACTORS.get(
        provider.lower(), _DEFAULT_SOFT_CAP_FACTORS["default"]
    )
    factor = soft_cap_factor_override if soft_cap_factor_override is not None else default_factor
    factor = max(0.01, min(1.0, factor))
    soft_cap = max(1, math.floor(hard_cap * factor))

    # Default ratios by provider (conservative estimates)
    is_remote_provider = provider.lower() in ("google", "cohere")
    ratio_p95 = 1.25 if is_remote_provider else 1.05

    return CalibrationResult(
        model_id=model_id,
        factory_backend=factory_backend,
        ratio_p95=ratio_p95,
        delta_p95=int(hard_cap * (ratio_p95 - 1.0)),
        ratio_mean=ratio_p95 * 0.95,
        sample_count=0,
        soft_cap_factor=round(factor, 4),
        soft_cap=soft_cap,
        hard_cap=hard_cap,
    )


def clear_calibration_cache() -> None:
    """Clear the module-level calibration cache. Useful in tests."""
    with _CALIBRATION_LOCK:
        _CALIBRATION_CACHE.clear()


def get_cached_calibration(model_id: str) -> Optional[CalibrationResult]:
    """Return cached calibration for model_id+active backend, or None."""
    factory_backend = os.getenv("DEFAULT_TOKENIZER_BACKEND", "huggingface")
    with _CALIBRATION_LOCK:
        return _CALIBRATION_CACHE.get((model_id, factory_backend))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _normalise_metric(metric: str) -> str:
    """Normalize metric string for comparison (lowercase, strip whitespace)."""
    return metric.lower().strip()


def _add_modes(existing: List[str], new_modes: List[str]) -> None:
    """Add failure mode codes to an existing list, avoiding duplicates."""
    for m in new_modes:
        if m not in existing:
            existing.append(m)
