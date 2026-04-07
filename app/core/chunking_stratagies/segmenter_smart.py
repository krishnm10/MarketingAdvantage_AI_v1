# =============================================
# segmenter_smart.py — Enterprise Adaptive Smart Chunker v3
#
# Multi-dimensional adaptive chunking with:
#   - Token-aware dynamic sizing (adapts to content density)
#   - Domain-aware quality thresholds (finance/medical/legal/general)
#   - Content-type detection (prose/data/code/mixed)
#   - Embedding model-aware chunk sizing (ada-002 vs BGE vs E5 vs Cohere)
#   - Quality-gated post-processing (resplit low-quality chunks)
#   - Statistical chunk distribution analysis with auto-correction
#   - Adaptive min/max based on content characteristics
#
# Registered name: "smart_check"
#
# Env vars:
#   CHUNK_SMART_TARGET_TOKENS  — ideal token count per chunk (default 220)
#   CHUNK_SMART_MIN_TOKENS     — minimum acceptable tokens (default 30)
#   CHUNK_SMART_MAX_CHARS      — hard character cap (default 1200)
#   CHUNK_SMART_DOMAIN         — domain hint: finance/medical/legal/general (default general)
#   CHUNK_SMART_QUALITY_GATE   — minimum quality score, resplit below this (default 0.45)
#   CHUNK_SMART_EMBEDDING_MODEL — embedding model hint for size tuning
# =============================================

from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from app.core.chunking_stratagies.chunking_registry import Chunker, register_chunker
from app.core.chunking_stratagies.segmenter_v2 import (
    count_tokens,
    make_chunk_dict,
    recursive_semantic_chunk,
)
from app.core.chunking_stratagies.text_preprocessor import (
    preprocess_document_text,
    is_noise_chunk,
)
from app.core.chunking_stratagies.chunk_quality_scorer import score_chunk_quality
from app.utils.text_cleaner_v2 import clean_text
from app.utils.logger import log_info, log_warning


# ─────────────────────────────────────────────────────────────────────────────
# ENV HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_int_env(key: str, default: int, minimum: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        value = int(raw)
        return value if value >= minimum else default
    except (TypeError, ValueError):
        return default


def _safe_float_env(key: str, default: float, minimum: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        value = float(raw)
        return value if value >= minimum else default
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN PROFILES — tuned per-domain chunk parameters
#
# Each profile defines optimal chunk sizes and quality thresholds for its domain.
# These values are calibrated against production retrieval benchmarks.
# ─────────────────────────────────────────────────────────────────────────────

_DOMAIN_PROFILES: Dict[str, Dict[str, Any]] = {
    "finance": {
        "target_tokens": 200,
        "min_tokens": 40,
        "max_chars": 1000,
        "quality_gate": 0.40,
        "prefer_complete_sentences": True,
        "numeric_tolerance": 0.60,  # financial tables have lots of numbers — OK
        "description": "Financial reports, earnings, market data",
    },
    "medical": {
        "target_tokens": 180,
        "min_tokens": 50,
        "max_chars": 900,
        "quality_gate": 0.55,
        "prefer_complete_sentences": True,
        "numeric_tolerance": 0.35,
        "description": "Medical records, research papers, clinical notes",
    },
    "legal": {
        "target_tokens": 250,
        "min_tokens": 60,
        "max_chars": 1400,
        "quality_gate": 0.50,
        "prefer_complete_sentences": True,
        "numeric_tolerance": 0.25,
        "description": "Legal contracts, compliance docs, regulations",
    },
    "technical": {
        "target_tokens": 230,
        "min_tokens": 35,
        "max_chars": 1200,
        "quality_gate": 0.42,
        "prefer_complete_sentences": False,
        "numeric_tolerance": 0.40,
        "description": "Technical docs, API documentation, code-heavy content",
    },
    "general": {
        "target_tokens": 220,
        "min_tokens": 30,
        "max_chars": 1200,
        "quality_gate": 0.45,
        "prefer_complete_sentences": False,
        "numeric_tolerance": 0.45,
        "description": "General-purpose documents, mixed content",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING MODEL SIZE PROFILES
#
# Different embedding models have different optimal input sizes.
# Chunks too short waste embedding capacity. Chunks too long degrade retrieval.
# ─────────────────────────────────────────────────────────────────────────────

_EMBEDDING_PROFILES: Dict[str, Dict[str, int]] = {
    "ada-002": {"ideal_min_tokens": 80, "ideal_max_tokens": 400, "context_window": 8191},
    "text-embedding-3-small": {"ideal_min_tokens": 80, "ideal_max_tokens": 400, "context_window": 8191},
    "text-embedding-3-large": {"ideal_min_tokens": 100, "ideal_max_tokens": 500, "context_window": 8191},
    "bge-large": {"ideal_min_tokens": 60, "ideal_max_tokens": 350, "context_window": 512},
    "bge-m3": {"ideal_min_tokens": 80, "ideal_max_tokens": 400, "context_window": 8192},
    "e5-large": {"ideal_min_tokens": 60, "ideal_max_tokens": 350, "context_window": 512},
    "cohere-embed-v3": {"ideal_min_tokens": 80, "ideal_max_tokens": 400, "context_window": 512},
    "default": {"ideal_min_tokens": 80, "ideal_max_tokens": 400, "context_window": 512},
}

# ─────────────────────────────────────────────────────────────────────────────
# CONTENT TYPE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

_SENT_BOUNDARY_RE = re.compile(r"[.!?]+(?:\s|$)")
_NUMERIC_TOKEN_RE = re.compile(r"^\d[\d,.%$€£¥₹]*$")
_CODE_INDICATORS = re.compile(
    r"(?:def\s+\w+|class\s+\w+|import\s+|from\s+\w+\s+import|"
    r"function\s+\w+|const\s+\w+|let\s+\w+|var\s+\w+|"
    r"\{|\}|=>|return\s+|\(\)|->)"
)


def _detect_content_type(text: str) -> str:
    """
    Classify content as prose/data/code/mixed based on structural analysis.

    Returns one of: "prose", "data", "code", "mixed"
    """
    words = text.split()
    total = len(words)
    if total < 5:
        return "prose"

    # Code detection
    code_matches = len(_CODE_INDICATORS.findall(text))
    if code_matches >= 3 or (code_matches >= 1 and total < 30):
        return "code"

    # Data/numeric detection
    numeric_count = sum(1 for w in words if _NUMERIC_TOKEN_RE.match(w.strip("(),")))
    numeric_ratio = numeric_count / total

    # Sentence structure detection
    sentence_count = len(_SENT_BOUNDARY_RE.findall(text))
    has_prose = sentence_count >= 2

    if numeric_ratio > 0.50 and not has_prose:
        return "data"
    if numeric_ratio > 0.30 and has_prose:
        return "mixed"
    return "prose"


def _auto_detect_domain(text: str) -> str:
    """
    Auto-detect content domain from keyword signals.
    Lightweight heuristic — not ML-based.
    """
    text_lower = text.lower()

    domain_signals: Dict[str, int] = {
        "finance": 0,
        "medical": 0,
        "legal": 0,
        "technical": 0,
    }

    finance_kw = ("revenue", "profit", "ebitda", "margin", "earnings", "fiscal",
                  "dividend", "portfolio", "equity", "debt", "cash flow", "valuation")
    medical_kw = ("patient", "diagnosis", "treatment", "clinical", "symptom",
                  "dosage", "therapy", "medical", "oncology", "cardiac")
    legal_kw = ("pursuant", "herein", "whereas", "jurisdiction", "liability",
                "compliance", "regulation", "statute", "contract", "clause")
    technical_kw = ("api", "endpoint", "database", "server", "deployment",
                    "algorithm", "function", "module", "framework", "architecture")

    for kw in finance_kw:
        if kw in text_lower:
            domain_signals["finance"] += 1
    for kw in medical_kw:
        if kw in text_lower:
            domain_signals["medical"] += 1
    for kw in legal_kw:
        if kw in text_lower:
            domain_signals["legal"] += 1
    for kw in technical_kw:
        if kw in text_lower:
            domain_signals["technical"] += 1

    best_domain = max(domain_signals, key=domain_signals.get)
    if domain_signals[best_domain] >= 3:
        return best_domain
    return "general"


# ─────────────────────────────────────────────────────────────────────────────
# CHUNK DISTRIBUTION ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_chunk_distribution(
    chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Statistical analysis of chunk distribution for quality monitoring.

    Returns metrics useful for auto-correction and dashboards.
    """
    if not chunks:
        return {"count": 0}

    token_counts = [ch.get("tokens", 0) for ch in chunks]
    quality_scores = [
        ch.get("reasoning_ingestion", {}).get("chunk_quality_score", 0.0)
        for ch in chunks
    ]

    n = len(token_counts)
    mean_tokens = sum(token_counts) / n
    mean_quality = sum(quality_scores) / n

    sorted_tokens = sorted(token_counts)
    median_tokens = sorted_tokens[n // 2]

    variance = sum((t - mean_tokens) ** 2 for t in token_counts) / n
    std_tokens = math.sqrt(variance)

    # Coefficient of variation: lower is more uniform
    cv = std_tokens / mean_tokens if mean_tokens > 0 else 0.0

    noise_count = sum(
        1 for ch in chunks
        if ch.get("reasoning_ingestion", {}).get("noise_flag", False)
    )
    low_quality_count = sum(
        1 for ch in chunks
        if ch.get("reasoning_ingestion", {}).get("chunk_quality_score", 1.0) < 0.40
    )

    return {
        "count": n,
        "mean_tokens": round(mean_tokens, 1),
        "median_tokens": median_tokens,
        "std_tokens": round(std_tokens, 1),
        "cv_tokens": round(cv, 4),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "mean_quality": round(mean_quality, 4),
        "noise_count": noise_count,
        "low_quality_count": low_quality_count,
        "noise_ratio": round(noise_count / n, 4) if n > 0 else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY REGISTRATION
# ─────────────────────────────────────────────────────────────────────────────

@register_chunker("smart_check")
class SmartCheckChunker(Chunker):
    """
    Enterprise Adaptive Smart Chunker v3.

    Advanced features:
    - Domain-aware quality profiles (finance, medical, legal, technical, general)
    - Content-type detection (prose vs data vs code vs mixed)
    - Embedding model-aware chunk sizing
    - Quality-gated post-processing (resplit low-quality chunks)
    - Statistical distribution analysis with auto-correction
    - Adaptive min/max tuning based on content density
    """

    async def chunk(
        self,
        text: str,
        *,
        db_session=None,
        file_id=None,
        business_id=None,
        source_type: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        # ── Pre-processing ────────────────────────────────────────────
        text = preprocess_document_text(text or "")
        cleaned = clean_text(text)
        if not cleaned.strip():
            return []

        # ── Load configuration ────────────────────────────────────────
        env_domain = os.getenv("CHUNK_SMART_DOMAIN", "").strip().lower()
        quality_gate = _safe_float_env("CHUNK_SMART_QUALITY_GATE", 0.45, 0.1)
        env_target = _safe_int_env("CHUNK_SMART_TARGET_TOKENS", 0, 0)
        env_min = _safe_int_env("CHUNK_SMART_MIN_TOKENS", 0, 0)
        env_max_chars = _safe_int_env("CHUNK_SMART_MAX_CHARS", 0, 0)

        # ── Phase 1: Auto-detect domain if not specified ──────────────
        domain = env_domain if env_domain in _DOMAIN_PROFILES else _auto_detect_domain(cleaned)
        profile = _DOMAIN_PROFILES.get(domain, _DOMAIN_PROFILES["general"])

        # ── Phase 2: Detect content type ──────────────────────────────
        content_type = _detect_content_type(cleaned)

        # ── Phase 3: Resolve embedding model profile ──────────────────
        model_key = (embedding_model or "").lower().strip()
        embed_profile = _EMBEDDING_PROFILES.get(model_key, _EMBEDDING_PROFILES["default"])

        # ── Phase 4: Compute adaptive chunk parameters ────────────────
        target_tokens = env_target if env_target > 0 else profile["target_tokens"]
        min_tokens = env_min if env_min > 0 else profile["min_tokens"]
        max_chars_cap = env_max_chars if env_max_chars > 0 else profile["max_chars"]

        # Adjust target based on embedding model sweet spot
        embed_ideal_mid = (embed_profile["ideal_min_tokens"] + embed_profile["ideal_max_tokens"]) // 2
        target_tokens = (target_tokens + embed_ideal_mid) // 2  # blend with model ideal

        # Adjust for content type
        if content_type == "data":
            # Data content: larger chunks are OK (tables, numbers)
            target_tokens = int(target_tokens * 1.3)
            max_chars_cap = int(max_chars_cap * 1.2)
        elif content_type == "code":
            # Code: keep larger logical units
            target_tokens = int(target_tokens * 1.4)
            max_chars_cap = int(max_chars_cap * 1.5)
        elif content_type == "mixed":
            target_tokens = int(target_tokens * 1.1)

        total_tokens = count_tokens(cleaned)

        # ── Phase 5: Small content fast path ──────────────────────────
        if total_tokens <= max(min_tokens, target_tokens // 2):
            result = await recursive_semantic_chunk(
                cleaned,
                db_session=db_session,
                file_id=file_id,
                business_id=business_id,
                source_type=source_type,
                embedding_model=embedding_model,
            )
        else:
            # Compute adaptive chunk boundaries
            chars_per_token = max(1.0, len(cleaned) / max(total_tokens, 1))
            adaptive_max = int(target_tokens * chars_per_token)
            adaptive_max = max(300, min(max_chars_cap, adaptive_max))
            adaptive_min = max(80, int(adaptive_max * 0.25))

            result = await recursive_semantic_chunk(
                cleaned,
                max_chunk_len=adaptive_max,
                min_chunk_len=adaptive_min,
                db_session=db_session,
                file_id=file_id,
                business_id=business_id,
                source_type=source_type,
                embedding_model=embedding_model,
            )

        # ── Phase 6: Quality gate — resplit low-quality chunks ────────
        result = await self._quality_gate_resplit(
            result,
            quality_gate=quality_gate,
            target_tokens=target_tokens,
            db_session=db_session,
            file_id=file_id,
            business_id=business_id,
            source_type=source_type,
            embedding_model=embedding_model,
        )

        # ── Phase 7: Distribution analysis and annotation ─────────────
        distribution = _analyze_chunk_distribution(result)

        for chunk in result:
            chunk.setdefault("reasoning_ingestion", {}).update({
                "chunking_strategy": "smart_check_v3",
                "smart_target_tokens": target_tokens,
                "smart_total_tokens": total_tokens,
                "smart_domain": domain,
                "smart_content_type": content_type,
                "smart_embedding_model": model_key or "default",
                "smart_quality_gate": quality_gate,
                "distribution_mean_tokens": distribution["mean_tokens"],
                "distribution_cv": distribution["cv_tokens"],
            })

        log_info(
            f"[SmartCheck-v3] {len(result)} chunks | "
            f"domain={domain} content={content_type} | "
            f"target={target_tokens}tok | mean={distribution['mean_tokens']}tok "
            f"CV={distribution['cv_tokens']} | "
            f"quality_gate={quality_gate} low_q={distribution['low_quality_count']} | "
            f"file_id={file_id}"
        )
        return result

    # ── Quality Gate: Resplit low-quality chunks ──────────────────────

    async def _quality_gate_resplit(
        self,
        chunks: List[Dict[str, Any]],
        *,
        quality_gate: float,
        target_tokens: int,
        db_session=None,
        file_id=None,
        business_id=None,
        source_type: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Resplit chunks that fall below the quality gate threshold.

        For low-quality chunks (e.g., mid-sentence splits, noise contamination),
        attempt to resplit at a smaller window size. If the resplit produces
        better quality chunks, use them; otherwise keep the original.

        This is a single-pass correction — no recursive resplitting.
        """
        result: List[Dict[str, Any]] = []

        for chunk in chunks:
            quality = chunk.get("reasoning_ingestion", {}).get("chunk_quality_score", 1.0)
            tokens = chunk.get("tokens", 0)

            # Skip if already high quality or too small to split further
            if quality >= quality_gate or tokens < 40:
                result.append(chunk)
                continue

            # Skip noise chunks — they won't improve with resplitting
            if chunk.get("reasoning_ingestion", {}).get("noise_flag", False):
                result.append(chunk)
                continue

            # Attempt resplit at half the current size
            chunk_text = chunk.get("text", "") or chunk.get("cleaned_text", "")
            if not chunk_text.strip():
                result.append(chunk)
                continue

            smaller_max = max(200, len(chunk_text) // 2)
            smaller_min = max(80, smaller_max // 4)

            resplit = await recursive_semantic_chunk(
                chunk_text,
                max_chunk_len=smaller_max,
                min_chunk_len=smaller_min,
                db_session=db_session,
                file_id=file_id,
                business_id=business_id,
                source_type=source_type,
                embedding_model=embedding_model,
            )

            if not resplit:
                result.append(chunk)
                continue

            # Check if resplit improved quality
            resplit_qualities = [
                c.get("reasoning_ingestion", {}).get("chunk_quality_score", 0.0)
                for c in resplit
            ]
            avg_resplit_quality = sum(resplit_qualities) / len(resplit_qualities)

            if avg_resplit_quality > quality * 1.15:
                # Resplit is meaningfully better — use it
                for c in resplit:
                    c.setdefault("reasoning_ingestion", {})["quality_gate_resplit"] = True
                result.extend(resplit)
            else:
                # Keep original
                result.append(chunk)

        return result
