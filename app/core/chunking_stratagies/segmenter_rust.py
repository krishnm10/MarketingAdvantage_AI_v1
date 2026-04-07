# =============================================
# segmenter_rust.py — Enterprise Rust-Backed Chunker v3
#
# High-performance chunker with Rust extension integration:
#   - Primary: PyO3/maturin compiled Rust extension for O(N) chunking
#   - Fallback cascade: Rust → structure_aware → semantic (graceful degradation)
#   - Quality validation of Rust output (enterprise trust-but-verify)
#   - Chunk normalization: ensures Rust output meets pipeline schema
#   - Performance telemetry: timing, throughput, fallback tracking
#   - Configurable Rust extension parameters via env vars
#   - Health check for Rust extension availability
#
# Registered name: "rust"
#
# Env vars:
#   CHUNK_RUST_FALLBACK        — fallback strategy: structure_aware|semantic (default structure_aware)
#   CHUNK_RUST_QUALITY_GATE    — minimum avg quality from Rust output (default 0.35)
#   CHUNK_RUST_MAX_CHUNK_CHARS — pass to Rust extension as max chars (default 800)
#   CHUNK_RUST_TELEMETRY       — enable performance telemetry logging (default true)
# =============================================

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from app.core.chunking_stratagies.chunking_registry import Chunker, register_chunker
from app.core.chunking_stratagies.segmenter_v2 import (
    make_chunk_dict,
    recursive_semantic_chunk,
    count_tokens,
)
from app.core.chunking_stratagies.text_preprocessor import preprocess_document_text
from app.core.chunking_stratagies.chunk_quality_scorer import score_chunk_quality
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


def _safe_bool_env(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("true", "1", "yes", "on")


# ─────────────────────────────────────────────────────────────────────────────
# RUST EXTENSION HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

_RUST_AVAILABLE: Optional[bool] = None
_RUST_VERSION: Optional[str] = None


def _check_rust_availability() -> bool:
    """
    Check if the Rust extension is available and functional.
    Caches result after first check to avoid repeated import attempts.
    """
    global _RUST_AVAILABLE, _RUST_VERSION
    if _RUST_AVAILABLE is not None:
        return _RUST_AVAILABLE

    try:
        import segmenter_rust as rust_ext  # type: ignore
        # Validate the extension has the expected interface
        if not hasattr(rust_ext, "chunk"):
            log_warning("[RustChunker] segmenter_rust module lacks 'chunk' function")
            _RUST_AVAILABLE = False
            return False

        _RUST_VERSION = getattr(rust_ext, "__version__", "unknown")
        _RUST_AVAILABLE = True
        log_info(f"[RustChunker] Rust extension available (version={_RUST_VERSION})")
        return True
    except ImportError:
        _RUST_AVAILABLE = False
        log_warning("[RustChunker] Rust extension not available (ImportError)")
        return False
    except Exception as exc:
        _RUST_AVAILABLE = False
        log_warning(f"[RustChunker] Rust extension check failed: {exc}")
        return False


def is_rust_chunker_available() -> bool:
    """Public API: check if Rust chunker is available without side effects."""
    return _check_rust_availability()


# ─────────────────────────────────────────────────────────────────────────────
# CHUNK VALIDATION & NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def _validate_rust_chunk(item: Any) -> bool:
    """
    Validate a single Rust output item meets minimum requirements.
    Accepts dict with text content or a non-empty string.
    """
    if isinstance(item, dict):
        text = item.get("text") or item.get("cleaned_text") or ""
        return bool(text.strip())
    if isinstance(item, str):
        return bool(item.strip())
    return False


def _normalize_rust_output(
    rust_output: list,
    *,
    db_session=None,
    file_id=None,
    business_id=None,
    source_type: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Normalize Rust extension output to match pipeline schema.

    The Rust extension may return:
    - List of dicts with text/cleaned_text/semantic_hash
    - List of strings
    - Mixed list

    This function normalizes all formats to make_chunk_dict() output,
    ensuring consistent schema for downstream processing.
    """
    chunks: List[Dict[str, Any]] = []
    kw = dict(
        db_session=db_session, file_id=file_id, business_id=business_id,
        source_type=source_type, embedding_model=embedding_model,
    )

    for item in rust_output:
        if not _validate_rust_chunk(item):
            continue

        if isinstance(item, dict):
            # Check if Rust produced a complete chunk dict
            if "semantic_hash" in item and "cleaned_text" in item:
                # Trust Rust's output but ensure all pipeline fields exist
                chunk = dict(item)
                chunk.setdefault("source_type", source_type)
                chunk.setdefault("embedding_model", embedding_model)
                chunk.setdefault("confidence", 1.0)
                chunk.setdefault("global_content_id", None)

                # Ensure quality score exists
                if "reasoning_ingestion" not in chunk:
                    cleaned = chunk.get("cleaned_text", "")
                    tokens = chunk.get("tokens", count_tokens(cleaned))
                    chunk["reasoning_ingestion"] = {}
                    chunk["reasoning_ingestion"]["chunk_quality_score"] = score_chunk_quality(
                        cleaned, tokens
                    )
                chunks.append(chunk)
            else:
                # Partial dict — extract text and run through make_chunk_dict
                text_val = item.get("text") or item.get("cleaned_text") or ""
                chunk = make_chunk_dict(text_val, **kw)
                if chunk:
                    chunks.append(chunk)
        else:
            # String output
            chunk = make_chunk_dict(str(item), **kw)
            if chunk:
                chunks.append(chunk)

    return chunks


def _validate_rust_quality(
    chunks: List[Dict[str, Any]],
    quality_gate: float,
) -> bool:
    """
    Validate that Rust output meets minimum quality threshold.

    Returns True if average chunk quality is above the gate.
    If Rust produces garbage (e.g., bad boundary detection), this
    triggers fallback to Python-based chunking.
    """
    if not chunks:
        return False

    qualities = [
        ch.get("reasoning_ingestion", {}).get("chunk_quality_score", 0.0)
        for ch in chunks
    ]
    avg_quality = sum(qualities) / len(qualities)
    return avg_quality >= quality_gate


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK CASCADE
# ─────────────────────────────────────────────────────────────────────────────

async def _fallback_structure_aware(
    text: str,
    *,
    db_session=None,
    file_id=None,
    business_id=None,
    source_type: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fallback to structure-aware chunker (preferred fallback)."""
    try:
        from app.core.chunking_stratagies.segmenter_structure_aware import StructureAwareChunker

        chunker = StructureAwareChunker()
        chunks = await chunker.chunk(
            text,
            db_session=db_session,
            file_id=file_id,
            business_id=business_id,
            source_type=source_type,
            embedding_model=embedding_model,
        )
        for ch in chunks:
            ch.setdefault("reasoning_ingestion", {})["chunking_strategy"] = "rust_fallback_structure_aware"
        return chunks
    except Exception as exc:
        log_warning(f"[RustChunker] Structure-aware fallback failed: {exc}")
        return []


async def _fallback_semantic(
    text: str,
    *,
    db_session=None,
    file_id=None,
    business_id=None,
    source_type: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Final fallback to basic semantic chunker."""
    chunks = await recursive_semantic_chunk(
        text,
        db_session=db_session,
        file_id=file_id,
        business_id=business_id,
        source_type=source_type,
        embedding_model=embedding_model,
    )
    for ch in chunks:
        ch.setdefault("reasoning_ingestion", {})["chunking_strategy"] = "rust_fallback_semantic"
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY REGISTRATION
# ─────────────────────────────────────────────────────────────────────────────

@register_chunker("rust")
class RustChunker(Chunker):
    """
    Enterprise Rust-Backed Chunker v3.

    Advanced features:
    - Cached Rust extension availability check
    - Quality validation of Rust output (trust-but-verify)
    - Graceful degradation cascade: Rust → structure_aware → semantic
    - Performance telemetry with timing metrics
    - Configurable Rust extension parameters
    - Schema normalization for mixed Rust output formats
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
        # ── Configuration ─────────────────────────────────────────────
        fallback_strategy = os.getenv("CHUNK_RUST_FALLBACK", "structure_aware").strip().lower()
        quality_gate = _safe_float_env("CHUNK_RUST_QUALITY_GATE", 0.35, 0.0)
        max_chunk_chars = _safe_int_env("CHUNK_RUST_MAX_CHUNK_CHARS", 800, 100)
        telemetry = _safe_bool_env("CHUNK_RUST_TELEMETRY", True)

        kw = dict(
            db_session=db_session, file_id=file_id,
            business_id=business_id, source_type=source_type,
            embedding_model=embedding_model,
        )

        # ── Phase 1: Pre-process text ─────────────────────────────────
        preprocessed = preprocess_document_text(text or "")
        if not preprocessed.strip():
            return []

        # ── Phase 2: Attempt Rust extension ───────────────────────────
        if _check_rust_availability():
            try:
                chunks = await self._run_rust_chunker(
                    preprocessed,
                    max_chunk_chars=max_chunk_chars,
                    quality_gate=quality_gate,
                    telemetry=telemetry,
                    **kw,
                )
                if chunks:
                    return chunks
                # Rust returned no valid chunks — fall through to fallback
                log_warning(
                    f"[RustChunker] Rust extension returned no valid chunks, "
                    f"falling back to {fallback_strategy} | file_id={file_id}"
                )
            except Exception as exc:
                log_warning(f"[RustChunker] Rust execution failed ({exc})")

        # ── Phase 3: Fallback cascade ─────────────────────────────────
        return await self._execute_fallback(
            preprocessed, fallback_strategy, telemetry=telemetry, **kw,
        )

    async def _run_rust_chunker(
        self,
        text: str,
        *,
        max_chunk_chars: int,
        quality_gate: float,
        telemetry: bool,
        db_session=None,
        file_id=None,
        business_id=None,
        source_type: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute Rust chunker with validation and telemetry."""
        import segmenter_rust as rust_ext  # type: ignore

        start_time = time.perf_counter()

        # Call Rust extension
        if hasattr(rust_ext, "chunk_with_params"):
            rust_output = rust_ext.chunk_with_params(text, max_chars=max_chunk_chars)
        else:
            rust_output = rust_ext.chunk(text)

        rust_elapsed = time.perf_counter() - start_time

        if not isinstance(rust_output, list):
            raise TypeError(f"segmenter_rust.chunk() returned {type(rust_output)}, expected list")

        # Normalize and validate
        kw = dict(
            db_session=db_session, file_id=file_id, business_id=business_id,
            source_type=source_type, embedding_model=embedding_model,
        )
        chunks = _normalize_rust_output(rust_output, **kw)

        if not chunks:
            return []

        # Quality validation
        if not _validate_rust_quality(chunks, quality_gate):
            qualities = [
                ch.get("reasoning_ingestion", {}).get("chunk_quality_score", 0.0)
                for ch in chunks
            ]
            avg_q = sum(qualities) / len(qualities)
            log_warning(
                f"[RustChunker] Rust output below quality gate "
                f"(avg={avg_q:.3f} < gate={quality_gate}) | "
                f"file_id={file_id}"
            )
            return []  # Trigger fallback

        # Annotate all chunks
        for ch in chunks:
            ch.setdefault("reasoning_ingestion", {}).update({
                "chunking_strategy": "rust_v3",
                "rust_extension_version": _RUST_VERSION or "unknown",
                "rust_max_chunk_chars": max_chunk_chars,
            })

        if telemetry:
            total_chars = len(text)
            throughput = total_chars / rust_elapsed if rust_elapsed > 0 else 0

            qualities = [
                ch.get("reasoning_ingestion", {}).get("chunk_quality_score", 0.0)
                for ch in chunks
            ]
            avg_quality = sum(qualities) / len(qualities) if qualities else 0.0

            log_info(
                f"[RustChunker] Rust completed: {len(chunks)} chunks from "
                f"{total_chars:,} chars in {rust_elapsed*1000:.1f}ms "
                f"({throughput:,.0f} chars/sec) | "
                f"avg_quality={avg_quality:.3f} | "
                f"file_id={file_id}"
            )

        return chunks

    async def _execute_fallback(
        self,
        text: str,
        fallback_strategy: str,
        *,
        telemetry: bool,
        db_session=None,
        file_id=None,
        business_id=None,
        source_type: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute fallback cascade with telemetry."""
        kw = dict(
            db_session=db_session, file_id=file_id,
            business_id=business_id, source_type=source_type,
            embedding_model=embedding_model,
        )

        start_time = time.perf_counter()

        # Try preferred fallback first
        if fallback_strategy == "structure_aware":
            chunks = await _fallback_structure_aware(text, **kw)
            if chunks:
                if telemetry:
                    elapsed = time.perf_counter() - start_time
                    log_info(
                        f"[RustChunker] Fallback (structure_aware): "
                        f"{len(chunks)} chunks in {elapsed*1000:.1f}ms | "
                        f"file_id={file_id}"
                    )
                return chunks

        # Final fallback: semantic
        chunks = await _fallback_semantic(text, **kw)
        if telemetry:
            elapsed = time.perf_counter() - start_time
            log_info(
                f"[RustChunker] Fallback (semantic): "
                f"{len(chunks)} chunks in {elapsed*1000:.1f}ms | "
                f"file_id={file_id}"
            )
        return chunks
